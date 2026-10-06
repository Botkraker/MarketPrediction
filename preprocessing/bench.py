"""ADR-001 P0: one test bench for every phase, and the controls that prove it works.

Generalises h3.walk_forward_proba and h3.compare (ADR-001 §8, P0 card). h3.py stays
untouched: it is the frozen implementation of PREREG_H3 and must keep reproducing its
numbers. test_bench.py checks that walk_forward below reproduces h3.walk_forward_proba
and baseline.walk_forward exactly before it is trusted with anything new.

WHY THE NULL IS k RANDOM FEATURES, NOT ZERO (ADR T2)
H1 and H3a both reported "news makes the forecast worse". With no true effect, every
fitted parameter still costs out-of-sample loss, so a nested comparison read against
zero calls estimation noise "harm". Every nested comparison here is also read against
the same model with k random features in place of the k news features (k = the arm's
size, 100 seeds), and MSE targets get Clark-West, which corrects the same bias.

The random features must drift as slowly as the news does. Run 1 (2026-10-06) used iid
N(0,1) columns and failed G0: news moved a year or more out of date "beat" them in 19 of
100 runs, because slow-moving series beat white noise by luck. Since then (owner's
choice, 2026-10-06) each random block is a phase-randomised surrogate of the real news
block, and the planted signal is calibrated on 50 runs (10 overshot: median 4.73, not 2.8).
Run 2 still failed by one (10 false finds, limit 9), and a planted signal swung 1.76x the
surrogate spread. Since then (owner, 2026-10-06) the yardstick for log-loss and QLIKE is
that spread combined with the 20-session block-bootstrap SE of the same comparison
(1.80x on the design window); the ADR's P1 card names both. MSE keeps Clark-West.

WHEN THE HARNESS IS TRUSTED (gate G0; bands fixed by the owner 2026-10-05, before any run)
  positive control   a feature planted at the MDE is found in 72-88 of 100 runs
  negative controls  labels shuffled in 20-session blocks, and news moved 250+ sessions
                     away, show a difference in at most 9 of 100 runs
Every test is two-sided at 5%, the setting MDE = 2.8 x SE assumes (80% power).

THE SEAL (owner's answer to ADR Q1, 2026-10-05)
`seal` drops index sessions from 2024-01-01 and firm sessions from 2021-01-01 (P2 card),
and blanks targets that would read them, unless --confirm is given AND the git tag
prereg-<phase>-v1 exists. H1-H3 replications are exempt and do not call it.

    OMP_NUM_THREADS=1 python3 preprocessing/bench.py --jobs 10   # P0 -> data/curated/p0_results.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import h3
from baseline import walk_forward as h1_walk_forward
from features import DEFAULT_CALENDAR, DEFAULT_PRICES, load_prices, phantom_sessions
from hypothesis_tests import (BASELINE_FEATURES, DEFAULT_INPUT as H1_INPUT,
                              DEFAULT_OUTPUT as H1_OUTPUT, diebold_mariano, holm,
                              paired_test, required_columns)

OUTPUT = h3.CURATED / "p0_results.json"
SEAL, FIRM_SEAL, FIRM_END = "2024-01-01", "2021-01-01", "2022-12-30"
MIN_TRAIN, REFIT, N_BOOT = 500, 20, 2000            # as H1 and H3
SEED, SEEDS = 20261005, 100                         # ADR P0: >= 100 seeds
CALIBRATION_RUNS = 50                               # owner, 2026-10-06 (run 1: 10)
POWER_BAND, FALSE_MAX = (72, 88), 9                 # runs out of SEEDS; owner, 2026-10-05
Z = float(stats.norm.isf(0.025))                    # two-sided 5%


# ------------------------------------------------------------------ the seal
def tag_exists(tag: str) -> bool:
    out = subprocess.run(["git", "tag", "-l", tag], cwd=h3.ROOT, capture_output=True, text=True)
    return out.stdout.strip() == tag


def seal(frame: pd.DataFrame, phase: str, confirm: bool = False, boundary: str = SEAL,
         forward: dict[str, int] | None = None) -> pd.DataFrame:
    """Sessions >= boundary exist only in the one confirmation run. `forward` names target
    columns that look h sessions ahead: on the last h design rows they would read a sealed
    session, so they are blanked."""
    if confirm:
        if not tag_exists(f"prereg-{phase}-v1"):
            raise SystemExit(f"--confirm needs the git tag prereg-{phase}-v1, which does not exist")
        return frame
    cut = pd.Timestamp(boundary)
    out = frame[frame.session < cut].reset_index(drop=True)
    if (frame.session >= cut).any():
        for col, h in (forward or {}).items():
            out.loc[out.index[-h:], col] = np.nan
    return out


# ------------------------------------------------------------------ walk-forward
def residualiser(col: str, controls: list[str]):
    """In-fold transform: `col` becomes its OLS residual on `controls`, coefficients fitted
    on the training rows only. The controls need not be model features: on a model feature
    this is a no-op, which is why H1's and H3's orthogonal arms were redundant (§H3)."""
    def fit(d: pd.DataFrame, rows: slice) -> pd.DataFrame:
        A = np.c_[np.ones(len(d)), d[controls].to_numpy(float)]
        v = d[col].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A[rows], v[rows], rcond=None)
        return d.assign(**{col: v - A @ beta})
    fit.needs = list(controls)
    return fit


def walk_forward(frame: pd.DataFrame, features: list[str], target: str, h: int = 1,
                 kind: str = "regress", transform=None, min_train: int = MIN_TRAIN,
                 refit: int = REFIT) -> pd.DataFrame:
    """Expanding walk-forward: rows j < i - max(5, h) train the model that predicts row i,
    so every training target is observed before row i is predicted (embargo >= horizon).

    kind  regress   OLS forecast of `target`                          (H1: next return)
          logvar    OLS on log(target), forecast exp(fit) x smearing   (P1: variance)
          terciles  drop / flat / rise from in-fold terciles           (H3a)
          sign      down / up, binary logistic                         (P3)
    Everything that learns from data is refitted on the training slice at each refit: the
    scaler, the tercile cut points, the smearing factor and transform(frame, rows) (the
    residualiser now; a vectoriser, token screen or PCA later)."""
    embargo = max(5, h)
    need = list(dict.fromkeys(features + [target] + getattr(transform, "needs", [])))
    data = frame.dropna(subset=need).reset_index(drop=True)
    r = data[target].to_numpy(float)
    rows, model = [], None
    for i in range(min_train, len(data)):
        if model is None or (i - min_train) % refit == 0:
            cut = i - embargo
            X = (transform(data, slice(0, cut)) if transform else data)[features].to_numpy(float)
            if kind in ("regress", "logvar"):
                y = np.log(r[:cut]) if kind == "logvar" else r[:cut]
                model = make_pipeline(StandardScaler(), LinearRegression()).fit(X[:cut], y)
                smear = np.exp(y - model.predict(X[:cut])).mean() if kind == "logvar" else 1.0
            else:
                q = np.percentile(r[:cut], [100 / 3, 200 / 3])
                label = ((lambda v, q=q: h3.tercile_class(v, q)) if kind == "terciles"
                         else (lambda v: (v > 0).astype(int)))
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
                                      ).fit(X[:cut], label(r[:cut]))
        row = {"session": data.session.iloc[i]}
        if kind in ("regress", "logvar"):
            f = float(model.predict(X[i:i + 1])[0])
            row |= {"y": r[i], "pred": np.exp(f) * smear if kind == "logvar" else f}
        else:
            p = np.zeros(3 if kind == "terciles" else 2)   # a class absent from training gets 0
            p[model.classes_] = model.predict_proba(X[i:i + 1])[0]
            row |= {"y": int(label(r[i:i + 1])[0])} | {f"p{c}": v for c, v in enumerate(p)}
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ evaluation
def losses(pred: pd.DataFrame, loss: str) -> np.ndarray:
    y = pred.y.to_numpy()
    if loss == "mse":
        return (pred.pred.to_numpy() - y) ** 2
    if loss == "qlike":                                 # Patton (2011); P1's primary loss
        q = y / pred.pred.to_numpy()
        return q - np.log(q) - 1
    P = pred.filter(regex=r"^p\d$").to_numpy()
    return -np.log(np.clip(P[np.arange(len(P)), y.astype(int)], 1e-12, 1))


def hac_se(d: np.ndarray, lags: int) -> float:
    """Newey-West standard error of mean(d), with h3.compare's lag rule."""
    dbar = d.mean()
    var = ((d - dbar) ** 2).mean()
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
    return float(np.sqrt(max(var, 0.0) / len(d)))


def clark_west(treat: pd.DataFrame, base: pd.DataFrame, h: int) -> dict:
    """Clark & West (2007) for nested MSE comparisons: the bigger model is credited back the
    squared gap between the two forecasts, the part of its loss that is estimation noise
    under the null. Positive statistic: the bigger model is better."""
    y, b, t = base.y.to_numpy(), base.pred.to_numpy(), treat.pred.to_numpy()
    f = (y - b) ** 2 - ((y - t) ** 2 - (b - t) ** 2)
    se = hac_se(f, int(4 * (len(f) / 100) ** (2 / 9)) + h - 1)
    z = float(f.mean() / se)
    return {"cw_stat": z, "cw_p": float(2 * stats.norm.sf(abs(z))), "cw_mde": 2.8 * se}


def against(d: float, null: np.ndarray, se_boot: float = 0.0) -> dict:
    """A loss differential read against the matched-noise null (ADR T2): centred on the
    surrogates' mean loss, scaled by their spread combined with the comparison's own
    block-bootstrap SE (run 2). z < 0: the block beats random series of the same drift;
    z > 0: it does worse than they would."""
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    scale = float(np.hypot(sd, se_boot))
    z = (d - mu) / scale
    return {"noise_mean": mu, "noise_sd": sd, "se_boot_used": se_boot, "z_noise": z,
            "p_noise": float(2 * stats.norm.sf(abs(z))), "mde_noise": 2.8 * scale,
            "noise_runs_at_or_below": int((null <= d).sum()), "noise_runs": len(null)}


def compare(treat: pd.DataFrame, base: pd.DataFrame, h: int = 1, loss: str = "logloss",
            null: np.ndarray | None = None, n_boot: int = N_BOOT) -> dict:
    """h3.compare generalised: loss differential (treatment - baseline), circular block
    bootstrap (block max(20, 2h)), MDE = 2.8 x SE, Clark-West for MSE, and the
    matched-noise reading when a null is given. n_boot=0 skips the bootstrap."""
    if not treat.session.equals(base.session):
        raise ValueError("arms not aligned on the same sessions")
    d = losses(treat, loss) - losses(base, loss)
    out = {"n": len(d), "d": float(d.mean())}
    if n_boot:
        block, rng = max(20, 2 * h), np.random.default_rng(SEED)
        boot = np.array([d[h3.block_indices(len(d), block, rng)].mean() for _ in range(n_boot)])
        out |= {"ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "block": block,
                "p_boot": float(np.mean(np.abs(boot - boot.mean()) >= abs(out["d"]))),
                "se_boot": float(boot.std()), "mde_boot": 2.8 * float(boot.std())}
    if loss == "mse":
        out |= clark_west(treat, base, h)
    if null is not None:
        out |= against(out["d"], null, out.get("se_boot", 0.0))
    return out


def boots(loss: str) -> int:
    """Control runs need the bootstrap only where it enters the decision (not for MSE)."""
    return 0 if loss == "mse" else N_BOOT


def score(res: dict, loss: str) -> float:
    """The decision statistic in SE units, positive when the block beats the baseline
    beyond estimation noise: Clark-West for MSE, the matched-noise z otherwise."""
    return res["cw_stat"] if loss == "mse" else -res["z_noise"]


# ------------------------------------------------------------------ matched noise and controls
_TASK = None


def _call(x):
    return _TASK(x)


def pmap(fn, items, jobs: int = 1) -> list:
    """Map over fork()ed workers that inherit `fn`, so closures are never pickled. (joblib
    1.1.1, the version installed here, hangs under Python 3.12.) Workers run jobs=1."""
    global _TASK
    if jobs == 1:
        return [fn(x) for x in items]
    _TASK = fn
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(_call, list(items))


def surrogate(block: np.ndarray, seed: int) -> np.ndarray:
    """Phase-randomised copy of a (rows x k) block. Each column keeps its power spectrum,
    i.e. how slowly it drifts, and shared phases keep the correlation between columns;
    values are then mapped back by rank onto the column's real values, so the distribution
    is exact (a 0/1 flag stays 0/1). Only its timing against the target is random."""
    n = len(block)
    spec = np.fft.rfft(block - block.mean(0), axis=0)
    phase = np.exp(2j * np.pi * np.random.default_rng(seed).random(len(spec)))
    phase[0] = 1
    if n % 2 == 0:
        phase[-1] = 1                                   # the Nyquist term must stay real
    s = np.fft.irfft(spec * phase[:, None], n=n, axis=0)
    return np.take_along_axis(np.sort(block, 0), s.argsort(0).argsort(0), 0)


def with_surrogate(frame: pd.DataFrame, news: list[str], seed: int, k: int | None = None,
                   prefix: str = "noise"):
    block = frame[news].to_numpy(float)
    if np.isnan(block).any():
        raise ValueError(f"news block {news} has missing values")
    s = surrogate(block, seed)[:, :k]
    cols = [f"{prefix}{j}" for j in range(s.shape[1])]
    return frame.assign(**{c: s[:, j] for j, c in enumerate(cols)}), cols


def noise_null(frame, features, news, fit, loss, base, seeds=SEEDS, offset=0, jobs=1) -> np.ndarray:
    """ADR P0 matched noise: the baseline plus a surrogate of the news block (same size,
    same drift), one walk-forward per seed. Returns mean(loss - baseline loss) per seed."""
    def one(s):
        f, cols = with_surrogate(frame, news, SEED + offset + s)
        return compare(fit(f, features + cols), base, loss=loss, n_boot=0)["d"]
    return np.array(pmap(one, range(seeds), jobs))


def positive_control(frame, features, news, fit, target, h, loss, null, jobs=1, log=False) -> dict:
    """Plant z ~ N(0,1) in a COPY of the data (target += beta x sd(target) x z) and give z
    to the model, padded with k-1 surrogate columns so the arm keeps its size. beta is set
    by bisection so the median statistic over CALIBRATION_RUNS runs is 2.8, i.e. the effect
    sits at the MDE; 100 fresh runs must then find it in POWER_BAND of them, which holds
    only if the harness's standard error is right. log=True plants on the log scale, so a
    variance target stays positive (P1)."""
    sd = (np.log(frame[target]) if log else frame[target]).std()

    def one(beta, seed):
        f = frame.assign(planted=np.random.default_rng(seed).standard_normal(len(frame)))
        shift = beta * sd * f.planted
        f[target] = f[target] * np.exp(shift) if log else f[target] + shift
        f, pad = with_surrogate(f, news, seed + 500_000, k=len(news) - 1, prefix="pad")
        return score(compare(fit(f, features + ["planted"] + pad), fit(f, features), h, loss,
                             null, n_boot=boots(loss)), loss)

    lo, hi = 0.005, 1.0
    for _ in range(8):
        mid = float(np.sqrt(lo * hi))
        med = np.median(pmap(lambda s: one(mid, SEED + 10_000 + s), range(CALIBRATION_RUNS), jobs))
        lo, hi = (mid, hi) if med < 2.8 else (lo, mid)
    beta = float(np.sqrt(lo * hi))
    stat = np.array(pmap(lambda s: one(beta, SEED + 20_000 + s), range(SEEDS), jobs))
    found = int((stat > Z).sum())
    return {"beta": beta, "found": found, "of": SEEDS, "band": list(POWER_BAND),
            "pass": POWER_BAND[0] <= found <= POWER_BAND[1], "median_stat": float(np.median(stat))}


def block_permute(v: np.ndarray, block: int, rng) -> np.ndarray:
    """Whole blocks of `block` sessions in random order: no link to the features is left,
    short-range dependence inside each block is kept."""
    blocks = np.array_split(v, np.arange(block, len(v), block))
    return np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])


def negative_controls(frame, features, news, fit, target, h, loss, base, null, jobs=1) -> dict:
    """ADR P0: two ways to make the news meaningless, 100 runs each. Permuted labels get
    their own matched-noise null per run; stale news (circularly at least 250 sessions
    from its own date, both ways) is read against the real-label null."""
    step = max(1, (len(frame) - 500) // SEEDS)

    def permuted(s):
        f = frame.copy()
        m = f[target].notna().to_numpy()
        f.loc[m, target] = block_permute(f.loc[m, target].to_numpy(), 20,
                                         np.random.default_rng(SEED + 30_000 + s))
        b = fit(f, features)
        nul = None if loss == "mse" else noise_null(f, features, news, fit, loss, b,
                                                    offset=100_000 + 1000 * s)
        return score(compare(fit(f, features + news), b, h, loss, nul, n_boot=boots(loss)), loss)

    def stale(s):
        f = frame.copy()
        f[news] = np.roll(f[news].to_numpy(), 250 + step * s, axis=0)
        return score(compare(fit(f, features + news), base, h, loss, null, n_boot=boots(loss)), loss)

    out = {}
    for name, fn in (("permuted_labels", permuted), ("stale_news", stale)):
        stat = np.array(pmap(fn, range(SEEDS), jobs))
        found = int((np.abs(stat) > Z).sum())
        out[name] = {"found": found, "of": SEEDS, "max": FALSE_MAX, "pass": found <= FALSE_MAX,
                     "found_better": int((stat > Z).sum()), "found_worse": int((stat < -Z).sum())}
    return out


def controls(frame, features, news, fit, target, h, loss, jobs, log=False) -> dict:
    base = fit(frame, features)
    null = noise_null(frame, features, news, fit, loss, base, jobs=jobs)
    f, cols = with_surrogate(frame, news, SEED)          # bootstrap SE of a null comparison
    se = compare(fit(f, features + cols), base, h, loss)["se_boot"]
    return {"n": len(base), "news": news,
            "matched_noise": {"k": len(news), "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                              "se_boot_probe": se, "mde": 2.8 * float(np.hypot(null.std(ddof=1), se))},
            "positive": positive_control(frame, features, news, fit, target, h, loss, null, jobs, log),
            **negative_controls(frame, features, news, fit, target, h, loss, base, null, jobs)}


# ------------------------------------------------------------------ H1 on v3 (replication: exempt from the seal)
H1_ARMS = ["all", "ex_price", "placebo", "orthogonal"]
ORTHO_CONTROLS = ["abs_ret_lag0", "range_pct_lag0", "log_headlines_lag0"]   # owner, 2026-10-05


def h1_frame(scored=h3.SCORED_V3, start: str = h3.START) -> pd.DataFrame:
    """H1's columns (features._add_sentiment) from the v3 scores on the phantom-free
    calendar, built in memory: features.build writes data/ and keeps phantom sessions.
    log_headlines counts scored headlines, as h3 does."""
    f = h3.index_frame().merge(load_prices()[["session", "range_pct"]], on="session", how="left")
    f["abs_ret_lag0"], f["range_pct_lag0"] = f.ret_lag0.abs(), f.range_pct
    news = h3.headlines(scored)
    f["log_headlines_lag0"] = np.log1p(f.session.map(news.groupby("session").size()).fillna(0))
    for suffix, sub in (("", news), ("_ex_price", news[~news.is_price_report]),
                        ("_price_only", news[news.is_price_report])):
        g = sub.assign(pos=sub.sent_score > 0, neg=sub.sent_score < 0).groupby("session")
        agg = pd.DataFrame({f"sent_mean{suffix}_lag1": g.sent_score.mean(),
                            f"sent_pos_share{suffix}_lag1": g.pos.mean(),
                            f"sent_neg_share{suffix}_lag1": g.neg.mean()})
        f = f.merge(agg, left_on="session", right_index=True, how="left")
        f[f"has_sent{suffix}_lag1"] = f[f"sent_mean{suffix}_lag1"].notna().astype(float)
        f[list(agg.columns)] = f[list(agg.columns)].fillna(0.0)
    return f[f.session >= pd.Timestamp(start)].reset_index(drop=True)


def h1_arm(arm: str):
    if arm == "orthogonal":
        return ["sent_mean_lag1"], residualiser("sent_mean_lag1", ORTHO_CONTROLS)
    return required_columns(arm), None


def as_h1(p: pd.DataFrame) -> pd.DataFrame:
    """bench output -> the columns hypothesis_tests.paired_test / diebold_mariano read."""
    return p.assign(y_true=(p.y > 0).astype(int), y_pred=(p.pred > 0).astype(int),
                    score=p.pred, ret_next=p.y)


def run_h1_v3(jobs: int) -> dict:
    """ADR P0 gap: H1 as pre-registered (OLS on the next return, ret_lag0..2, four arms,
    McNemar and Diebold-Mariano, Holm), on v3 scores from 2016, with the orthogonal arm
    residualised in-fold on controls that are not in the model. Clark-West and the
    matched-noise reading are added; Clark-West decides for this MSE target."""
    frame = h1_frame()
    fit = lambda f, cols: walk_forward(f, cols, "fwd1")
    base = fit(frame, BASELINE_FEATURES)
    auc0 = roc_auc_score(base.y > 0, base.pred)
    arms = {}
    for arm in H1_ARMS:
        cols, tf = h1_arm(arm)
        t = walk_forward(frame, BASELINE_FEATURES + cols, "fwd1", transform=tf)
        auc = roc_auc_score(t.y > 0, t.pred)
        # the orthogonal arm's null is a surrogate of the raw column (descriptive: CW decides)
        null = noise_null(frame, BASELINE_FEATURES, cols, fit, "mse", base, jobs=jobs)
        arms[arm] = {**compare(t, base, 1, "mse", null), "k": len(cols),
                     "auc": auc, "d_auc": auc - auc0, "tripwire_auc_gain": bool(auc - auc0 > 0.02),
                     "mcnemar": paired_test(as_h1(t), as_h1(base)),
                     "dm": diebold_mariano(as_h1(t), as_h1(base))}
    holm_p = {name: holm({a: get(arms[a]) for a in arms}) for name, get in (
        ("cw", lambda v: v["cw_p"]), ("mcnemar", lambda v: v["mcnemar"]["p_value"]),
        ("dm", lambda v: v["dm"]["p_value"]))}
    return {"window": f"{frame.session.min().date()}..{frame.session.max().date()}",
            "n": len(base), "baseline_auc": auc0, "arms": arms, "holm": holm_p,
            "orthogonal_controls": ORTHO_CONTROLS}


def reread_h1(jobs: int) -> dict:
    """ADR P0 gap (T2): the original H1 (v2 scores, 2014+, daily_features.parquet as
    committed), rerun with its own walk-forward, read against matched noise and Clark-West."""
    frame = pd.read_parquet(H1_INPUT)
    fit = lambda f, cols: h1_walk_forward(f, cols, kind="regress").rename(
        columns={"score": "pred", "ret_next": "y"})
    base = fit(frame, BASELINE_FEATURES)
    stored = json.loads(H1_OUTPUT.read_text())["h1"]
    out = {}
    for arm in H1_ARMS:
        cols = required_columns(arm)
        null = noise_null(frame, BASELINE_FEATURES, cols, fit, "mse", base, jobs=jobs)
        out[arm] = {**compare(fit(frame, BASELINE_FEATURES + cols), base, 1, "mse", null),
                    "stored_dm_mean_loss_differential":
                        stored[arm]["vs_baseline_continuous"]["mean_loss_differential"]}
    return out


def reread_h3a(jobs: int) -> dict:
    """ADR P0 gap (T2): H3a's stored log-loss differentials against 3 random features on the
    same frame and the same harness (h3.walk_forward_proba, unchanged)."""
    frame = h3.direction_frame()
    fit = lambda f, cols: h3.walk_forward_proba(f, cols, 1)
    base = fit(frame, h3.F0)
    null = noise_null(frame, h3.F0, h3.NEWS, fit, "logloss", base, jobs=jobs)
    stored = json.loads(h3.OUTPUT.read_text())["H3a"]["arms"]
    # H3's own block-bootstrap SE for each arm: its stored mde_80 / 2.8
    return {arm: {"stored_d_log_loss": v["d_log_loss"], **against(v["d_log_loss"], null, v["mde_80"] / 2.8)}
            for arm, v in stored.items()}


# ------------------------------------------------------------------ MDE table (design windows only)
P1_BASE = ["har1", "har5", "har22", "abs_ret_lag0", "dow_next_mon", "dow_next_tue",
           "dow_next_wed", "dow_next_thu", "gap_next"]


def calendar_count(start: str, end: str | None = None) -> int:
    """Sessions in the raw trading calendar: dates only, no price is read."""
    cal = pd.read_csv(DEFAULT_CALENDAR, parse_dates=["session_date"]).session_date
    return int(((cal >= start) & (cal <= (end or cal.max()))).sum())


def p1_frame() -> pd.DataFrame:
    """Provisional P1 target and baseline, built only for the MDE table (the P1 prereg fixes
    both). Target: next-session Parkinson variance (ln H/L)^2 / (4 ln 2), no open. HAR on log
    range: last session, 5- and 22-session means, with a 1bp floor because some sessions
    have high == low; those sessions have no variance to score and are not targets."""
    px = pd.read_csv(DEFAULT_PRICES, encoding="utf-8-sig")
    px["session"] = pd.to_datetime(px.date).dt.normalize()
    px = px[~px.session.isin(phantom_sessions())].sort_values("session").reset_index(drop=True)
    hl = np.log(px.high / px.low)
    lr = np.log(hl.clip(lower=1e-4))
    nxt = px.session.shift(-1)
    pv_next = (hl ** 2 / (4 * np.log(2))).shift(-1)
    f = pd.DataFrame({"session": px.session, "har1": lr, "har5": lr.rolling(5).mean(),
                      "har22": lr.rolling(22).mean(), "abs_ret_lag0": px.close.pct_change().abs(),
                      "gap_next": (nxt - px.session).dt.days, "pv_next": pv_next.where(pv_next > 0)})
    for d, name in enumerate(["mon", "tue", "wed", "thu"]):
        f[f"dow_next_{name}"] = (nxt.dt.dayofweek == d).astype(float)
    return f[f.session >= pd.Timestamp(h3.START)].reset_index(drop=True)


def firm_panel(ks=(1, 5)) -> pd.DataFrame:
    """Provisional P2 / P3a unit table, built only for the MDE table (the P2 prereg fixes it).
    P2 card: traded days only (ALL_DATA.csv, duplicates dropped, not the carried-forward H3c
    panel); trade-to-trade return; issuer headline dated after the previous trade and up to
    this one, price reports excluded; market-adjusted return over the next k trades. Prices
    from FIRM_SEAL on are blanked before anything is computed, so 2021-22 rows carry unit
    counts only. `gap` = index sessions to the firm's next trade."""
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv")
    a["session"] = pd.to_datetime(a.Date).dt.normalize()
    a = a.drop_duplicates(["Ticker", "session"], keep="last").sort_values(["Ticker", "session"])
    px = pd.read_csv(DEFAULT_PRICES, encoding="utf-8-sig")
    idx = pd.Series(px.close.to_numpy(float), index=pd.to_datetime(px.date).dt.normalize())
    idx = idx[idx.index < FIRM_SEAL]
    cal = h3.calendar()
    news = h3.headlines(h3.SCORED_V3)
    news = news[~news.is_price_report]
    low = news.headline_clean.astype(str).str.lower()
    parts = []
    for ticker, pat in h3.issuer_patterns().items():
        g = a[a.Ticker == ticker]
        if len(g) < 2:
            continue
        t = pd.DatetimeIndex(g.session)
        c = pd.Series(np.where(t < FIRM_SEAL, g.Close.to_numpy(float), np.nan))
        j = t.searchsorted(pd.DatetimeIndex(news.day[low.str.contains(pat)]), side="left")
        f = pd.DataFrame({"ticker": ticker, "session": t,
                          "news": np.bincount(j[j < len(t)], minlength=len(t)) > 0,
                          "r": (c / c.shift() - 1).to_numpy(),
                          "r_m": idx.reindex(t).to_numpy() / idx.reindex(pd.Series(t).shift()).to_numpy() - 1,
                          "gap": np.r_[np.diff(cal.searchsorted(t)), np.nan]})
        for k in ks:
            end = pd.Series(t).shift(-k)
            f[f"ar{k}"] = ((c.shift(-k) / c - 1)
                           - (idx.reindex(end).to_numpy() / idx.reindex(t).to_numpy() - 1)).to_numpy()
        parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    return p[(p.session >= pd.Timestamp(h3.START)) & (p.session <= FIRM_END)].reset_index(drop=True)


def p2_mde(panel: pd.DataFrame, k: int, jobs: int) -> dict:
    """SE of c in AR = a + b r + c r.news + d news + e r_m, date-clustered, with the news flag
    replaced by random flags at the real design rate (matched noise, 100 seeds)."""
    design = panel[(panel.session < FIRM_SEAL) & panel.r.notna()]
    d = design[(design.gap <= 5) & design[f"ar{k}"].notna() & design.r_m.notna()]
    rate, groups = float(d.news.mean()), pd.factorize(d.session)[0]

    def one(s):
        z = (np.random.default_rng(SEED + 60_000 + s).random(len(d)) < rate).astype(float)
        X = sm.add_constant(np.c_[d.r, d.r * z, z, d.r_m])
        res = sm.OLS(d[f"ar{k}"].to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        return res.params[2], res.bse[2]

    est = np.array(pmap(one, range(SEEDS), jobs))
    se = float(np.median(est[:, 1]))
    dates_confirm = calendar_count(FIRM_SEAL, FIRM_END)
    return {"test": f"P2 drift-vs-reversal term c, AR over {k} trade(s)", "unit": "firm trade day",
            "n_design": len(d), "dates_design": int(d.session.nunique()), "news_rate": rate,
            "dropped_no_trade_within_5": int((design.gap > 5).sum()),
            "se_design": se, "sd_c_across_noise": float(est[:, 0].std(ddof=1)),
            "mde_design": 2.8 * se, "dates_confirm": dates_confirm,
            "mde_confirm": 2.8 * se * np.sqrt(d.session.nunique() / dates_confirm),
            "window_design": f"{h3.START}..{FIRM_SEAL} (exclusive)", "window_confirm": f"{FIRM_SEAL}..{FIRM_END}"}


def mde_table(jobs: int) -> list[dict]:
    """ADR P0 output: the MDE of every planned primary test, design and confirm windows.
    Confirm-window MDEs scale the design SE by sqrt(design units / confirm units); confirm
    units are counted from dates and headline counts, never from sealed prices."""
    rows = []
    # P1's own news block needs novelty, which P1 builds; H3's 3-feature block stands in
    stand_in = h3.direction_frame()[["session"] + h3.NEWS]
    d = seal(p1_frame().merge(stand_in, on="session", how="left"), "p1", forward={"pv_next": 1})
    fit = lambda f, cols: walk_forward(f, cols, "pv_next", 1, "logvar")
    base = fit(d, P1_BASE)
    null = noise_null(d, P1_BASE, h3.NEWS, fit, "qlike", base, jobs=jobs)
    f, cols = with_surrogate(d, h3.NEWS, SEED)           # bootstrap SE of a null comparison,
    se = compare(fit(f, P1_BASE + cols), base, 1, "qlike")["se_boot"]   # never of real news
    scale = float(np.hypot(null.std(ddof=1), se))
    n_confirm = calendar_count(SEAL) - 1
    rows.append({"test": "P1 news block (3 features), d QLIKE vs HAR", "unit": "index session",
                 "null": "surrogates of H3's tone, has_news, log_n (stand-in until P1 builds its block)",
                 "n_design": len(base), "noise_mean": float(null.mean()), "se_boot_probe": se,
                 "mde_design": 2.8 * scale, "n_confirm": n_confirm,
                 "mde_confirm": 2.8 * scale * np.sqrt(len(base) / n_confirm),
                 "zero_range_sessions_dropped": int(d.pv_next.isna().sum() - 1),
                 "window_design": f"{h3.START}..{SEAL} (exclusive)", "window_confirm": f"{SEAL}.."})
    panel = firm_panel()
    rows += [p2_mde(panel, k, jobs) for k in (1, 5)]
    events = panel[panel.news]
    n3a = (int((events.session < FIRM_SEAL).sum()), int((events.session >= FIRM_SEAL).sum()))
    sessions = h3.headlines(h3.SCORED_V3)
    sessions = sessions[~sessions.is_price_report].session.drop_duplicates()
    n3b = (int(((sessions >= h3.START) & (sessions < SEAL)).sum()), int((sessions >= SEAL).sum()))
    for test, unit, (n_d, n_c) in (("P3a lexicon score, firm events", "firm event with issuer news", n3a),
                                   ("P3b lexicon score, index sessions", "session with non-price news", n3b)):
        rows.append({"test": test, "unit": unit, "method": "ADR T1: |r| >= 2.8 / sqrt(n), one pre-specified score",
                     "n_design": n_d, "mde_design": 2.8 / np.sqrt(n_d),
                     "n_confirm": n_c, "mde_confirm": 2.8 / np.sqrt(n_c)})
    return rows


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirm", action="store_true", help="read sealed sessions (needs the prereg tag)")
    ap.add_argument("--phase", default="p0")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--output", default=str(OUTPUT), help="each run keeps its own file")
    args = ap.parse_args()
    if args.confirm:
        seal(pd.DataFrame({"session": pd.to_datetime([])}), args.phase, confirm=True)
        raise SystemExit("P0 has no confirmation run")
    j = args.jobs
    out = {"plan": "ADR-001 P0", "seal": SEAL, "firm_seal": FIRM_SEAL, "seeds": SEEDS,
           "power_band": POWER_BAND, "false_max": FALSE_MAX, "null": "surrogate + block bootstrap",
           "calibration_runs": CALIBRATION_RUNS}

    print("controls: direction, log-loss (H3a setup, 2016-2023)...", flush=True)
    frame = seal(h3.direction_frame(), "p0", forward={"fwd1": 1, "fwd5": 5, "fwd20": 20})
    out["controls_logloss"] = controls(frame, h3.F0, h3.NEWS,
                                       lambda f, cols: walk_forward(f, cols, "fwd1", 1, "terciles"),
                                       "fwd1", 1, "logloss", j)
    print("controls: next return, MSE (H1 setup on v3, 2016-2023)...", flush=True)
    frame = seal(h1_frame(), "p0", forward={"fwd1": 1, "fwd5": 5, "fwd20": 20})
    out["controls_mse"] = controls(frame, BASELINE_FEATURES, required_columns("all"),
                                   lambda f, cols: walk_forward(f, cols, "fwd1"), "fwd1", 1, "mse", j)
    gates = [out[c][g]["pass"] for c in ("controls_logloss", "controls_mse")
             for g in ("positive", "permuted_labels", "stale_news")]
    out["g0_controls_pass"] = all(gates)

    print("H1 on v3 (replication)...", flush=True)
    out["h1_v3"] = run_h1_v3(j)
    print("H1 and H3a re-read against matched noise...", flush=True)
    out["h1_reread"], out["h3a_reread"] = reread_h1(j), reread_h3a(j)
    print("MDE table...", flush=True)
    out["mde_table"] = mde_table(j)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"g0_controls_pass": out["g0_controls_pass"]}), f"-> {args.output}")


if __name__ == "__main__":
    main()
