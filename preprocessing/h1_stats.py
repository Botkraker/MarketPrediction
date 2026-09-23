"""H1 statistics added by amendment prereg-h1-v1-a1 (blueprint 6.1-6.3).

Declared before any H1 run on the v2 (CamemBERT) scores. What this adds, and the
decision rule it fixes in advance:

  block bootstrap   paired, circular moving-block bootstrap of dAUC = AUC(arm) -
                    AUC(F0 baseline) over the walk-forward predictions. Blocks of
                    BLOCK_LENGTH sessions keep volatility clustering (+0.387) and
                    return autocorrelation inside each resample; an iid bootstrap
                    would understate the variance.
  per-block         dAUC in every 3-month test block (blueprint 6.1), and the share
                    of blocks where the arm beats F0.
  COVID-2020        dAUC in 2020 alone, and the main dAUC with 2020 removed, so a
                    single shock cannot carry the result (blueprint 6.3).
  purged k-fold     K contiguous folds, EMBARGO sessions purged on each side of the
                    test fold. A DIAGNOSTIC only: it trains on the future, so it
                    says whether the walk-forward number is an artefact of one
                    particular ordering, never whether H1 holds.

GO / NO-GO, per arm (fixed here, before the run):
  pass  <=>  Holm-adjusted bootstrap p < 0.05 across the four arms
             AND dAUC > 0 in at least 2/3 of the scorable 3-month blocks.
The verdict then mirrors the pre-registered ladder in hypothesis_tests._interpret:
ex_price and orthogonal must both pass for SUPPORTED. A null is reported as a
null; none of these constants may be changed after seeing the result.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from baseline import EMBARGO_SESSIONS, make_model

BLOCK_LENGTH = 20            # ~1 trading month
N_BOOT = 2000
SEED = 20260923
TEST_BLOCK = "QS"            # 3-month test blocks, blueprint 6.1
MIN_BLOCK_N = 30             # a block needs this many sessions (and both classes)
BLOCK_SHARE_REQUIRED = 2 / 3
ALPHA = 0.05
COVID_YEAR = 2020
K_FOLDS = 5


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def auc_rows(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ROC-AUC of every row of (y, s) at once, by the Mann-Whitney rank formula
    (ties get average ranks, as in roc_auc_score). NaN where a row has one class."""
    ranks = rankdata(s, axis=1)
    pos = y.sum(axis=1).astype(float)
    neg = y.shape[1] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = ((ranks * y).sum(axis=1) - pos * (pos + 1) / 2) / (pos * neg)
    return np.where((pos > 0) & (neg > 0), auc, np.nan)


def _paired(treat: pd.DataFrame, base: pd.DataFrame):
    if not treat.session.reset_index(drop=True).equals(base.session.reset_index(drop=True)):
        raise ValueError("Arms are not aligned on the same sessions; cannot pair.")
    return (treat.y_true.to_numpy(), treat.score.to_numpy(), base.score.to_numpy())


def block_indices(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular moving-block resamples of 0..n-1, shape (n_boot, n)."""
    starts = rng.integers(0, n, (n_boot, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def delta_auc(treat: pd.DataFrame, base: pd.DataFrame, block: int = BLOCK_LENGTH,
              n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    y, st, sb = _paired(treat, base)
    point = _auc(y, st) - _auc(y, sb)
    if len(y) < 2 * block:
        return {"n": int(len(y)), "delta_auc": round(point, 4), "skipped": "too few sessions"}
    idx = block_indices(len(y), block, n_boot, np.random.default_rng(seed))
    draws = auc_rows(y[idx], st[idx]) - auc_rows(y[idx], sb[idx])
    draws = draws[~np.isnan(draws)]
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"n": int(len(y)), "auc_arm": round(_auc(y, st), 4), "auc_f0": round(_auc(y, sb), 4),
            "delta_auc": round(point, 4), "ci95": [round(low, 4), round(high, 4)],
            "p_bootstrap": round(float(p), 4), "block_length": block, "n_boot": n_boot}


def per_block(treat: pd.DataFrame, base: pd.DataFrame, freq: str = TEST_BLOCK,
              min_n: int = MIN_BLOCK_N) -> dict:
    y, st, sb = _paired(treat, base)
    period = pd.to_datetime(treat.session).dt.to_period(freq[0]).astype(str).to_numpy()
    blocks = []
    for label in pd.unique(period):
        m = period == label
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        blocks.append({"block": label, "n": int(m.sum()),
                       "delta_auc": round(_auc(y[m], st[m]) - _auc(y[m], sb[m]), 4)})
    share = float(np.mean([b["delta_auc"] > 0 for b in blocks])) if blocks else float("nan")
    return {"blocks": blocks, "n_blocks": len(blocks),
            "share_positive": round(share, 4), "min_block_n": min_n}


def covid_split(treat: pd.DataFrame, base: pd.DataFrame, year: int = COVID_YEAR,
                **boot) -> dict:
    in_year = pd.to_datetime(treat.session).dt.year.eq(year).to_numpy()
    part = lambda m: delta_auc(treat[m].reset_index(drop=True),
                               base[m].reset_index(drop=True), **boot)
    return {f"only_{year}": part(in_year), f"excluding_{year}": part(~in_year)}


def purged_kfold(frame: pd.DataFrame, features: list[str], kind: str = "regress",
                 k: int = K_FOLDS, embargo: int = EMBARGO_SESSIONS) -> dict:
    data = frame.dropna(subset=features + ["ret_next"]).reset_index(drop=True)
    X, ret = data[features].to_numpy(), data["ret_next"].to_numpy()
    y = (ret > 0).astype(int)
    aucs = []
    for fold in np.array_split(np.arange(len(data)), k):
        lo, hi = fold[0], fold[-1] + 1
        train = np.r_[0:max(0, lo - embargo), min(len(data), hi + embargo):len(data)]
        model = make_model(kind).fit(X[train], y[train] if kind == "classify" else ret[train])
        s = (model.predict_proba(X[fold])[:, 1] if kind == "classify"
             else model.predict(X[fold]))
        aucs.append(round(_auc(y[fold], s), 4))
    return {"fold_auc": aucs, "mean_auc": round(float(np.nanmean(aucs)), 4),
            "k": k, "embargo": embargo}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni. Three or four arms are tested against one baseline on the
    same data; reporting the smallest raw p as if it were the only test is how a
    null becomes a finding."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, adjusted, running = len(ordered), {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = round(running, 4)
    return adjusted


def verdict(passes: dict[str, bool]) -> str:
    """The pre-registered ladder of hypothesis_tests._interpret, on dAUC."""
    if passes.get("ex_price") and passes.get("orthogonal"):
        return "GO: H1 SUPPORTED on dAUC (survives price-report exclusion and orthogonalisation)."
    if passes.get("ex_price"):
        return ("PARTIAL: ex_price passes but the orthogonalised arm does not -- the gain "
                "may be momentum entering through unflagged headlines.")
    if passes.get("all"):
        return "NO-GO: MOMENTUM LAUNDERING -- `all` passes, ex_price does not."
    if passes.get("placebo"):
        return "NO-GO: PLACEBO FIRED -- price-report headlines alone carry the gain."
    return ("NO-GO: H1 NOT SUPPORTED on dAUC. Report the null; do not tune it away.")


def analyse(arm_preds: dict[str, pd.DataFrame], base_preds: pd.DataFrame,
            frame: pd.DataFrame, arm_features: dict[str, list[str]],
            base_features: list[str], kind: str = "regress") -> dict:
    """Everything above, for every arm that ran, plus the go/no-go."""
    out = {"arms": {}}
    for arm, preds in arm_preds.items():
        out["arms"][arm] = {"delta_auc": delta_auc(preds, base_preds),
                            "per_block": per_block(preds, base_preds),
                            "covid": covid_split(preds, base_preds),
                            "purged_kfold": purged_kfold(frame, arm_features[arm], kind)}
    raw = {a: r["delta_auc"]["p_bootstrap"] for a, r in out["arms"].items()
           if "p_bootstrap" in r["delta_auc"]}
    adjusted = holm(raw)
    passes = {}
    for arm, r in out["arms"].items():
        r["delta_auc"]["p_holm"] = adjusted.get(arm)
        share = r["per_block"]["share_positive"]
        passes[arm] = bool(adjusted.get(arm, 1.0) < ALPHA
                           and r["delta_auc"]["delta_auc"] > 0
                           and share >= BLOCK_SHARE_REQUIRED)
        r["passes"] = passes[arm]
    out["purged_kfold_f0"] = purged_kfold(frame, base_features, kind)
    out["rule"] = (f"pass = Holm-adjusted block-bootstrap p < {ALPHA} on dAUC > 0 AND "
                   f"dAUC > 0 in >= {BLOCK_SHARE_REQUIRED:.0%} of 3-month blocks")
    out["verdict"] = verdict(passes)
    return out
