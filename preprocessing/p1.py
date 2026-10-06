"""ADR-001 P1: does news flow predict the size of the next index move?

Plan: ADR-001 §8, P1 card. The owner's decisions (2026-10-06) are in STATUS.md and are
fixed in audit/PREREG_P1.md before the design run. Harness: bench.py, validated at gate
G0 (AUDIT_REPORT §P0). Its seal keeps sessions from 2024-01-01 out of every fit until the
tag prereg-p1-v1 exists and --confirm is given.

TARGET    next-session Parkinson variance (ln H/L)^2 / (4 ln 2); `open` is never read.
          A session with high == low has no variance to score: it is not a target, and
          is counted (owner). Secondary targets: the next squared return, and the sum of
          the next 5 squared returns (h = 5), with the same rule for zeros.
BASELINE  HAR on log range (last session, 5- and 22-session means, 1bp floor), the last
          |return|, the weekday of the target session and the days to it (bench.p1_frame).
NEWS      three features per arm, from the headlines mapped to the row's session (dated
          before it, rule 2); 0 on a session without any:
            log_n      log(1 + headlines)
            neg_share  share labelled negative by the v3 instrument
            novelty    mean surprise per headline, in bits per token, under a word-bigram
                       model of every headline of the previous 250 sessions
ARMS      ex_price decides G1 (owner); all and price_only are reported (rule 6).
TEST      d QLIKE of the block against the baseline, read against 100 surrogates of the
          block and scaled by sqrt(surrogate spread^2 + block-bootstrap SE^2) (G0). SESOI:
          3% of the baseline's mean QLIKE in the same run (owner).

    OMP_NUM_THREADS=1 python3 preprocessing/p1.py --controls --jobs 10  # P1's own controls
    OMP_NUM_THREADS=1 python3 preprocessing/p1.py --jobs 10             # design 2016-2023
    OMP_NUM_THREADS=1 python3 preprocessing/p1.py --confirm --jobs 10   # once, after the tag
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import bench
import h3
from config import PRICE_REPORT_PATTERN
from dedup import _template_key
from features import map_to_next_session

LM_WINDOW, LAMBDA = 250, 0.5              # previous 250 sessions (owner); lambda fixed
SESOI = 0.03                              # owner, 2026-10-06
ARMS = ("ex_price", "all", "price_only")
TARGETS = {"pv_next": 1, "r2_next": 1, "rv5_next": 5}


def block(arm: str) -> list[str]:
    return [f"log_n_{arm}", f"neg_share_{arm}", f"novelty_{arm}"]


# ------------------------------------------------------------------ news
def headlines() -> pd.DataFrame:
    """Every canonical relevant headline, scored or not (novelty needs the 2015 history),
    on the first session strictly after its date (rule 2)."""
    s = pd.read_parquet(h3.SCORED_V3)
    s["session"] = map_to_next_session(pd.to_datetime(s.published_date).dt.normalize(), h3.calendar())
    s = s.dropna(subset=["session"]).reset_index(drop=True)
    s["is_price_report"] = s.headline_clean.astype(str).str.contains(
        PRICE_REPORT_PATTERN, case=False, regex=True, na=False)
    s["negative"] = (s.sent_label.astype(str) == "negative").astype(float)
    s["tokens"] = s.headline_clean.astype(str).map(lambda t: _template_key(t).split())
    return s


def novelty(tokens: pd.Series, sessions: pd.Series, calendar: pd.DatetimeIndex,
            window: int = LM_WINDOW) -> pd.Series:
    """Surprise of each headline, in bits per token, under a word-bigram model of the
    headlines of the previous `window` calendar sessions, never its own session:
        p(w | v) = LAMBDA c(v, w) / c(v) + (1 - LAMBDA) p1(w)   if v was seen, else p1(w)
        p1(w)    = (c(w) + 1) / (N + V + 1)                     add-one; unseen words too
    NaN while the window holds no headline. Tokens come from dedup's template key: lower
    case, numbers and date words masked, so a new figure or a new month is not 'novel'."""
    toks = list(tokens)
    uni, bi, ctx, size = defaultdict(int), defaultdict(int), defaultdict(int), {"n": 0, "v": 0}

    def update(rows, sign):
        for j in rows:
            prev = "<s>"
            for w in toks[j]:
                size["v"] += (sign > 0 and uni[w] == 0) - (sign < 0 and uni[w] == 1)
                uni[w] += sign
                bi[(prev, w)] += sign
                ctx[prev] += sign
                size["n"] += sign
                prev = w

    def surprise(t):
        if not t or size["n"] == 0:
            return np.nan
        bits, prev, denom = 0.0, "<s>", size["n"] + size["v"] + 1
        for w in t:
            p = (uni[w] + 1) / denom
            if ctx[prev] > 0:
                p = LAMBDA * bi[(prev, w)] / ctx[prev] + (1 - LAMBDA) * p
            bits -= np.log2(p)
            prev = w
        return bits / len(t)

    rows_of = pd.Series(np.arange(len(toks))).groupby(sessions.to_numpy()).agg(list).to_dict()
    cal = calendar[calendar >= sessions.min()]
    out = np.full(len(toks), np.nan)
    for i, s in enumerate(cal):
        if i >= 1:
            update(rows_of.get(cal[i - 1], []), +1)
        if i > window:
            update(rows_of.get(cal[i - 1 - window], []), -1)
        for j in rows_of.get(s, []):
            out[j] = surprise(toks[j])
    return pd.Series(out, index=tokens.index)


def news_block(news: pd.DataFrame) -> pd.DataFrame:
    """The three features of every arm, one row per session that has headlines."""
    news = news.assign(novelty=novelty(news.tokens, news.session, h3.calendar()))
    out = []
    for arm, sub in (("ex_price", news[~news.is_price_report]), ("all", news),
                     ("price_only", news[news.is_price_report])):
        g = sub.groupby("session")
        out.append(pd.DataFrame({f"log_n_{arm}": np.log1p(g.size()),
                                 f"neg_share_{arm}": g.negative.mean(),
                                 f"novelty_{arm}": g.novelty.mean()}))
    return pd.concat(out, axis=1).rename_axis("session").reset_index()


def p1_frame(news: pd.DataFrame | None = None) -> pd.DataFrame:
    """bench.p1_frame (target and HAR baseline), the secondary targets and the three arms."""
    f = bench.p1_frame()
    f["zero_range_next"] = (f.pv_next.isna() & f.gap_next.notna()).astype(float)
    r2 = h3.index_frame().set_index("session").ret ** 2
    for col, v in (("r2_next", r2.shift(-1)), ("rv5_next", sum(r2.shift(-j) for j in range(1, 6)))):
        v = v.reindex(f.session).to_numpy()
        f[col] = np.where(v > 0, v, np.nan)                 # nothing to score at zero (owner)
    f = f.merge(news_block(headlines() if news is None else news), on="session", how="left")
    cols = [c for arm in ARMS for c in block(arm)]
    f[cols] = f[cols].fillna(0.0)
    return f


# ------------------------------------------------------------------ tests
def fit_for(target: str, h: int, kind: str = "logvar", keep: str | None = None):
    """Walk-forward on the whole frame; the confirmation run scores only sessions >= keep."""
    def fit(g, cols):
        p = bench.walk_forward(g, cols, target, h, kind)
        return p if keep is None else p[p.session >= keep].reset_index(drop=True)
    return fit


def verdict(res: dict, base: pd.DataFrame) -> dict:
    """The pre-registered reading of one comparison (audit/PREREG_P1.md)."""
    effect = res["d"] - res["noise_mean"]
    half = bench.Z * res["mde_noise"] / 2.8                 # mde_noise = 2.8 x scale
    sesoi = SESOI * float(bench.losses(base, "qlike").mean())
    lo, hi = effect - half, effect + half
    word = ("NEWS IMPROVES" if hi < 0 else "NEWS WORSE THAN NOISE" if lo > 0
            else "NO EFFECT WORTH HAVING" if lo > -sesoi else "INCONCLUSIVE")
    return {"effect_vs_noise": effect, "ci95_vs_noise": [lo, hi], "sesoi": sesoi, "verdict": word}


def sealed_frame(confirm: bool) -> pd.DataFrame:
    return bench.seal(p1_frame(), "p1", confirm, forward=TARGETS | {"zero_range_next": 1})


def run(confirm: bool, jobs: int) -> dict:
    f = sealed_frame(confirm)
    keep = bench.SEAL if confirm else None
    fit = fit_for("pv_next", 1, keep=keep)
    base = fit(f, bench.P1_BASE)
    window = f[f.session >= bench.SEAL] if confirm else f
    out = {"mode": "confirm" if confirm else "design", "n": len(base),
           "window": f"{base.session.min().date()}..{base.session.max().date()}",
           "baseline_mean_qlike": float(bench.losses(base, "qlike").mean()),
           "zero_range_sessions_not_targets": int(window.zero_range_next.sum())}
    arms = {}
    for arm in ARMS:
        cols = block(arm)
        null = bench.noise_null(f, bench.P1_BASE, cols, fit, "qlike", base, jobs=jobs)
        res = bench.compare(fit(f, bench.P1_BASE + cols), base, 1, "qlike", null)
        arms[arm] = res | verdict(res, base)
    out["primary"] = {"arm": "ex_price", **arms["ex_price"]}
    out["arms"] = arms

    single = {}                                             # secondary: each feature alone
    for col in block("ex_price"):
        null = bench.noise_null(f, bench.P1_BASE, [col], fit, "qlike", base, jobs=jobs)
        single[col] = bench.compare(fit(f, bench.P1_BASE + [col]), base, 1, "qlike", null)
    for col, p in bench.holm({c: v["p_noise"] for c, v in single.items()}).items():
        single[col]["p_holm"] = p
    out["single_features"] = single

    g = f.assign(log_pv_next=np.log(f.pv_next))             # secondary: MSE on log variance
    fit_log = fit_for("log_pv_next", 1, "regress", keep)
    out["log_variance_clark_west"] = bench.compare(
        fit_log(g, bench.P1_BASE + block("ex_price")), fit_log(g, bench.P1_BASE), 1, "mse")

    secondary = {}                                          # secondary targets
    for target in ("r2_next", "rv5_next"):
        h = TARGETS[target]
        fit_t = fit_for(target, h, keep=keep)
        b = fit_t(f, bench.P1_BASE)
        null = bench.noise_null(f, bench.P1_BASE, block("ex_price"), fit_t, "qlike", b, jobs=jobs)
        res = bench.compare(fit_t(f, bench.P1_BASE + block("ex_price")), b, h, "qlike", null)
        secondary[target] = res | verdict(res, b)
    for target, p in bench.holm({k: v["p_noise"] for k, v in secondary.items()}).items():
        secondary[target]["p_holm"] = p
    out["secondary_targets"] = secondary
    return out


def controls(jobs: int) -> dict:
    """bench's G0 controls on P1's own pipeline and decisive arm, design window only."""
    out = bench.controls(sealed_frame(False), bench.P1_BASE, block("ex_price"),
                         fit_for("pv_next", 1), "pv_next", 1, "qlike", jobs, log=True)
    out["pass"] = all(out[k]["pass"] for k in ("positive", "permuted_labels", "stale_news"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controls", action="store_true", help="P1's own controls, design window")
    ap.add_argument("--confirm", action="store_true", help="the one sealed run; needs prereg-p1-v1")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    name = "p1_controls" if args.controls else "p1_confirm" if args.confirm else "p1_design"
    path = h3.CURATED / f"{name}.json"
    if args.confirm and path.exists():
        raise SystemExit(f"{path.name} exists: the confirmation run is made once")
    out = controls(args.jobs) if args.controls else run(args.confirm, args.jobs)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out.get("pass", out.get("primary", {}).get("verdict")), f"-> {path}")


if __name__ == "__main__":
    main()
