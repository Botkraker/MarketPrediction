"""ADR-002 F2: one replication of P2, P2t and W on firm prices from 2023 (D2).

Step 1, `--power` (ADR-002 F2, first box): go / no-go from headline counts only, before any
price after 2022 is downloaded or read.

Each P2 / P2t MDE, as a share of the mean |AR_1|, is projected onto 2023-01-01..END by
sqrt(anchor news sessions / new news sessions). Sessions come from the BVMT calendar rather
than firm trades, for every window alike: there are no firm prices after 2022. The script
also runs the projection design -> sealed against the sealed run's real MDE, as a check on
the method.

GO if the larger projected g_post MDE (design or sealed anchor) is below the smaller lower
95% bound of g_post in the design and sealed runs (the claim to confirm). g_pre and P2's
pooled test are reported, never decisive. W is price-only: no news count bears on it.

Step 2, `--overlap` (ADR-002 F2, D2): the new ilboursa firm prices against ALL_DATA on
2022-H2 (same sessions, same closes, back-adjusted or not). For 2023 on, row counts only.

Step 3, the replication (audit/PREREG_F.md, tag prereg-f-v1). The tagged P2, P2t and W
code runs unchanged on firm sessions F_START..END: bench.FIRM_SEAL / FIRM_END are swapped and
every read of ALL_DATA.csv returns ALL_DATA up to 2022-12-30 plus D2 from 2023 (`prices`).
`--check` first runs that same path on the ADR-001 sealed window and must reproduce
p2_confirm.json, p2t_confirm.json and w_results.json; it reads no price after 2022.

Run: python3 preprocessing/f.py --power     -> data/curated/f_power.json
     python3 preprocessing/f.py --overlap   -> data/curated/f_d2_overlap.json
     OMP_NUM_THREADS=1 python3 preprocessing/f.py --check     -> data/curated/f_check.json
     OMP_NUM_THREADS=1 python3 preprocessing/f.py --confirm   -> data/curated/f_confirm.json, once
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import bench
import h3
import p2
import p2t
import w
from features import DEFAULT_CALENDAR, phantom_sessions
from p2_posthoc import patched
from p3 import code_matches_tag

READ = pd.read_csv                 # the real reader, kept before any swap
TAG, F_START = "prereg-f-v1-a1", "2023-01-01"   # a1: zero gaps clipped (PREREG_F §9)
ADR1 = ("2021-01-01", "2022-12-30")          # bench.FIRM_SEAL, FIRM_END as tagged: the check
GAPS = ["log_gap", "log_back_gap"]           # trading-gap secondary (owner)

END = "2026-09-15"                 # last canonical headline; PREREG_F fixes the D2 end date
WINDOWS = {"design": ("2016-01-04", "2020-12-31"), "sealed": ("2021-01-04", "2022-12-30"),
           "f": ("2023-01-01", END)}
RESULTS = {"design": "p2t_design.json", "sealed": "p2t_confirm.json"}
P2_RESULTS = {"design": "p2_design.json", "sealed": "p2_confirm.json"}
D2 = h3.ROOT / "data" / "raw" / "bvmt_d2"
OVERLAP = ("2022-07-01", "2022-12-30")


def project(share: float, n_anchor: int, n_new: int) -> float:
    """An MDE share moved to a window with n_new flagged sessions instead of n_anchor."""
    return share * np.sqrt(n_anchor / n_new)


def counts() -> dict:
    """Flagged (firm, session) pairs per window, price reports excluded (P2t's primary arm)."""
    news = p2t.timed_news()
    news = news[~news.is_price_report]
    cal = pd.DatetimeIndex(pd.read_csv(DEFAULT_CALENDAR).session_date).difference(phantom_sessions())
    f = p2t.flags3(news, {t: cal for t in news.ticker.unique()})
    out = {}
    for w, (a, b) in WINDOWS.items():
        g = f[(f.session >= a) & (f.session <= b)]
        out[w] = {k: int(g[k].sum()) for k in p2t.FLAGS}
        out[w]["any"] = int((g[list(p2t.FLAGS)].max(axis=1) > 0).sum())
        out[w]["sessions"] = int(((cal >= a) & (cal <= b)).sum())
    return out


def anchors() -> dict:
    """MDE and lower 95% bound, as shares of the mean |AR_1|, from the design and sealed runs."""
    out = {}
    for w in RESULTS:
        r = json.load(open(h3.CURATED / RESULTS[w]))["primary"]
        v = json.load(open(h3.CURATED / P2_RESULTS[w]))["primary"]["tests"]["vol"]
        out[w] = {k: {"mde": r[k]["mde"] / r["mean_abs_ar1"], "low": r[k]["ci95"][0] / r["mean_abs_ar1"]}
                  for k in p2t.DECISIVE}
        out[w]["any"] = {"mde": v["mde"] / v["mean_abs_ar1"], "low": v["ci95"][0] / v["mean_abs_ar1"]}
    return out


def power() -> dict:
    n, a = counts(), anchors()
    tests = {}
    for k in ("post", "pre", "any"):
        tests[k] = {
            "backtest_design_to_sealed": {"projected": project(a["design"][k]["mde"], n["design"][k], n["sealed"][k]),
                                          "actual": a["sealed"][k]["mde"]},
            "projected_f": {w: project(a[w][k]["mde"], n[w][k], n["f"][k]) for w in RESULTS},
            "lower_bounds": {w: a[w][k]["low"] for w in RESULTS},
        }
    worst, bar = max(tests["post"]["projected_f"].values()), min(tests["post"]["lower_bounds"].values())
    return {"counts": n, "tests": tests, "go": bool(worst < bar),
            "rule": f"projected g_post MDE {worst:.3f} vs smallest lower bound {bar:.3f} (shares of mean |AR_1|)",
            "sesoi_share": p2t.SESOI, "end": END,
            "note": "counts on the BVMT calendar, not firm trades; W is price-only and has no news count"}


def overlap() -> dict:
    """Each D2 file against ALL_DATA on 2022-H2. Duplicate ALL_DATA rows (data rule 11) count
    as a match if any copy matches. From 2023 on: rows per year, nothing else."""
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv")
    a = a.assign(date=pd.to_datetime(a.Date))
    a = a[a.date.between(*OVERLAP)]
    out = {}
    for path in sorted(D2.glob("*.csv")):
        t, d = path.stem, pd.read_csv(path, parse_dates=["date"])
        new = d[d.date.between(*OVERLAP)].set_index("date")
        old = a[a.Ticker == t]
        m = old.merge(new, left_on="date", right_index=True)
        hit = m.assign(close_eq=(m.Close.round(3) == m.close.round(3)), vol_eq=(m.Volume == m.volume))
        by = hit.groupby("date")[["close_eq", "vol_eq"]].any()
        out[t] = {"sessions_d2": len(new), "sessions_all_data": int(old.date.nunique()),
                  "common": len(by), "close_equal": int(by.close_eq.sum()), "volume_equal": int(by.vol_eq.sum()),
                  "only_d2": [str(x.date()) for x in new.index.difference(old.date)][:10],
                  "only_all_data": [str(x.date()) for x in pd.DatetimeIndex(old.date.unique()).difference(new.index)][:10],
                  "close_ratio": [float((m.close / m.Close).min()), float((m.close / m.Close).max())] if len(m) else None,
                  "rows_by_year_2023_on": {int(y): int(n) for y, n in
                                           d[d.date >= "2023-01-01"].date.dt.year.value_counts().sort_index().items()}}
    return out


# ------------------------------------------------------------------ the replication
def prices() -> pd.DataFrame:
    """ALL_DATA.csv up to 2022-12-30, then D2 from F_START, in ALL_DATA's layout. On 2022-H2
    the two agree, except that ilboursa back-adjusts for later splits and free shares: those
    tickers' D2 closes are ALL_DATA's times a constant (f_d2_overlap.json). Their ALL_DATA
    prices are put on D2's basis (owner), which leaves every earlier return unchanged."""
    a = READ(h3.RAW / "ALL_DATA.csv")
    a = a[a.Date <= ADR1[1]]
    d = pd.concat([READ(f).assign(Ticker=f.stem) for f in sorted(D2.glob("*.csv"))]).rename(columns=str.capitalize)
    both = a.drop_duplicates(["Ticker", "Date"], keep="last").merge(d, on=["Ticker", "Date"], suffixes=("", "_d2"))
    both = both[both.Date >= OVERLAP[0]]
    ratio = (both.Close_d2 / both.Close).groupby(both.Ticker).agg(["min", "max"])
    assert np.allclose(ratio["min"], ratio["max"], rtol=1e-9, atol=0), "a factor varies inside 2022-H2"
    scale = ratio["min"][~np.isclose(ratio["min"], 1.0, rtol=1e-9, atol=0)]
    k = a.Ticker.map(scale).fillna(1.0).to_numpy()
    a = a.assign(**{c: a[c] * k for c in ("Open", "High", "Low", "Close")})
    return pd.concat([a, d[d.Date >= F_START][a.columns]], ignore_index=True)


@contextmanager
def frozen_code_on(start: str, end: str, gaps: bool = False, frame: pd.DataFrame | None = None):
    """The tagged code on firm sessions start..end with `frame` (default prices()) read as
    ALL_DATA.csv. p2t's own tag guard is replaced by F's (main). gaps adds the two log
    trading gaps to P2's HAR terms, which P2t also uses (secondary). A trade dated on a day
    the calendar lacks sits 0 sessions from its neighbour; it is at least 1 (amendment a1)."""
    frame = prices() if frame is None else frame

    def read_csv(path, *args, **kw):
        if Path(str(path)).name != "ALL_DATA.csv":
            return READ(path, *args, **kw)
        cols = kw.get("usecols")
        return (frame[[c for c in frame.columns if c in cols]] if cols else frame).copy()

    swaps = [(pd, "read_csv", read_csv), (bench, "FIRM_SEAL", start), (bench, "FIRM_END", end),
             (p2t, "code_matches_tag", lambda tag: True)]
    if gaps:
        panel = p2.panel
        swaps += [(p2, "panel", lambda *a, **k: panel(*a, **k).pipe(
                      lambda p: p.assign(log_gap=np.log(p.gap.clip(lower=1)), log_back_gap=np.log(p.back_gap.clip(lower=1))))),
                  (p2, "HAR", p2.HAR + GAPS)]
    with ExitStack() as stack:
        for obj, name, value in swaps:
            stack.enter_context(patched(obj, name, value))
        yield


def w_run(start: str, end: str) -> dict:
    """W as in w.main (H3c, calibrators on earlier predictions only), scored from `start`."""
    panel = h3.firm_panel(end)
    pred = h3.walk_forward_panel(panel, h3.FIRM_BASE).merge(w.jump_windows(end), on=["ticker", "session"], how="left")
    pred["jump_window"] = pred.jump_window.fillna(False).astype(bool)
    out = {}
    for name, d in (("screened", pred[~pred.jump_window].reset_index(drop=True)), ("as_published", pred)):
        d = d.assign(platt=w.calibrate(d, "platt"), isotonic=w.calibrate(d, "isotonic"))
        out[name] = w.analyse(d[d.platt.notna() & (d.session >= start)].reset_index(drop=True), panel)
    low, base = out["screened"]["bootstrap"]["pr_auc_ci95"][0], out["screened"]["scores"]["raw"]["event_rate"]
    out["rule"] = {"pr_auc_lower_95": low, "event_rate": base,
                   "verdict": "REPLICATES" if low > base else "DOES NOT REPLICATE"}
    return out


def verdict(v: dict, sesoi: float, p_holm: float) -> str:
    lo, hi = v["ci95"]
    return ("NO EFFECT WORTH HAVING" if -sesoi < lo and hi < sesoi
            else "CHANNEL" if p_holm < 0.05 else "INCONCLUSIVE")


def replicate(start: str = F_START, end: str = END) -> dict:
    """PREREG_F: P2 (all of it), P2t (all of it), then Holm across P2 vol, g_post and g_pre."""
    with frozen_code_on(start, end):
        r2, r2t = p2.run(True), p2t.run(True)
    with frozen_code_on(start, end, gaps=True):
        gap = {"p2_vol": p2.family(p2.panel(True), True)["vol"],
               "p2t": p2t.evaluate(p2t.units(*p2t.data(True), True))}
    tests = {"p2_vol": r2["primary"]["tests"]["vol"], "post": r2t["primary"]["post"], "pre": r2t["primary"]["pre"]}
    sesoi = {"p2_vol": tests["p2_vol"]["sesoi"], "post": r2t["primary"]["sesoi"], "pre": r2t["primary"]["sesoi"]}
    family = {}
    for k, ph in bench.holm({k: v["p"] for k, v in tests.items()}).items():
        v = tests[k]
        family[k] = {"effect": v["effect_vs_noise"], "ci95": v["ci95"], "mde": v["mde"], "sesoi": sesoi[k],
                     "p": v["p"], "p_holm_f": ph, "verdict": verdict(v, sesoi[k], ph)}
    post = family["post"]
    claim = ("CONFIRMED: after-close firm news is followed by a bigger next move, on untouched data"
             if post["verdict"] == "CHANNEL" and post["effect"] > 0
             else "NOT REPLICATED: no effect worth having after the close (a design-window finding)"
             if post["verdict"] == "NO EFFECT WORTH HAVING"
             else f"NOT CONFIRMED, NOT REFUTED (after-close {post['verdict']}, effect {post['effect']:+.5f})")
    return {"window": [start, end], "family": family, "reading": claim,
            "p2": r2, "p2t": r2t, "secondary_trading_gaps": gap, "dividend_adjusted": "not run (owner)"}


def check() -> dict:
    """The replication path on the ADR-001 sealed window against the sealed results."""
    with frozen_code_on(*ADR1):
        r2, r2t = p2.run(True), p2t.run(True)
        rw = w_run("1900-01-01", ADR1[1])
    ref = {n: json.load(open(h3.CURATED / f"{n}.json")) for n in ("p2_confirm", "p2t_confirm", "w_results")}
    same = lambda a, b, keys: all(np.allclose(a[k], b[k], rtol=1e-9, atol=0) for k in keys)
    out = {"p2_vol": same(r2["primary"]["tests"]["vol"], ref["p2_confirm"]["primary"]["tests"]["vol"],
                          ["n", "effect_vs_noise", "se_boot", "ci95", "p"]),
           "p2t": all(same(r2t["primary"][f], ref["p2t_confirm"]["primary"][f], ["effect_vs_noise", "scale", "ci95", "p"])
                      for f in p2t.FLAGS) and r2t["primary"]["n"] == ref["p2t_confirm"]["primary"]["n"],
           "w": same(rw["screened"]["scores"]["raw"], ref["w_results"]["screened"]["scores"]["raw"], ["pr_auc", "brier"])
                and same(rw["screened"]["bootstrap"], ref["w_results"]["screened"]["bootstrap"], ["pr_auc_ci95"])
                and rw["screened"]["n"] == ref["w_results"]["screened"]["n"]}
    out["pass"] = all(out.values())
    return out


def save(name: str, res: dict) -> Path:
    path = h3.CURATED / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--check", action="store_true", help="reproduce the ADR-001 sealed results")
    ap.add_argument("--confirm", action="store_true", help="the one replication run; needs prereg-f-v1")
    args = ap.parse_args()
    if args.check:
        res = check()
        print(res, "->", save("f_check", res))
    if args.confirm:
        if not bench.tag_exists(TAG) or not code_matches_tag(TAG):
            raise SystemExit(f"--confirm needs the tag {TAG} and preprocessing/ identical to it")
        if (h3.CURATED / "f_confirm.json").exists():
            raise SystemExit("f_confirm.json exists: the replication is run once")
        res = replicate()
        print(res["reading"], {k: v["verdict"] for k, v in res["family"].items()}, "->", save("f_confirm", res))
    if args.overlap:
        res = overlap()
        (h3.CURATED / "f_d2_overlap.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    if args.power:
        res = power()
        (h3.CURATED / "f_power.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
