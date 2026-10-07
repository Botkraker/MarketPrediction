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

Run: python3 preprocessing/f.py --power   -> data/curated/f_power.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import h3
import p2t
from features import DEFAULT_CALENDAR, phantom_sessions

END = "2026-09-15"                 # last canonical headline; PREREG_F fixes the D2 end date
WINDOWS = {"design": ("2016-01-04", "2020-12-31"), "sealed": ("2021-01-04", "2022-12-30"),
           "f": ("2023-01-01", END)}
RESULTS = {"design": "p2t_design.json", "sealed": "p2t_confirm.json"}
P2_RESULTS = {"design": "p2_design.json", "sealed": "p2_confirm.json"}


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", action="store_true")
    args = ap.parse_args()
    if args.power:
        res = power()
        (h3.CURATED / "f_power.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
