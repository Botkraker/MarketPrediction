"""Test a candidate F1 control before it is allowed into the H1 baseline.

Blueprint 5.2's F1 rung adds macro/FX controls: Brent, EUR/TND, and European
index returns. Brent was tested this way by hand (AUDIT_REPORT 8g) and found
immaterial. This file makes that test a script, so every candidate gets the same
check under the same rule, and so the Brent numbers can be reproduced.

ALIGNMENT
A series dated D is matched to Tunindex session D by as-of join (the latest
value on or before D). European markets and Brent settle after the ~14:10 Tunis
close, so a day-D move is known before session D+1 opens. That makes it a
legitimate predictor of ret_next. It is NOT a legitimate predictor of the same
session's return, which is reported only as a description.

DECISION RULE (amendment prereg-h1-v1-a1, fixed before any H1 run)
A candidate enters the H1 baseline as part of F1 iff its move predicts the next
session's Tunindex return at p < 0.05 by OLS with Newey-West (HAC, 5 lags)
standard errors, over the H1 window (2014 onward). Otherwise H1 is reported
against F0 and the candidate is documented as tested and immaterial.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import load_prices

ROOT = Path(__file__).resolve().parent.parent
MACRO = ROOT / "data" / "raw" / "macro"
DEFAULT_OUTPUT = ROOT / "data" / "curated" / "macro_controls.json"
START = "2014-01-01"
HAC_LAGS = 5
ALPHA = 0.05


def load_series(path: Path, value_column: str | None = None) -> pd.DataFrame:
    """A CSV with a `date` column and one level column (price or rate)."""
    frame = pd.read_csv(path)
    value_column = value_column or [c for c in frame.columns if c != "date"][0]
    frame = frame.assign(day=pd.to_datetime(frame["date"], errors="coerce"),
                         level=pd.to_numeric(frame[value_column], errors="coerce"))
    frame = frame.dropna(subset=["day", "level"]).sort_values("day")
    frame = frame[frame.level > 0]
    frame["move"] = np.log(frame.level).diff()
    return frame[["day", "move"]].dropna()


def align(prices: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    px = prices[["session", "ret"]].assign(ret_next=prices["ret"].shift(-1))
    joined = pd.merge_asof(px.sort_values("session"), series,
                           left_on="session", right_on="day", direction="backward")
    # a stale value (series gap longer than a week) is missing, not a zero move
    stale = (joined.session - joined.day) > pd.Timedelta(days=7)
    joined.loc[stale, "move"] = np.nan
    return joined


def _hac(y: pd.Series, x: pd.Series) -> dict:
    data = pd.concat([y, x], axis=1).dropna()
    fit = sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {"r": round(float(data.corr().iloc[0, 1]), 4),
            "r2": round(float(fit.rsquared), 5),
            "p_hac": round(float(fit.pvalues.iloc[1]), 4), "n": int(len(data))}


def test_candidate(name: str, series: pd.DataFrame, prices: pd.DataFrame,
                   start: str = START) -> dict:
    joined = align(prices, series)
    joined = joined[joined.session >= pd.Timestamp(start)]
    next_session = _hac(joined.ret_next, joined.move)
    return {
        "candidate": name,
        "session_coverage": round(float(joined.move.notna().mean()), 4),
        "same_session_descriptive": _hac(joined.ret, joined.move),
        "next_session": next_session,
        "abs_next_session": _hac(joined.ret_next.abs(), joined.move.abs()),
        "ret_lag0_r2_for_scale": _hac(joined.ret_next, joined.ret)["r2"],
        "enters_f1": bool(next_session["p_hac"] < ALPHA),
        "rule": f"next-session OLS, Newey-West {HAC_LAGS} lags, p < {ALPHA}, from {start}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", nargs="*", type=Path,
                        help="CSV files with a date column and one level column "
                             "(default: every CSV in data/raw/macro)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = args.series or sorted(MACRO.glob("*.csv"))
    prices = load_prices()
    results = [test_candidate(p.stem, load_series(p), prices) for p in paths]
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'candidate':<22}{'cover':>7}{'r next':>9}{'p HAC':>8}{'R2':>9}"
          f"{'R2 ret_lag0':>13}  F1?")
    for r in results:
        n = r["next_session"]
        print(f"{r['candidate']:<22}{r['session_coverage']:>7.1%}{n['r']:>+9.4f}"
              f"{n['p_hac']:>8.4f}{n['r2']:>9.5f}{r['ret_lag0_r2_for_scale']:>13.5f}"
              f"  {'ENTERS' if r['enters_f1'] else 'no'}")


if __name__ == "__main__":
    main()
