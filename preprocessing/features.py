"""Align headlines to BVMT trading sessions and build daily features.

THE TWO RULES THIS FILE EXISTS TO ENFORCE
-----------------------------------------
1. Returns are CLOSE-TO-CLOSE. The `open` column in tunindex_2010_today.csv is
   the previous session's close for ~33% of rows (AUDIT_REPORT.md section 8b), so
   (close - open)/open silently mixes two different quantities. Never use it.

2. News dated day D may only predict sessions STRICTLY AFTER D. Headlines carry
   no time-of-day, so we cannot tell whether a headline landed before or after
   the close on its own date. Mapping D -> session D would leak: a headline
   published at 18:00 would "predict" a close that already happened. The cost of
   this conservatism is losing same-day intraday signal; the cost of getting it
   wrong is an unpublishable result.

News on non-session days (weekends, the 191 holidays) accumulates forward to the
next session, so nothing is discarded.

RELAXING RULE 2 LATER
`scrape_ilboursa.py` now emits `published_at` (full timestamp) and `url`, but the
26,596 rows currently in data/raw/ are still date-only -- the fix applies from the
next extraction. Once ilboursa is re-scraped, its headlines can be split around the
~14:10 Tunis close and pre-close items aligned to the SAME session, recovering
intraday signal on ~35% of the corpus. Do this per-source: the other sources have no
timestamps and must keep the strict rule, so the alignment becomes mixed and the
feature table needs a column recording which rule each headline used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PRICE_REPORT_PATTERN

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
AUDIT = ROOT / "audit"
RAW = ROOT / "data" / "raw"

DEFAULT_HEADLINES = CURATED / "03_dedup.parquet"
DEFAULT_PRICES = RAW / "bvmt" / "tunindex_2010_today.csv"
DEFAULT_CALENDAR = AUDIT / "trading_calendar.csv"
DEFAULT_OUTPUT = CURATED / "daily_features.parquet"


def load_prices(path: Path = DEFAULT_PRICES) -> pd.DataFrame:
    """Sessions with close-to-close returns. `open` is deliberately dropped."""
    px = pd.read_csv(path, encoding="utf-8-sig")
    px["session"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    px = px.dropna(subset=["session"]).sort_values("session").reset_index(drop=True)
    px["ret"] = px["close"].pct_change()
    px["log_ret"] = np.log(px["close"]).diff()
    px["abs_ret"] = px["ret"].abs()                 # realised volatility proxy
    px["range_pct"] = (px["high"] - px["low"]) / px["close"]
    px["log_volume"] = np.log1p(px["volume"])
    return px[["session", "close", "volume", "log_volume",
               "ret", "log_ret", "abs_ret", "range_pct"]]


def map_to_next_session(days: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Each publication day -> the first trading session STRICTLY after it.

    searchsorted with side='right' gives the first session > day, which is the
    no-look-ahead guarantee. Days after the last session map to NaT and are
    dropped by the caller.
    """
    idx = sessions.searchsorted(pd.DatetimeIndex(days), side="right")
    out = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns]")
    valid = idx < len(sessions)
    out.loc[valid] = sessions[idx[valid]]
    return out


def build(headlines_path: Path = DEFAULT_HEADLINES,
          prices_path: Path = DEFAULT_PRICES,
          calendar_path: Path = DEFAULT_CALENDAR,
          output_path: Path = DEFAULT_OUTPUT,
          start: str = "2014-01-01") -> pd.DataFrame:
    px = load_prices(prices_path)
    sessions = pd.DatetimeIndex(
        pd.read_csv(calendar_path, parse_dates=["session_date"])["session_date"]).sort_values()

    news = pd.read_parquet(headlines_path)
    news = news[news["is_canonical"] & (news["relevance_tag"] != "other")
                & news["date_parse_ok"]].copy()
    news["day"] = pd.to_datetime(news["published_date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["day"])
    news["session"] = map_to_next_session(news["day"], sessions)
    news = news.dropna(subset=["session"])

    # See config.PRICE_REPORT_PATTERN: these restate the index's own move, so
    # their sentiment is a proxy for ret_D. Counted separately so H1 can be run
    # with them, without them, and on them alone.
    news["is_price_report"] = news["headline_clean"].astype(str).str.contains(
        PRICE_REPORT_PATTERN, case=False, regex=True, na=False)

    daily = news.groupby("session").agg(
        n_headlines=("row_id", "size"),
        n_sources=("source", "nunique"),
        n_tunisia_econ=("relevance_tag", lambda s: int((s == "tunisia_econ").sum())),
        n_global_linked=("relevance_tag", lambda s: int((s == "global_linked").sum())),
        n_price_reports=("is_price_report", "sum"),
    ).reset_index()

    frame = px.merge(daily, on="session", how="left")
    for column in ("n_headlines", "n_sources", "n_tunisia_econ",
                   "n_global_linked", "n_price_reports"):
        frame[column] = frame[column].fillna(0).astype(int)
    frame["n_non_price"] = frame["n_headlines"] - frame["n_price_reports"]
    frame["log_headlines"] = np.log1p(frame["n_headlines"])
    frame["price_report_share"] = np.where(
        frame["n_headlines"] > 0, frame["n_price_reports"] / frame["n_headlines"], 0.0)

    # Lags: only information available BEFORE the session being predicted.
    for lag in (1, 2, 3):
        frame[f"ret_lag{lag}"] = frame["ret"].shift(lag)
        frame[f"log_headlines_lag{lag}"] = frame["log_headlines"].shift(lag)
        frame[f"abs_ret_lag{lag}"] = frame["abs_ret"].shift(lag)
    frame["ret_next"] = frame["ret"].shift(-1)      # the thing H1 wants to predict

    frame = frame[frame["session"] >= pd.Timestamp(start)].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = build(output_path=args.output, start=args.start)
    covered = int((frame["n_headlines"] > 0).sum())
    print(f"{len(frame)} sessions from {frame.session.min().date()} "
          f"to {frame.session.max().date()}")
    print(f"sessions with >=1 headline: {covered} ({covered/len(frame):.1%})")
    print(f"median headlines/session: {frame.n_headlines.median():.0f}")


if __name__ == "__main__":
    main()
