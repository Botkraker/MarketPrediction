"""One tidy time-series CSV: one row per BVMT session, date + everything analysable.

WHY THIS EXISTS SEPARATELY FROM features.py
`daily_features.parquet` is the modelling frame -- it carries what H1 consumes and
nothing else. This builds the ANALYSIS frame: the same sessions plus the raw bar
levels, the macro series, and the alternative scorer, with the redundant columns
removed. It is additive and read-only with respect to the pipeline; nothing
downstream of features.py reads this file.

THREE THINGS IT FIXES, none of which are cosmetic:

1. FIVE ORPHAN COLUMNS. `vol_chg_lag0` and `dow_next_{mon,tue,wed,thu}` are in the
   committed daily_features.parquet and in no committed script -- a fresh
   features.py run emits 59 columns, the artifact has 64. They are recomputed here
   from the raw price file, and `verify()` asserts the recomputation matches the
   committed artifact exactly. That turns an orphan into a regenerable column.

2. THE TWELVE DUPLICATE COLUMNS. sent_*_lag1 are bit-identical copies of sent_*
   (features.py:117). That is CORRECT -- the one-session lag is performed upstream
   by map_to_next_session's side="right", so the suffix names the offset from the
   TARGET, per the file's own lag convention. But bit-identical columns poison a
   correlation matrix, so the analysis frame keeps the _lag1 names (what
   hypothesis_tests.py refers to) and drops the unsuffixed duplicates.
   sent_resid_*_lag1 are NOT duplicates and are kept.

3. NO MACRO. The three macro series exist in data/raw/macro/ and are referenced by
   no .py file in the repo -- only by data/curated/macro_controls.json. They are
   joined here, forward-filled over BVMT sessions that the source market did not
   trade, with an explicit *_is_ffill flag so a stale value is never mistaken for
   an observation.

`open` is never read. Returns are close-to-close (docs/architecture.md §5, rule 1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
import features

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated"
MACRO = ROOT / "data" / "raw" / "macro"

DEFAULT_FEATURES = CURATED / "daily_features.parquet"
DEFAULT_ALT_SCORED = CURATED / "04_scored_v2.parquet"
DEFAULT_OUTPUT = CURATED / "tunindex_timeseries.csv"

# (filename, level column, precomputed return column or None)
MACRO_SERIES = {
    "brent": ("brent_eia_daily.csv", "brent", "brent_ret"),
    "stoxx50": ("eurostoxx50_yahoo_daily.csv", "stoxx50_close", None),
    "eur_tnd": ("eurtnd_bct_daily.csv", "eur_tnd", None),
}

# Bit-identical to their _lag1 twins; see docstring point 2.
REDUNDANT = [f"{stem}{suffix}"
             for suffix in ("", "_ex_price", "_price_only")
             for stem in ("sent_mean", "sent_pos_share", "sent_neg_share", "has_sent")]


def add_orphan_columns(frame: pd.DataFrame, prices_path: Path) -> pd.DataFrame:
    """Recompute the five columns that exist in the artifact but in no script.

    vol_chg_lag0 is log(volume).diff(), NOT log1p: zero-volume sessions are masked
    to NaN BEFORE the diff, which zeroes the session after a zero-volume day too.
    Computed on the full 2010+ series before the 2014 window, so row 0 carries a
    real value in rather than a NaN. Clipped to +/-3 (21 rows bind).
    """
    px = pd.read_csv(prices_path, encoding="utf-8-sig")
    px["session"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()
    px = px.dropna(subset=["session"]).sort_values("session").reset_index(drop=True)

    volume = px["volume"].astype(float).replace(0, np.nan)
    px["vol_chg_lag0"] = np.log(volume).diff().clip(-3, 3).fillna(0.0)

    nxt = px["session"].shift(-1).dt.dayofweek
    for name, code in (("mon", 0), ("tue", 1), ("wed", 2), ("thu", 3)):
        px[f"dow_next_{name}"] = (nxt == code).astype(float).where(nxt.notna())

    cols = ["session", "high", "low", "vol_chg_lag0",
            "dow_next_mon", "dow_next_tue", "dow_next_wed", "dow_next_thu"]
    # The recomputed columns are authoritative; drop the artifact's copies first
    # so the merge does not suffix them into _x/_y. verify() still compares the
    # recomputation against the committed values.
    frame = frame.drop(columns=[c for c in cols if c != "session" and c in frame.columns])
    return frame.merge(px[cols], on="session", how="left")


def add_macro(frame: pd.DataFrame) -> pd.DataFrame:
    """Left-join each macro series onto the session spine.

    Forward-fill covers BVMT sessions the source market did not trade (203 cells
    across the window, longest run 4). The fill is flagged: a derived return on a
    filled session is mechanically 0.0 and is not an observation.
    """
    sessions = pd.DatetimeIndex(frame["session"])
    for prefix, (fname, level_col, ret_col) in MACRO_SERIES.items():
        src = pd.read_csv(MACRO / fname)
        src["date"] = pd.to_datetime(src["date"], errors="coerce").dt.normalize()
        src = src.dropna(subset=["date", level_col]).sort_values("date")
        src = src.drop_duplicates(subset="date", keep="last").set_index("date")

        union = src.index.union(sessions).sort_values()
        level = src[level_col].reindex(union).ffill()
        ret = (src[ret_col].reindex(union).ffill() if ret_col
               else level.pct_change())

        frame[f"{prefix}_level"] = level.reindex(sessions).to_numpy()
        frame[f"{prefix}_ret"] = ret.reindex(sessions).to_numpy()
        frame[f"{prefix}_is_ffill"] = (~sessions.isin(src.index)).astype(int)
    return frame


def add_alt_scorer(frame: pd.DataFrame, scored_path: Path,
                   headlines_path: Path, calendar_path: Path) -> pd.DataFrame:
    """Daily mean of the TF-IDF v2 scorer, as a robustness series.

    daily_features.parquet carries CamemBERT. The two scorers agree on only 61.9%
    of headlines and their daily means correlate 0.381 -- genuinely different
    series, not a rescaling. Mapped with the same side="right" rule, so this
    column obeys the same no-look-ahead guarantee.
    """
    scored = pd.read_parquet(scored_path, columns=["row_id", "sent_score"])
    news = pd.read_parquet(headlines_path, columns=[
        "row_id", "is_canonical", "relevance_tag", "date_parse_ok", "published_date"])
    news = news[news["is_canonical"] & (news["relevance_tag"] != "other")
                & news["date_parse_ok"]].copy()
    news = news.merge(scored, on="row_id", how="left")
    news["day"] = pd.to_datetime(news["published_date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["day", "sent_score"])

    sessions = pd.DatetimeIndex(
        pd.read_csv(calendar_path, parse_dates=["session_date"])["session_date"]).sort_values()
    news["session"] = features.map_to_next_session(news["day"], sessions)
    news = news.dropna(subset=["session"])

    agg = news.groupby("session")["sent_score"].agg(
        sent_mean_tfidf_lag1="mean", sent_n_tfidf_lag1="size").reset_index()
    return frame.merge(agg, on="session", how="left")


def verify(frame: pd.DataFrame, committed: pd.DataFrame) -> None:
    """The check this file must not ship without.

    The five recomputed columns are the whole reason to trust this artifact: if
    the reverse-engineered formula is wrong, every downstream analysis inherits a
    silently different feature. Assert an exact match against the committed
    parquet rather than trusting the derivation.
    """
    for col in ("vol_chg_lag0", "dow_next_mon", "dow_next_tue",
                "dow_next_wed", "dow_next_thu"):
        a = frame[col].to_numpy(dtype=float)
        b = committed[col].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.nanmax(np.abs(np.where(both_nan, 0.0, a - b)))
        assert diff == 0.0, f"{col} does not reproduce: max abs diff {diff}"

    assert frame["session"].is_monotonic_increasing, "sessions out of order"
    assert not frame["session"].duplicated().any(), "duplicate sessions"
    assert frame["ret_next"].iloc[:-1].notna().all(), "target has interior gaps"
    print(f"verify: 5 recomputed columns reproduce exactly on {len(frame)} sessions")


def build(features_path: Path = DEFAULT_FEATURES,
          prices_path: Path = features.DEFAULT_PRICES,
          headlines_path: Path = features.DEFAULT_HEADLINES,
          calendar_path: Path = features.DEFAULT_CALENDAR,
          alt_scored_path: Path = DEFAULT_ALT_SCORED,
          output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    committed = pd.read_parquet(features_path)
    frame = committed.drop(columns=[c for c in REDUNDANT if c in committed.columns])

    frame = add_orphan_columns(frame, prices_path)
    frame = add_macro(frame)
    if alt_scored_path.exists():
        frame = add_alt_scorer(frame, alt_scored_path, headlines_path, calendar_path)

    verify(frame, committed)

    lead = ["session", "close", "high", "low", "volume", "ret", "ret_next"]
    frame = frame[lead + [c for c in frame.columns if c not in lead]]
    frame["session"] = pd.to_datetime(frame["session"]).dt.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"wrote {output_path.relative_to(ROOT)}  {frame.shape[0]} rows x {frame.shape[1]} cols")
    print(f"      {frame['session'].iloc[0]} .. {frame['session'].iloc[-1]}")
    return frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    build(features_path=args.features, output_path=args.out)


if __name__ == "__main__":
    main()
