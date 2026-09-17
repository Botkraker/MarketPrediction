"""
Stage 1: raw -> cleaned. Applies stage-1 character normalization
(normalize.clean_text), the audit's per-source usable date window
(config.SOURCE_WINDOWS), and drops rows whose "headline" is actually a
recurring section/rubric label (config.BOILERPLATE_TITLES, audit §3.3-3.4).

Writes ALL rows (kept and dropped) with boolean flags, not just survivors --
funnel.py needs the drop counts, and a curated file should let you see why a
row didn't make it through rather than silently disappearing.
"""
from pathlib import Path

import pandas as pd

from config import BOILERPLATE_TITLES, SOURCE_WINDOWS
from io_raw import load_source
from normalize import clean_text

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"


def clean_source(name: str) -> pd.DataFrame:
    df = load_source(name)
    df["headline_clean"] = df["headline_raw"].map(clean_text)

    start, end = SOURCE_WINDOWS[name]
    in_window = pd.Series(True, index=df.index)
    if start is not None:
        in_window &= df["published_date"].map(lambda d: pd.notna(d) and d >= start)
    if end is not None:
        in_window &= df["published_date"].map(lambda d: pd.notna(d) and d <= end)
    df["in_window"] = in_window

    boilerplate = BOILERPLATE_TITLES.get(name, set())
    df["is_boilerplate"] = df["headline_clean"].isin(boilerplate)

    df["kept"] = df["date_parse_ok"] & df["in_window"] & ~df["is_boilerplate"]
    return df


def run() -> pd.DataFrame:
    frames = [clean_source(name) for name in SOURCE_WINDOWS]
    out = pd.concat(frames, ignore_index=True)
    CURATED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CURATED / "01_cleaned.parquet", index=False)
    return out


if __name__ == "__main__":
    df = run()
    print(f"Wrote {CURATED / '01_cleaned.parquet'} ({len(df)} rows, {df['kept'].sum()} kept)")
    print(df.groupby("source")["kept"].agg(["sum", "count"]))
