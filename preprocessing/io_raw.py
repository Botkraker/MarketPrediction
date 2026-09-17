"""
Raw-CSV loading, mirroring audit/build_audit.py's SOURCES/delimiters/date
parsing so this pipeline's counts stay consistent with the audit report.
economist_tunisia_economy is not listed here at all -- audit §4 found it a
100%-overlapping subset of economist_tunisia_all, dropped rather than kept
and deduped, per the audit's "Source registry update" (§8).
"""
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# name -> (filename, delimiter)
SOURCES = {
    "assabah": ("assabah_headlines.csv", ";"),
    "economist_tunisia_all": ("economist_tunisia_all.csv", ";"),
    "guardian_tunisia": ("guardian_tunisia_headlines.csv", ";"),
    "ilboursa": ("ilboursa_headlines.csv", ";"),
    "kapitalis": ("kapitalis_headlines.csv", ";"),
    "lapresse": ("lapressheadlines.csv", ","),
    "leconomistmaghrebin": ("leconomistmaghrebin_headlines.csv", ","),
    "nyt_economy": ("nyt_economy_headlines.csv", ";"),
    "tap": ("tap_headlines.csv", ";"),
}

AR_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "أغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
AR_ABS_RE = re.compile(r"^(\d{1,2})\s+(\S+)\s+(\d{4})$")


def parse_assabah_date(raw: str):
    """Returns (date_or_None, category) where category in {abs, relative, other}."""
    if raw is None:
        return None, "other"
    raw = raw.strip()
    m = AR_ABS_RE.match(raw)
    if m and m.group(2) in AR_MONTHS:
        d, mo_word, y = m.groups()
        try:
            return date(int(y), AR_MONTHS[mo_word], int(d)), "abs"
        except ValueError:
            return None, "other"
    if raw.startswith("منذ"):
        return None, "relative"
    return None, "other"


def load_source(name: str) -> pd.DataFrame:
    fname, delim = SOURCES[name]
    path = RAW / fname
    df = pd.read_csv(path, delimiter=delim, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df.rename(columns={"headline": "headline_raw", "date": "date_raw"})
    df["source"] = name
    if name == "assabah":
        parsed = df["date_raw"].map(parse_assabah_date)
        df["published_date"] = [p[0] for p in parsed]
    else:
        pd_dates = pd.to_datetime(df["date_raw"], format="%Y-%m-%d", errors="coerce")
        df["published_date"] = pd_dates.dt.date
    df["date_parse_ok"] = df["published_date"].notna()
    df["row_id"] = df["source"] + "::" + df.index.astype(str)
    return df[["row_id", "source", "headline_raw", "date_raw", "published_date", "date_parse_ok"]]


def load_all() -> pd.DataFrame:
    return pd.concat([load_source(name) for name in SOURCES], ignore_index=True)
