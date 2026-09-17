"""
Per-source funnel: raw -> cleaned -> relevant -> canonical. Reads the three
stage outputs (run clean.py, relevance.py, dedup.py first) plus the raw
CSVs themselves, so it never has to trust that earlier counts didn't drift.
"""
from pathlib import Path

import pandas as pd

from io_raw import SOURCES, load_source

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"


def run() -> pd.DataFrame:
    cleaned = pd.read_parquet(CURATED / "01_cleaned.parquet")
    relevance = pd.read_parquet(CURATED / "02_relevance.parquet")
    dedup = pd.read_parquet(CURATED / "03_dedup.parquet")

    rows = []
    for name in SOURCES:
        raw_n = len(load_source(name))
        cleaned_n = int(cleaned[(cleaned["source"] == name) & cleaned["kept"]].shape[0])
        relevant_n = int(relevance[(relevance["source"] == name)
                                    & (relevance["relevance_tag"] != "other")].shape[0])
        canonical_n = int(dedup[(dedup["source"] == name) & dedup["is_canonical"]].shape[0])
        rows.append({
            "source": name, "raw": raw_n, "cleaned": cleaned_n,
            "relevant": relevant_n, "canonical": canonical_n,
        })

    out = pd.DataFrame(rows)
    out.loc["total"] = out.sum(numeric_only=True)
    out.loc["total", "source"] = "TOTAL"
    CURATED.mkdir(parents=True, exist_ok=True)
    out.to_csv(CURATED / "funnel.csv", index=False)
    return out


if __name__ == "__main__":
    out = run()
    print(f"Wrote {CURATED / 'funnel.csv'}")
    print(out.to_string(index=False))
