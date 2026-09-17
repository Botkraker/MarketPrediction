"""
Stage 3: relevant -> canonical-flagged. Marks duplicates without dropping
them: every row keeps dup_cluster_id and cluster_size, and exactly one row
per cluster gets is_canonical=True.

Clustering is exact-hash on the normalized headline (whitespace-collapsed,
lowercased), NOT MinHash. The audit ran MinHash near-dup matching for
wire-reprint detection (AUDIT_REPORT.md §5, French Tunisian outlets vs. TAP,
Jaccard >= 0.8) and got 0% matches everywhere despite the exact-hash check
finding real cross-source duplication -- its own conclusion was that
headline-only near-dup detection is too weak to trust and that the
exact-hash numbers are "the more reliable (if narrower) signal" until
article bodies exist. So exact-hash is what this stage uses; re-add MinHash
once bodies are captured.

Canonical choice per cluster: TAP row if present (wire original), else
earliest published_date, else lowest row_id for a deterministic tie-break.
"""
import re
from pathlib import Path

import pandas as pd

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"


def _norm(headline: str) -> str:
    return re.sub(r"\s+", " ", headline or "").strip().lower()


def _pick_canonical(cluster: pd.DataFrame) -> str:
    tap_rows = cluster[cluster["source"] == "tap"]
    pool = tap_rows if len(tap_rows) else cluster
    pool = pool.sort_values(["published_date", "row_id"])
    return pool.iloc[0]["row_id"]


def run() -> pd.DataFrame:
    df = pd.read_parquet(CURATED / "02_relevance.parquet")
    df = df[df["relevance_tag"] != "other"].copy()

    df["dup_cluster_id"] = df["headline_clean"].map(_norm).map(
        lambda h: __import__("hashlib").sha1(h.encode("utf-8")).hexdigest()
    )
    cluster_sizes = df.groupby("dup_cluster_id")["row_id"].transform("size")
    df["cluster_size"] = cluster_sizes

    canonical_ids = df.groupby("dup_cluster_id", group_keys=False).apply(_pick_canonical, include_groups=False)
    canonical_set = set(canonical_ids)
    df["is_canonical"] = df["row_id"].isin(canonical_set)

    CURATED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CURATED / "03_dedup.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    n_clusters = df["dup_cluster_id"].nunique()
    n_dupe_clusters = (df.groupby("dup_cluster_id")["row_id"].size() > 1).sum()
    print(f"Wrote {CURATED / '03_dedup.parquet'} ({len(df)} rows, {n_clusters} clusters, "
          f"{n_dupe_clusters} clusters with >1 row, {df['is_canonical'].sum()} canonical rows)")
