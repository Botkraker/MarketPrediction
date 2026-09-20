"""Build a deterministic sentiment-annotation template from curated headlines."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "03_dedup.parquet"
DEFAULT_OUTPUT = CURATED / "sentiment_gold_annotation.csv"
DEFAULT_METADATA = CURATED / "sentiment_gold_metadata.json"

LABELS = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]
REQUIRED_COLUMNS = [
    "row_id",
    "source",
    "lang",
    "published_date",
    "headline_clean",
    "relevance_tag",
    "dup_cluster_id",
]
ANNOTATION_COLUMNS = [
    "annotator_1_label",
    "annotator_2_label",
    "adjudicated_label",
    "annotation_status",
    "annotation_notes",
]


def _display_path(path: Path) -> str:
    """Repo-relative path for metadata, falling back to absolute when the
    path is outside the repo (e.g. a pytest tmp_path). Mirrors the same
    helper in split.py / adjudicate_from_annotator.py; gold.py cannot import
    theirs without a circular import (split.py imports LABELS from here)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _period(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(pd.Timestamp(value).year)


def _allocate_languages(available: pd.Series, target: int) -> dict[str, int]:
    """Allocate target rows proportionally, while preserving rare languages."""
    counts = available.value_counts().sort_index()
    if target >= int(counts.sum()):
        return counts.astype(int).to_dict()

    allocation = {lang: 0 for lang in counts.index}
    for lang in counts.index:
        if counts[lang] <= target // max(len(counts), 1):
            allocation[lang] = int(counts[lang])

    remaining = target - sum(allocation.values())
    if remaining <= 0:
        return allocation

    weights = counts.astype(float)
    for lang in allocation:
        weights[lang] = max(float(counts[lang] - allocation[lang]), 0.0)
    if weights.sum() == 0:
        return allocation

    raw = weights / weights.sum() * remaining
    extras = raw.astype(int)
    for lang, amount in extras.items():
        allocation[lang] += int(amount)
    remainder = target - sum(allocation.values())
    ranked = (raw - extras).sort_values(ascending=False)
    for lang in ranked.index[:remainder]:
        allocation[lang] += 1
    return allocation


def _sample_language(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame.copy()
    work = frame.copy()
    work["annotation_period"] = work["published_date"].map(_period)
    strata = work["source"] + "|" + work["relevance_tag"] + "|" + work["annotation_period"]
    work["_stratum"] = strata
    selected = []
    for _, group in work.groupby("_stratum", sort=True):
        selected.append(group.sample(n=1, random_state=seed))
    first_pass = pd.concat(selected, ignore_index=False) if selected else work.iloc[0:0]
    if len(first_pass) > n:
        first_pass = first_pass.sample(n=n, random_state=seed)
    elif len(first_pass) < n:
        remaining = work.drop(index=first_pass.index)
        extra = remaining.sample(n=n - len(first_pass), random_state=seed)
        first_pass = pd.concat([first_pass, extra])
    return first_pass.drop(columns=["_stratum", "annotation_period"])


def build_gold(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    target: int = 3000,
    seed: int = 20260918,
) -> pd.DataFrame:
    """Write a blank, provenance-preserving annotation template."""
    df = pd.read_parquet(input_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    eligible = df[
        df["is_canonical"]
        & (df["relevance_tag"] != "other")
        & df["date_parse_ok"]
        & ~df["is_boilerplate"]
    ].copy()
    eligible = eligible.dropna(subset=["headline_clean", "lang", "dup_cluster_id"])
    if eligible["row_id"].duplicated().any():
        raise ValueError("Eligible input contains duplicate row_id values")
    if target <= 0:
        raise ValueError("target must be positive")

    target = min(target, len(eligible))
    language_targets = _allocate_languages(eligible["lang"], target)
    samples = [
        _sample_language(
            eligible[eligible["lang"] == lang], n, seed + index
        )
        for index, (lang, n) in enumerate(sorted(language_targets.items()))
        if n
    ]
    sample = pd.concat(samples, ignore_index=True)
    sample = sample.sort_values(["lang", "published_date", "row_id"]).reset_index(drop=True)
    sample.insert(0, "gold_item_id", [f"gold-{index:05d}" for index in range(len(sample))])
    for column in ANNOTATION_COLUMNS:
        sample[column] = ""

    columns = ["gold_item_id"] + REQUIRED_COLUMNS + ANNOTATION_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(output_path, index=False)
    metadata = {
        "input": _display_path(input_path),
        "output": _display_path(output_path),
        "target_requested": int(target),
        "rows_written": len(sample),
        "seed": seed,
        "labels": LABELS,
        "selection": "canonical, relevant, dated, non-boilerplate; stratified by language, source, relevance tag, and year",
        "language_targets": language_targets,
        "language_counts": sample["lang"].value_counts().sort_index().astype(int).to_dict(),
        "duplicate_clusters": int(sample["dup_cluster_id"].nunique()),
        "annotation_complete": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sample[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    result = build_gold(args.input, args.output, args.metadata, args.target, args.seed)
    print(f"Wrote {args.output} ({len(result)} rows)")
    print(result["lang"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()