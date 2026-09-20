"""Validate the sentiment gold set and create a frozen evaluation split."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gold import LABELS

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
# adjudicate_from_annotator.py writes sentiment_gold_annotation1.csv; the
# upstream sentiment_gold_annotation.csv has an empty adjudicated_label
# column and fails validate_gold_complete().
DEFAULT_INPUT = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_OUTPUT = CURATED / "sentiment_gold_split.csv"
DEFAULT_METADATA = CURATED / "sentiment_gold_split_metadata.json"
LANGUAGES = ("ar", "en", "fr")
SPLITS = ("train", "validation", "evaluation")


def validate_gold_complete(frame: pd.DataFrame) -> None:
    """Raise a clear error unless the gold set is safe for model evaluation."""
    required = {
        "gold_item_id",
        "row_id",
        "lang",
        "dup_cluster_id",
        "adjudicated_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Gold set is missing required columns: {sorted(missing)}")

    blank = frame["adjudicated_label"].astype(str).str.strip().eq("")
    invalid = ~frame["adjudicated_label"].isin(LABELS)
    if blank.any() or invalid.any():
        raise ValueError(
            "Gold set is incomplete: every row needs a valid adjudicated_label"
        )
    invalid_languages = set(frame["lang"].dropna()) - set(LANGUAGES)
    if invalid_languages:
        raise ValueError(f"Gold set contains unsupported languages: {sorted(invalid_languages)}")
    for column in ("gold_item_id", "row_id", "dup_cluster_id"):
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(f"Gold set contains missing or duplicate {column} values")


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _assign_stratified(frame: pd.DataFrame, seed: int, evaluation_fraction: float,
                       validation_fraction: float) -> pd.Series:
    assignments = pd.Series(index=frame.index, dtype="object")
    for _, group in frame.groupby(["lang", "adjudicated_label"], sort=True):
        ordered = sorted(
            group.index,
            key=lambda index: _stable_hash(str(frame.at[index, "gold_item_id"]), seed),
        )
        evaluation_count = max(1, round(len(ordered) * evaluation_fraction))
        validation_count = max(1, round(len(ordered) * validation_fraction))
        if evaluation_count + validation_count >= len(ordered):
            validation_count = max(0, len(ordered) - evaluation_count - 1)
        assignments.loc[ordered[:evaluation_count]] = "evaluation"
        assignments.loc[ordered[evaluation_count:evaluation_count + validation_count]] = "validation"
        assignments.loc[ordered[evaluation_count + validation_count:]] = "train"
    return assignments


def build_split(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
    seed: int = 20260919,
    evaluation_fraction: float = 1 / 3,
    validation_fraction: float = 0.15,
) -> pd.DataFrame:
    """Write a deterministic split manifest without changing gold labels."""
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1 - evaluation_fraction:
        raise ValueError("validation_fraction leaves no training data")

    frame = pd.read_csv(input_path, keep_default_na=False)
    # Rows flagged out of study (see adjudicate_from_annotator.py) are kept on disk
    # but never split, trained on or evaluated against.
    excluded = 0
    if "in_study" in frame.columns:
        keep = frame["in_study"].astype(str).str.lower().isin(("true", "1"))
        excluded = int((~keep).sum())
        frame = frame[keep].reset_index(drop=True)
    validate_gold_complete(frame)
    frame = frame.copy()
    frame["split"] = _assign_stratified(
        frame, seed, evaluation_fraction, validation_fraction
    )
    if frame["split"].isna().any():
        raise ValueError("Could not assign every gold row to a split")
    cluster_splits = frame.groupby("dup_cluster_id")["split"].nunique()
    if (cluster_splits > 1).any():
        raise ValueError("A duplicate cluster crosses the split boundary")

    columns = ["gold_item_id", "row_id", "lang", "adjudicated_label", "dup_cluster_id", "split"]
    result = frame[columns].sort_values("gold_item_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata = {
        "input": _display_path(input_path),
        "output": _display_path(output_path),
        "seed": seed,
        "rows_excluded_not_in_study": excluded,
        "rows_split": len(result),
        "evaluation_fraction": evaluation_fraction,
        "validation_fraction_of_total": validation_fraction,
        "split_counts": result["split"].value_counts().reindex(SPLITS, fill_value=0).astype(int).to_dict(),
        "language_counts": result.groupby(["split", "lang"]).size().unstack(fill_value=0).to_dict(),
        "label_counts": result.groupby(["split", "adjudicated_label"]).size().unstack(fill_value=0).to_dict(),
        "id_sha256": hashlib.sha256("\n".join(result["gold_item_id"]).encode("utf-8")).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=int), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    result = build_split(args.input, args.output, args.metadata, args.seed)
    print(result["split"].value_counts().reindex(SPLITS, fill_value=0).to_string())


if __name__ == "__main__":
    main()