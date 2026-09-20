"""Create a provisional gold dataset from annotator 1 labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import SOURCE_WINDOWS
from gold import LABELS

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "sentiment_gold_annotation.csv"
DEFAULT_OUTPUT = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_METADATA = CURATED / "sentiment_gold_annotation1_metadata.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def create_dataset(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA,
) -> pd.DataFrame:
    """Copy annotator 1 labels into adjudicated_label without changing input."""
    frame = pd.read_csv(input_path, keep_default_na=False)
    required = {"annotator_1_label", "adjudicated_label", "annotation_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    labels = frame["annotator_1_label"].astype(str).str.strip()
    if labels.eq("").any() or (~labels.isin(LABELS)).any():
        raise ValueError("annotator_1_label must contain one valid label for every row")

    result = frame.copy()
    result["adjudicated_label"] = labels
    result["annotation_status"] = "adjudicated_from_annotator_1"
    # in_study gates a row out of training/eval WITHOUT deleting it. Derived from
    # config.SOURCE_WINDOWS (the single source of truth for source exclusion), so a
    # source dropped there is dropped here automatically -- no hardcoded id list.
    # Currently excludes the 68 `assabah` rows: scrape_assabah.py targets assabah.ma
    # (Morocco), not Tunisia. See AUDIT_REPORT.md section 6b.
    result["in_study"] = result["source"].isin(SOURCE_WINDOWS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    metadata = {
        "input": _display_path(input_path),
        "output": _display_path(output_path),
        "labels_source": "annotator_1_label",
        "adjudication_method": "copy annotator_1_label into adjudicated_label",
        "provisional": True,
        "rows_written": len(result),
        "in_study_counts": result["in_study"].value_counts().to_dict(),
        "excluded_sources": sorted(set(result.loc[~result["in_study"], "source"])),
        "label_counts": result["adjudicated_label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    result = create_dataset(args.input, args.output, args.metadata)
    print(f"Wrote {args.output} ({len(result)} rows)")
    print(result["adjudicated_label"].value_counts().reindex(LABELS, fill_value=0).to_string())


if __name__ == "__main__":
    main()