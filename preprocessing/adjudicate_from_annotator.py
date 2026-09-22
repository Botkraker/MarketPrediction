"""Create the adjudicated gold dataset.

Two methods. `annotator_1` copies one model's labels through and is PROVISIONAL
-- there is no reliability estimate behind it. `majority` takes a plurality vote
across every populated annotator_N_label column and leaves ties blank rather
than resolving them silently, so unresolved items stay visible instead of
becoming quiet noise in the training set.

Two optional refinements to `majority`, both recorded per row in
annotation_status and counted in the metadata:
  --human-annotator N  rows annotator N labelled take that label outright. Used
                       for the human anchor, whose rows become the evaluation set:
                       a classifier graded against the human must be graded
                       against the human's label, not a vote the human was one of.
  --tiebreak N         a tie takes annotator N's label instead of staying blank.
                       This is a DECISION, not a measurement, and the metadata
                       says so; see the tiebreak_note it writes.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from agreement import annotator_label_columns, majority_label
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
    method: str = "annotator_1",
    human_annotator: int | None = None,
    tiebreak: int | None = None,
    tiebreak_note: str = "",
) -> pd.DataFrame:
    """Copy annotator 1 labels into adjudicated_label without changing input."""
    frame = pd.read_csv(input_path, keep_default_na=False)
    # "source" is required because in_study is derived from it (see below).
    required = {"annotator_1_label", "adjudicated_label", "annotation_status",
                "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    result = frame.copy()
    if method == "annotator_1":
        labels = frame["annotator_1_label"].astype(str).str.strip()
        if labels.eq("").any() or (~labels.isin(LABELS)).any():
            raise ValueError("annotator_1_label must contain one valid label for every row")
        result["adjudicated_label"] = labels
        result["annotation_status"] = "adjudicated_from_annotator_1"
        ties = 0
    elif method == "majority":
        columns = annotator_label_columns(frame)
        if len(columns) < 2:
            raise ValueError(
                f"majority needs >=2 populated annotator_N_label columns, found {columns}")
        human_col = f"annotator_{human_annotator}_label" if human_annotator else None
        tiebreak_col = f"annotator_{tiebreak}_label" if tiebreak else None
        for col in (human_col, tiebreak_col):
            if col and col not in columns:
                raise ValueError(f"{col} is not a populated annotator column")
        # The human column never votes: where it is filled it decides outright,
        # and where it is blank it has nothing to add.
        voters = [c for c in columns if c != human_col]
        labels, statuses = [], []
        for _, row in frame.iterrows():
            human = str(row[human_col]).strip() if human_col else ""
            if human:
                labels.append(human)
                statuses.append(f"adjudicated_human_{human_col.split('_')[1]}")
                continue
            label, tied = majority_label(row, voters)
            if not tied:
                labels.append(label)
                statuses.append("adjudicated_majority")
            elif tiebreak_col and str(row[tiebreak_col]).strip():
                labels.append(str(row[tiebreak_col]).strip())
                statuses.append(f"adjudicated_tiebreak_{tiebreak_col.rsplit('_', 1)[0]}")
            else:
                # Left blank on purpose: split.py's validate_gold_complete refuses
                # the file until every tie is resolved, so ties cannot reach a model.
                labels.append("")
                statuses.append("adjudicated_tie_unresolved")
        result["adjudicated_label"] = labels
        result["annotation_status"] = statuses
        ties = statuses.count("adjudicated_tie_unresolved")
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'annotator_1' or 'majority'.")
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
        "adjudication_method": method,
        "labels_source": ("annotator_1_label" if method == "annotator_1"
                          else annotator_label_columns(frame)),
        # provisional == a single model's labels with no agreement statistic behind them
        "provisional": method == "annotator_1",
        "ties_unresolved": ties,
        "status_counts": result["annotation_status"].value_counts().sort_index().astype(int).to_dict(),
        "human_annotator": human_annotator,
        "tiebreak_annotator": tiebreak,
        "tiebreak_note": tiebreak_note or None,
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
    parser.add_argument("--method", choices=("annotator_1", "majority"),
                        default="annotator_1")
    parser.add_argument("--human-annotator", type=int, default=None,
                        help="annotator N whose label, where present, decides outright")
    parser.add_argument("--tiebreak", type=int, default=None,
                        help="annotator N whose label resolves a majority tie")
    parser.add_argument("--tiebreak-note", default="",
                        help="why this tiebreak was chosen; written to the metadata")
    args = parser.parse_args()
    if (args.human_annotator or args.tiebreak) and args.method != "majority":
        parser.error("--human-annotator and --tiebreak require --method majority")
    result = create_dataset(args.input, args.output, args.metadata, args.method,
                            args.human_annotator, args.tiebreak, args.tiebreak_note)
    print(f"Wrote {args.output} ({len(result)} rows)")
    print(result["adjudicated_label"].value_counts().reindex(LABELS, fill_value=0).to_string())


if __name__ == "__main__":
    main()