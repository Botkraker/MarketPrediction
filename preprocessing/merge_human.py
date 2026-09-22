"""Merge the human anchor worksheet into the gold file as an annotator column.

The human labels a SUBSET (150 of 1,000 by default), so this writes a normal
annotator_N_label column and leaves every other row blank. agreement.py compares
each pair on the rows that pair both labelled, so a partial column lowers no
model-vs-model statistic; Fleiss' kappa still uses only rows every annotator
labelled, and the JSON says how many those are.

Why the human matters even though the labels are LLM-produced: both LLM
annotators score against PROMPT_V2, and one candidate annotator's family wrote
it. Two models agreeing on a rubric they were both handed is not evidence the
rubric is right. The human anchor is the only outside measurement in the loop.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from gold import LABELS
from llm_annotate import annotator_columns

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_WORKSHEET = CURATED / "sentiment_gold_v2_human.csv"
DEFAULT_TARGET = CURATED / "sentiment_gold_v2.csv"
DEFAULT_METADATA = CURATED / "sentiment_gold_v2_human_metadata.json"


def merge_human(
    worksheet_path: Path = DEFAULT_WORKSHEET,
    target_path: Path = DEFAULT_TARGET,
    metadata_path: Path = DEFAULT_METADATA,
    annotator: int = 3,
    annotator_name: str = "human",
) -> pd.DataFrame:
    worksheet_path = Path(worksheet_path)
    if worksheet_path.suffix.lower() == ".xlsx":
        # The Excel copy from human_workbook.py; labels live on the "annotate" sheet.
        worksheet = pd.read_excel(worksheet_path, sheet_name="annotate",
                                  dtype=str).fillna("")
    else:
        worksheet = pd.read_csv(worksheet_path, keep_default_na=False)
    target = pd.read_csv(target_path, keep_default_na=False)
    for column in ("gold_item_id", "human_label"):
        if column not in worksheet.columns:
            raise ValueError(f"Worksheet is missing required column {column!r}")
    if worksheet["gold_item_id"].duplicated().any():
        raise ValueError("Worksheet contains duplicate gold_item_id values")

    filled = worksheet[worksheet["human_label"].astype(str).str.strip().ne("")].copy()
    filled["human_label"] = filled["human_label"].str.strip().str.lower()
    if filled.empty:
        raise SystemExit(f"No human_label filled in {worksheet_path.name}; nothing to merge.")
    bad = sorted(set(filled["human_label"]) - set(LABELS))
    if bad:
        raise ValueError(f"Worksheet contains labels outside {LABELS}: {bad}")
    unknown = sorted(set(filled["gold_item_id"]) - set(target["gold_item_id"]))
    if unknown:
        raise ValueError(f"Worksheet rows are not in the gold file: {unknown[:5]}")

    label_col, notes_col, model_col = annotator_columns(annotator)
    for column in (label_col, notes_col, model_col):
        if column not in target.columns:
            target[column] = ""
    prompt_col = f"annotator_{annotator}_prompt"
    if prompt_col not in target.columns:
        target[prompt_col] = ""

    notes = dict(zip(filled["gold_item_id"], filled.get("human_notes", pd.Series("", index=filled.index))))
    labels = dict(zip(filled["gold_item_id"], filled["human_label"]))
    is_human = target["gold_item_id"].isin(labels)
    target.loc[is_human, label_col] = target.loc[is_human, "gold_item_id"].map(labels)
    target.loc[is_human, notes_col] = target.loc[is_human, "gold_item_id"].map(notes).fillna("")
    target.loc[is_human, model_col] = annotator_name
    target.loc[is_human, prompt_col] = "human"
    target.to_csv(target_path, index=False)

    metadata = {
        "worksheet": worksheet_path.name,
        "target": target_path.name,
        "annotator": annotator,
        "annotator_name": annotator_name,
        "worksheet_rows": int(len(worksheet)),
        "rows_merged": int(is_human.sum()),
        "rows_blank_in_worksheet": int(len(worksheet) - len(filled)),
        "label_counts": filled["human_label"].value_counts().sort_index().astype(int).to_dict(),
        "coverage_note": "partial column by design; pairwise kappa uses each pair's "
                         "overlap, Fleiss uses rows every annotator labelled",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--annotator", type=int, default=3)
    parser.add_argument("--name", default="human")
    args = parser.parse_args()
    merge_human(args.worksheet, args.target, args.metadata, args.annotator, args.name)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    print(f"Merged {metadata['rows_merged']} human labels into {args.target.name} "
          f"as annotator_{args.annotator} ({metadata['rows_blank_in_worksheet']} blank)")
    print(metadata["label_counts"])


if __name__ == "__main__":
    main()
