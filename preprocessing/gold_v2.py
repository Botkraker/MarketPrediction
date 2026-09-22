"""Build the v2 sentiment gold set: a re-annotation sample drawn from the v1 gold set.

WHY A SUBSAMPLE OF THE OLD HEADLINES, NOT A FRESH DRAW
The 2,932 in-study v1 labels come from one annotator under a prompt that never
defined the task (AUDIT_REPORT.md section 8d). They are not reused as labels. The
HEADLINES are reused, because relabelling the same text under PROMPT_V2 lets v1 and
v2 be compared item by item: the prompt effect is then measured, not argued.

Blueprint section 4.3 sizes the gold set at ~600-1,000 double-annotated sentences;
the default target is the upper end because score_corpus.py --mode expanding needs
>= 200 gold rows dated before each block, and a smaller set leaves more of the early
corpus unscored.

WHAT IS EXCLUDED, AND WHY
  * rows outside the study (in_study == False: the wrong-country Assabah rows);
  * the 60 pilot rows (annotation_pilot_v2.csv) -- one candidate annotator has
    already labelled them under v2, so they are not blind;
  * any headline that appears as a worked example in PROMPT_V2 -- an annotator
    shown the answer in its prompt is not being measured on that item.

WHAT IS WRITTEN
  sentiment_gold_v2.csv            annotation file; every label column blank
  sentiment_gold_v2_v1labels.csv   the old v1 labels, kept OUT of the annotation
                                   file so a human annotator never sees them
  sentiment_gold_v2_human.csv      blind worksheet for the human anchor subset
  sentiment_gold_v2_metadata.json  counts, exclusions, seed
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from gold import LABELS
from llm_annotate import PROMPT_V2

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_PILOT = CURATED / "annotation_pilot_v2.csv"
DEFAULT_OUTPUT = CURATED / "sentiment_gold_v2.csv"
DEFAULT_V1_LABELS = CURATED / "sentiment_gold_v2_v1labels.csv"
DEFAULT_HUMAN = CURATED / "sentiment_gold_v2_human.csv"
DEFAULT_METADATA = CURATED / "sentiment_gold_v2_metadata.json"

PROVENANCE_COLUMNS = [
    "gold_item_id",
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
HUMAN_COLUMNS = ["gold_item_id", "source", "published_date", "headline_clean",
                 "human_label", "human_notes"]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalise_headline(text: str) -> str:
    """Comparison key: NFKC, casefold, typographic quotes/dashes unified,
    punctuation dropped, whitespace collapsed. Used only for matching, never
    written back."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.translate(str.maketrans("’‘«»“”–—", "''\"\"\"\"--"))
    text = re.sub(r"[^\w%]+", " ", text)
    return " ".join(text.split())


def prompt_example_headlines(prompt: str = PROMPT_V2) -> set[str]:
    """Normalised headlines quoted as worked examples ("..." -> label) in a prompt."""
    examples = re.findall(r'^"(.+)"\s*->', prompt, flags=re.MULTILINE)
    return {normalise_headline(example) for example in examples}


def allocate(counts: pd.Series, target: int) -> pd.Series:
    """Proportional allocation over strata by largest remainder, with every
    non-empty stratum guaranteed one row so thin source-years stay represented."""
    counts = counts[counts > 0]
    if target >= int(counts.sum()):
        return counts.astype(int)
    if target < len(counts):
        raise ValueError(f"target {target} is smaller than the {len(counts)} strata")
    allocation = pd.Series(1, index=counts.index)
    spare = (counts - 1).astype(float)
    remaining = target - int(allocation.sum())
    raw = spare / spare.sum() * remaining
    extras = raw.astype(int)
    allocation += extras
    leftover = target - int(allocation.sum())
    ranked = (raw - extras).sort_values(ascending=False, kind="mergesort")
    allocation[ranked.index[:leftover]] += 1
    return allocation.clip(upper=counts).astype(int)


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_gold_v2(
    input_path: Path = DEFAULT_INPUT,
    pilot_path: Path | None = DEFAULT_PILOT,
    output_path: Path = DEFAULT_OUTPUT,
    v1_labels_path: Path = DEFAULT_V1_LABELS,
    human_path: Path = DEFAULT_HUMAN,
    metadata_path: Path = DEFAULT_METADATA,
    target: int = 1000,
    human_target: int = 150,
    seed: int = 20260921,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path, keep_default_na=False)
    missing = set(PROVENANCE_COLUMNS + ["adjudicated_label"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if frame["gold_item_id"].duplicated().any():
        raise ValueError("Input contains duplicate gold_item_id values")

    exclusions: dict[str, int] = {}
    pool = frame
    if "in_study" in pool.columns:
        keep = pool["in_study"].astype(str).str.lower().isin(("true", "1"))
        exclusions["not_in_study"] = int((~keep).sum())
        pool = pool[keep]

    if pilot_path is not None and Path(pilot_path).exists():
        pilot_ids = set(pd.read_csv(pilot_path, keep_default_na=False)["gold_item_id"])
        in_pilot = pool["gold_item_id"].isin(pilot_ids)
        exclusions["annotation_pilot"] = int(in_pilot.sum())
        pool = pool[~in_pilot]
    else:
        exclusions["annotation_pilot"] = 0

    keys = pool["headline_clean"].map(normalise_headline)
    is_example = keys.isin(prompt_example_headlines())
    exclusions["prompt_v2_example"] = int(is_example.sum())
    pool = pool[~is_example]

    # Two gold items with the same text would be two draws of one item.
    keys = pool["headline_clean"].map(normalise_headline)
    repeated = keys.duplicated(keep="first")
    exclusions["repeated_headline"] = int(repeated.sum())
    pool = pool[~repeated].copy()

    if target <= 0 or human_target < 0:
        raise ValueError("target must be positive and human_target non-negative")
    target = min(target, len(pool))

    year = pd.to_datetime(pool["published_date"], errors="coerce").dt.year
    pool["_stratum"] = pool["source"] + "|" + year.fillna(0).astype(int).astype(str)
    allocation = allocate(pool["_stratum"].value_counts().sort_index(), target)

    parts = [
        group.sample(n=int(allocation[stratum]),
                     random_state=_stable_seed(seed, stratum))
        for stratum, group in pool.groupby("_stratum", sort=True)
        if allocation.get(stratum, 0)
    ]
    sample = (pd.concat(parts)
              .sort_values(["published_date", "gold_item_id"])
              .reset_index(drop=True))

    v1 = sample[["gold_item_id", "adjudicated_label"]].rename(
        columns={"adjudicated_label": "v1_label"})
    annotation = sample[PROVENANCE_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        annotation[column] = ""
    annotation["in_study"] = True

    # Human anchor: stratified by source only (year strata are too thin at 150),
    # then shuffled so the worksheet does not run in date order.
    human_target = min(human_target, len(sample))
    human_alloc = allocate(sample["source"].value_counts().sort_index(), human_target) \
        if human_target else pd.Series(dtype=int)
    human_parts = [
        group.sample(n=int(human_alloc[source]),
                     random_state=_stable_seed(seed, f"human|{source}"))
        for source, group in sample.groupby("source", sort=True)
        if human_alloc.get(source, 0)
    ]
    human = (pd.concat(human_parts) if human_parts else sample.iloc[0:0])
    human = human.sample(frac=1, random_state=_stable_seed(seed, "human-order"))
    human = human[["gold_item_id", "source", "published_date", "headline_clean"]].copy()
    human["human_label"] = ""
    human["human_notes"] = ""

    for path in (output_path, v1_labels_path, human_path, metadata_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(output_path, index=False)
    v1.to_csv(v1_labels_path, index=False)
    human[HUMAN_COLUMNS].to_csv(human_path, index=False)

    metadata = {
        "input": _display_path(Path(input_path)),
        "output": _display_path(Path(output_path)),
        "v1_labels": _display_path(Path(v1_labels_path)),
        "human_worksheet": _display_path(Path(human_path)),
        "seed": seed,
        "labels": LABELS,
        "target_requested": int(target),
        "rows_written": int(len(annotation)),
        "human_rows": int(len(human)),
        "eligible_pool": int(len(pool)),
        "exclusions": exclusions,
        "strata": "source x publication year; >=1 row per non-empty stratum, "
                  "remainder proportional (largest remainder)",
        "source_counts": annotation["source"].value_counts().sort_index().astype(int).to_dict(),
        "lang_counts": annotation["lang"].value_counts().sort_index().astype(int).to_dict(),
        "human_source_counts": human["source"].value_counts().sort_index().astype(int).to_dict(),
        "v1_label_counts_in_sample": v1["v1_label"].value_counts().sort_index().astype(int).to_dict(),
        "annotation_complete": False,
        "note": "v1 labels are for item-level v1-vs-v2 comparison only; they are "
                "not an annotator and must not enter agreement or adjudication.",
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return annotation


DEFAULT_EXTENSION = CURATED / "sentiment_gold_v2_ext.csv"
DEFAULT_EXTENSION_METADATA = CURATED / "sentiment_gold_v2_ext_metadata.json"


def build_extension(
    input_path: Path = DEFAULT_INPUT,
    existing_path: Path = DEFAULT_OUTPUT,
    output_path: Path = DEFAULT_EXTENSION,
    metadata_path: Path = DEFAULT_EXTENSION_METADATA,
) -> pd.DataFrame:
    """Every remaining in-study v1 headline, blank-labelled, for the two local
    annotators. The 1,000-row sample stays the measured core (agreement, human
    anchor); the extension adds training volume and early-year coverage, which
    score_corpus.py --mode expanding needs.

    Unlike the core sample, pilot rows are NOT excluded: they were held out only
    because Claude had labelled them, and Claude is not an annotator here.
    PROMPT_V2's worked examples still are.
    """
    frame = pd.read_csv(input_path, keep_default_na=False)
    existing = pd.read_csv(existing_path, keep_default_na=False)
    exclusions: dict[str, int] = {}
    pool = frame
    if "in_study" in pool.columns:
        keep = pool["in_study"].astype(str).str.lower().isin(("true", "1"))
        exclusions["not_in_study"] = int((~keep).sum())
        pool = pool[keep]
    in_core = pool["gold_item_id"].isin(existing["gold_item_id"])
    exclusions["in_core_sample"] = int(in_core.sum())
    pool = pool[~in_core]
    keys = pool["headline_clean"].map(normalise_headline)
    is_example = keys.isin(prompt_example_headlines())
    exclusions["prompt_v2_example"] = int(is_example.sum())
    pool = pool[~is_example]
    core_keys = set(existing["headline_clean"].map(normalise_headline))
    keys = pool["headline_clean"].map(normalise_headline)
    repeated = keys.duplicated(keep="first") | keys.isin(core_keys)
    exclusions["repeated_headline"] = int(repeated.sum())
    pool = pool[~repeated]

    extension = pool[PROVENANCE_COLUMNS].sort_values(["published_date", "gold_item_id"]).copy()
    for column in ANNOTATION_COLUMNS:
        extension[column] = ""
    extension["in_study"] = True
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    extension.to_csv(output_path, index=False)
    Path(metadata_path).write_text(json.dumps({
        "input": _display_path(Path(input_path)),
        "core_sample": _display_path(Path(existing_path)),
        "output": _display_path(Path(output_path)),
        "rows_written": int(len(extension)),
        "exclusions": exclusions,
        "source_counts": extension["source"].value_counts().sort_index().astype(int).to_dict(),
        "note": "no human anchor rows; labels come from the two local annotators only",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return extension.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v1-labels", type=Path, default=DEFAULT_V1_LABELS)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--human-target", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--extend", action="store_true",
                        help="write every remaining in-study headline to --extension")
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--extension-metadata", type=Path, default=DEFAULT_EXTENSION_METADATA)
    args = parser.parse_args()
    if args.extend:
        ext = build_extension(args.input, args.output, args.extension, args.extension_metadata)
        print(f"Wrote {args.extension} ({len(ext)} rows)")
        return
    result = build_gold_v2(args.input, args.pilot, args.output, args.v1_labels,
                           args.human, args.metadata, args.target,
                           args.human_target, args.seed)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    print(f"Wrote {args.output} ({len(result)} rows); "
          f"human worksheet {metadata['human_rows']} rows")
    print(f"exclusions: {metadata['exclusions']}")
    print(result["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
