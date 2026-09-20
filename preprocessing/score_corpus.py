"""Score every relevant headline with the sentiment classifier.

PROVENANCE WARNING -- READ BEFORE USING ANY OUTPUT OF THIS FILE
The classifier is trained on `adjudicated_label`, which currently comes from a
single 7B model prompted with PROMPT_V1. v1 never defined the task: its labels
track growth-flavoured vocabulary rather than expected market impact
(AUDIT_REPORT.md section 8d). Scores produced from v1 labels are a PIPELINE
DEMONSTRATION, not a measurement of sentiment. The metadata written alongside
records this, and `hypothesis_tests.py` surfaces it in its output.

Re-run this after re-annotating under PROMPT_V2 with >=2 annotators and a
majority adjudication; nothing else in the pipeline needs to change.

TEMPORAL LEAKAGE -- WHY --mode=expanding IS THE DEFAULT
split.py assigns splits by hash within (lang, adjudicated_label) strata, i.e. a
RANDOM split. The `train` split therefore spans 2005-2026. Fitting once on it and
scoring the whole corpus means a 2014 headline is scored by a model that saw 2026
labels, and `baseline.walk_forward` then "predicts" 2016 using a feature whose
generating function saw the future. walk_forward cannot detect this: it only sees
the finished column. Every no-look-ahead guarantee elsewhere in the pipeline is
void while that holds.

  expanding (default) : refit at each cutoff on gold rows dated STRICTLY BEFORE
                        it, then score only the headlines in that period. Slower,
                        and it cannot score the earliest period (no training data
                        yet) -- those rows get sent_label = "" and are dropped
                        downstream rather than silently imputed.
  static              : the old single-fit behaviour. Leaks. Kept only so the
                        difference can be quantified; never use it for a result.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from gold import LABELS
from sentiment_baseline import build_model

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_CORPUS = CURATED / "03_dedup.parquet"
DEFAULT_GOLD = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_SPLIT = CURATED / "sentiment_gold_split.csv"
DEFAULT_OUTPUT = CURATED / "04_scored.parquet"
DEFAULT_METADATA = CURATED / "04_scored_metadata.json"

# Ordinal labels -> a signed scale. Equal spacing is an assumption, stated here
# rather than buried: it treats very_positive as exactly twice positive.
LABEL_SCORE = {"very_negative": -2.0, "negative": -1.0, "neutral": 0.0,
               "positive": 1.0, "very_positive": 2.0}


def prompt_versions_used(gold: pd.DataFrame) -> list[str]:
    columns = [c for c in gold.columns if c.endswith("_prompt")]
    seen = set()
    for column in columns:
        seen |= set(gold[column].astype(str).str.strip()) - {"", "nan"}
    return sorted(seen) or ["v1 (assumed - no annotator_N_prompt column present)"]


def monotonicity_check(labelled: pd.DataFrame) -> dict:
    """Is LABEL_SCORE's ordering supported by the data it will be applied to?

    Reported, not enforced -- the extremes are thinly populated, so an apparent
    inversion there is usually an unpowered estimate rather than evidence the
    scale is wrong. Publishing the equal-spacing assumption without this check is
    what would be indefensible.
    """
    frame = labelled.dropna(subset=["adjudicated_label"])
    counts = frame.adjudicated_label.value_counts().reindex(LABELS, fill_value=0)
    return {"n_per_label": counts.astype(int).to_dict(),
            "underpowered_labels": [l for l in LABELS if counts[l] < 200],
            "note": ("Equal spacing (-2..+2) is an ASSUMPTION. Validate it against "
                     "realised returns before reporting any coefficient on sent_score; "
                     "consider collapsing to 3 classes, which loses nothing measurable.")}


def _fit(train: pd.DataFrame):
    return build_model().fit(train.headline_clean, train.adjudicated_label)


def run(corpus_path: Path = DEFAULT_CORPUS, gold_path: Path = DEFAULT_GOLD,
        split_path: Path = DEFAULT_SPLIT, output_path: Path = DEFAULT_OUTPUT,
        metadata_path: Path = DEFAULT_METADATA, mode: str = "expanding",
        min_train: int = 200, freq: str = "YS") -> pd.DataFrame:
    if mode not in ("expanding", "static"):
        raise ValueError("mode must be 'expanding' or 'static'")
    gold = pd.read_csv(gold_path, keep_default_na=False)
    split = pd.read_csv(split_path)[["gold_item_id", "split"]]
    labelled = gold.merge(split, on="gold_item_id", how="inner")
    labelled["day"] = pd.to_datetime(labelled.published_date, errors="coerce")
    train_pool = labelled[labelled.split == "train"].dropna(subset=["day"])
    if train_pool.empty:
        raise SystemExit("No training rows -- run split.py first.")

    corpus = pd.read_parquet(corpus_path)
    corpus = corpus[corpus.is_canonical & (corpus.relevance_tag != "other")
                    & corpus.date_parse_ok].copy()
    corpus["day"] = pd.to_datetime(corpus.published_date, errors="coerce")
    corpus = corpus.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    unscored = 0
    if mode == "static":
        model = _fit(train_pool)
        corpus["sent_label"] = model.predict(corpus.headline_clean.astype(str))
        corpus["sent_model_train_end"] = "ALL (leaks)"
    else:
        corpus["sent_label"] = ""
        corpus["sent_model_train_end"] = ""
        cutoffs = pd.date_range(corpus.day.min(), corpus.day.max(), freq=freq)
        for start, end in zip(cutoffs, list(cutoffs[1:]) + [corpus.day.max() + pd.Timedelta(days=1)]):
            # strictly before `start`: nothing at or after it may inform the fit
            past = train_pool[train_pool.day < start]
            block = (corpus.day >= start) & (corpus.day < end)
            if not block.any():
                continue
            if len(past) < min_train or past.adjudicated_label.nunique() < 2:
                unscored += int(block.sum())      # left blank, never imputed
                continue
            model = _fit(past)
            corpus.loc[block, "sent_label"] = model.predict(
                corpus.loc[block, "headline_clean"].astype(str))
            corpus.loc[block, "sent_model_train_end"] = start.date().isoformat()
    corpus["sent_score"] = corpus.sent_label.map(LABEL_SCORE)

    keep = ["row_id", "source", "lang", "published_date", "headline_clean",
            "relevance_tag", "sent_label", "sent_score", "sent_model_train_end"]
    scored = corpus[keep].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)

    metadata = {
        "corpus": corpus_path.name,
        "rows_scored": int(len(scored)),
        "train_pool_rows": int(len(train_pool)),
        "model": "TfidfVectorizer(char_wb 2-5) + LogisticRegression(balanced)",
        "label_scale": LABEL_SCORE,
        "mode": mode,
        "leakage_free": mode == "expanding",
        "rows_unscored_insufficient_history": unscored,
        "min_train": min_train,
        "refit_frequency": freq,
        "monotonicity": monotonicity_check(labelled),
        "prompt_versions_in_training_labels": prompt_versions_used(labelled),
        "provisional": True,
        "warning": ("Labels derive from a single 7B annotator under PROMPT_V1, which "
                    "did not define the task (AUDIT_REPORT 8d). These scores are a "
                    "pipeline demonstration, NOT a sentiment measurement."),
        "label_distribution": scored.sent_label.value_counts()
                                    .reindex(LABELS, fill_value=0).astype(int).to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("expanding", "static"), default="expanding",
                        help="expanding = leakage-free refit; static LEAKS, diagnostic only")
    parser.add_argument("--min-train", type=int, default=200)
    args = parser.parse_args()
    scored = run(output_path=args.output, mode=args.mode, min_train=args.min_train)
    blank = int((scored.sent_label == "").sum())
    print(f"mode={args.mode}  scored {len(scored)-blank:,} of {len(scored):,} headlines"
          f"  ({blank:,} left unscored: insufficient prior labels)")
    print(scored.sent_label.value_counts().reindex(LABELS, fill_value=0).to_string())
    print(f"\nmean score {scored.sent_score.mean():+.3f} "
          f"(0 = neutral; positive drift indicates the v1 label skew)")


if __name__ == "__main__":
    main()
