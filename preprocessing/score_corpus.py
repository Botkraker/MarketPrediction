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


MODELS = {"tfidf": "TfidfVectorizer(char_wb 2-5) + LogisticRegression(balanced)",
          "camembert": "almanach/camembert-base fine-tuned, 3 classes (camembert_clf.py)"}


def _fit(train: pd.DataFrame, model: str = "tfidf"):
    if model == "camembert":
        from camembert_clf import CamembertClassifier   # lazy: torch only when asked
        return CamembertClassifier().fit(train.headline_clean, train.adjudicated_label)
    return build_model().fit(train.headline_clean, train.adjudicated_label)


def run(corpus_path: Path = DEFAULT_CORPUS, gold_path: Path = DEFAULT_GOLD,
        split_path: Path = DEFAULT_SPLIT, output_path: Path = DEFAULT_OUTPUT,
        metadata_path: Path = DEFAULT_METADATA, mode: str = "expanding",
        min_train: int = 200, freq: str = "YS", model: str = "tfidf",
        cache_dir: Path | None = None, years: set[int] | None = None) -> pd.DataFrame | None:
    """`cache_dir` makes the expanding run resumable and splittable: each block's
    scores are written to cache_dir/<start>.parquet as soon as they exist, and a
    block already there is loaded instead of refitted. `years` restricts the run to
    those blocks and writes only the cache (returns None), so separate machines can
    take separate years; a final run without `years` assembles the output."""
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
        fitted = _fit(train_pool, model)
        corpus["sent_label"] = fitted.predict(corpus.headline_clean.astype(str))
        corpus["sent_model_train_end"] = "ALL (leaks)"
    else:
        corpus["sent_label"] = ""
        corpus["sent_model_train_end"] = ""
        cutoffs = pd.date_range(corpus.day.min(), corpus.day.max(), freq=freq)
        for start, end in zip(cutoffs, list(cutoffs[1:]) + [corpus.day.max() + pd.Timedelta(days=1)]):
            # strictly before `start`: nothing at or after it may inform the fit
            past = train_pool[train_pool.day < start]
            block = (corpus.day >= start) & (corpus.day < end)
            if not block.any() or (years and start.year not in years):
                continue
            if len(past) < min_train or past.adjudicated_label.nunique() < 2:
                unscored += int(block.sum())      # left blank, never imputed
                continue
            cached = cache_dir / f"{start.date()}.parquet" if cache_dir else None
            if cached and cached.exists():
                part = pd.read_parquet(cached)
                assert part.row_id.tolist() == corpus.loc[block, "row_id"].tolist(), \
                    f"{cached} was scored on a different corpus"
            else:
                fitted = _fit(past, model)
                print(f"  {start.date()}: fit on {len(past):,} gold rows, scoring {int(block.sum()):,}", flush=True)
                part = corpus.loc[block, ["row_id"]].copy()
                texts = corpus.loc[block, "headline_clean"].astype(str)
                if model == "camembert":    # keep the probabilities, not only the argmax
                    from camembert_clf import CLASSES
                    proba = fitted.predict_proba(texts)
                    part["sent_label"] = [CLASSES[i] for i in proba.argmax(1)]
                    for i, name in enumerate(CLASSES):
                        part[f"p_{name}"] = proba[:, i].round(4)
                else:
                    part["sent_label"] = fitted.predict(texts)
                part["train_n"] = len(past)
                if cached:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    part.to_parquet(cached.with_suffix(".tmp"), index=False)
                    cached.with_suffix(".tmp").replace(cached)   # atomic: a kick-out never leaves half a file
            for col in part.columns.drop(["row_id", "train_n"]):
                corpus.loc[block, col] = part[col].to_numpy()
            corpus.loc[block, "sent_model_train_end"] = start.date().isoformat()
        if years:
            return None
    corpus["sent_score"] = corpus.sent_label.map(LABEL_SCORE)

    keep = ["row_id", "source", "lang", "published_date", "headline_clean",
            "relevance_tag", "sent_label", "sent_score", "sent_model_train_end"]
    keep += [c for c in corpus.columns if c.startswith("p_")]
    scored = corpus[keep].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)

    metadata = {
        "corpus": corpus_path.name,
        "rows_scored": int(len(scored)),
        "train_pool_rows": int(len(train_pool)),
        "model": MODELS[model],
        "gold": gold_path.name,
        "split": split_path.name,
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
                    "pipeline demonstration, NOT a sentiment measurement."
                    if any("v1" in v for v in prompt_versions_used(labelled)) else
                    "Labels are PROMPT_V2, two local annotators (qwen2.5-7b, ministral-8b) "
                    "with a 150-row human anchor; qwen breaks model ties, which decides "
                    "~28% of rows (AUDIT_REPORT 8h). Grade the classifier against the "
                    "human evaluation split before treating these scores as a measurement."),
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
    parser.add_argument("--model", choices=sorted(MODELS), default="tfidf")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-dir", type=Path, help="per-year cache: resumable, splittable")
    parser.add_argument("--years", type=lambda s: {int(y) for y in s.split(",")},
                        help="score only these years into --cache-dir, e.g. 2016,2017")
    args = parser.parse_args()
    if args.years and not args.cache_dir:
        parser.error("--years needs --cache-dir")
    scored = run(gold_path=args.gold, split_path=args.split, output_path=args.output,
                 metadata_path=args.metadata, mode=args.mode, min_train=args.min_train,
                 model=args.model, cache_dir=args.cache_dir, years=args.years)
    if scored is None:
        print(f"cached years {sorted(args.years)} -> {args.cache_dir}")
        return
    blank = int((scored.sent_label == "").sum())
    print(f"mode={args.mode}  scored {len(scored)-blank:,} of {len(scored):,} headlines"
          f"  ({blank:,} left unscored: insufficient prior labels)")
    print(scored.sent_label.value_counts().reindex(LABELS, fill_value=0).to_string())
    print(f"\nmean score {scored.sent_score.mean():+.3f} "
          f"(0 = neutral; positive drift indicates the v1 label skew)")


if __name__ == "__main__":
    main()
