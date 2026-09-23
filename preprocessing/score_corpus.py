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

SCORERS (--scorer, one or more; several are averaged at the probability level)
  tfidf     char-ngram TF-IDF + logistic regression (sentiment_baseline.py)
  finbert   linear head on frozen FinBERT embeddings (finbert_head.py). Needs two
            caches from finbert_embed.py: the gold set (keyed by gold_item_id)
            and the corpus (keyed by row_id, --id-column row_id).
  camembert fine-tuned French encoder (camembert_finetune.py). --ft-epochs is
            fixed rather than re-searched each year, and should come from the
            full-train dev search that camembert_finetune.py reports.
Every scorer is refit inside each expanding window, so the leakage guarantee
above holds for all of them, not only TF-IDF.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gold import LABELS
from sentiment_baseline import COLLAPSE_3, LABELS_3, build_model

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_CORPUS = CURATED / "03_dedup.parquet"
DEFAULT_GOLD = CURATED / "sentiment_gold_v2_full.csv"
DEFAULT_SPLIT = CURATED / "sentiment_gold_v2_full_split.csv"
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


def _provenance_warning(versions: list[str]) -> str:
    """The caveat must describe the labels actually used, not the ones this file
    was first written for."""
    if any(v.startswith("v1") for v in versions):
        return ("Labels derive from a single 7B annotator under PROMPT_V1, which did "
                "not define the task (AUDIT_REPORT 8d). These scores are a pipeline "
                "demonstration, NOT a sentiment measurement.")
    return ("Labels are PROMPT_V2, two local annotators (qwen2.5-7b, ministral-8b) "
            "with a 150-row human anchor; qwen breaks model ties, which decides ~28% "
            "of rows (AUDIT_REPORT 8h). Grade the classifier against the human "
            "evaluation split before treating these scores as a measurement.")


def _as_frame(probs: np.ndarray, classes, labels: list[str]) -> pd.DataFrame:
    """Probabilities on the full label list. An early window can lack a class
    entirely; it gets probability 0 there instead of shifting the columns."""
    return pd.DataFrame(probs, columns=list(classes)).reindex(columns=labels, fill_value=0.0)


def tfidf_scorer(labels: list[str]):
    def fit(train: pd.DataFrame):
        model = build_model().fit(train.headline_clean, train.adjudicated_label)
        return lambda rows: _as_frame(model.predict_proba(rows.headline_clean.astype(str)),
                                      model.classes_, labels)
    return fit


def finbert_scorer(labels: list[str], gold_cache: Path, corpus_cache: Path):
    import finbert_embed
    from finbert_head import fit_head

    def table(cache):
        ids, vectors, _, meta = finbert_embed.load(cache)
        return pd.Series(np.arange(len(ids)), index=ids), vectors, meta

    gold_pos, gold_vec, gold_meta = table(gold_cache)
    corpus_pos, corpus_vec, corpus_meta = table(corpus_cache)
    if gold_meta["translated"] != corpus_meta["translated"]:
        raise ValueError("gold and corpus embeddings differ in translation -- not comparable")

    def lookup(position, vectors, keys):
        keys = keys.astype(str)
        missing = ~keys.isin(position.index)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} rows have no embedding")
        return vectors[position.loc[keys].to_numpy()]

    def fit(train: pd.DataFrame):
        head, _, _ = fit_head(lookup(gold_pos, gold_vec, train.gold_item_id),
                              train.adjudicated_label.to_numpy(),
                              [l for l in labels if l in set(train.adjudicated_label)])
        return lambda rows: _as_frame(
            head.predict_proba(lookup(corpus_pos, corpus_vec, rows.row_id)),
            head.classes_, labels)
    fit.meta = {"translated": gold_meta["translated"],
                "encoder_revision": gold_meta["encoder_revision"],
                "translator_revision": gold_meta.get("translator_revision")}
    return fit


def camembert_scorer(labels: list[str], model_key: str, epochs: int, seeds: int):
    import camembert_finetune

    def fit(train: pd.DataFrame):
        present = [l for l in labels if l in set(train.adjudicated_label)]
        model = camembert_finetune.fit(train.headline_clean, train.adjudicated_label,
                                       present, model_key, seeds, epochs)
        return lambda rows: _as_frame(model.predict_proba(rows.headline_clean.astype(str)),
                                      model.classes_, labels)
    fit.meta = {"model": camembert_finetune.MODELS.get(model_key, model_key),
                "epochs": epochs, "seeds": seeds}
    return fit


def averaged(fitters):
    """Fit every scorer on the same window; average their probabilities."""
    def fit(train: pd.DataFrame):
        predictors = [f(train) for f in fitters]
        return lambda rows: sum(p(rows) for p in predictors) / len(predictors)
    return fit


def _predict(predictor, rows: pd.DataFrame) -> np.ndarray:
    probs = predictor(rows)
    return probs.columns.to_numpy()[probs.to_numpy().argmax(axis=1)]


def run(corpus_path: Path = DEFAULT_CORPUS, gold_path: Path = DEFAULT_GOLD,
        split_path: Path = DEFAULT_SPLIT, output_path: Path = DEFAULT_OUTPUT,
        metadata_path: Path = DEFAULT_METADATA, mode: str = "expanding",
        min_train: int = 200, freq: str = "YS", classes: int = 3,
        scorers: tuple[str, ...] = ("tfidf",), gold_embeddings: Path | None = None,
        corpus_embeddings: Path | None = None, ft_model: str = "camembert",
        ft_epochs: int = 4, ft_seeds: int = 3) -> pd.DataFrame:
    if mode not in ("expanding", "static"):
        raise ValueError("mode must be 'expanding' or 'static'")
    if classes not in (3, 5):
        raise ValueError("classes must be 3 or 5")
    gold = pd.read_csv(gold_path, keep_default_na=False)
    split = pd.read_csv(split_path)[["gold_item_id", "split"]]
    labelled = gold.merge(split, on="gold_item_id", how="inner")
    if classes == 3:
        labelled["adjudicated_label"] = labelled["adjudicated_label"].map(COLLAPSE_3)
    labelled["day"] = pd.to_datetime(labelled.published_date, errors="coerce")
    train_pool = labelled[labelled.split == "train"].dropna(subset=["day"])
    if train_pool.empty:
        raise SystemExit("No training rows -- run split.py first.")

    labels = LABELS_3 if classes == 3 else LABELS
    fitters, scorer_meta = [], {}
    for name in scorers:
        if name == "tfidf":
            fitters.append(tfidf_scorer(labels))
            scorer_meta[name] = "TfidfVectorizer(char_wb 2-5) + LogisticRegression(balanced)"
        elif name == "finbert":
            if not (gold_embeddings and corpus_embeddings):
                raise SystemExit("--scorer finbert needs --gold-embeddings and --corpus-embeddings")
            fitters.append(finbert_scorer(labels, gold_embeddings, corpus_embeddings))
            scorer_meta[name] = {"head": "StandardScaler + LogisticRegression(balanced), "
                                         "C by 5-fold CV inside each window",
                                 **fitters[-1].meta}
        elif name == "camembert":
            fitters.append(camembert_scorer(labels, ft_model, ft_epochs, ft_seeds))
            scorer_meta[name] = fitters[-1].meta
        else:
            raise ValueError(f"unknown scorer {name!r}")
    fit = averaged(fitters)

    corpus = pd.read_parquet(corpus_path)
    corpus = corpus[corpus.is_canonical & (corpus.relevance_tag != "other")
                    & corpus.date_parse_ok].copy()
    corpus["day"] = pd.to_datetime(corpus.published_date, errors="coerce")
    corpus = corpus.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)

    unscored = 0
    if mode == "static":
        corpus["sent_label"] = _predict(fit(train_pool), corpus)
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
            corpus.loc[block, "sent_label"] = _predict(fit(past), corpus.loc[block])
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
        "model": " + ".join(scorers) + (" (probabilities averaged)" if len(scorers) > 1 else ""),
        "scorers": scorer_meta,
        "label_scale": LABEL_SCORE,
        "mode": mode,
        "leakage_free": mode == "expanding",
        "rows_unscored_insufficient_history": unscored,
        "min_train": min_train,
        "refit_frequency": freq,
        "monotonicity": monotonicity_check(labelled),
        "prompt_versions_in_training_labels": prompt_versions_used(labelled),
        "classes": classes,
        "gold": gold_path.name,
        "provisional": True,
        "warning": _provenance_warning(prompt_versions_used(labelled)),
        "label_distribution": scored.sent_label.value_counts()
                                    .reindex(LABELS_3 if classes == 3 else LABELS,
                                             fill_value=0).astype(int).to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mode", choices=("expanding", "static"), default="expanding",
                        help="expanding = leakage-free refit; static LEAKS, diagnostic only")
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--classes", type=int, choices=(3, 5), default=3)
    parser.add_argument("--scorer", nargs="+", default=["tfidf"],
                        choices=("tfidf", "finbert", "camembert"),
                        help="several = average of their class probabilities")
    parser.add_argument("--gold-embeddings", type=Path,
                        help="finbert_embed.py cache of the gold set (finbert scorer)")
    parser.add_argument("--corpus-embeddings", type=Path,
                        help="finbert_embed.py cache of the corpus, --id-column row_id")
    parser.add_argument("--ft-model", default="camembert", help="camembert or xlmr")
    parser.add_argument("--ft-epochs", type=int, default=4)
    parser.add_argument("--ft-seeds", type=int, default=3)
    args = parser.parse_args()
    scored = run(corpus_path=args.corpus, gold_path=args.gold, split_path=args.split, output_path=args.output,
                 metadata_path=args.metadata, mode=args.mode, min_train=args.min_train,
                 classes=args.classes, scorers=tuple(args.scorer),
                 gold_embeddings=args.gold_embeddings,
                 corpus_embeddings=args.corpus_embeddings, ft_model=args.ft_model,
                 ft_epochs=args.ft_epochs, ft_seeds=args.ft_seeds)
    blank = int((scored.sent_label == "").sum())
    print(f"mode={args.mode}  scored {len(scored)-blank:,} of {len(scored):,} headlines"
          f"  ({blank:,} left unscored: insufficient prior labels)")
    print(scored.sent_label.value_counts().reindex(
        LABELS_3 if args.classes == 3 else LABELS, fill_value=0).to_string())
    print(f"\nmean score {scored.sent_score.mean():+.3f} "
          f"(0 = neutral)")


if __name__ == "__main__":
    main()
