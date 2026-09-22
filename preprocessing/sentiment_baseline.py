"""TF-IDF + logistic sentiment classifier: the bar CamemBERT has to beat.

WHY THIS BEFORE A TRANSFORMER
-----------------------------
Fine-tuning CamemBERT costs a dependency stack, GPU time and a training loop. A
character-ngram TF-IDF model costs three seconds and zero new dependencies. If
the transformer only matches it, the ceiling is set by LABEL QUALITY, not model
capacity -- and no amount of GPU fixes that. Run this first so that comparison is
available.

METRICS
-------
Labels are ORDINAL (very_negative < ... < very_positive), so the headline metric
is quadratic-weighted kappa: it penalises very_negative->positive far more than
negative->neutral. Plain accuracy is reported too, always against the
majority-class floor, because 58.8% of the training labels are `positive` and a
model that always says `positive` already scores that.

The frozen `evaluation` split is NOT touched here. Model selection happens on
`validation`; evaluation is spent once, at the end, on the final model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, f1_score)
from sklearn.pipeline import make_pipeline

from gold import LABELS

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_GOLD = CURATED / "sentiment_gold_annotation1.csv"
DEFAULT_SPLIT = CURATED / "sentiment_gold_split.csv"
DEFAULT_OUTPUT = CURATED / "sentiment_baseline_results.json"


def load(gold_path: Path = DEFAULT_GOLD,
         split_path: Path = DEFAULT_SPLIT) -> pd.DataFrame:
    gold = pd.read_csv(gold_path, keep_default_na=False)
    split = pd.read_csv(split_path)[["gold_item_id", "split"]]
    frame = gold.merge(split, on="gold_item_id", how="inner")
    if frame.empty:
        raise SystemExit("No rows after joining gold to split -- run split.py first.")
    return frame


def build_model() -> object:
    # Word + char ngrams: char ngrams carry French morphology (recul/recule/
    # reculent) that a word model treats as unrelated tokens on a 2k-row set.
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        min_df=2, sublinear_tf=True, max_features=60000),
        LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"),
    )


def score(y_true, y_pred, majority: str) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "majority_floor": round(float((y_true == majority).mean()), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=LABELS, zero_division=0)), 4),
        "qwk_ordinal": round(float(cohen_kappa_score(y_true, y_pred, labels=LABELS,
                                                     weights="quadratic")), 4),
        "kappa_nominal": round(float(cohen_kappa_score(y_true, y_pred,
                                                       labels=LABELS)), 4),
    }


def run(gold_path: Path = DEFAULT_GOLD, split_path: Path = DEFAULT_SPLIT,
        output_path: Path = DEFAULT_OUTPUT, use_evaluation: bool = False) -> dict:
    frame = load(gold_path, split_path)
    train = frame[frame.split == "train"]
    held = frame[frame.split == ("evaluation" if use_evaluation else "validation")]

    model = build_model()
    model.fit(train.headline_clean, train.adjudicated_label)
    majority = train.adjudicated_label.value_counts().idxmax()

    results = {
        "held_out_split": "evaluation" if use_evaluation else "validation",
        "train_n": int(len(train)),
        "majority_class": majority,
        "train": score(train.adjudicated_label,
                       model.predict(train.headline_clean), majority),
        "held_out": score(held.adjudicated_label,
                          model.predict(held.headline_clean), majority),
        "per_class": classification_report(
            held.adjudicated_label, model.predict(held.headline_clean),
            labels=LABELS, output_dict=True, zero_division=0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="store_true",
                        help="spend the frozen evaluation split (do this ONCE)")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    r = run(args.gold, args.split, output_path=args.output, use_evaluation=args.evaluation)

    print(f"train n={r['train_n']}  majority class = {r['majority_class']}")
    print(f"{'':<12}{'acc':>9}{'floor':>9}{'macroF1':>10}{'QWK':>9}{'kappa':>9}")
    print("-" * 58)
    for name in ("train", "held_out"):
        s = r[name]
        tag = name if name == "train" else r["held_out_split"]
        print(f"{tag:<12}{s['accuracy']:>9.4f}{s['majority_floor']:>9.4f}"
              f"{s['macro_f1']:>10.4f}{s['qwk_ordinal']:>9.4f}{s['kappa_nominal']:>9.4f}")
    print("\nper-class F1 on held-out:")
    for label in LABELS:
        d = r["per_class"][label]
        print(f"  {label:<15} f1={d['f1-score']:.3f}  support={int(d['support'])}")


if __name__ == "__main__":
    main()
