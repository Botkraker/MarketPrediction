"""Train a classifier head on frozen FinBERT embeddings, and compare it fairly.

The head is a linear probe: standardised embeddings -> class-balanced logistic
regression. Its regularisation strength C is chosen by 5-fold cross-validation
INSIDE the training split, scored by quadratic-weighted kappa, so the validation
number reported afterwards was never used to pick anything.

Every variant is scored on the same held-out rows with the same metrics:
  tfidf               the existing char-ngram baseline (sentiment_baseline.py)
  finbert_zero_shot   FinBERT's own head, no Tunisian training (reference only)
  finbert_head_*      the trained probe, one per embedding cache passed in

The frozen `evaluation` split -- the human anchor's 150 rows -- is not touched
unless --evaluation is given. Spend it once, on the variant that wins here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import finbert_embed
from gold import LABELS
from sentiment_baseline import COLLAPSE_3, LABELS_3, build_model, score

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_GOLD = CURATED / "sentiment_gold_v2_full.csv"
DEFAULT_SPLIT = CURATED / "sentiment_gold_v2_full_split.csv"
DEFAULT_OUTPUT = CURATED / "finbert_head_results.json"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEED = 20260923


def load_frame(gold_path: Path, split_path: Path, classes: int) -> tuple[pd.DataFrame, list[str]]:
    gold = pd.read_csv(gold_path, keep_default_na=False)
    split = pd.read_csv(split_path)[["gold_item_id", "split"]]
    frame = gold.merge(split, on="gold_item_id", how="inner")
    if frame.empty:
        raise SystemExit("No rows after joining gold to split -- run split.py first.")
    if classes == 3:
        frame["adjudicated_label"] = frame["adjudicated_label"].map(COLLAPSE_3)
        return frame, LABELS_3
    return frame, LABELS


def aligned(frame: pd.DataFrame, cache: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embedding and zero-shot rows in `frame` order; refuses any missing id."""
    ids, vectors, zero_shot, meta = finbert_embed.load(cache)
    position = pd.Series(np.arange(len(ids)), index=ids)
    wanted = frame["gold_item_id"].astype(str)
    missing = set(wanted) - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} gold rows have no embedding in {cache.name}")
    rows = position.loc[wanted].to_numpy()
    return vectors[rows], zero_shot[rows], meta


def fit_head(x_train, y_train, labels: list[str]):
    qwk = make_scorer(cohen_kappa_score, weights="quadratic", labels=labels)
    search = GridSearchCV(
        make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=5000)),
        {"logisticregression__C": list(C_GRID)},
        scoring=qwk, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    search.fit(x_train, y_train)
    return search.best_estimator_, float(search.best_params_["logisticregression__C"]), \
        round(float(search.best_score_), 4)


def zero_shot_labels(probs: np.ndarray, zs_order: list[str]) -> np.ndarray:
    """FinBERT's own argmax. Its labels (positive/negative/neutral) are already
    valid names at both 3 and 5 classes; it simply never predicts very_*."""
    return np.array(zs_order)[probs.argmax(axis=1)]


def run(gold_path: Path = DEFAULT_GOLD, split_path: Path = DEFAULT_SPLIT,
        caches: list[Path] | None = None, output_path: Path = DEFAULT_OUTPUT,
        classes: int = 3, use_evaluation: bool = False) -> dict:
    frame, labels = load_frame(gold_path, split_path, classes)
    held_name = "evaluation" if use_evaluation else "validation"
    train = frame["split"].eq("train").to_numpy()
    held = frame["split"].eq(held_name).to_numpy()
    y = frame["adjudicated_label"].to_numpy()
    majority = pd.Series(y[train]).value_counts().idxmax()

    results = {"held_out_split": held_name, "classes": classes,
               "train_n": int(train.sum()), "held_out_n": int(held.sum()),
               "majority_class": majority, "variants": {}}

    tfidf = build_model().fit(frame.loc[train, "headline_clean"], y[train])
    results["variants"]["tfidf"] = score(
        y[held], tfidf.predict(frame.loc[held, "headline_clean"]), majority, labels)

    for cache in caches or []:
        cache = Path(cache)
        x, zero_shot, meta = aligned(frame, cache)
        tag = "translated" if meta["translated"] else "raw"
        if f"finbert_zero_shot_{tag}" not in results["variants"]:
            results["variants"][f"finbert_zero_shot_{tag}"] = score(
                y[held], zero_shot_labels(zero_shot[held], meta["zero_shot_labels"]),
                majority, labels)
        head, best_c, cv_qwk = fit_head(x[train], y[train], labels)
        entry = score(y[held], head.predict(x[held]), majority, labels)
        entry.update({"C": best_c, "cv_qwk_on_train": cv_qwk,
                      "encoder_revision": meta["encoder_revision"],
                      "translator_revision": meta.get("translator_revision")})
        results["variants"][f"finbert_head_{tag}"] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True,
                        help="one or more caches written by finbert_embed.py")
    parser.add_argument("--classes", type=int, choices=(3, 5), default=3)
    parser.add_argument("--evaluation", action="store_true",
                        help="spend the frozen human evaluation split (do this ONCE)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    r = run(args.gold, args.split, args.embeddings, args.output, args.classes, args.evaluation)
    print(f"{r['held_out_split']} n={r['held_out_n']}  train n={r['train_n']}  "
          f"classes={r['classes']}  floor={r['majority_class']}")
    print(f"{'variant':<28}{'QWK':>8}{'macroF1':>9}{'acc':>8}{'floor':>8}   notes")
    for name, s in r["variants"].items():
        note = f"C={s['C']} cv_qwk={s['cv_qwk_on_train']}" if "C" in s else ""
        print(f"{name:<28}{s['qwk_ordinal']:>8.4f}{s['macro_f1']:>9.4f}"
              f"{s['accuracy']:>8.4f}{s['majority_floor']:>8.4f}   {note}")


if __name__ == "__main__":
    main()
