"""Fine-tuned CamemBERT sentiment classifier behind a scikit-learn fit/predict face.

Reconstructs the scorer behind 04_scored_v2_camembert.parquet, which was trained
off-repo and shipped without code or weights. What the metadata records is
reproduced exactly: almanach/camembert-base, 3 classes (very_* collapsed into
their neighbour), 5 epochs, 3 seeds with probabilities averaged, 15% of the
training rows held out as a dev set to log QWK per epoch. What it does NOT record
(learning rate, batch size, max length) is set in HYPERPARAMS and written into
every output, so a guess is never mistaken for the original.

    python3 preprocessing/camembert_clf.py                  # train split -> 150-row human evaluation
    python3 preprocessing/camembert_clf.py --save DIR       # train + validation -> saved model
    python3 preprocessing/camembert_clf.py --score-corpus   # saved model -> 04_scored_v2_camembert_static.parquet

    clf = CamembertClassifier.load("data/models/camembert_3class")
    clf.predict(["La BCT maintient son taux directeur"])     # -> array(['neutral'], ...)

Evaluation rows are never trained on, in either mode. The saved model is for
scoring NEW headlines; research scores come from score_corpus.py --mode expanding,
which refits on the past only. Never use the saved model to score the historical
corpus: it has seen labels from 2026.

--score-corpus does exactly that, on purpose and labelled as such: its output is
marked leakage_free = false and exists only so build_headline_series.py can fill
the years the expanding run cannot score. It is a descriptive fill, never a
feature for a result.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "almanach/camembert-base"
CLASSES = ["negative", "neutral", "positive"]
COLLAPSE = {"very_negative": "negative", "very_positive": "positive"}
HYPERPARAMS = {"epochs": 5, "seeds": [0, 1, 2], "dev_share": 0.15,
               "lr": 2e-5, "batch_size": 16, "max_len": 64, "weight_decay": 0.01,
               "unrecorded_in_original": ["lr", "batch_size", "max_len", "weight_decay"]}


def collapse(labels) -> list[str]:
    return [COLLAPSE.get(l, l) for l in labels]


class CamembertClassifier:
    def __init__(self, **overrides):
        self.hp = {**HYPERPARAMS, **overrides}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.models, self.dev_qwk_curves = [], []

    def _batches(self, texts, labels=None, order=None):
        order = np.arange(len(texts)) if order is None else order
        for i in range(0, len(texts), self.hp["batch_size"]):
            idx = order[i:i + self.hp["batch_size"]]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                           max_length=self.hp["max_len"], return_tensors="pt").to(self.device)
            if labels is not None:
                enc["labels"] = torch.tensor(labels[idx], device=self.device)
            yield enc

    @torch.no_grad()
    def _proba(self, model, texts) -> np.ndarray:
        model.to(self.device).eval()
        out = []
        for enc in self._batches(texts):
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(CLASSES)))

    def fit(self, X, y):
        texts = [str(t) for t in X]
        y = np.array([CLASSES.index(l) for l in collapse(y)])
        self.models, self.dev_qwk_curves = [], []
        for seed in self.hp["seeds"]:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            stratify = y if np.bincount(y, minlength=len(CLASSES)).min() > 1 else None
            tr, dev = train_test_split(np.arange(len(texts)), test_size=self.hp["dev_share"],
                                       random_state=seed, stratify=stratify)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=len(CLASSES)).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.hp["lr"],
                                    weight_decay=self.hp["weight_decay"])
            steps = self.hp["epochs"] * -(-len(tr) // self.hp["batch_size"])
            warmup = max(1, int(0.1 * steps))   # linear warmup, then linear decay to 0
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / (steps - warmup))))
            scaler = torch.amp.GradScaler(enabled=self.device == "cuda")
            tr_texts, dev_texts = [texts[i] for i in tr], [texts[i] for i in dev]
            curve = []
            for _ in range(self.hp["epochs"]):
                model.train()
                for enc in self._batches(tr_texts, y[tr], order=rng.permutation(len(tr))):
                    with torch.autocast(self.device, enabled=self.device == "cuda"):
                        loss = model(**enc).loss
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                pred = self._proba(model, dev_texts).argmax(1)
                curve.append(round(float(cohen_kappa_score(y[dev], pred, weights="quadratic")), 4))
            model.to("cpu")
            self.models.append(model)
            self.dev_qwk_curves.append(curve)
            del opt, scaler
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, X) -> np.ndarray:
        texts = [str(t) for t in X]
        probs = []
        for m in self.models:
            probs.append(self._proba(m, texts))
            m.to("cpu")
        return np.mean(probs, axis=0)

    def predict(self, X) -> np.ndarray:
        return np.array(CLASSES)[self.predict_proba(X).argmax(1)]

    def save(self, path, **provenance) -> None:
        path = Path(path)
        for i, m in enumerate(self.models):
            m.save_pretrained(path / f"seed{i}")
        self.tok.save_pretrained(path / "tokenizer")
        (path / "camembert_clf.json").write_text(json.dumps(
            {"base_model": MODEL_NAME, "classes": CLASSES, "hyperparams": self.hp,
             "dev_qwk_curves": self.dev_qwk_curves, **provenance}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CamembertClassifier":
        path = Path(path)
        meta = json.loads((path / "camembert_clf.json").read_text(encoding="utf-8"))
        clf = cls.__new__(cls)
        clf.hp, clf.dev_qwk_curves = meta["hyperparams"], meta["dev_qwk_curves"]
        clf.device = "cuda" if torch.cuda.is_available() else "cpu"
        clf.tok = AutoTokenizer.from_pretrained(path / "tokenizer")
        clf.models = [AutoModelForSequenceClassification.from_pretrained(p)
                      for p in sorted(path.glob("seed*"))]
        return clf


def main() -> None:
    from score_corpus import CURATED
    from sentiment_baseline import load, score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="train on train+validation and save here")
    parser.add_argument("--score-corpus", action="store_true",
                        help="score every relevant headline with the saved model (LEAKS: static fit)")
    parser.add_argument("--model-dir", type=Path, default=CURATED.parent / "models" / "camembert_3class")
    parser.add_argument("--gold", type=Path, default=CURATED / "sentiment_gold_v2_full.csv")
    parser.add_argument("--split", type=Path, default=CURATED / "sentiment_gold_v2_full_split.csv")
    parser.add_argument("--out", type=Path,
                        default=CURATED / "finetune" / "camembert_3class_evaluation_repro.json")
    args = parser.parse_args()
    if args.score_corpus:
        import pandas as pd
        from score_corpus import DEFAULT_CORPUS, LABEL_SCORE
        corpus = pd.read_parquet(DEFAULT_CORPUS)
        corpus = corpus[corpus.is_canonical & (corpus.relevance_tag != "other")
                        & corpus.date_parse_ok][["row_id", "headline_clean"]].reset_index(drop=True)
        clf = CamembertClassifier.load(args.model_dir)
        proba = clf.predict_proba(corpus.headline_clean)
        scored = corpus[["row_id"]].copy()
        scored["sent_label"] = np.array(CLASSES)[proba.argmax(1)]
        scored["sent_score"] = scored.sent_label.map(LABEL_SCORE)
        for i, name in enumerate(CLASSES):
            scored[f"p_{name}"] = proba[:, i].round(4)
        out = CURATED / "04_scored_v2_camembert_static.parquet"
        scored.to_parquet(out, index=False)
        out.with_name(out.stem + "_metadata.json").write_text(json.dumps({
            "model_dir": str(args.model_dir.relative_to(CURATED.parent.parent)), "mode": "static",
            "leakage_free": False, "rows_scored": int(len(scored)),
            "warning": "One model fitted on gold labels from every year (train+validation) scored "
                       "the whole corpus, so a 2014 headline is scored by a model that saw 2026 "
                       "labels, and ~2,780 headlines are scored in-sample. Descriptive fill only.",
            "label_distribution": scored.sent_label.value_counts().to_dict()}, indent=2), encoding="utf-8")
        print(f"scored {len(scored):,} headlines -> {out}")
        print(scored.sent_label.value_counts().to_string())
        return
    frame = load(args.gold, args.split)
    if args.save:
        fit_rows = frame[frame.split.isin(["train", "validation"])]
        assert not fit_rows.gold_item_id.isin(frame[frame.split == "evaluation"].gold_item_id).any()
        start = time.time()
        clf = CamembertClassifier().fit(fit_rows.headline_clean, fit_rows.adjudicated_label)
        clf.save(args.save, trained_on="sentiment_gold_v2_full.csv, split in {train, validation}",
                 train_n=int(len(fit_rows)), excluded="evaluation (150 human rows)",
                 expected_quality="QWK ~0.62 on the human split, measured on a train-only fit "
                                  "(finetune/camembert_3class_evaluation_repro.json)",
                 minutes=round((time.time() - start) / 60, 1))
        print(f"saved {len(clf.models)} seeds, {len(fit_rows):,} rows -> {args.save}")
        return
    train, held = frame[frame.split == "train"], frame[frame.split == "evaluation"]
    start = time.time()
    clf = CamembertClassifier().fit(train.headline_clean, train.adjudicated_label)
    majority = max(CLASSES, key=collapse(list(train.adjudicated_label)).count)
    truth = np.array(collapse(held.adjudicated_label))
    texts = list(held.headline_clean)
    per_seed = []
    for m in clf.models:
        pred = np.array(CLASSES)[clf._proba(m, texts).argmax(1)]
        m.to("cpu")
        per_seed.append(round(float(cohen_kappa_score(truth, pred, weights="quadratic")), 4))
    result = {"held_out_split": "evaluation", "classes": 3, "model": MODEL_NAME,
              "train_n": int(len(train)), "held_out_n": int(len(held)),
              "majority_class": majority, "hyperparams": clf.hp,
              "dev_qwk_curves": clf.dev_qwk_curves,
              "minutes": round((time.time() - start) / 60, 1),
              "camembert_finetuned": {**score(truth, clf.predict(texts), majority),
                                      "per_seed_qwk": per_seed},
              "original_committed": {"qwk_ordinal": 0.6349, "accuracy": 0.66,
                                     "source": "finetune/camembert_3class_evaluation.json"}}
    result["gold"] = args.gold.name
    out = args.out
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["camembert_finetuned"], indent=2), f"\n-> {out}")


if __name__ == "__main__":
    main()
