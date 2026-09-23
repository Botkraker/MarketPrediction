"""Fine-tune a native French encoder (CamemBERT, or XLM-R) as the sentiment scorer.

WHY
Blueprint section 4.2 option E. FinBERT only reaches this corpus through
OPUS-MT, and translation mangles exactly the words that carry sentiment
("sans entrain" -> "without training"). A French encoder reads the headline
as written.

NO VALIDATION NUMBER IS USED TO CHOOSE ANYTHING
The epoch count is picked on a stratified DEV fold carved out of `train`
(DEV_SHARE of it), never on `validation`. The model is then refit on the whole
of `train` for that many epochs, so validation stays as clean for this variant
as it is for finbert_head.py's CV-chosen C. Everything else (learning rate,
batch size, max length) is a fixed, standard setting, not searched.

SEED VARIANCE
Fine-tuning ~2.3k rows moves a point or two of QWK with the seed alone, which is
about the gap between the variants being compared. So N_SEEDS models are trained
and their class probabilities averaged. The per-seed scores are reported so the
spread is visible, and the seed average is the variant that gets compared.

The validation-row probabilities are written out so an ensemble with TF-IDF can
be scored without retraining. The evaluation split is not touched unless
--evaluation is given.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from finbert_head import load_frame
from sentiment_baseline import build_model, score

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_GOLD = CURATED / "sentiment_gold_v2_full.csv"
DEFAULT_SPLIT = CURATED / "sentiment_gold_v2_full_split.csv"
DEFAULT_OUTPUT_DIR = CURATED / "finetune"
MODELS = {"camembert": "almanach/camembert-base",
          "xlmr": "FacebookAI/xlm-roberta-base"}
MAX_LENGTH = 64
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1
MAX_EPOCHS = 6
DEV_SHARE = 0.15
N_SEEDS = 3
SEED = 20260923


def class_weights(y: np.ndarray, labels: list[str]) -> np.ndarray:
    """sklearn's 'balanced' weights, so the loss treats classes as the TF-IDF
    and FinBERT-head baselines do."""
    counts = np.array([(y == label).sum() for label in labels], dtype=float)
    return len(y) / (len(labels) * np.maximum(counts, 1.0))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Scorer:
    """A fine-tuned model plus its tokenizer; predict_proba over raw text.

    `classes_` mirrors sklearn so the average_proba helper in finbert_head.py
    can combine it with the TF-IDF pipeline directly."""

    def __init__(self, model, tokenizer, labels: list[str], meta: dict):
        self.model, self.tokenizer = model, tokenizer
        self.classes_ = np.array(labels)
        self.meta = meta

    def predict_proba(self, texts, batch_size: int = 128) -> np.ndarray:
        import torch
        texts = [str(t) for t in texts]
        device = next(self.model.parameters()).device
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=MAX_LENGTH).to(device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                logits = self.model(**batch).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.empty((0, len(self.classes_)))

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def _load_pretrained(model_name: str, n_labels: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=n_labels)
    return model, tokenizer


def train_one(texts, y, labels: list[str], model_name: str, epochs: int, seed: int,
              eval_texts=None, eval_y=None, loader=_load_pretrained) -> tuple[Scorer, list]:
    """Fine-tune for `epochs`. If eval rows are given, score them after every
    epoch (QWK) and return that curve; the caller picks the epoch from it."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(_device())
    model, tokenizer = loader(model_name, len(labels))
    model.to(device)
    index = {label: i for i, label in enumerate(labels)}
    targets = torch.tensor([index[label] for label in y], dtype=torch.long)
    texts = [str(t) for t in texts]
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(np.asarray(y), labels), dtype=torch.float32,
                            device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    steps = epochs * int(np.ceil(len(texts) / BATCH_SIZE))
    schedule = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * steps), steps)
    scorer = Scorer(model, tokenizer, labels, {"model": model_name, "epochs": epochs,
                                               "seed": seed})
    generator = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(epochs):
        model.train()
        for batch_idx in torch.randperm(len(texts), generator=generator).split(BATCH_SIZE):
            batch = tokenizer([texts[i] for i in batch_idx], return_tensors="pt",
                              padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), targets[batch_idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
        if eval_texts is not None:
            majority = pd.Series(y).value_counts().idxmax()
            curve.append(score(np.asarray(eval_y), scorer.predict(eval_texts),
                               majority, labels)["qwk_ordinal"])
    return scorer, curve


def choose_epochs(curves: list[list[float]]) -> int:
    """Epoch count with the best mean dev QWK across seeds; ties go to fewer."""
    mean = np.mean(np.array(curves), axis=0)
    return int(np.argmax(mean)) + 1


def fit(texts, y, labels: list[str], model_key: str = "camembert", seeds: int = N_SEEDS,
        epochs: int | None = None, loader=_load_pretrained) -> "SeedEnsemble":
    """Choose epochs on a dev fold of (texts, y) unless given, then refit on all
    of it once per seed. This is the unit score_corpus.py refits per year."""
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    model_name = MODELS.get(model_key, model_key)
    curves = []
    if epochs is None:
        fit_idx, dev_idx = train_test_split(np.arange(len(y)), test_size=DEV_SHARE,
                                            stratify=y, random_state=SEED)
        for s in range(seeds):
            _, curve = train_one(texts[fit_idx], y[fit_idx], labels, model_name,
                                 MAX_EPOCHS, SEED + s, texts[dev_idx], y[dev_idx], loader)
            curves.append(curve)
        epochs = choose_epochs(curves)
    members = [train_one(texts, y, labels, model_name, epochs, SEED + s, loader=loader)[0]
               for s in range(seeds)]
    return SeedEnsemble(members, labels, {"model": model_name, "epochs": epochs,
                                          "seeds": seeds, "dev_qwk_curves": curves})


class SeedEnsemble:
    def __init__(self, members: list[Scorer], labels: list[str], meta: dict):
        self.members, self.meta = members, meta
        self.classes_ = np.array(labels)

    def predict_proba(self, texts) -> np.ndarray:
        return np.mean([m.predict_proba(texts) for m in self.members], axis=0)

    def predict(self, texts) -> np.ndarray:
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def run(gold_path: Path = DEFAULT_GOLD, split_path: Path = DEFAULT_SPLIT,
        model_key: str = "camembert", output_dir: Path = DEFAULT_OUTPUT_DIR,
        classes: int = 3, seeds: int = N_SEEDS, use_evaluation: bool = False,
        loader=_load_pretrained) -> dict:
    frame, labels = load_frame(gold_path, split_path, classes)
    held_name = "evaluation" if use_evaluation else "validation"
    train = frame[frame.split == "train"]
    held = frame[frame.split == held_name]
    y_train, y_held = train.adjudicated_label.to_numpy(), held.adjudicated_label.to_numpy()
    majority = pd.Series(y_train).value_counts().idxmax()

    started = time.time()
    ensemble = fit(train.headline_clean, y_train, labels, model_key, seeds, loader=loader)
    probs = ensemble.predict_proba(held.headline_clean)
    per_seed = [score(y_held, m.predict(held.headline_clean), majority, labels)["qwk_ordinal"]
                for m in ensemble.members]

    tfidf = build_model().fit(train.headline_clean, y_train)
    if list(tfidf.classes_) != labels:
        raise ValueError("TF-IDF class order differs from the label list")
    blend = (probs + tfidf.predict_proba(held.headline_clean)) / 2

    variants = {
        f"{model_key}_finetuned": {**score(y_held, ensemble.classes_[probs.argmax(1)],
                                           majority, labels),
                                   "per_seed_qwk": per_seed, "epochs": ensemble.meta["epochs"]},
        f"ensemble_tfidf_{model_key}": score(y_held, np.array(labels)[blend.argmax(1)],
                                             majority, labels),
    }
    results = {"held_out_split": held_name, "classes": classes, "model": ensemble.meta["model"],
               "train_n": int(len(train)), "held_out_n": int(len(held)),
               "majority_class": majority, "seeds": seeds,
               "dev_share": DEV_SHARE, "dev_qwk_curves": ensemble.meta["dev_qwk_curves"],
               "minutes": round((time.time() - started) / 60, 1), "variants": variants}

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_key}_{classes}class_{held_name}"
    (output_dir / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(probs, columns=[f"p_{label}" for label in labels]).assign(
        gold_item_id=held.gold_item_id.to_numpy(), label=y_held).to_csv(
        output_dir / f"{stem}_probs.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="camembert",
                        help=f"one of {sorted(MODELS)} or a Hugging Face model id")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--classes", type=int, choices=(3, 5), default=3)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--evaluation", action="store_true",
                        help="spend the frozen human evaluation split (do this ONCE)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    r = run(args.gold, args.split, args.model, args.output_dir, args.classes, args.seeds,
            args.evaluation)
    print(f"{r['model']}  {r['held_out_split']} n={r['held_out_n']}  train n={r['train_n']}"
          f"  {r['minutes']} min")
    print("dev QWK by epoch, per seed:")
    for curve in r["dev_qwk_curves"]:
        print("  " + "  ".join(f"{q:.3f}" for q in curve))
    print(f"{'variant':<30}{'QWK':>8}{'macroF1':>9}{'acc':>8}{'floor':>8}")
    for name, s in r["variants"].items():
        print(f"{name:<30}{s['qwk_ordinal']:>8.4f}{s['macro_f1']:>9.4f}"
              f"{s['accuracy']:>8.4f}{s['majority_floor']:>8.4f}"
              + (f"   seeds {s['per_seed_qwk']} epochs={s['epochs']}" if "epochs" in s else ""))


if __name__ == "__main__":
    main()
