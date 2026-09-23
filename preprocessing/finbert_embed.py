"""Embed headlines with a frozen FinBERT encoder, optionally after FR->EN translation.

WHY THE TRANSLATE SWITCH EXISTS
FinBERT (ProsusAI/finbert) is English-only; ~97% of this corpus is French.
Blueprint section 4.1: applied to French it produces "confident but meaningless
scores" -- its WordPiece vocabulary shreds French into sub-word fragments.
Section 4.2 Option A is the blueprint's route to using FinBERT at all: translate
with a free local MT model first. Both variants are embedded so the difference
is measured rather than asserted.

WHAT ONE PASS PRODUCES (cached, so the GPU step runs once per variant)
  embedding    mean-pooled last hidden state, 768-d -- input to finbert_head.py
  zero-shot    FinBERT's own positive/negative/neutral probabilities, a free
               reference point: what FinBERT says with no Tunisian training
  translation  the English text actually embedded (translate variant only),
               written out so translation quality can be read, not assumed

Model revisions (commit hashes) are recorded with the cache: blueprint
section 4.4 requires every score to name the scorer version that produced it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
DEFAULT_INPUT = CURATED / "sentiment_gold_v2_full.csv"
DEFAULT_CACHE_DIR = CURATED / "embeddings"
ENCODER = "ProsusAI/finbert"
TRANSLATOR = "Helsinki-NLP/opus-mt-fr-en"
MAX_LENGTH = 64          # headlines; the longest in the gold set is well under this


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def translate(texts: list[str], langs: list[str], model_name: str = TRANSLATOR,
              batch_size: int = 32) -> tuple[list[str], str]:
    """Translate the French rows; other languages pass through unchanged.
    Returns (texts, model revision)."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = _device()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device).eval()
    out = list(texts)
    french = [i for i, lang in enumerate(langs) if lang == "fr"]
    for start in range(0, len(french), batch_size):
        index = french[start:start + batch_size]
        batch = tokenizer([texts[i] for i in index], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LENGTH * 2).to(device)
        with torch.no_grad():
            generated = model.generate(**batch, num_beams=4, max_new_tokens=MAX_LENGTH * 2)
        for i, text in zip(index, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            out[i] = text
    return out, getattr(model.config, "_commit_hash", None) or "unknown"


def mean_pool(hidden, attention_mask):
    """Average the token vectors, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode(texts: list[str], model_name: str = ENCODER,
           batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """(embeddings, zero-shot probabilities, zero-shot label order, revision)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=True).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    vectors, probs = [], []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            output = model(**batch)
        vectors.append(mean_pool(output.hidden_states[-1], batch["attention_mask"]).cpu().numpy())
        probs.append(torch.softmax(output.logits, dim=-1).cpu().numpy())
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    return np.vstack(vectors), np.vstack(probs), labels, revision


def cache_path(cache_dir: Path, translate_first: bool, stem: str) -> Path:
    return Path(cache_dir) / f"{stem}__finbert_{'translated' if translate_first else 'raw'}.npz"


def build(input_path: Path = DEFAULT_INPUT, cache_dir: Path = DEFAULT_CACHE_DIR,
          translate_first: bool = False, id_column: str = "gold_item_id",
          text_column: str = "headline_clean", encoder=encode, translator=translate) -> Path:
    if Path(input_path).suffix == ".parquet":       # the corpus, for score_corpus.py
        frame = pd.read_parquet(input_path).fillna({text_column: ""})
    else:
        frame = pd.read_csv(input_path, keep_default_na=False)
    ids = frame[id_column].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{id_column} is not unique")
    texts = frame[text_column].astype(str).tolist()
    langs = frame["lang"].astype(str).tolist() if "lang" in frame else ["fr"] * len(frame)

    translator_revision = None
    if translate_first:
        texts, translator_revision = translator(texts, langs)
    vectors, probs, zs_labels, encoder_revision = encoder(texts)

    path = cache_path(cache_dir, translate_first, Path(input_path).stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "input": Path(input_path).name, "rows": len(ids), "id_column": id_column,
        "encoder": ENCODER, "encoder_revision": encoder_revision,
        "translated": translate_first,
        "translator": TRANSLATOR if translate_first else None,
        "translator_revision": translator_revision,
        "pooling": "mean over last hidden state, padding masked",
        "max_length": MAX_LENGTH, "zero_shot_labels": zs_labels,
    }
    np.savez_compressed(path, ids=np.array(ids), embeddings=vectors.astype(np.float32),
                        zero_shot=probs.astype(np.float32), meta=json.dumps(meta))
    if translate_first:
        pd.DataFrame({id_column: ids, "original": frame[text_column],
                      "translated": texts}).to_csv(path.with_suffix(".translations.csv"),
                                                   index=False)
    return path


def load(path: Path) -> tuple[pd.Index, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    return (pd.Index(data["ids"].astype(str)), data["embeddings"], data["zero_shot"],
            json.loads(str(data["meta"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--translate", action="store_true",
                        help="translate French to English with OPUS-MT before FinBERT")
    parser.add_argument("--id-column", default="gold_item_id")
    args = parser.parse_args()
    path = build(args.input, args.cache_dir, args.translate, args.id_column)
    _, vectors, _, meta = load(path)
    print(f"Wrote {path}  {vectors.shape}  encoder rev {meta['encoder_revision'][:8]}"
          + (f"  translator rev {meta['translator_revision'][:8]}" if meta["translated"] else ""))


if __name__ == "__main__":
    main()
