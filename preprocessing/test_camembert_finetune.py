import json
import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import camembert_finetune as cf

LABELS3 = ["negative", "neutral", "positive"]


class TinyTokenizer:
    """One token per word, hashed into a small vocabulary."""

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=64):
        ids = [[zlib.crc32(w.encode()) % 997 + 1 for w in t.split()][:max_length] or [0] for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.tensor([i + [0] * (width - len(i)) for i in ids])
        return transformers.BatchEncoding({"input_ids": input_ids,
                                           "attention_mask": (input_ids > 0).long()})


class TinyModel(torch.nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(998, n_labels, mode="sum")
        torch.nn.init.zeros_(self.bag.weight)       # unseen words contribute nothing

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.bag(input_ids,
                                               per_sample_weights=attention_mask.float()))


def tiny_loader(model_name, n_labels):
    return TinyModel(n_labels), TinyTokenizer()


def _gold(tmp_path, n=150):
    labels = np.array(LABELS3)[np.arange(n) % 3]
    frame = pd.DataFrame({
        "gold_item_id": [f"gold-{i:05d}" for i in range(n)],
        "lang": "fr",
        "headline_clean": [f"titre {label} numero{i}" for i, label in enumerate(labels)],
        "adjudicated_label": labels,
    })
    split = frame[["gold_item_id"]].assign(
        split=np.where(np.arange(n) < 105, "train",
                       np.where(np.arange(n) < 125, "validation", "evaluation")))
    frame.to_csv(tmp_path / "gold.csv", index=False)
    split.to_csv(tmp_path / "split.csv", index=False)
    return tmp_path / "gold.csv", tmp_path / "split.csv"


def test_class_weights_match_sklearn_balanced():
    from sklearn.utils.class_weight import compute_class_weight
    y = np.array(["neutral"] * 6 + ["positive"] * 3 + ["negative"])
    expected = compute_class_weight("balanced", classes=np.array(LABELS3), y=y)
    assert np.allclose(cf.class_weights(y, LABELS3), expected)


def test_choose_epochs_uses_mean_curve_and_prefers_fewer_on_ties():
    assert cf.choose_epochs([[0.1, 0.5, 0.4], [0.2, 0.3, 0.6]]) == 3
    assert cf.choose_epochs([[0.5, 0.5, 0.2]]) == 1


def test_run_learns_separable_text_and_writes_validation_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "LEARNING_RATE", 0.5)
    monkeypatch.setattr(cf, "MAX_EPOCHS", 3)
    gold, split = _gold(tmp_path)
    r = cf.run(gold, split, "tiny", tmp_path / "out", seeds=2, loader=tiny_loader)

    assert r["held_out_split"] == "validation" and r["held_out_n"] == 20
    assert len(r["dev_qwk_curves"]) == 2 and all(len(c) == 3 for c in r["dev_qwk_curves"])
    v = r["variants"]["tiny_finetuned"]
    assert v["accuracy"] == 1.0 and len(v["per_seed_qwk"]) == 2
    assert "ensemble_tfidf_tiny" in r["variants"]
    probs = pd.read_csv(tmp_path / "out" / "tiny_3class_validation_probs.csv")
    assert len(probs) == 20 and np.allclose(probs.filter(like="p_").sum(axis=1), 1, atol=1e-3)
    assert json.loads((tmp_path / "out" / "tiny_3class_validation.json").read_text())


def test_fit_with_fixed_epochs_skips_dev_search():
    texts = [f"mot {l}" for l in LABELS3 * 10]
    y = np.array(LABELS3 * 10)
    model = cf.fit(texts, y, LABELS3, "tiny", seeds=1, epochs=2, loader=tiny_loader)
    assert model.meta["epochs"] == 2 and model.meta["dev_qwk_curves"] == []
    assert model.predict_proba(texts[:4]).shape == (4, 3)
