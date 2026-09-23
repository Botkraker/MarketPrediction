import json

import numpy as np
import pandas as pd
import pytest

import finbert_embed
import finbert_head

LABELS3 = ["negative", "neutral", "positive"]


def _gold(tmp_path, n=90):
    rng = np.random.default_rng(0)
    labels = np.array(LABELS3)[np.arange(n) % 3]
    frame = pd.DataFrame({
        "gold_item_id": [f"gold-{i:05d}" for i in range(n)],
        "row_id": [f"r{i}" for i in range(n)],
        "lang": ["fr"] * (n - 3) + ["en"] * 3,
        "dup_cluster_id": [f"c{i}" for i in range(n)],
        "headline_clean": [f"titre {label} {i} {rng.integers(1000)}"
                           for i, label in enumerate(labels)],
        "adjudicated_label": labels,
    })
    split = frame[["gold_item_id"]].copy()
    split["split"] = np.where(np.arange(n) < 60, "train",
                              np.where(np.arange(n) < 75, "validation", "evaluation"))
    gold_path, split_path = tmp_path / "gold.csv", tmp_path / "split.csv"
    frame.to_csv(gold_path, index=False)
    split.to_csv(split_path, index=False)
    return frame, gold_path, split_path


def _fake_encoder(texts):
    """Separable by construction: one-hot on the label word in the text, plus noise."""
    rng = np.random.default_rng(1)
    vectors = rng.normal(0, 0.1, (len(texts), 8))
    for i, text in enumerate(texts):
        for j, label in enumerate(LABELS3):
            if label in text:
                vectors[i, j] += 3.0
    probs = np.tile([0.2, 0.2, 0.6], (len(texts), 1))       # always "neutral"
    return vectors, probs, ["positive", "negative", "neutral"], "enc-rev"


def test_translate_only_touches_french_rows(tmp_path):
    frame, gold_path, _ = _gold(tmp_path)
    seen = {}

    def fake_translator(texts, langs):
        out = [f"EN:{t}" if lang == "fr" else t for t, lang in zip(texts, langs)]
        seen["out"] = out
        return out, "mt-rev"

    path = finbert_embed.build(gold_path, tmp_path / "emb", translate_first=True,
                               encoder=_fake_encoder, translator=fake_translator)

    ids, vectors, zero_shot, meta = finbert_embed.load(path)
    assert list(ids) == frame["gold_item_id"].tolist()
    assert vectors.shape == (len(frame), 8) and zero_shot.shape == (len(frame), 3)
    assert meta["translated"] and meta["translator_revision"] == "mt-rev"
    assert meta["encoder_revision"] == "enc-rev"
    translations = pd.read_csv(path.with_suffix(".translations.csv"), keep_default_na=False)
    assert translations["translated"].str.startswith("EN:").sum() == len(frame) - 3


def test_duplicate_ids_are_refused(tmp_path):
    frame, gold_path, _ = _gold(tmp_path)
    frame.loc[1, "gold_item_id"] = frame.loc[0, "gold_item_id"]
    frame.to_csv(gold_path, index=False)
    with pytest.raises(ValueError, match="not unique"):
        finbert_embed.build(gold_path, tmp_path / "emb", encoder=_fake_encoder)


def test_head_learns_separable_embeddings_and_reports_every_variant(tmp_path):
    _, gold_path, split_path = _gold(tmp_path)
    cache = finbert_embed.build(gold_path, tmp_path / "emb", encoder=_fake_encoder)

    result = finbert_head.run(gold_path, split_path, [cache], tmp_path / "r.json")

    assert result["held_out_split"] == "validation" and result["held_out_n"] == 15
    variants = result["variants"]
    assert set(variants) == {"tfidf", "finbert_zero_shot_raw", "finbert_head_raw"}
    assert variants["finbert_head_raw"]["accuracy"] == 1.0
    assert variants["finbert_zero_shot_raw"]["accuracy"] == pytest.approx(1 / 3, abs=0.01)
    assert json.loads((tmp_path / "r.json").read_text())["classes"] == 3


def test_evaluation_split_is_untouched_by_default(tmp_path):
    _, gold_path, split_path = _gold(tmp_path)
    cache = finbert_embed.build(gold_path, tmp_path / "emb", encoder=_fake_encoder)
    default = finbert_head.run(gold_path, split_path, [cache], tmp_path / "r.json")
    spent = finbert_head.run(gold_path, split_path, [cache], tmp_path / "e.json",
                             use_evaluation=True)
    assert default["held_out_split"] == "validation"
    assert spent["held_out_split"] == "evaluation" and spent["held_out_n"] == 15


def test_missing_embeddings_are_refused(tmp_path):
    frame, gold_path, split_path = _gold(tmp_path)
    partial = tmp_path / "partial.csv"
    frame.iloc[:-1].to_csv(partial, index=False)
    cache = finbert_embed.build(partial, tmp_path / "emb", encoder=_fake_encoder)
    with pytest.raises(ValueError, match="no embedding"):
        finbert_head.run(gold_path, split_path, [cache], tmp_path / "r.json")


def test_mean_pool_ignores_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert finbert_embed.mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]
