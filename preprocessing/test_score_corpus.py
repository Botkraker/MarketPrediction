import json

import numpy as np
import pandas as pd
import pytest

import finbert_embed
import score_corpus
from score_corpus import _as_frame, _provenance_warning

LABELS3 = ["negative", "neutral", "positive"]


def test_warning_follows_the_labels_actually_used():
    assert "PROMPT_V1" in _provenance_warning(["v1"])
    assert "PROMPT_V1" in _provenance_warning(["v1 (assumed - no annotator_N_prompt column present)"])
    v2 = _provenance_warning(["human", "v2"])
    assert "PROMPT_V2" in v2 and "PROMPT_V1" not in v2


def test_missing_class_gets_zero_probability_not_a_shifted_column():
    probs = _as_frame(np.array([[0.3, 0.7]]), ["negative", "positive"], LABELS3)
    assert list(probs.columns) == LABELS3
    assert probs.iloc[0].tolist() == [0.3, 0.0, 0.7]


def _world(tmp_path, years=range(2016, 2021), per_year=90):
    rows, gold = [], []
    for y in years:
        for i in range(per_year):
            label = LABELS3[i % 3]
            row_id = f"r{y}_{i}"
            text = f"bourse {label} {y} {i}"
            rows.append({"row_id": row_id, "source": "s", "lang": "fr",
                         "published_date": f"{y}-03-{i % 28 + 1:02d}",
                         "headline_clean": text, "relevance_tag": "tunisia",
                         "is_canonical": True, "date_parse_ok": True})
            if i % 2 == 0:
                gold.append({"gold_item_id": f"g{y}_{i}", "row_id": row_id,
                             "published_date": f"{y}-03-{i % 28 + 1:02d}",
                             "headline_clean": text, "adjudicated_label": label,
                             "annotator_1_prompt": "v2", "lang": "fr"})
    corpus, gold = pd.DataFrame(rows), pd.DataFrame(gold)
    corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
    gold.to_csv(tmp_path / "gold.csv", index=False)
    gold[["gold_item_id"]].assign(split="train").to_csv(tmp_path / "split.csv", index=False)
    return tmp_path / "corpus.parquet", tmp_path / "gold.csv", tmp_path / "split.csv"


def _run(tmp_path, paths, **kw):
    corpus, gold, split = paths
    return score_corpus.run(corpus, gold, split, tmp_path / "out.parquet",
                            tmp_path / "meta.json", min_train=40, **kw)


def test_every_window_is_fit_only_on_labels_dated_before_it(tmp_path, monkeypatch):
    seen = []
    real = score_corpus.tfidf_scorer

    def spy(labels):
        inner = real(labels)

        def fit(train):
            seen.append(train.day.max())
            return inner(train)
        return fit
    monkeypatch.setattr(score_corpus, "tfidf_scorer", spy)

    scored = _run(tmp_path, _world(tmp_path))
    ends = sorted(scored.sent_model_train_end.replace("", np.nan).dropna().unique())
    assert len(seen) == len(ends) == 4                    # 2016 has no past: unscored
    for last_label_day, window in zip(seen, ends):
        assert last_label_day < pd.Timestamp(window)
    assert (scored.sent_label[scored.published_date.str.startswith("2016")] == "").all()


def test_tfidf_plus_finbert_average_scores_and_records_both(tmp_path):
    paths = _world(tmp_path)

    def encoder(texts):
        v = np.random.default_rng(0).normal(0, 0.1, (len(texts), 6))
        for i, t in enumerate(texts):
            for j, label in enumerate(LABELS3):
                v[i, j] += 3.0 * (label in t)
        return v, np.full((len(texts), 3), 1 / 3), LABELS3, "rev"

    gold_cache = finbert_embed.build(paths[1], tmp_path / "emb", encoder=encoder)
    corpus_cache = finbert_embed.build(paths[0], tmp_path / "emb", id_column="row_id",
                                       encoder=encoder)
    scored = _run(tmp_path, paths, scorers=("tfidf", "finbert"),
                  gold_embeddings=gold_cache, corpus_embeddings=corpus_cache)

    done = scored[scored.sent_label != ""]
    truth = done.headline_clean.str.split().str[1]
    assert (done.sent_label == truth).mean() > 0.95
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert set(meta["scorers"]) == {"tfidf", "finbert"} and meta["leakage_free"]
    assert "averaged" in meta["model"]


def test_finbert_scorer_refuses_without_caches(tmp_path):
    with pytest.raises(SystemExit, match="embeddings"):
        _run(tmp_path, _world(tmp_path), scorers=("finbert",))
