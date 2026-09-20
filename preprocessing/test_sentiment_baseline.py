import pandas as pd
import pytest

from sentiment_baseline import build_model, load, run, score


def _gold_and_split(tmp_path, n_per_label=12):
    labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    rows, splits = [], []
    for li, label in enumerate(labels):
        for i in range(n_per_label):
            gid = f"gold-{li:02d}{i:03d}"
            rows.append({"gold_item_id": gid,
                         "headline_clean": f"{label} marche dinar bourse {i}",
                         "adjudicated_label": label})
            splits.append({"gold_item_id": gid,
                           "split": "train" if i < n_per_label - 4
                           else ("validation" if i < n_per_label - 2 else "evaluation")})
    g, s = tmp_path / "gold.csv", tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(g, index=False)
    pd.DataFrame(splits).to_csv(s, index=False)
    return g, s


def test_evaluation_split_is_untouched_by_default(tmp_path):
    """The frozen evaluation split must be spent deliberately, never by accident."""
    g, s = _gold_and_split(tmp_path)
    default = run(g, s, tmp_path / "a.json")
    assert default["held_out_split"] == "validation"
    explicit = run(g, s, tmp_path / "b.json", use_evaluation=True)
    assert explicit["held_out_split"] == "evaluation"


def test_load_keeps_only_rows_present_in_the_split(tmp_path):
    g, s = _gold_and_split(tmp_path)
    gold = pd.read_csv(g)
    gold.loc[len(gold)] = {"gold_item_id": "gold-NOT-IN-SPLIT",
                           "headline_clean": "orphan", "adjudicated_label": "neutral"}
    gold.to_csv(g, index=False)
    assert "gold-NOT-IN-SPLIT" not in set(load(g, s).gold_item_id)


def test_score_reports_floor_and_ordinal_kappa():
    y_true = pd.Series(["positive"] * 8 + ["negative"] * 2)
    s = score(y_true, pd.Series(["positive"] * 10), majority="positive")
    assert s["majority_floor"] == 0.8
    assert s["accuracy"] == 0.8          # a constant model exactly matches the floor
    assert s["qwk_ordinal"] == 0.0       # ...and adds no ordinal information

def test_ordinal_weighting_punishes_distant_errors_harder():
    y_true = pd.Series(["very_negative"] * 4)
    near = score(y_true, pd.Series(["negative"] * 4), majority="very_negative")
    far = score(y_true, pd.Series(["very_positive"] * 4), majority="very_negative")
    assert near["qwk_ordinal"] >= far["qwk_ordinal"]


def test_model_trains_and_predicts_valid_labels(tmp_path):
    g, s = _gold_and_split(tmp_path)
    frame = load(g, s)
    train = frame[frame.split == "train"]
    model = build_model().fit(train.headline_clean, train.adjudicated_label)
    preds = model.predict(frame.headline_clean)
    assert set(preds) <= set(frame.adjudicated_label.unique())


def test_run_refuses_when_split_does_not_match_gold(tmp_path):
    g, s = _gold_and_split(tmp_path)
    pd.DataFrame({"gold_item_id": ["nope"], "split": ["train"]}).to_csv(s, index=False)
    with pytest.raises(SystemExit, match="run split.py"):
        run(g, s, tmp_path / "x.json")
