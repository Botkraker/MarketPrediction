import json
import pandas as pd
import pytest

from split import build_split, validate_gold_complete


def _gold_frame() -> pd.DataFrame:
    rows = []
    for language in ("ar", "en", "fr"):
        for label in ("very_negative", "negative", "neutral", "positive", "very_positive"):
            for index in range(6):
                rows.append(
                    {
                        "gold_item_id": f"gold-{language}-{label}-{index}",
                        "row_id": f"row-{language}-{label}-{index}",
                        "lang": language,
                        "adjudicated_label": label,
                        "dup_cluster_id": f"cluster-{language}-{label}-{index}",
                    }
                )
    return pd.DataFrame(rows)


def test_incomplete_gold_is_rejected():
    frame = _gold_frame()
    frame.loc[0, "adjudicated_label"] = ""
    with pytest.raises(ValueError, match="incomplete"):
        validate_gold_complete(frame)


def test_split_is_deterministic_and_disjoint(tmp_path):
    input_path = tmp_path / "gold.csv"
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    frame = _gold_frame()
    frame.to_csv(input_path, index=False)

    first = build_split(input_path, first_path, tmp_path / "first.json", seed=7)
    second = build_split(input_path, second_path, tmp_path / "second.json", seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"train", "validation", "evaluation"}
    assert first.groupby("dup_cluster_id")["split"].nunique().max() == 1
    assert first.groupby("split")["lang"].nunique().min() == 3

def _with_human(frame, human_rows):
    frame = frame.copy()
    frame["annotator_3_label"] = ""
    frame.loc[human_rows, "annotator_3_label"] = frame.loc[human_rows, "adjudicated_label"]
    return frame


def test_evaluation_can_be_fixed_to_the_human_rows(tmp_path):
    frame = _with_human(_gold_frame(), list(range(0, 90, 3)))
    input_path = tmp_path / "gold.csv"
    frame.to_csv(input_path, index=False)

    result = build_split(input_path, tmp_path / "s.csv", tmp_path / "s.json",
                         seed=7, evaluation_from_annotator=3)

    human_ids = set(frame.loc[list(range(0, 90, 3)), "gold_item_id"])
    assert set(result.loc[result["split"] == "evaluation", "gold_item_id"]) == human_ids
    assert set(result["split"]) == {"train", "validation", "evaluation"}
    meta = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert meta["evaluation_source"] == "annotator_3_label (fixed rows)"


def test_fixed_evaluation_refuses_labels_that_are_not_the_humans(tmp_path):
    frame = _with_human(_gold_frame(), [0, 1, 2])
    frame.loc[0, "annotator_3_label"] = "neutral" if frame.loc[0, "adjudicated_label"] != "neutral" else "positive"
    input_path = tmp_path / "gold.csv"
    frame.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="--human-annotator 3"):
        build_split(input_path, tmp_path / "s.csv", tmp_path / "s.json",
                    evaluation_from_annotator=3)
