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