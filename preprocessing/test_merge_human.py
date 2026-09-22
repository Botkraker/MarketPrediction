import json

import pandas as pd
import pytest

from merge_human import merge_human


def _target(tmp_path):
    path = tmp_path / "gold.csv"
    pd.DataFrame({
        "gold_item_id": [f"gold-{i:05d}" for i in range(4)],
        "headline_clean": list("abcd"),
        "annotator_1_label": ["positive"] * 4,
        "annotator_2_label": ["neutral"] * 4,
    }).to_csv(path, index=False)
    return path


def _worksheet(tmp_path, labels, ids=None, notes=None):
    path = tmp_path / "human.csv"
    ids = ids or [f"gold-{i:05d}" for i in range(len(labels))]
    pd.DataFrame({
        "gold_item_id": ids,
        "headline_clean": list("abcd")[:len(labels)],
        "human_label": labels,
        "human_notes": notes or [""] * len(labels),
    }).to_csv(path, index=False)
    return path


def test_partial_human_column_leaves_other_rows_blank(tmp_path):
    target = _target(tmp_path)
    worksheet = _worksheet(tmp_path, ["negative", "", "NEUTRAL ", ""],
                           notes=["rate hike", "", "procedural", ""])

    merge_human(worksheet, target, tmp_path / "meta.json")

    out = pd.read_csv(target, keep_default_na=False)
    assert out["annotator_3_label"].tolist() == ["negative", "", "neutral", ""]
    assert out["annotator_3_model"].tolist() == ["human", "", "human", ""]
    assert out["annotator_3_notes"].tolist() == ["rate hike", "", "procedural", ""]
    assert out["annotator_1_label"].tolist() == ["positive"] * 4
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["rows_merged"] == 2 and meta["rows_blank_in_worksheet"] == 2


def test_label_outside_the_scale_is_rejected(tmp_path):
    worksheet = _worksheet(tmp_path, ["bullish"])
    with pytest.raises(ValueError, match="outside"):
        merge_human(worksheet, _target(tmp_path), tmp_path / "meta.json")


def test_unknown_gold_item_is_rejected(tmp_path):
    worksheet = _worksheet(tmp_path, ["negative"], ids=["gold-99999"])
    with pytest.raises(ValueError, match="not in the gold file"):
        merge_human(worksheet, _target(tmp_path), tmp_path / "meta.json")


def test_empty_worksheet_stops_with_a_clear_message(tmp_path):
    worksheet = _worksheet(tmp_path, ["", ""])
    with pytest.raises(SystemExit):
        merge_human(worksheet, _target(tmp_path), tmp_path / "meta.json")
