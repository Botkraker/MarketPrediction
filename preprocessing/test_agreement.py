import pandas as pd
import pytest

from agreement import (annotator_label_columns, compute, interpret,
                       majority_label)


def _frame(**overrides):
    base = pd.DataFrame({
        "gold_item_id": [f"gold-{i:05d}" for i in range(6)],
        "in_study": [True] * 6,
        "annotator_1_label": ["positive", "negative", "neutral",
                              "positive", "very_negative", "positive"],
        "annotator_2_label": ["positive", "negative", "positive",
                              "positive", "very_negative", "negative"],
        "annotator_3_label": ["positive", "neutral", "negative",
                              "positive", "very_negative", "very_positive"],
        "annotator_1_model": ["m1"] * 6,
        "annotator_2_model": ["m2"] * 6,
        "annotator_3_model": ["m3"] * 6,
    })
    base.update(pd.DataFrame(overrides))
    return base


def test_only_populated_annotator_columns_are_found():
    frame = _frame()
    frame["annotator_4_label"] = ""          # declared but never run
    assert annotator_label_columns(frame) == [
        "annotator_1_label", "annotator_2_label", "annotator_3_label"]


def test_majority_needs_a_strict_winner():
    columns = ["annotator_1_label", "annotator_2_label", "annotator_3_label"]
    two_to_one = pd.Series({"annotator_1_label": "positive",
                            "annotator_2_label": "positive",
                            "annotator_3_label": "negative"})
    assert majority_label(two_to_one, columns) == ("positive", False)

    # three-way split has no majority -> must NOT silently pick one
    three_way = pd.Series({"annotator_1_label": "positive",
                           "annotator_2_label": "negative",
                           "annotator_3_label": "neutral"})
    label, tied = majority_label(three_way, columns)
    assert label == "" and tied is True


def test_compute_reports_kappas_and_counts(tmp_path):
    path = tmp_path / "gold.csv"
    _frame().to_csv(path, index=False)
    result = compute(path, tmp_path / "agreement.json")

    assert result["rows_all_annotators_labelled"] == 6
    assert result["unanimous_rows"] == 3          # rows 0, 3, 4
    assert len(result["pairwise"]) == 3           # 3 annotators -> 3 pairs
    # ordinal weighting must not equal nominal when disagreements differ in size
    pair = result["pairwise"][0]
    assert pair["cohen_kappa_quadratic"] != pair["cohen_kappa_nominal"]
    assert -1.0 <= result["fleiss_kappa_nominal"] <= 1.0


def test_excluded_rows_never_enter_the_statistic(tmp_path):
    frame = _frame()
    frame.loc[0:2, "in_study"] = False
    path = tmp_path / "gold.csv"
    frame.to_csv(path, index=False)
    result = compute(path, tmp_path / "agreement.json")
    assert result["rows_all_annotators_labelled"] == 3


def test_refuses_with_a_single_annotator(tmp_path):
    frame = _frame()
    frame["annotator_2_label"] = ""
    frame["annotator_3_label"] = ""
    path = tmp_path / "gold.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="Need >=2"):
        compute(path, tmp_path / "agreement.json")


def test_interpret_brackets():
    assert interpret(0.9) == "almost perfect"
    assert interpret(0.7) == "substantial"
    assert interpret(-0.1) == "poor (worse than chance)"
