import json

import pandas as pd
import pytest

from adjudicate_from_annotator import create_dataset


def test_create_dataset_copies_annotator_one(tmp_path):
    source = pd.DataFrame(
        {
            "source": ["ilboursa", "kapitalis"],
            "annotator_1_label": ["negative", "positive"],
            "adjudicated_label": ["", ""],
            "annotation_status": ["llm_annotated", "llm_annotated"],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    source.to_csv(input_path, index=False)

    result = create_dataset(input_path, output_path, tmp_path / "metadata.json")

    assert result["adjudicated_label"].tolist() == ["negative", "positive"]
    assert set(result["annotation_status"]) == {"adjudicated_from_annotator_1"}
    assert pd.read_csv(input_path, keep_default_na=False)["adjudicated_label"].tolist() == ["", ""]


def test_create_dataset_rejects_blank_annotation_one(tmp_path):
    source = pd.DataFrame(
        {
            "source": ["ilboursa"],
            "annotator_1_label": [""],
            "adjudicated_label": [""],
            "annotation_status": ["llm_annotated"],
        }
    )
    input_path = tmp_path / "input.csv"
    source.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="annotator_1_label"):
        create_dataset(input_path, tmp_path / "output.csv")

def _multi_annotator_frame():
    return pd.DataFrame({
        "source": ["ilboursa", "kapitalis", "lapresse"],
        "annotator_1_label": ["positive", "negative", "positive"],
        "annotator_2_label": ["positive", "negative", "negative"],
        "annotator_3_label": ["neutral", "negative", "very_positive"],
        "adjudicated_label": ["", "", ""],
        "annotation_status": ["llm_annotated"] * 3,
    })


def test_majority_vote_resolves_and_flags_ties(tmp_path):
    input_path = tmp_path / "input.csv"
    _multi_annotator_frame().to_csv(input_path, index=False)

    result = create_dataset(input_path, tmp_path / "out.csv",
                            tmp_path / "meta.json", method="majority")

    # 2-of-3 wins; the three-way split stays blank rather than being guessed
    assert result["adjudicated_label"].tolist() == ["positive", "negative", ""]
    assert result["annotation_status"].tolist() == [
        "adjudicated_majority", "adjudicated_majority", "adjudicated_tie_unresolved"]

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["ties_unresolved"] == 1
    assert meta["provisional"] is False


def test_majority_refuses_with_one_annotator(tmp_path):
    frame = _multi_annotator_frame()
    frame["annotator_2_label"] = ""
    frame["annotator_3_label"] = ""
    input_path = tmp_path / "input.csv"
    frame.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="majority needs"):
        create_dataset(input_path, tmp_path / "out.csv",
                       tmp_path / "meta.json", method="majority")


def test_unresolved_tie_is_rejected_by_split(tmp_path):
    """A tie must never reach a model: split.py has to refuse the file."""
    from split import validate_gold_complete

    input_path = tmp_path / "input.csv"
    _multi_annotator_frame().to_csv(input_path, index=False)
    result = create_dataset(input_path, tmp_path / "out.csv",
                            tmp_path / "meta.json", method="majority")
    result["gold_item_id"] = ["g-0", "g-1", "g-2"]
    result["row_id"] = ["r-0", "r-1", "r-2"]
    result["lang"] = ["fr", "fr", "fr"]
    result["dup_cluster_id"] = ["c-0", "c-1", "c-2"]

    with pytest.raises(ValueError, match="incomplete"):
        validate_gold_complete(result)


def _two_models_and_human():
    return pd.DataFrame({
        "source": ["ilboursa"] * 4,
        "annotator_1_label": ["positive", "negative", "positive", "neutral"],
        "annotator_2_label": ["positive", "neutral", "neutral", "positive"],
        "annotator_3_label": ["", "", "", "negative"],
        "adjudicated_label": [""] * 4,
        "annotation_status": ["llm_annotated"] * 4,
    })


def test_human_decides_its_rows_and_tiebreak_resolves_the_rest(tmp_path):
    input_path = tmp_path / "input.csv"
    _two_models_and_human().to_csv(input_path, index=False)

    result = create_dataset(input_path, tmp_path / "out.csv", tmp_path / "meta.json",
                            method="majority", human_annotator=3, tiebreak=1,
                            tiebreak_note="qwen preferred")

    # row 0 agreement; rows 1-2 model ties -> annotator 1; row 3 human overrides both
    assert result["adjudicated_label"].tolist() == ["positive", "negative", "positive", "negative"]
    assert result["annotation_status"].tolist() == [
        "adjudicated_majority", "adjudicated_tiebreak_annotator_1",
        "adjudicated_tiebreak_annotator_1", "adjudicated_human_3"]
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["ties_unresolved"] == 0
    assert meta["tiebreak_note"] == "qwen preferred"
    assert meta["status_counts"]["adjudicated_tiebreak_annotator_1"] == 2


def test_without_tiebreak_model_ties_stay_blank_even_with_a_human(tmp_path):
    input_path = tmp_path / "input.csv"
    _two_models_and_human().to_csv(input_path, index=False)

    result = create_dataset(input_path, tmp_path / "out.csv", tmp_path / "meta.json",
                            method="majority", human_annotator=3)

    assert result["adjudicated_label"].tolist() == ["positive", "", "", "negative"]
