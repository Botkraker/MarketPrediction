import pandas as pd
import pytest

from adjudicate_from_annotator import create_dataset


def test_create_dataset_copies_annotator_one(tmp_path):
    source = pd.DataFrame(
        {
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
    assert pd.read_csv(input_path)["adjudicated_label"].tolist() == ["", ""]


def test_create_dataset_rejects_blank_annotation_one(tmp_path):
    source = pd.DataFrame(
        {
            "annotator_1_label": [""],
            "adjudicated_label": [""],
            "annotation_status": ["llm_annotated"],
        }
    )
    input_path = tmp_path / "input.csv"
    source.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="annotator_1_label"):
        create_dataset(input_path, tmp_path / "output.csv")