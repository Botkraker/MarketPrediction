import pandas as pd
import pytest

from llm_annotate import _parse_annotation, _response_text


def test_response_text_reads_lm_studio_native_chat_response():
    response = {"output": [{"type": "message", "content": '{"label":"positive","reason":"test"}'}]}

    assert _parse_annotation(_response_text(response)) == ("positive", "test")


def test_parse_annotation_accepts_prefixed_fenced_json():
    content = 'Analysis complete.\n```json\n{"label":"negative","reason":"test"}\n```'

    assert _parse_annotation(content) == ("negative", "test")


def test_parse_annotation_accepts_quoted_pseudo_json():
    content = '"value: \\"very_negative\\", reason: \\"test explanation.\\""'

    assert _parse_annotation(content) == ("very_negative", "test explanation.")


def test_parse_annotation_rejects_truncated_json():
    content = '```json\n{\n  "label": "neutral",\n  "reason": "incomplete'

    try:
        _parse_annotation(content)
    except ValueError as error:
        assert "invalid JSON" in str(error)
    else:
        raise AssertionError("Expected truncated JSON to be rejected")

def test_annotator_columns_are_isolated_per_annotator():
    from llm_annotate import annotator_columns

    assert annotator_columns(1) == (
        "annotator_1_label", "annotation_notes", "annotator_1_model")
    assert annotator_columns(2) == (
        "annotator_2_label", "annotator_2_notes", "annotator_2_model")
    # no column name is shared between two annotators -> no silent overwrite
    assert not set(annotator_columns(1)) & set(annotator_columns(2))
    with pytest.raises(ValueError):
        annotator_columns(0)


def test_annotate_file_writes_only_its_own_annotator(tmp_path, monkeypatch):
    import llm_annotate

    frame = pd.DataFrame({
        "headline_clean": ["Le dinar recule", "La BCT releve son taux"],
        "lang": ["fr", "fr"],
        "relevance_tag": ["tunisia_econ", "tunisia_econ"],
        "annotator_1_label": ["negative", "positive"],
        "annotation_notes": ["prior run", "prior run"],
    })
    path = tmp_path / "gold.csv"
    frame.to_csv(path, index=False)

    monkeypatch.setattr(llm_annotate, "annotate_headline",
                        lambda *a, **k: ("neutral", "second opinion"))
    result = llm_annotate.annotate_file(path, model="other-model", annotator=2)

    # annotator 1's labels survive untouched
    assert result["annotator_1_label"].tolist() == ["negative", "positive"]
    assert result["annotation_notes"].tolist() == ["prior run", "prior run"]
    # annotator 2 is written into its own columns, model recorded
    assert result["annotator_2_label"].tolist() == ["neutral", "neutral"]
    assert result["annotator_2_model"].tolist() == ["other-model", "other-model"]
