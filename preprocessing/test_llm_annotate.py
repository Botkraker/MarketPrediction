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


def test_v1_prompt_is_frozen_for_reproducibility():
    """The original 3,000 labels were produced with v1. Editing it would silently
    invalidate their provenance."""
    from llm_annotate import PROMPTS
    assert PROMPTS["v1"] == (
        "You annotate news headline sentiment for the Tunisian economy. "
        "Return ONLY one complete valid JSON object with exactly two fields: label and reason. "
        "label must be one of ['very_negative', 'negative', 'neutral', 'positive', "
        "'very_positive']. reason must be one short sentence of at most 15 words. "
        "Do not use Markdown, prefixes, suffixes, or extra quotes.")


def test_v2_prompt_fixes_the_documented_v1_failures():
    from llm_annotate import PROMPTS
    v2 = PROMPTS["v2"]
    # defines the construct as market impact, not tone
    assert "MARKET" in v2 and "optimistic" in v2
    # makes neutral the default rather than a last resort
    assert "NEUTRAL IS THE DEFAULT" in v2
    # names the exact traps found in the v1 labels
    for trap in ("en cours d'élaboration", "inchangé", "dépassent", "Météo"):
        assert trap in v2
    # every label is defined, not just listed
    for label in ("very_negative", "negative", "neutral", "positive", "very_positive"):
        assert f"- {label}:" in v2


def test_unknown_prompt_version_is_rejected():
    import llm_annotate
    with pytest.raises(ValueError, match="Unknown prompt_version"):
        llm_annotate.annotate_headline("x", "fr", "tunisia_econ", prompt_version="v9")


def test_prompt_version_is_recorded_per_row(tmp_path, monkeypatch):
    import llm_annotate
    frame = pd.DataFrame({
        "headline_clean": ["Le dinar recule"], "lang": ["fr"],
        "relevance_tag": ["tunisia_econ"],
    })
    path = tmp_path / "g.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(llm_annotate, "annotate_headline",
                        lambda *a, **k: ("negative", "currency weakness"))
    out = llm_annotate.annotate_file(path, annotator=2, prompt_version="v2")
    assert out["annotator_2_prompt"].tolist() == ["v2"]
