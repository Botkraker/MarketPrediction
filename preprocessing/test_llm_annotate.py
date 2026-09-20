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