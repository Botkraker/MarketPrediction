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


def test_parse_annotation_accepts_bare_label_reply():
    # qwen2.5-7b under PROMPT_V2 on short headlines: no JSON wrapper at all.
    assert _parse_annotation("neutral meeting discussed cooperation opportunities") == \
        ("neutral", "meeting discussed cooperation opportunities")
    assert _parse_annotation("very_negative: sovereign default risk") == \
        ("very_negative", "sovereign default risk")


def test_parse_annotation_bare_label_needs_exact_leading_token():
    for content in ("neutrality of the board", "The label is neutral", "neutral"):
        with pytest.raises(ValueError):
            _parse_annotation(content)


def test_annotate_file_continues_past_unparseable_row(tmp_path, monkeypatch):
    import llm_annotate
    frame = pd.DataFrame({
        "headline_clean": ["ok one", "broken", "ok two"], "lang": ["fr"] * 3,
        "relevance_tag": ["tunisia_econ"] * 3,
    })
    path = tmp_path / "g.csv"
    frame.to_csv(path, index=False)

    def fake(headline, *a, **k):
        if headline == "broken":
            raise ValueError("invalid annotation")
        return "neutral", "no clear effect"

    monkeypatch.setattr(llm_annotate, "annotate_headline", fake)
    out = llm_annotate.annotate_file(path, annotator=1)
    assert out["annotator_1_label"].tolist() == ["neutral", "", "neutral"]
    assert out["annotation_status"].tolist() == ["llm_annotated", "llm_failed", "llm_annotated"]


def test_annotate_file_still_aborts_when_server_unreachable(tmp_path, monkeypatch):
    import llm_annotate
    path = tmp_path / "g.csv"
    pd.DataFrame({"headline_clean": ["x"], "lang": ["fr"],
                  "relevance_tag": ["tunisia_econ"]}).to_csv(path, index=False)

    def down(*a, **k):
        raise RuntimeError("Could not reach LM Studio")

    monkeypatch.setattr(llm_annotate, "annotate_headline", down)
    with pytest.raises(RuntimeError):
        llm_annotate.annotate_file(path)


def test_parse_annotation_accepts_unquoted_key_value_lines():
    assert _parse_annotation("label: negative\nreason: index decline signals pessimism.") == \
        ("negative", "index decline signals pessimism.")


def test_parse_annotation_key_value_needs_both_keys():
    with pytest.raises(ValueError):
        _parse_annotation("label: negative")


def test_request_timeout_fails_the_row_not_the_run(tmp_path, monkeypatch):
    import llm_annotate
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm_annotate, "urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        llm_annotate.annotate_headline("x", "fr", "tunisia_econ")
    assert len(calls) == llm_annotate.ANNOTATION_ATTEMPTS


def test_dead_server_still_aborts(tmp_path, monkeypatch):
    import llm_annotate
    from urllib.error import URLError

    def fake_urlopen(request, timeout=None):
        raise URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(llm_annotate, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Could not reach"):
        llm_annotate.annotate_headline("x", "fr", "tunisia_econ")


def test_response_text_reads_openai_dialect():
    response = {"choices": [{"message": {"content": '{"label": "neutral", "reason": "x"}'}}]}
    assert _response_text(response) == '{"label": "neutral", "reason": "x"}'


def test_response_text_ignores_reasoning_content():
    # A chain of thought weighs several labels; parsing it could return a rejected one.
    response = {"choices": [{"message": {
        "content": "",
        "reasoning_content": 'maybe {"label": "positive", "reason": "..."} -- no, neutral'}}]}
    with pytest.raises(ValueError):
        _response_text(response)


def test_openai_payload_and_bearer_header(monkeypatch):
    import json
    import llm_annotate
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": '{"label": "negative", "reason": "rate hike"}'}}]}).encode()

    def fake_urlopen(request, timeout=None):
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setenv("TEST_KEY", "secret-token")
    monkeypatch.setattr(llm_annotate, "urlopen", fake_urlopen)
    label, _ = llm_annotate.annotate_headline(
        "La BCT relève son taux", "fr", "tunisia_econ", endpoint="https://x/v1/chat/completions",
        model="vendor/model", api="openai", api_key_env="TEST_KEY")

    assert label == "negative"
    assert seen["headers"]["Authorization"] == "Bearer secret-token"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["temperature"] == 0


def test_missing_api_key_env_aborts(monkeypatch):
    import llm_annotate
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ABSENT_KEY"):
        llm_annotate.annotate_headline("x", "fr", "tunisia_econ", api="openai",
                                       api_key_env="ABSENT_KEY")


def test_api_key_never_reaches_the_csv(tmp_path, monkeypatch):
    import llm_annotate
    path = tmp_path / "g.csv"
    pd.DataFrame({"headline_clean": ["x"], "lang": ["fr"],
                  "relevance_tag": ["tunisia_econ"]}).to_csv(path, index=False)
    monkeypatch.setenv("TEST_KEY", "secret-token")
    monkeypatch.setattr(llm_annotate, "annotate_headline",
                        lambda *a, **k: ("neutral", "no effect"))
    llm_annotate.annotate_file(path, annotator=3, api="openai", api_key_env="TEST_KEY",
                               model="vendor/model")
    assert "secret-token" not in path.read_text(encoding="utf-8")


def _http_error(code):
    from urllib.error import HTTPError
    return HTTPError("https://x", code, "err", {}, None)


def test_transient_http_errors_are_retried_with_backoff(monkeypatch):
    import json
    import llm_annotate
    attempts, slept = [], []

    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"ok": 1}).encode()

    def flaky(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise _http_error(503 if len(attempts) == 1 else 429)
        return Ok()

    monkeypatch.setattr(llm_annotate, "urlopen", flaky)
    result = llm_annotate._post_with_backoff(object(), 10, sleep=slept.append)
    assert result == {"ok": 1}
    assert slept == [2, 5]


def test_configuration_http_errors_abort_immediately(monkeypatch):
    import llm_annotate
    from urllib.error import HTTPError

    def not_found(request, timeout=None):
        raise _http_error(404)

    monkeypatch.setattr(llm_annotate, "urlopen", not_found)
    with pytest.raises(HTTPError):
        llm_annotate._post_with_backoff(object(), 10, sleep=lambda s: None)
