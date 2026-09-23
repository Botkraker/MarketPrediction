from score_corpus import _provenance_warning


def test_warning_follows_the_labels_actually_used():
    assert "PROMPT_V1" in _provenance_warning(["v1"])
    assert "PROMPT_V1" in _provenance_warning(["v1 (assumed - no annotator_N_prompt column present)"])
    v2 = _provenance_warning(["human", "v2"])
    assert "PROMPT_V2" in v2 and "PROMPT_V1" not in v2
