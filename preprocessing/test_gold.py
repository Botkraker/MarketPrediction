import json

import pandas as pd
from gold import ANNOTATION_COLUMNS, LABELS, build_gold


def test_build_gold_is_deterministic_and_preserves_provenance(tmp_path):
    source = pd.DataFrame(
        {
            "row_id": ["ar-1", "fr-1", "fr-2", "en-1", "other-1"],
            "source": ["assabah", "lapresse", "lapresse", "guardian", "guardian"],
            "lang": ["ar", "fr", "fr", "en", "en"],
            "published_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02", "2024-01-03", "2024-01-04"]
            ),
            "headline_clean": ["a", "b", "c", "d", "e"],
            "relevance_tag": [
                "tunisia_econ",
                "global_linked",
                "tunisia_econ",
                "global_linked",
                "other",
            ],
            "dup_cluster_id": ["c1", "c2", "c3", "c4", "c5"],
            "date_parse_ok": [True] * 5,
            "is_canonical": [True] * 5,
            "is_boilerplate": [False] * 5,
        }
    )
    input_path = tmp_path / "input.parquet"
    output_one = tmp_path / "one.csv"
    metadata_one = tmp_path / "one.json"
    output_two = tmp_path / "two.csv"
    metadata_two = tmp_path / "two.json"
    source.to_parquet(input_path)

    first = build_gold(input_path, output_one, metadata_one, target=4, seed=7)
    second = build_gold(input_path, output_two, metadata_two, target=4, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["lang"]) == {"ar", "en", "fr"}
    assert first["row_id"].is_unique
    assert first["dup_cluster_id"].is_unique
    assert list(first[ANNOTATION_COLUMNS].columns) == ANNOTATION_COLUMNS
    metadata = json.loads(metadata_one.read_text(encoding="utf-8"))
    assert metadata["labels"] == LABELS
    assert metadata["annotation_complete"] is False


def test_build_gold_rejects_missing_columns(tmp_path):
    input_path = tmp_path / "input.parquet"
    pd.DataFrame({"row_id": ["one"]}).to_parquet(input_path)
    try:
        build_gold(input_path, tmp_path / "out.csv", tmp_path / "out.json")
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("Expected missing-column validation to fail")
