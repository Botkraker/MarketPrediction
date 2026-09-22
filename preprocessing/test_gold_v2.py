import json

import pandas as pd
import pytest

from gold_v2 import (
    ANNOTATION_COLUMNS,
    allocate,
    build_gold_v2,
    normalise_headline,
    prompt_example_headlines,
)
from llm_annotate import PROMPT_V2


def _gold(n_per_source=20):
    rows = []
    for source in ("ilboursa", "kapitalis", "tap"):
        for i in range(n_per_source):
            rows.append({
                "gold_item_id": f"gold-{source}-{i:03d}",
                "row_id": f"{source}::{i}",
                "source": source,
                "lang": "fr",
                "published_date": f"{2015 + i % 4}-03-0{1 + i % 9}",
                "headline_clean": f"{source} titre numéro {i}",
                "relevance_tag": "tunisia_econ",
                "dup_cluster_id": f"c-{source}-{i}",
                "annotator_1_label": "positive",
                "annotator_2_label": "",
                "adjudicated_label": "positive",
                "annotation_status": "adjudicated_from_annotator_1",
                "annotation_notes": "v1 reason",
                "in_study": True,
            })
    return pd.DataFrame(rows)


def _paths(tmp_path):
    return dict(
        output_path=tmp_path / "v2.csv",
        v1_labels_path=tmp_path / "v1.csv",
        human_path=tmp_path / "human.csv",
        metadata_path=tmp_path / "meta.json",
    )


def test_prompt_examples_are_found():
    examples = prompt_example_headlines(PROMPT_V2)
    assert len(examples) == 10
    assert normalise_headline("Tunisie - Météo : quel temps fera-t-il mardi ?") in examples


def test_normalise_headline_unifies_typography():
    assert normalise_headline("Fitch maintient la note « B- »") == \
        normalise_headline("fitch  maintient la note \"B-\"")


def test_allocate_guarantees_one_per_stratum_and_hits_target():
    counts = pd.Series({"a": 1000, "b": 5, "c": 1})
    result = allocate(counts, 100)
    assert result.sum() == 100
    assert (result >= 1).all()
    assert result["c"] == 1


def test_allocate_rejects_target_below_strata():
    with pytest.raises(ValueError):
        allocate(pd.Series({"a": 5, "b": 5, "c": 5}), 2)


def test_build_excludes_pilot_examples_out_of_study_and_blanks_labels(tmp_path):
    gold = _gold()
    gold.loc[0, "headline_clean"] = "Tunisie - Météo : quel temps fera-t-il mardi ?"
    gold.loc[1, "in_study"] = False
    gold.loc[3, "headline_clean"] = gold.loc[2, "headline_clean"].upper()
    input_path = tmp_path / "gold.csv"
    pilot_path = tmp_path / "pilot.csv"
    gold.to_csv(input_path, index=False)
    gold.iloc[[4, 5]][["gold_item_id"]].to_csv(pilot_path, index=False)

    out = build_gold_v2(input_path, pilot_path, target=30, human_target=9,
                        **_paths(tmp_path))

    excluded = set(gold.loc[[0, 1, 3, 4, 5], "gold_item_id"])
    assert not excluded & set(out["gold_item_id"])
    assert len(out) == 30
    assert (out[ANNOTATION_COLUMNS] == "").all().all()
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["exclusions"] == {"not_in_study": 1, "annotation_pilot": 2,
                                  "prompt_v2_example": 1, "repeated_headline": 1}


def test_v1_labels_kept_out_of_annotation_and_human_files(tmp_path):
    input_path = tmp_path / "gold.csv"
    _gold().to_csv(input_path, index=False)
    out = build_gold_v2(input_path, None, target=30, human_target=9,
                        **_paths(tmp_path))
    human = pd.read_csv(tmp_path / "human.csv", keep_default_na=False)
    v1 = pd.read_csv(tmp_path / "v1.csv", keep_default_na=False)

    assert "v1_label" not in out.columns
    assert list(human.columns) == ["gold_item_id", "source", "published_date",
                                   "headline_clean", "human_label", "human_notes"]
    assert len(human) == 9 and set(human["source"]) == {"ilboursa", "kapitalis", "tap"}
    assert set(human["gold_item_id"]) <= set(out["gold_item_id"])
    assert set(v1["gold_item_id"]) == set(out["gold_item_id"])


def test_build_is_deterministic(tmp_path):
    input_path = tmp_path / "gold.csv"
    _gold().to_csv(input_path, index=False)
    first = build_gold_v2(input_path, None, target=25, **_paths(tmp_path))
    second = build_gold_v2(input_path, None, target=25, **_paths(tmp_path))
    pd.testing.assert_frame_equal(first, second)


def test_extension_is_the_complement_of_the_core_sample(tmp_path):
    from gold_v2 import build_extension
    gold = _gold()
    gold.loc[0, "headline_clean"] = "Tunisie - Météo : quel temps fera-t-il mardi ?"
    gold.loc[1, "in_study"] = False
    input_path = tmp_path / "gold.csv"
    gold.to_csv(input_path, index=False)
    core = build_gold_v2(input_path, None, target=20, human_target=0, **_paths(tmp_path))

    ext = build_extension(input_path, tmp_path / "v2.csv", tmp_path / "ext.csv",
                          tmp_path / "ext.json")

    assert not set(ext["gold_item_id"]) & set(core["gold_item_id"])
    assert set(ext["gold_item_id"]) | set(core["gold_item_id"]) == \
        set(gold["gold_item_id"]) - set(gold.loc[[0, 1], "gold_item_id"])
    assert (ext[ANNOTATION_COLUMNS] == "").all().all()
