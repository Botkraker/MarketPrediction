"""Gold v3: pre-2019 gold expansion labelled by Claude Haiku (H3, 2026-10-03).

The 2016-18 yearly CamemBERT models saw 298-582 gold rows and collapsed (AUDIT_REPORT
§8i.2). This adds ~1,800 Haiku labels from 2014-18 so every yearly model from 2016 on
trains on > 700 rows.

    python3 preprocessing/gold_haiku.py sample   # writes chunk_XX/headlines.csv + key.csv
    (each chunk is labelled by a Haiku subagent under PROMPT_V2 -> chunk_XX/labels.csv)
    python3 preprocessing/gold_haiku.py merge    # -> sentiment_gold_v3{,_split}.csv

Haiku sees headline, language and relevance tag only: no date, no source, so it has
fewer cues to apply hindsight. Rows sharing a duplicate cluster with any v2 gold row are
excluded, so none can leak into the 150-row human evaluation split. Every Haiku row goes
to `train`; validation and evaluation are the v2 rows, unchanged. Labels are an LLM's
judgment and are not regenerable bit-for-bit; the raw label files are the record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_annotate import PROMPT_V2

CURATED = Path("data/curated")
OUT = CURATED / "haiku_labels"
PER_YEAR = {2014: 300, 2015: 325, 2016: 375, 2017: 375, 2018: 325}
CHUNK = 100
SEED = 20261003
MODEL = "claude-haiku-4-5 (Claude Code subagent)"
LABELS = {"very_negative", "negative", "neutral", "positive", "very_positive"}


def sample() -> None:
    d = pd.read_parquet(CURATED / "03_dedup.parquet")
    d = d[d.is_canonical & d.kept & d.date_parse_ok & (d.relevance_tag != "other")].copy()
    gold = pd.read_csv(CURATED / "sentiment_gold_v2_full.csv")
    pilot = pd.read_csv(OUT / "pilot_key.csv")
    taken = set(gold.row_id) | set(pilot.row_id)
    taken_clusters = set(gold.dup_cluster_id) | set(d[d.row_id.isin(taken)].dup_cluster_id)
    d = d[~d.dup_cluster_id.isin(taken_clusters)]
    d["year"] = pd.to_datetime(d.published_date).dt.year
    s = pd.concat(d[d.year == y].sample(n, random_state=SEED) for y, n in PER_YEAR.items())
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)   # mix years across chunks
    s["pid"] = [f"h{i:04d}" for i in range(len(s))]
    s[["pid", "row_id", "year", "source"]].to_csv(OUT / "key.csv", index=False)
    for c in range(0, len(s), CHUNK):
        folder = OUT / f"chunk_{c // CHUNK + 1:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "prompt_v2.txt").write_text(PROMPT_V2)
        (s.iloc[c:c + CHUNK][["pid", "headline_clean", "lang", "relevance_tag"]]
         .rename(columns={"lang": "language"}).to_csv(folder / "headlines.csv", index=False))
    print(f"{len(s)} rows in {-(-len(s) // CHUNK)} chunks -> {OUT}")


def read_labels(path: Path) -> pd.DataFrame:
    """Split each line at its first two commas only: some subagents left commas in the
    free-text reason unquoted, and pid and label never contain one."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            pid, label, reason = (line.split(",", 2) + [""])[:3]
            rows.append((pid.strip(), label.strip().strip('"'), reason.strip().strip('"')))
    return pd.DataFrame(rows, columns=["pid", "label", "reason"])


def merge() -> None:
    key = pd.concat([pd.read_csv(OUT / "key.csv"), pd.read_csv(OUT / "pilot_key.csv")])
    labels = pd.concat(read_labels(f) for f in sorted(OUT.glob("*/labels.csv")))
    assert labels.pid.is_unique, "duplicate pid across label files"
    bad = set(labels.label) - LABELS
    assert not bad, f"labels outside the scale: {bad}"
    missing = set(key.pid) - set(labels.pid)
    assert not missing, f"{len(missing)} rows unlabelled, e.g. {sorted(missing)[:5]}"

    d = pd.read_parquet(CURATED / "03_dedup.parquet")
    new = (key.merge(labels, on="pid").merge(
        d[["row_id", "lang", "published_date", "headline_clean", "relevance_tag", "dup_cluster_id"]],
        on="row_id"))
    assert len(new) == len(key)
    new = new.assign(gold_item_id="haiku-" + new.pid, adjudicated_label=new.label,
                     annotation_status="haiku_single", annotator_1_label=new.label,
                     annotator_1_model=MODEL, annotator_1_prompt="v2",
                     annotation_notes=new.reason, gold_part="v3_haiku", in_study=True)

    v2 = pd.read_csv(CURATED / "sentiment_gold_v2_full.csv", keep_default_na=False)
    assert not set(new.dup_cluster_id) & set(v2.dup_cluster_id), "cluster overlaps v2 gold"
    gold = pd.concat([v2, new[[c for c in v2.columns if c in new.columns]]], ignore_index=True)
    split = pd.concat([pd.read_csv(CURATED / "sentiment_gold_v2_full_split.csv"),
                       pd.DataFrame({"gold_item_id": new.gold_item_id, "row_id": new.row_id,
                                     "lang": new.lang, "adjudicated_label": new.label,
                                     "dup_cluster_id": new.dup_cluster_id, "split": "train"})],
                      ignore_index=True)
    gold.to_csv(CURATED / "sentiment_gold_v3.csv", index=False)
    split.to_csv(CURATED / "sentiment_gold_v3_split.csv", index=False)
    meta = {"rows": len(gold), "haiku_rows": len(new), "annotator": MODEL, "prompt": "v2",
            "per_year": new.year.value_counts().sort_index().to_dict(),
            "labels": new.label.value_counts().to_dict(),
            "split": split.split.value_counts().to_dict(),
            "note": "Haiku rows are single-annotator LLM labels, train split only. "
                    "Human-150 agreement of the same setup: QWK3 0.756 (qwen 0.676)."}
    (CURATED / "sentiment_gold_v3_metadata.json").write_text(json.dumps(meta, indent=2, default=int))
    print(json.dumps(meta, indent=2, default=int))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["sample", "merge"])
    {"sample": sample, "merge": merge}[p.parse_args().step]()
