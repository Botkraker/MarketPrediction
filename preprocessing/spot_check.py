"""
Two manual-validation aids the plan calls for -- both need a human, this
just prepares the material:

1. 30 random cleaned articles (stratified: ~3-4 per source) for eyeballing
   that Arabic renders correctly and figures like "+1,2 %" survived. Also
   runs an automated proxy (script-char counts, a percent-figure regex) so
   a skim of the printed summary catches gross breakage before the human
   pass.
2. The relevance tagger run over the audit's 200-headline hand-label sample
   (audit/language_relevance_sample.csv), joined side by side with the
   automatic tag -- ready to compare the moment the "*_fill_by_hand"
   columns get filled in. They're still blank as of this run (confirmed
   below), so the 90%-catch-rate validation itself stays blocked on that
   labeling, same as it was in the audit.
"""
import re
from pathlib import Path

import pandas as pd

from relevance import tag_row
from config import SOURCE_LANG

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
AUDIT = ROOT / "audit"

_ARABIC_RE = re.compile(r"[؀-ۿ]")
_FR_PERCENT_RE = re.compile(r"\d[\d,.]*\s?%")


def sample_cleaned(n_per_source: int = 4, seed: int = 42) -> pd.DataFrame:
    cleaned = pd.read_parquet(CURATED / "01_cleaned.parquet")
    kept = cleaned[cleaned["kept"]]
    sample = kept.groupby("source", group_keys=False)[kept.columns].apply(
        lambda g: g.sample(n=min(n_per_source, len(g)), random_state=seed)
    )
    sample = sample[["row_id", "source", "headline_raw", "headline_clean", "published_date"]].copy()
    sample["has_arabic_script"] = sample["headline_clean"].map(lambda t: bool(_ARABIC_RE.search(t)))
    sample["has_percent_figure"] = sample["headline_clean"].map(lambda t: bool(_FR_PERCENT_RE.search(t)))
    sample["manual_ok (fill by hand)"] = ""
    sample["manual_notes (fill by hand)"] = ""
    CURATED.mkdir(parents=True, exist_ok=True)
    sample.to_csv(CURATED / "spot_check_sample_30.csv", index=False)
    return sample


def tag_audit_sample() -> pd.DataFrame:
    path = AUDIT / "language_relevance_sample.csv"
    audit_sample = pd.read_csv(path)
    hand_cols = [c for c in audit_sample.columns if "fill by hand" in c]
    filled = audit_sample[hand_cols].apply(lambda c: c.notna() & (c.astype(str).str.strip() != "")).any().any()

    audit_sample["lang"] = audit_sample["source"].map(SOURCE_LANG)
    tags = audit_sample.apply(lambda r: tag_row(str(r["headline"]), r["lang"]), axis=1)
    audit_sample["auto_relevance_tag"] = [t[0] for t in tags]
    audit_sample["auto_matched_keyword"] = [t[1] for t in tags]

    out_path = CURATED / "relevance_validation_sample.csv"
    audit_sample.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(audit_sample)} rows)")
    if filled:
        print("Hand-label columns have entries -- go compare auto_relevance_tag against them.")
    else:
        print("Hand-label columns in audit/language_relevance_sample.csv are still all blank -- "
              "90%-catch-rate validation is still blocked on human labeling, same as the audit "
              "itself flagged. This file is ready the moment they're filled in.")
    return audit_sample


if __name__ == "__main__":
    s = sample_cleaned()
    print(f"Wrote {CURATED / 'spot_check_sample_30.csv'} ({len(s)} rows)")
    print(f"  has_arabic_script: {s['has_arabic_script'].sum()}, "
          f"has_percent_figure: {s['has_percent_figure'].sum()}")
    tag_audit_sample()
