"""
Stage 3: relevant -> canonical-flagged. Marks duplicates without dropping
them: every row keeps dup_cluster_id and cluster_size, and exactly one row
per cluster gets is_canonical=True.

Clustering is exact-hash on a TEMPLATE key: the headline whitespace-collapsed,
lowercased, then with every number/percentage and every French date word
(weekday, month) masked. Plain exact-hash on the raw normalized headline is
not enough here. The template-driven outlets -- ilboursa above all -- emit the
same sentence every day with only the figure changing ("Le Tunindex grimpe de
0,46%" / "... 0,48%"), and under exact hashing those land in different
clusters, so split.py's guard lets them straddle the train/validation
boundary. The sentiment baseline vectorizes char 2-5grams, for which such a
pair is memorization, not generalization.

The masking is deliberately conservative: it only erases numbers and date
words, never verbs. "Le Tunindex gagne #" and "Le Tunindex perd #" stay
separate clusters, as do headlines with the same shape but different
entities. Over-merging destroys real training data, and the obvious next
step -- sorting the tokens of the key so clause reorderings collapse -- was
measured and rejected: on the current corpus it merges 100 more groups, of
which two are semantic opposites ("Fitch revise la perspective de negative a
stable" vs "... de stable a negative", and a dinar apprecie/deprecie pair).
A 2% error rate landing exactly on the sentiment-bearing cases is not worth
~100 extra merges, so clause order is left significant.

MinHash is still not used. The audit ran it for wire-reprint detection
(AUDIT_REPORT.md section 5, Jaccard >= 0.8) and got 0% matches; template
masking is both cheaper and, on this corpus, strictly more precise.

Canonical choice per cluster: TAP row if present (wire original), else
earliest published_date, else lowest row_id for a deterministic tie-break.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"


# A number, with an optional sign, optional ',' '.' or thousands space inside
# it, and an optional trailing percent: "0,46%", "-0,36%", "7 %", "2 700".
# The sign is kept in the mask: "a cloture la seance a -0,36%" and
# "... a +0,21%" are opposite-direction headlines whose only direction marker
# is that character. Cost of keeping it: 16 template groups on the current
# corpus stay split over typographic "+0,59%" vs "0,19%". Worth it.
_SIGN_MASK = {"": "#", "+": "up#", "-": "dn#"}
_NUMBER = re.compile(r"([-+]?)\d+(?:[.,\u00a0\u202f ]\d+)*(?:\s*%)?")
# French weekday and month names: the other thing that varies between two
# otherwise identical daily-report headlines.
_DATE_WORD = re.compile(
    r"\b(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"
    r"|janvier|f[\u00e9e]vrier|mars|avril|mai|juin|juillet|ao[\u00fbu]t"
    r"|septembre|octobre|novembre|d[\u00e9e]cembre)\b"
)
# Everything that is not a word character or a mask character. Outlets are
# inconsistent about "Tunis :" vs "Tunis:", ' vs \u2019, and "quasi-stable" vs
# "quasi stable"; none of that distinguishes two articles.
_PUNCT = re.compile(r"[^\w#@]+", re.UNICODE)


def _norm(headline: str) -> str:
    return re.sub(r"\s+", " ", headline or "").strip().lower()


def _template_key(headline: str) -> str:
    """_norm with numbers, date words and punctuation masked, so headlines
    generated from the same outlet template collapse onto one key."""
    masked = _NUMBER.sub(lambda m: _SIGN_MASK[m.group(1)], _norm(headline))
    masked = _DATE_WORD.sub("@", masked)
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", masked.replace("_", " "))).strip()


def _cluster_id(headline: str) -> str:
    return hashlib.sha1(_template_key(headline).encode("utf-8")).hexdigest()


def _pick_canonical(cluster: pd.DataFrame) -> str:
    tap_rows = cluster[cluster["source"] == "tap"]
    pool = tap_rows if len(tap_rows) else cluster
    pool = pool.sort_values(["published_date", "row_id"])
    return pool.iloc[0]["row_id"]


def run() -> pd.DataFrame:
    df = pd.read_parquet(CURATED / "02_relevance.parquet")
    df = df[df["relevance_tag"] != "other"].copy()

    df["dup_cluster_id"] = df["headline_clean"].map(_cluster_id)
    cluster_sizes = df.groupby("dup_cluster_id")["row_id"].transform("size")
    df["cluster_size"] = cluster_sizes

    canonical_ids = df.groupby("dup_cluster_id", group_keys=False).apply(_pick_canonical, include_groups=False)
    canonical_set = set(canonical_ids)
    df["is_canonical"] = df["row_id"].isin(canonical_set)

    CURATED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CURATED / "03_dedup.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    n_clusters = df["dup_cluster_id"].nunique()
    n_dupe_clusters = (df.groupby("dup_cluster_id")["row_id"].size() > 1).sum()
    print(f"Wrote {CURATED / '03_dedup.parquet'} ({len(df)} rows, {n_clusters} clusters, "
          f"{n_dupe_clusters} clusters with >1 row, {df['is_canonical'].sum()} canonical rows)")
