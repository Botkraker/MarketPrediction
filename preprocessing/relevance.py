"""
Stage 2: cleaned -> relevance-tagged. Matches config.KEYWORDS_* against the
headline (there are no paragraphs/section names in this headline-only
corpus, so matching is headline-only rather than "title + first two
paragraphs + section" as in the original body-based plan).

Language is looked up per-source (config.SOURCE_LANG, from the audit's
corpus-level langdetect pass) rather than detected per-article -- see
config.py's comment on that tradeoff.

Arabic matching runs on normalize.arabic_normalize_for_matching(headline),
never on the stored/displayed text, per the plan.
"""
from pathlib import Path

import pandas as pd

from config import (KEYWORDS_GLOBAL_LINKED, KEYWORDS_ISSUERS,
                    KEYWORDS_TUNISIA_ECON, SOURCE_LANG)
from normalize import arabic_normalize_for_matching
from relevance_geo import is_foreign_context

CURATED = Path(__file__).resolve().parent.parent / "data" / "curated"


def _match(text_norm: str, keywords: list[str]) -> str | None:
    for kw in keywords:
        if kw in text_norm:
            return kw
    return None


def tag_row(headline_clean: str, lang: str) -> tuple[str, str | None]:
    """Returns (relevance_tag, matched_keyword)."""
    if lang == "ar":
        text_norm = arabic_normalize_for_matching(headline_clean)
    else:
        text_norm = headline_clean.lower()

    econ_kws = KEYWORDS_TUNISIA_ECON.get(lang, [])
    econ_kws = [arabic_normalize_for_matching(k) if lang == "ar" else k.lower() for k in econ_kws]
    hit = _match(text_norm, econ_kws)
    # Country negation (audit finding S6): most KEYWORDS_TUNISIA_ECON entries
    # are bare generic economics terms, so a foreign-country headline matches
    # them too ("L'Algerie realise une croissance de..."). A headline naming a
    # foreign country and no Tunisian entity cannot be tunisia_econ; it falls
    # through to global_linked/other. Headlines carrying a Tunisia-specific
    # term (tunisie/dinar/bvmt/tunindex/bct/bourse de tunis/تونس/...) are never
    # blocked, so the Tunisia-specific keywords behave exactly as before.
    if hit and not is_foreign_context(headline_clean, lang):
        return "tunisia_econ", hit

    # Listed-issuer names (audit finding S6, false-negative side). The keyword
    # lists are TOPIC words, so a headline about a BVMT constituent carries none
    # of them -- 7,215 rows naming an issuer were being discarded as `other`
    # ("Le benefice net de Land'Or bondit de 80 %"). Issuer names are matched on
    # the raw lowercased text in any language, since company names are not
    # translated, and country negation still applies so a foreign story
    # mentioning a Tunisian issuer is not swept in.
    issuer_hit = _match(headline_clean.lower(), KEYWORDS_ISSUERS)
    if issuer_hit and not is_foreign_context(headline_clean, lang):
        return "tunisia_econ", issuer_hit

    global_kws = KEYWORDS_GLOBAL_LINKED.get(lang, [])
    global_kws = [arabic_normalize_for_matching(k) if lang == "ar" else k.lower() for k in global_kws]
    hit = _match(text_norm, global_kws)
    if hit:
        return "global_linked", hit

    return "other", None


def run() -> pd.DataFrame:
    cleaned = pd.read_parquet(CURATED / "01_cleaned.parquet")
    df = cleaned[cleaned["kept"]].copy()
    df["lang"] = df["source"].map(SOURCE_LANG)

    tags = df.apply(lambda r: tag_row(r["headline_clean"], r["lang"]), axis=1)
    df["relevance_tag"] = [t[0] for t in tags]
    df["matched_keyword"] = [t[1] for t in tags]

    CURATED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CURATED / "02_relevance.parquet", index=False)
    return df


if __name__ == "__main__":
    df = run()
    print(f"Wrote {CURATED / '02_relevance.parquet'} ({len(df)} rows)")
    print(pd.crosstab(df["source"], df["relevance_tag"]))
