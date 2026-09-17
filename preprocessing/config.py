"""
Editorial decisions from audit/AUDIT_REPORT.md, made explicit and versioned
here instead of hardcoded inline. Change the audit report's decision ->
change it here -> re-run the pipeline.
"""
from datetime import date

# source -> (window_start or None, window_end or None). None means "use
# whatever the raw data has". Sources not listed here are excluded (the
# audit's "drop" decision -- currently only economist_tunisia_economy, a
# 100%-overlapping subset of economist_tunisia_all).
SOURCE_WINDOWS = {
    "assabah": (None, None),
    "ilboursa": (None, None),
    "kapitalis": (None, None),
    "leconomistmaghrebin": (None, None),
    "lapresse": (date(2024, 10, 1), None),  # volume ramps up here and stays continuous
    "economist_tunisia_all": (date(2020, 5, 1), None),
    "guardian_tunisia": (None, None),  # sparse/topical, kept as supplementary signal
    "nyt_economy": (None, None),  # short by construction (recent topical scrape)
    "tap": (None, None),  # currently only ~1 month; re-scrape recommended, not blocking
}

# Sources whose "headline" is sometimes a recurring section/rubric label
# rather than an actual article title (found in audit/repeated_headlines.csv
# -- see AUDIT_REPORT.md §3). Exact match, case-sensitive (source text is
# kept as-is; matching happens on the normalized string).
BOILERPLATE_TITLES = {
    "assabah": {"مختصرات"},
    "lapresse": {
        "Express", "EXPRESS", "Des faits et des chiffres", "Ils ont dit",
        "Kiosque international", "High tech & Innovation",
    },
}

# fastText lid.176 confidence threshold below which a Latin-script headline
# is treated as low-confidence / candidate Arabizi rather than trusted as
# the predicted language. PLACEHOLDER: the plan calls for tuning this
# against the audit's hand labels (audit/language_relevance_sample.csv,
# the *_fill_by_hand columns) which are not filled in yet. Re-tune once
# they are -- see preprocessing/tune_lid_threshold.py.
FASTTEXT_CONFIDENCE_THRESHOLD = 0.5

# Minimum count of Arabizi-flagged "words" (see normalize.count_arabizi_words)
# for a whole headline to be flagged is_arabizi.
ARABIZI_MIN_WORDS = 2

# Relevance keyword lists. FR/EN lists match on lowercased text; AR lists
# match on arabic_normalize_for_matching() output. FIRST-PASS lists based on
# domain knowledge, not yet validated against hand labels (see Track A step
# "test the filter on the 200 audit-labelled articles" -- blocked on the
# same hand-labeling dependency as the fastText threshold above).
KEYWORDS_TUNISIA_ECON = {
    "fr": [
        "tunisie", "tunisien", "bvmt", "tunindex", "bct", "banque centrale",
        "dinar", "tnd", "taux directeur", "inflation", "pib", "budget",
        "cmf", "aneti", "cepex", "fmi", "endettement", "exportations",
        "importations", "croissance", "chomage", "investissement", "impot",
        "fiscal", "bourse de tunis", "deficit", "balance commerciale",
    ],
    # No English list existed for this category even though 3 kept sources
    # (economist_tunisia_all, guardian_tunisia, nyt_economy) are English --
    # added so those sources don't fall through to "other" by default.
    "en": [
        "tunisia", "tunisian", "bvmt", "tunindex", "central bank of tunisia",
        "tunisian dinar", "policy rate", "inflation", "gdp", "budget",
        "imf", "debt", "exports", "imports", "growth", "unemployment",
        "investment", "tax", "tunis stock exchange", "deficit",
        "trade balance", "trade deficit",
    ],
    "ar": [
        "تونس", "تونسي", "البورصة", "الدينار", "البنك المركزي", "التضخم",
        "الناتج المحلي", "الميزانية", "الصندوق النقد", "الصادرات",
        "الواردات", "النمو", "البطالة", "الاستثمار", "الضريبة", "العجز",
        "الديون", "المالية العمومية",
    ],
}
KEYWORDS_GLOBAL_LINKED = {
    "fr": [
        "fed", "reserve federale", "bce", "banque centrale europeenne",
        "wall street", "petrole", "opep", "dollar", "euro", "chine",
        "etats-unis", "recession mondiale", "marches mondiaux", "opep+",
        "taux d'interet americain", "inflation mondiale",
    ],
    "en": [
        "federal reserve", "the fed", "ecb", "wall street", "oil price",
        "opec", "u.s. economy", "global markets", "inflation", "recession",
        "interest rate", "china economy",
    ],
    "ar": [
        "الاحتياطي الفدرالي", "البنك المركزي الاوروبي", "اسعار النفط",
        "اوبك", "الاقتصاد العالمي", "التضخم العالمي", "الاسواق العالمية",
    ],
}

# Per-source dominant language, taken directly from the audit's corpus-level
# langdetect pass (AUDIT_REPORT.md §6 -- 93-100% single-language per source).
# Used in place of per-article language detection for relevance matching:
# cheap, and correct often enough at this stage that a per-article fastText
# pass isn't worth doing before the keyword lists themselves are validated
# against hand labels. Revisit if a source's automatic tag rate looks off.
SOURCE_LANG = {
    "assabah": "ar",
    "ilboursa": "fr",
    "kapitalis": "fr",
    "leconomistmaghrebin": "fr",
    "lapresse": "fr",
    "tap": "fr",
    "economist_tunisia_all": "en",
    "guardian_tunisia": "en",
    "nyt_economy": "en",
}

CURATED_DIR_NAME = "curated"
