"""
Editorial decisions from audit/AUDIT_REPORT.md, made explicit and versioned
here instead of hardcoded inline. Change the audit report's decision ->
change it here -> re-run the pipeline.
"""
from datetime import date
from pathlib import Path

# source -> (window_start or None, window_end or None). None means "use
# whatever the raw data has". Sources not listed here are excluded (the
# audit's "drop" decision). Currently excluded:
#   - economist_tunisia_economy: 100%-overlapping subset of economist_tunisia_all.
#   - assabah: WRONG COUNTRY. scrape_assabah.py targets assabah.ma (Morocco),
#     section /حوادث/ (crime), not the Tunisian assabah.com.tn. Evidence in
#     audit/topical_geography.csv (flag_wrong_country=True): 11,860 Morocco
#     mentions vs 27 Tunisia; 175 dirham vs 0 dinar; 0 Sfax/Carthage/Bizerte.
#     Raw file and scraper are RETAINED so the audit stays reproducible --
#     this key is the only thing gating it out of the pipeline. Re-add once
#     scrape_assabah.py is retargeted to assabah.com.tn.
SOURCE_WINDOWS = {
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
    # assabah retained for reference only -- excluded via SOURCE_WINDOWS above.
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


# --- Price-report headlines: the momentum-laundering confound -----------------
# ~10% of relevant headlines (4,244 of 42,645) report the index's own move:
# "Le Tunindex termine sur une note stable (+0,08%)". Their sentiment is a
# restatement of ret_D, so a sentiment feature built on them re-encodes lagged
# returns and can appear to "predict" ret_D+1 purely via the 0.263 return
# autocorrelation -- momentum laundered through a text channel.
#
# They are NOT dropped: a market-report headline is real news, and excluding it
# by default would be an unjustified editorial choice. They are FLAGGED so H1 can
# be reported three ways: all headlines, excluding price reports, and price
# reports only (a placebo -- if sentiment only works there, it is momentum).
# Concentrated in ilboursa (11.9%), kapitalis (11.2%), leconomistmaghrebin (9.3%).
# v2. v1 was `tunindex|bourse de tunis|clôtur|cloture|séance du|seance du|
# en hausse de|en baisse de|points` and was mis-specified in both directions:
#
#   OVER: the bare token `points` matched 96 headlines that are central-bank
#   RATE DECISIONS -- "La BCT abaisse de 50 points son taux directeur à 7 %",
#   "La BCE relève ses taux de 25 points de base". Those are the most
#   market-relevant headlines in the corpus. v1 removed them from the ex_price
#   TREATMENT arm and put them in the PLACEBO, inverting the control's logic.
#   `bourse de tunis` alone matched 1,908 rows including governance and IPO news
#   ("Bilel Sahnoun nommé DG", "Maille Club prépare son entrée à la bourse").
#
#   UNDER: French-only (1 of 1,048 English rows flagged) and missing `bvmt`,
#   so "BVMT : Le rebond continue" went unflagged.
#
# v2 requires CO-OCCURRENCE: an index/market token AND a move/close verb. A
# headline that merely mentions the exchange is not a price report; one that
# says what the index did is.
PRICE_REPORT_INDEX = r"tunindex|bvmt|bourse de tunis|tunis stock exchange|l'indice|the index"
PRICE_REPORT_MOVE = (
    r"cl\u00f4tur|cloture|clos|termine|finit|s\u00e9ance du|seance du|"
    r"en hausse|en baisse|gagne|perd|recul|progress|rebond|repli|chute|grimpe|"
    r"c\u00e8de|s'appr\u00e9cie|chutes?|chute|stable|inchang|"
    r"clos(?:e|ed)|end(?:s|ed)|gain(?:s|ed)|los(?:es|t)|ris(?:es|en)|f(?:all|ell)|"
    r"\d+[.,]\d+\s*%|[+-]\s*\d"
)
PRICE_REPORT_PATTERN = rf"(?=.*(?:{PRICE_REPORT_INDEX}))(?=.*(?:{PRICE_REPORT_MOVE}))"


# --- BVMT issuer names: the filter's largest FALSE-NEGATIVE source -------------
# The keyword lists above are topic words (inflation, dinar, bourse...). A
# headline about a LISTED COMPANY carries none of them, so 7,215 rows naming a
# BVMT issuer were tagged `other` and discarded -- e.g. "Le benefice net de
# Land'Or bondit de 80 %", "MPBS : Le benefice semestriel recule de 4%",
# "Carthage Cement confirme la consolidation de sa structure financiere".
# Those are the most index-relevant headlines in the corpus.
#
# Loaded from data/raw/bvmt/sotcks_list.csv so the list cannot drift from the
# actual index constituents. NAMES ONLY, >=4 characters: tickers like AB, BT,
# CC, SAH are short enough to match inside ordinary words and inside other
# issuers' names. Country negation still applies on top (relevance_geo.py).
ISSUER_STOPWORDS = {"tunisie", "tunis", "banque", "credit", "societe", "assurances"}


def _load_issuer_names() -> list[str]:
    path = (Path(__file__).resolve().parent.parent
            / "data" / "raw" / "bvmt" / "sotcks_list.csv")
    if not path.exists():
        return []
    import csv
    names = set()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip().lower()
            if len(name) >= 4 and name not in ISSUER_STOPWORDS:
                names.add(name)
    return sorted(names)


KEYWORDS_ISSUERS = _load_issuer_names()
