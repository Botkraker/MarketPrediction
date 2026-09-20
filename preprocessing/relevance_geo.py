"""
Country-negation for the relevance filter (audit finding S6).

KEYWORDS_TUNISIA_ECON contains bare generic economics terms (inflation,
croissance, budget, growth, gdp, tax, debt...). Matching on those alone tags
any foreign-country economics headline as `tunisia_econ` -- the same
wrong-country error that got the Moroccan `assabah` source excluded
(AUDIT_REPORT.md 6b), except sitting inside the KEPT corpus.

Rule implemented here: a headline that names a foreign country/city and names
no Tunisian entity is NOT about the Tunisian economy. `relevance.tag_row`
applies it before accepting an econ keyword hit; a headline containing any
Tunisia term is never blocked, so Tunisia-specific keywords (tunisie, bvmt,
tunindex, dinar, BCT, bourse de tunis, تونس ...) keep their old behaviour.

Matching: accent-folded, word-boundary for Latin scripts; plain substring for
Arabic (Arabic glues the article/prepositions onto the noun, so \b is wrong
there). Lists are deliberately surface-form based -- no NER dependency.
"""
import re
import unicodedata

from normalize import arabic_normalize_for_matching

# Tunisian: country, currency, market/institution tickers, governorates and
# the few listed companies whose names carry no "tunis" string. Any of these
# present => the headline is never treated as foreign-only.
TUNISIA_TERMS_LATIN = [
    "tunisie", "tunisia", "tunisien", "tunisienne", "tunisian", "tunis",
    "tunisiens", "tunisiennes", "tunisians",
    "bvmt", "tunindex", "bct", "cmf", "aneti", "cepex", "utica", "ugtt",
    "dinar", "dinars", "tnd", "millimes",
    "sfax", "sousse", "bizerte", "kairouan", "gabes", "monastir", "nabeul",
    "djerba", "jerba", "carthage", "ariana", "ben arous", "medenine",
    "tataouine", "kasserine", "gafsa", "tozeur", "kebili", "mahdia", "beja",
    "jendouba", "siliana", "zaghouan", "manouba", "zarzis", "hammamet",
    "rades", "la goulette", "la marsa",
    "tunisair", "poulina", "sfbt", "steg", "sonede", "etap", "sotuver",
    "sotipapier", "telnet", "carthage cement", "delice holding",
]
TUNISIA_TERMS_AR = [
    "تونس", "تونسي", "الدينار", "البورصه التونسيه", "البنك المركزي التونسي",
    "صفاقس", "سوسه", "بنزرت", "القيروان", "قابس", "المنستير", "نابل",
    "جربه", "قرطاج", "اريانه", "بن عروس", "مدنين", "تطاوين", "القصرين",
    "قفصه", "توزر", "قبلي", "المهديه", "باجه", "جندوبه", "سليانه",
    "زغوان", "منوبه", "الخطوط التونسيه",
]

# Foreign countries/nationalities/cities. Washington, New York, London and
# Paris are deliberately ABSENT: they appear as datelines on Tunisian stories
# ("Washington : Marouane El Abassi designe meilleur gouverneur...", "Bouden
# plaide a Paris pour...") and cost more false negatives than they buy. Their
# country names are still listed. Latin forms are written unaccented
# (the matcher folds accents); Arabic forms are written in the
# arabic_normalize_for_matching shape (ة->ه, alef forms unified).
FOREIGN_TERMS_LATIN = [
    # Maghreb / Africa
    "maroc", "marocain", "marocaine", "morocco", "moroccan", "casablanca",
    "rabat", "marrakech", "marrakesh", "tanger", "tangier", "agadir",
    "algerie", "algerien", "algerienne", "algeria", "algerian", "alger",
    "algiers", "oran",
    "libye", "libyen", "libya", "libyan", "tripoli", "benghazi",
    "mauritanie", "mauritania", "nouakchott",
    "egypte", "egyptien", "egypt", "egyptian", "le caire", "cairo",
    "soudan", "sudan", "khartoum",
    "senegal", "senegalais", "senegalese", "dakar",
    "nigeria", "nigerian", "nigeria n", "lagos", "abuja",
    "ghana", "ghaneen", "accra", "kenya", "kenyan", "nairobi",
    "cote d'ivoire", "ivoirien", "ivory coast", "abidjan",
    "mali", "malien", "bamako", "niger", "burkina", "ouagadougou",
    "ethiopie", "ethiopia", "ethiopian", "addis",
    "afrique du sud", "south africa", "south african", "johannesburg",
    "cameroun", "cameroon", "gabon", "congo", "angola", "zimbabwe",
    # Middle East
    "arabie saoudite", "saoudien", "saudi", "riyad", "riyadh", "djeddah",
    "emirats", "emirates", "dubai", "abou dhabi", "abu dhabi",
    "qatar", "qatari", "doha", "koweit", "kuwait", "bahrein", "bahrain",
    "oman", "yemen", "liban", "libanais", "lebanon", "lebanese", "beyrouth",
    "beirut", "syrie", "syrien", "syria", "syrian", "damas", "damascus",
    "irak", "iraq", "iraqi", "bagdad", "baghdad", "iran", "iranien",
    "iranian", "teheran", "tehran", "jordanie", "jordan", "amman",
    "israel", "israelien", "israeli", "tel aviv",
    "palestine", "palestinien", "palestinian", "gaza",
    "turquie", "turc", "turque", "turkey", "turkish", "istanbul", "ankara",
    # Europe / Americas / Asia
    "france", "francais", "french", "marseille", "lyon",
    "italie", "italien", "italy", "italian", "rome", "milan",
    "espagne", "espagnol", "spain", "spanish", "madrid", "barcelone",
    "allemagne", "allemand", "germany", "german", "berlin",
    "royaume-uni", "britannique", "britain", "british", "angleterre", "england",
    "portugal", "grece", "greece", "greek", "athenes", "athens",
    "belgique", "belgium", "pays-bas", "netherlands", "suisse",
    "switzerland", "autriche", "austria", "suede", "sweden", "norvege",
    "norway", "pologne", "poland", "roumanie", "romania",
    "russie", "russe", "russia", "russian", "moscou", "moscow",
    "ukraine", "ukrainien", "ukrainian", "kiev", "kyiv",
    "etats-unis", "americain",
    "canada", "canadien", "canadian", "ottawa", "toronto",
    "mexique", "mexico", "mexican", "bresil", "brazil", "brazilian",
    "argentine", "argentina", "argentinian", "buenos aires",
    "venezuela", "colombie", "colombia", "chili", "chile", "perou", "peru",
    "chine", "chinois", "china", "chinese", "pekin", "beijing", "shanghai",
    "japon", "japonais", "japan", "japanese", "tokyo",
    "inde", "indien", "india", "indian", "new delhi", "mumbai",
    "pakistan", "pakistani", "bangladesh", "indonesie", "indonesia",
    "vietnam", "thailande", "thailand", "philippines", "malaisie",
    "malaysia", "singapour", "singapore",
    "coree du sud", "south korea", "korean", "seoul",
    "australie", "australia", "australian", "sydney",
    # Supranational institutions that are unambiguously not Tunisian. The
    # generic keyword "banque centrale" otherwise tags every ECB/Fed headline
    # as tunisia_econ. ("the fed" and not bare "fed": "fed up".)
    "bce", "banque centrale europeenne", "european central bank",
    "reserve federale", "federal reserve", "the fed", "wall street",
    "opep", "opec",
]
FOREIGN_TERMS_AR = [
    "المغرب", "مغربي", "الرباط", "الدار البيضاء", "مراكش", "طنجه", "اكادير",
    "الجزاير", "جزايري", "ليبيا", "ليبي", "طرابلس", "موريتانيا",
    "مصر", "مصري", "القاهره", "السودان", "الخرطوم",
    "السنغال", "داكار", "نيجيريا", "غانا", "كينيا", "اثيوبيا", "مالي",
    "جنوب افريقيا", "الكاميرون",
    "السعوديه", "سعودي", "الرياض", "جده", "الامارات", "دبي", "ابوظبي",
    "قطر", "قطري", "الدوحه", "الكويت", "البحرين", "عمان", "اليمن",
    "لبنان", "لبناني", "بيروت", "سوريا", "سوري", "دمشق", "العراق",
    "عراقي", "بغداد", "ايران", "ايراني", "طهران", "الاردن", "عمان",
    "اسراييل", "فلسطين", "غزه", "تركيا", "تركي", "اسطنبول", "انقره",
    "فرنسا", "فرنسي", "باريس", "ايطاليا", "روما", "اسبانيا", "مدريد",
    "المانيا", "برلين", "بريطانيا", "لندن", "اليونان", "بلجيكا",
    "روسيا", "روسي", "موسكو", "اوكرانيا", "كييف",
    "الولايات المتحده", "امريكي", "واشنطن", "نيويورك", "كندا",
    "المكسيك", "البرازيل", "الارجنتين", "فنزويلا",
    "الصين", "صيني", "بكين", "اليابان", "طوكيو", "الهند", "هندي",
    "باكستان", "بنغلاديش", "اندونيسيا", "فيتنام", "ماليزيا", "سنغافوره",
    "كوريا", "سيول", "استراليا",
]

_COMBINING = re.compile(r"[̀-ͯ]")


def fold(text: str) -> str:
    """Lowercase + strip accents, so 'L'Algérie' matches 'algerie'."""
    if not text:
        return ""
    return _COMBINING.sub("", unicodedata.normalize("NFD", text.lower()))


def _boundary_re(terms: list[str]) -> re.Pattern:
    # trailing "s" tolerated so "Indians"/"Moroccans" match "indian"/"moroccan"
    return re.compile(r"(?<!\w)(?:%s)s?(?!\w)" % "|".join(sorted(map(re.escape, terms), key=len, reverse=True)))


_TUNISIA_RE = _boundary_re([fold(t) for t in TUNISIA_TERMS_LATIN])
_FOREIGN_RE = _boundary_re([fold(t) for t in FOREIGN_TERMS_LATIN])
_TUNISIA_AR = [arabic_normalize_for_matching(t) for t in TUNISIA_TERMS_AR]
_FOREIGN_AR = [arabic_normalize_for_matching(t) for t in FOREIGN_TERMS_AR]


def mentions_tunisia(headline: str, lang: str) -> bool:
    if lang == "ar":
        t = arabic_normalize_for_matching(headline or "")
        return any(k in t for k in _TUNISIA_AR)
    return bool(_TUNISIA_RE.search(fold(headline)))


def mentions_foreign(headline: str, lang: str) -> bool:
    if lang == "ar":
        t = arabic_normalize_for_matching(headline or "")
        return any(k in t for k in _FOREIGN_AR)
    return bool(_FOREIGN_RE.search(fold(headline)))


def is_foreign_context(headline: str, lang: str) -> bool:
    """True when the headline names a foreign country/city and no Tunisian
    entity -- i.e. an econ keyword hit on it is a wrong-country match."""
    return mentions_foreign(headline, lang) and not mentions_tunisia(headline, lang)
