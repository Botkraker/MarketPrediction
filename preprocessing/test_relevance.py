from relevance import tag_row


def test_french_tunisia_econ_headline():
    tag, kw = tag_row("Le dinar tunisien poursuit sa depreciation face a l'euro", "fr")
    assert tag == "tunisia_econ"
    assert kw is not None


def test_french_global_linked_not_tunisia():
    tag, kw = tag_row("La Fed maintient ses taux d'interet inchanges", "fr")
    assert tag == "global_linked"


def test_arabic_matches_on_normalized_form():
    # ة vs ه and alef forms differ from the keyword list's exact spelling --
    # must still match via arabic_normalize_for_matching.
    tag, kw = tag_row("الحكومة التونسيّة تراجع ميزانية الدولة", "ar")
    assert tag == "tunisia_econ"


def test_english_tunisia_econ_headline():
    tag, kw = tag_row("Tunisia's central bank holds interest rates steady", "en")
    assert tag == "tunisia_econ"


def test_irrelevant_headline_tagged_other():
    tag, kw = tag_row("L'equipe nationale de football remporte le match amical", "fr")
    assert tag == "other"
    assert kw is None


# --- country negation (audit finding S6) --------------------------------------
# KEYWORDS_TUNISIA_ECON is mostly bare generic economics terms, so before the
# fix any foreign-country economics headline was tagged tunisia_econ.

def test_foreign_country_generic_keyword_is_not_tunisia_econ():
    # "croissance" is in KEYWORDS_TUNISIA_ECON but this is Senegal.
    tag, _ = tag_row("L'Algérie réalise une croissance économique de 4,1%", "fr")
    assert tag != "tunisia_econ"
    tag, _ = tag_row("Le Sénégal affiche une croissance de 5% au premier trimestre", "fr")
    assert tag != "tunisia_econ"


def test_same_headline_with_tunisia_term_is_still_tunisia_econ():
    tag, kw = tag_row("La Tunisie et l'Algérie visent une croissance de 4,1%", "fr")
    assert tag == "tunisia_econ"
    tag, kw = tag_row("Le Maroc et le dinar tunisien : croissance comparée", "fr")
    assert tag == "tunisia_econ"


def test_english_foreign_country_generic_keyword():
    tag, _ = tag_row("Why Indians are unhappy about 7.8% economic growth", "en")
    assert tag != "tunisia_econ"
    tag, _ = tag_row("Argentina needs less shouting and more growth", "en")
    assert tag != "tunisia_econ"
    # ... but a Tunisia mention keeps it.
    tag, _ = tag_row("Tunisia and Argentina both need growth", "en")
    assert tag == "tunisia_econ"


def test_arabic_foreign_country_generic_keyword():
    tag, _ = tag_row("النمو الاقتصادي في المغرب يبلغ 4 بالمائة", "ar")
    assert tag != "tunisia_econ"
    tag, _ = tag_row("تونس والمغرب: النمو الاقتصادي في البلدين", "ar")
    assert tag == "tunisia_econ"


def test_tunisian_entity_without_the_word_tunisia_survives_foreign_mention():
    # Tunisian ticker / city anchors the headline even though Libya is named.
    tag, _ = tag_row("Sfax : les exportations vers la Libye en hausse", "fr")
    assert tag == "tunisia_econ"
    tag, _ = tag_row("Le Tunindex clôture en hausse malgré la crise en Égypte", "fr")
    assert tag == "tunisia_econ"


def test_ecb_headline_is_not_tunisia_econ():
    # "banque centrale" is a tunisia_econ keyword; the ECB is not Tunisia's.
    tag, _ = tag_row("La Banque centrale européenne maintient ses taux inchangés", "fr")
    assert tag != "tunisia_econ"


def test_dateline_city_does_not_block_a_tunisian_story():
    # Washington/Paris/London/New York are deliberately not foreign markers.
    tag, _ = tag_row(
        "Washington : Marouane El Abassi désigné meilleur gouverneur de banque centrale", "fr")
    assert tag == "tunisia_econ"


def test_listed_issuer_names_are_relevant_even_without_topic_keywords():
    """S6 false-negative side: a headline about a BVMT constituent carries no
    topic keyword (no 'tunisie'/'dinar'/'bourse'), so it was being discarded."""
    for headline in [
        "Le benefice net de Land'Or bondit de 80 % au premier semestre",
        "Carthage Cement confirme la consolidation de sa structure financiere",
        "Amen Bank publie ses resultats annuels",
    ]:
        tag, hit = tag_row(headline, "fr")
        assert tag == "tunisia_econ", f"{headline!r} -> {tag}"
        assert hit


def test_tunisian_issuer_operating_abroad_is_still_tunisian_news():
    """A Tunisian listed company expanding into Morocco IS Tunisian economic
    news. relevance_geo treats a named Tunisian issuer as a Tunisia anchor, so
    country negation must NOT fire here -- earning this row is the whole point
    of the issuer list."""
    tag, hit = tag_row("Delice Holding etend ses activites au Maroc et en Algerie", "fr")
    assert tag == "tunisia_econ"
    assert hit == "delice holding"


def test_foreign_company_news_is_not_swept_in_by_the_issuer_path():
    tag, _ = tag_row("L'inflation au Maroc recule de 0,6% en juillet", "fr")
    assert tag != "tunisia_econ"


def test_short_tickers_are_not_used_as_keywords():
    """AB, BT, CC, SAH would match inside ordinary words and other issuer names."""
    from config import KEYWORDS_ISSUERS
    assert all(len(k) >= 4 for k in KEYWORDS_ISSUERS)
    assert "ab" not in KEYWORDS_ISSUERS and "bt" not in KEYWORDS_ISSUERS
