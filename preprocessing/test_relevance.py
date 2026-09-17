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
