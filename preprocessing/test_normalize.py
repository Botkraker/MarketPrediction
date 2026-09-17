from normalize import (
    arabic_normalize_for_matching,
    clean_text,
    count_arabizi_words,
    script_counts,
)


def test_french_percentage_survives():
    # non-breaking space before %, and a comma decimal -- both common in FR
    # financial text -- must survive stage-1 normalization intact or
    # collapsed to a plain space, never dropped.
    raw = "Le Tunindex gagne +1,2 % cette semaine"
    out = clean_text(raw)
    assert "+1,2" in out
    assert "%" in out
    assert " " not in out  # NFKC folds NBSP to a normal space


def test_arabic_with_tatweel_untouched_by_stage1():
    raw = "الاقتصـــاد التونسي"  # tatweel-elongated
    out = clean_text(raw)
    assert "ـ" in out  # stage 1 keeps punctuation/elongation as-is


def test_arabic_with_tatweel_stripped_for_matching():
    raw = "الاقتصـــاد التونسي"
    out = arabic_normalize_for_matching(raw)
    assert "ـ" not in out
    assert "الاقتصاد" in out.replace(" ", "") or "التونسي" in out


def test_alef_and_ta_marbuta_unified_for_matching():
    assert arabic_normalize_for_matching("أحمد") == arabic_normalize_for_matching("احمد")
    assert arabic_normalize_for_matching("دولة") == arabic_normalize_for_matching("دوله")


def test_arabizi_line_flagged():
    raw = "chneya 7ata tounes ken najem net3allem barcha 7ajet"
    assert count_arabizi_words(raw) >= 1


def test_ordinal_not_flagged_as_arabizi():
    raw = "Le 3e trimestre a ete difficile"
    assert count_arabizi_words(raw) == 0


def test_zero_width_and_direction_marks_removed():
    raw = "Tunisie​‏: croissance‪ de 2%‬"
    out = clean_text(raw)
    for cp in "​‏‪‬":
        assert cp not in out
    assert "Tunisie" in out and "croissance" in out


def test_html_entities_unescaped():
    raw = "Tunisie &amp; BVMT: taux &gt; 8&nbsp;%"
    out = clean_text(raw)
    assert "&amp;" not in out and "&" in out
    assert "&gt;" not in out and ">" in out
    assert " " not in out


def test_script_counts():
    ar, la = script_counts("Tunisie تونس")
    assert ar == 4  # ت و ن س
    assert la == 7  # Tunisie
