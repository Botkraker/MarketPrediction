import pandas as pd

from dedup import _cluster_id, _norm, _pick_canonical, _template_key


def test_norm_collapses_whitespace_and_case():
    assert _norm("  Le   Dinar   Chute  ") == "le dinar chute"


def test_canonical_prefers_tap_over_earlier_non_tap():
    cluster = pd.DataFrame([
        {"row_id": "ilboursa::1", "source": "ilboursa", "published_date": pd.Timestamp("2024-01-01").date()},
        {"row_id": "tap::5", "source": "tap", "published_date": pd.Timestamp("2024-01-03").date()},
    ])
    assert _pick_canonical(cluster) == "tap::5"


def test_canonical_falls_back_to_earliest_when_no_tap():
    cluster = pd.DataFrame([
        {"row_id": "kapitalis::9", "source": "kapitalis", "published_date": pd.Timestamp("2024-02-02").date()},
        {"row_id": "ilboursa::1", "source": "ilboursa", "published_date": pd.Timestamp("2024-01-01").date()},
    ])
    assert _pick_canonical(cluster) == "ilboursa::1"


def test_canonical_tiebreak_is_deterministic():
    d = pd.Timestamp("2024-01-01").date()
    cluster = pd.DataFrame([
        {"row_id": "leconomistmaghrebin::3", "source": "leconomistmaghrebin", "published_date": d},
        {"row_id": "ilboursa::2", "source": "ilboursa", "published_date": d},
    ])
    assert _pick_canonical(cluster) == "ilboursa::2"


# --- near-duplicate (template) clustering ---------------------------------
# ilboursa & co. generate headlines from templates where only the figure or
# the date changes. Exact hashing put those in different clusters, so
# split.py's guard let them straddle the train/validation boundary.

def _same_cluster(a, b):
    return _cluster_id(a) == _cluster_id(b)


def test_same_template_different_number_shares_a_cluster():
    assert _same_cluster(
        "Bourse de Tunis : Le Tunindex grimpe de 0,46%",
        "Bourse de Tunis : Le Tunindex grimpe de 0,48%",
    )


def test_percent_spacing_and_decimal_width_do_not_split_a_cluster():
    assert _same_cluster(
        "La BCT maintient inchangé son taux directeur à 7 %",
        "La BCT maintient inchangé son taux directeur à 7,50%",
    )


def test_same_template_different_year_shares_a_cluster():
    assert _same_cluster(
        "Attijari bank annonce un bénéfice net de 132 millions de dinars en 2020",
        "Attijari bank annonce un bénéfice net de 174 millions de dinars en 2019",
    )


def test_same_template_different_date_word_shares_a_cluster():
    assert _same_cluster(
        "La BVMT clôture la séance du jeudi 22 septembre 2022 dans le rouge",
        "La BVMT clôture la séance du mardi 6 octobre 2022 dans le rouge",
    )


def test_opposite_direction_verbs_stay_in_different_clusters():
    # Same template, opposite sentiment: merging these would destroy label signal.
    assert not _same_cluster(
        "Bourse de Tunis : Le Tunindex gagne 0,46%",
        "Bourse de Tunis : Le Tunindex perd 0,46%",
    )


def test_different_entity_same_shape_stays_in_different_clusters():
    assert not _same_cluster(
        "La BIAT émet un emprunt obligataire de 100 millions de dinars",
        "Amen Bank émet un emprunt obligataire de 100 millions de dinars",
    )


def test_unrelated_headlines_stay_in_different_clusters():
    assert not _same_cluster(
        "Le dinar tunisien se déprécie face à l'euro",
        "Bourse de Tunis : Le Tunindex grimpe de 0,46%",
    )


def test_exact_duplicate_behaviour_is_preserved():
    # Case and whitespace collapsing still cluster, as before the template key.
    assert _same_cluster("  Le   Dinar   Chute  ", "le dinar chute")


def test_cluster_id_is_deterministic_across_calls():
    headline = "Bourse de Tunis : Le Tunindex en hausse de 0,46%"
    assert _cluster_id(headline) == _cluster_id(headline)
    assert _template_key(headline) == "bourse de tunis le tunindex en hausse de #"


def test_canonical_is_independent_of_row_order():
    rows = [
        {"row_id": "ilboursa::1", "source": "ilboursa", "published_date": pd.Timestamp("2024-01-01").date()},
        {"row_id": "kapitalis::7", "source": "kapitalis", "published_date": pd.Timestamp("2024-01-01").date()},
        {"row_id": "lapresse::4", "source": "lapresse", "published_date": pd.Timestamp("2024-01-02").date()},
    ]
    forward = _pick_canonical(pd.DataFrame(rows))
    backward = _pick_canonical(pd.DataFrame(rows[::-1]))
    assert forward == backward == "ilboursa::1"


def test_punctuation_and_apostrophe_variants_share_a_cluster():
    assert _same_cluster(
        "Bourse de Tunis : le Tunindex clôture quasi-stable",
        "Bourse de Tunis: le Tunindex clôture quasi stable",
    )


def test_sign_only_opposition_stays_in_different_clusters():
    # No direction word here -- the sign is the whole signal, so it survives
    # into the template key.
    assert not _same_cluster(
        "Bourse de Tunis : l’indice Tunindex a clôturé la séance du 25 février 2020 à -0,36%",
        "Bourse de Tunis : l’indice Tunindex a clôturé la séance du 27 février 2020 à +0,21%",
    )


def test_clause_reordering_is_not_merged():
    # Deliberate: token-sorting the key would merge these, but it also merges
    # genuine opposites like Fitch negative->stable vs stable->negative.
    assert not _same_cluster(
        "Fitch Ratings révise la perspective de la Tunisie de négative à stable",
        "Fitch Ratings révise la perspective de la Tunisie de stable à négative",
    )
