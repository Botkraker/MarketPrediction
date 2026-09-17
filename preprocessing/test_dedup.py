import pandas as pd

from dedup import _norm, _pick_canonical


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
