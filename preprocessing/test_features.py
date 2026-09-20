import numpy as np
import pandas as pd

from features import build, load_prices, map_to_next_session

SESSIONS = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04",
                             "2024-01-08", "2024-01-09"])


def test_news_never_maps_to_its_own_session():
    """The whole no-look-ahead guarantee: a headline dated D must map STRICTLY
    after D, because we cannot tell if it was published before or after the close."""
    days = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    mapped = map_to_next_session(days, SESSIONS)
    assert (mapped.values > days.values).all()
    assert mapped.tolist() == [pd.Timestamp("2024-01-03"),
                               pd.Timestamp("2024-01-04"),
                               pd.Timestamp("2024-01-08")]


def test_non_session_days_accumulate_forward():
    """Weekend/holiday news is carried to the next session, not discarded."""
    weekend = pd.Series(pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"]))
    assert map_to_next_session(weekend, SESSIONS).tolist() == [
        pd.Timestamp("2024-01-08")] * 3


def test_days_past_the_last_session_are_dropped_not_clamped():
    later = pd.Series(pd.to_datetime(["2024-01-09", "2024-02-01"]))
    mapped = map_to_next_session(later, SESSIONS)
    assert mapped.isna().all()   # nothing silently pinned to the final session


def test_returns_are_close_to_close_and_open_is_not_exposed(tmp_path):
    """open is the previous close for ~33% of rows (AUDIT_REPORT 8b) -- it must
    not leak into the feature table, and ret must come from close alone."""
    path = tmp_path / "px.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "open": [100.0, 110.0, 121.0],      # == previous close, the trap
        "high": [111.0, 122.0, 133.0],
        "low": [99.0, 109.0, 120.0],
        "close": [110.0, 121.0, 133.1],
        "volume": [1000, 2000, 3000],
    }).to_csv(path, index=False)

    px = load_prices(path)
    assert "open" not in px.columns
    assert np.isclose(px["ret"].iloc[1], 0.10)      # 110 -> 121
    assert np.isclose(px["ret"].iloc[2], 0.10)      # 121 -> 133.1
    assert pd.isna(px["ret"].iloc[0])


def test_ret_next_is_the_following_session_return():
    px = load_prices()
    assert px["ret"].notna().sum() == len(px) - 1


def test_price_report_headlines_are_flagged_not_dropped():
    """The momentum-laundering confound must stay visible and countable.

    Price-report headlines restate the index's own move ("Le Tunindex termine
    a +0,08%"), so their sentiment is a proxy for that day's return. They are
    real news and are NOT removed; they are counted separately so H1 can be run
    with them, without them, and on them alone.
    """
    import pandas as pd
    from config import PRICE_REPORT_PATTERN

    headlines = pd.Series([
        "Bourse de Tunis : Le Tunindex termine sur une note stable (+0,08%)",
        "Le Tunindex a cloture la seance du 20 fevrier en hausse de 0,5%",
        "Hydrogene vert : des entreprises allemandes attendues en Tunisie",
        "La BCT releve son taux directeur de 75 points de base",
    ])
    flagged = headlines.str.contains(PRICE_REPORT_PATTERN, case=False, regex=True)
    assert flagged.tolist() == [True, True, False, True]


def test_feature_table_separates_price_reports_from_other_news():
    import pandas as pd
    frame = pd.read_parquet(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "data" / "curated" / "daily_features.parquet")
    for column in ("n_price_reports", "n_non_price", "price_report_share"):
        assert column in frame.columns
    # the split is exact and never negative
    assert (frame.n_price_reports + frame.n_non_price == frame.n_headlines).all()
    assert (frame.n_non_price >= 0).all()
    assert frame.price_report_share.between(0, 1).all()
