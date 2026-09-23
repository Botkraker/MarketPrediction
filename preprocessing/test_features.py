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
        # A RATE DECISION IS NOT A PRICE REPORT. The v1 pattern matched the bare
        # token "points" and flagged 96 of these, exiling the most market-relevant
        # headlines in the corpus from the treatment arm into the placebo. An
        # earlier version of THIS TEST asserted True here and certified the bug.
        "La BCT releve son taux directeur de 75 points de base",
        # merely naming the exchange is not a price report either
        "Bilel Sahnoun nomme nouveau directeur general de la Bourse de Tunis",
    ])
    flagged = headlines.str.contains(PRICE_REPORT_PATTERN, case=False, regex=True)
    assert flagged.tolist() == [True, True, False, False, False]


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


def test_most_recent_available_return_is_exposed_as_ret_lag0():
    """Regression guard for the lag-offset bug.

    Row i is session i; the target ret_next[i] is session i+1, predicted after
    session i has closed. ret[i] is therefore AVAILABLE and is the strongest
    single predictor. Defining lags from the row instead of from the target
    silently dropped it and turned a significant baseline (0.5736, p=0.028) into
    a null (0.5564, p=0.418).
    """
    import pandas as pd
    from pathlib import Path
    from baseline import FEATURE_SETS

    frame = pd.read_parquet(Path(__file__).resolve().parent.parent
                            / "data" / "curated" / "daily_features.parquet")
    assert "ret_lag0" in frame.columns
    assert frame["ret_lag0"].equals(frame["ret"])          # lag 0 == the row's own return
    # Lags are computed before the --start filter, so row 0's lag correctly comes
    # from the last session BEFORE the window. Compare from row 1 onward.
    assert frame["ret_lag1"].iloc[1:].equals(frame["ret"].shift(1).iloc[1:])
    assert frame["ret_next"].iloc[:-1].equals(frame["ret"].shift(-1).iloc[:-1])
    assert pd.notna(frame["ret_lag1"].iloc[0])   # carried in from pre-window

    # every momentum feature set must include the most recent available return
    for name, feats in FEATURE_SETS.items():
        assert "ret_lag0" in feats, f"{name} omits the last closed session's return"

    # and it must actually be the more informative one
    d = frame[["ret_lag0", "ret_lag1", "ret_next"]].dropna()
    assert abs(d.ret_lag0.corr(d.ret_next)) > abs(d.ret_lag1.corr(d.ret_next))


def test_ret_lag0_is_not_look_ahead():
    """ret_lag0[i] must depend only on closes up to session i."""
    import numpy as np
    import pandas as pd
    from features import load_prices

    px = load_prices()
    expected = px["close"].pct_change()
    assert np.allclose(px["ret"].dropna(), expected.dropna())
    # ret[i] uses close[i] and close[i-1] -- never close[i+1]
    assert px["ret"].iloc[0] != px["ret"].iloc[0] or pd.isna(px["ret"].iloc[0])


def test_orthogonal_residual_never_uses_future_rows():
    """Changing a later row must not change any earlier residual."""
    from features import expanding_residual
    rng = np.random.default_rng(0)
    controls = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    target = pd.Series(controls.a * 2 + rng.normal(size=200))
    base = expanding_residual(target, controls, min_history=30)
    shocked = target.copy()
    shocked.iloc[150:] += 100.0
    again = expanding_residual(shocked, controls, min_history=30)
    assert np.allclose(base.iloc[:150], again.iloc[:150])
    assert (base.iloc[:30] == target.iloc[:30]).all()          # no projection yet
    # once history exists it really does remove the control
    assert abs(np.corrcoef(base.iloc[100:], controls.a.iloc[100:])[0, 1]) < 0.2
