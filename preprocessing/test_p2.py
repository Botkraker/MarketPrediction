import numpy as np
import pandas as pd
import pytest

import bench
from p2 import by_date, flags, panel, solve, within

NO_NEWS = pd.DataFrame({"ticker": pd.Series(dtype=str), "day": pd.to_datetime([]),
                        "is_price_report": pd.Series(dtype=bool)})


def test_news_is_dated_after_the_previous_trade_and_up_to_this_one():
    """P2 card: a headline belongs to the first trade on or after its date, never earlier."""
    trades = pd.DatetimeIndex(["2016-01-04", "2016-01-06", "2016-01-11"])
    assert flags(trades, pd.to_datetime(["2016-01-05", "2016-01-09"])).tolist() == [False, True, True]
    assert flags(trades, pd.to_datetime(["2016-01-04"])).tolist() == [True, False, False]
    assert not flags(trades, pd.to_datetime(["2016-01-12"])).any()   # after the last trade


def test_resampling_whole_dates_equals_ols_on_the_resampled_rows():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(30), 4)
    X = np.c_[np.ones(120), rng.standard_normal((120, 2))]
    y = X @ [0.1, 2.0, -1.0] + rng.standard_normal(120)
    XtX, Xty = by_date(y, X, dates)
    w = rng.integers(0, 3, size=(1, 30)).astype(float)
    picked = np.repeat(np.arange(120), np.repeat(w[0], 4).astype(int))
    assert np.allclose(solve(XtX, Xty, w)[0], np.linalg.lstsq(X[picked], y[picked], rcond=None)[0])
    assert np.allclose(solve(XtX, Xty), np.linalg.lstsq(X, y, rcond=None)[0])


def test_demeaning_within_firm_equals_firm_intercepts():
    rng = np.random.default_rng(1)
    firm = np.repeat([0, 1, 2], 20)
    X = rng.standard_normal((60, 2))
    y = X @ [1.5, -0.5] + 3.0 * firm + rng.standard_normal(60)
    dummies = np.linalg.lstsq(np.c_[X, np.eye(3)[firm]], y, rcond=None)[0][:2]
    assert np.allclose(np.linalg.lstsq(within(X, firm), within(y, firm), rcond=None)[0], dummies)


def test_design_panel_reads_no_firm_price_from_2021():
    """The seal (P2 card: confirm 2021-2022): before the tag, nothing priced from 2021 on."""
    p = panel(news=NO_NEWS)
    late = p.session >= bench.FIRM_SEAL
    assert late.any() and p.loc[late, ["r", "r_m", "ar1", "ar5", "ar1_dimson"]].isna().all().all()
    last = p[~late].groupby("ticker").tail(1)                 # their targets would read 2021
    assert last[["ar1", "ar5"]].isna().all().all()
    assert p.loc[~late, "r"].notna().mean() > 0.9
    if not bench.tag_exists("prereg-p2-v1"):
        with pytest.raises(SystemExit):
            panel(confirm=True, news=NO_NEWS)
