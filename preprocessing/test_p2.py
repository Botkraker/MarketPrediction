import numpy as np
import pandas as pd
import pytest

import bench
from p2 import JUMP, by_date, dimson_betas, flags, panel, read, solve, within

NO_NEWS = pd.DataFrame({"ticker": pd.Series(dtype=str), "day": pd.to_datetime([]),
                        "is_price_report": pd.Series(dtype=bool)})


@pytest.fixture(scope="module")
def design_panel():
    return panel(news=NO_NEWS)


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


def test_design_panel_reads_no_firm_price_from_2021(design_panel):
    """The seal (P2 card: confirm 2021-2022): before the tag, nothing priced from 2021 on."""
    p = design_panel
    late = p.session >= bench.FIRM_SEAL
    assert late.any() and p.loc[late, ["r", "r_m", "ar1", "ar5", "ar1_dimson"]].isna().all().all()
    last = p[~late].groupby("ticker").tail(1)                 # their targets would read 2021
    assert last[["ar1", "ar5"]].isna().all().all()
    assert p.loc[~late, "r"].notna().mean() > 0.9


def test_confirm_needs_the_tag(monkeypatch):
    monkeypatch.setattr(bench, "tag_exists", lambda tag: False)
    with pytest.raises(SystemExit):
        panel(confirm=True, news=NO_NEWS)


def test_move_screen_drops_the_move_and_every_window_that_contains_it(design_panel):
    p = design_panel
    jump = (p.r.abs() > JUMP).astype(float)
    for k in (1, 5):
        ahead = jump.groupby(p.ticker).transform(
            lambda s: s[::-1].rolling(k, min_periods=1).max()[::-1].shift(-1)).fillna(0)
        assert ((jump == 0) & (ahead == 0)).equals(p[f"clean{k}"])


def test_dimson_beta_uses_only_the_past():
    cal = pd.bdate_range("2014-01-01", "2019-12-31")
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0, 0.01, len(cal)), index=cal)
    close = pd.Series(100 * np.cumprod(1 + 0.8 * m.to_numpy() + rng.normal(0, 0.01, len(cal))), index=cal)
    before = dimson_betas(close, m, cal, range(2015, 2019))
    late = cal >= "2017-01-01"
    after = dimson_betas(close.where(~late, close * 1.5), m.where(~late, -3 * m), cal, range(2015, 2019))
    assert [before[y] for y in (2015, 2016, 2017)] == [after[y] for y in (2015, 2016, 2017)]
    assert before[2018] != after[2018]


def test_verdict_order_puts_no_effect_worth_having_first():
    """PREREG_P2 §6: inside +-SESOI wins even when the interval excludes zero."""
    res = {"c1": {"p": 1e-6, "ci95": [0.05, 0.15], "effect_vs_noise": 0.10},
           "c5": {"p": 1e-6, "ci95": [0.30, 0.90], "effect_vs_noise": 0.60},
           "vol": {"p": 0.5, "ci95": [-0.004, 0.004], "effect_vs_noise": 0.0}}
    out = read(res, {"c1": 0.20, "c5": 0.40, "vol": 0.001})
    assert [out[t]["verdict"] for t in ("c1", "c5", "vol")] == [
        "NO EFFECT WORTH HAVING", "CHANNEL", "INCONCLUSIVE"]
