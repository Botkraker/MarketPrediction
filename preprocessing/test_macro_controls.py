import numpy as np
import pandas as pd

import macro_controls as mc


def _prices(n=900, seed=0):
    rng = np.random.default_rng(seed)
    session = pd.bdate_range("2014-01-01", periods=n)
    return pd.DataFrame({"session": session, "ret": rng.normal(0, 0.005, n)})


def _series(days, moves):
    return pd.DataFrame({"day": pd.DatetimeIndex(days), "move": moves})


def test_a_move_that_predicts_the_next_session_enters_f1():
    px = _prices()
    # the series' day-D move equals Tunindex's D+1 return, plus noise
    lead = px["ret"].shift(-1).fillna(0).to_numpy() + np.random.default_rng(1).normal(0, 0.005, len(px))
    r = mc.test_candidate("lead", _series(px.session, lead), px)
    assert r["enters_f1"] and r["next_session"]["r"] > 0.5


def test_noise_does_not_enter_f1():
    px = _prices()
    r = mc.test_candidate("noise", _series(px.session, np.random.default_rng(2).normal(0, 1, len(px))), px)
    assert not r["enters_f1"]


def test_alignment_is_backward_only_and_stale_values_are_missing():
    px = _prices(n=20)
    series = _series(["2014-01-01", "2014-01-03"], [0.1, 0.2])
    joined = mc.align(px, series)
    by_day = joined.set_index("session").move
    assert by_day[pd.Timestamp("2014-01-02")] == 0.1            # carried forward, never back
    assert by_day[pd.Timestamp("2014-01-03")] == 0.2
    assert pd.isna(by_day[pd.Timestamp("2014-01-13")])          # >7 days stale


def test_load_series_takes_log_moves_and_drops_bad_rows(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "bad", "2020-01-03"],
                  "level": [100.0, 110.0, 5.0, 0.0]}).to_csv(path, index=False)
    s = mc.load_series(path)
    assert len(s) == 1 and np.isclose(s.move.iloc[0], np.log(1.1))
