import numpy as np
import pandas as pd

import w


def frame(seed: int = 0, sessions: int = 420, firms: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=sessions)
    d = pd.DataFrame({"session": np.repeat(days, firms), "ticker": np.tile([f"F{i}" for i in range(firms)], sessions)})
    d["prob"] = rng.uniform(0.2, 0.8, len(d))                       # inflated, as balanced weights make them
    d["event"] = (rng.random(len(d)) < d.prob / 5).astype(int)      # true rate is a fifth of that
    return d


def test_calibration_uses_only_outcomes_known_before_each_block():
    d = frame()
    pos = np.searchsorted(np.sort(d.session.unique()), d.session.to_numpy())
    flipped = d.assign(event=np.where(pos >= 340, 1 - d.event, d.event))
    a, b = w.calibrate(d, "platt"), w.calibrate(flipped, "platt")
    early = pos < 380                                  # block 360 sees outcomes of sessions < 340 only
    assert np.allclose(a[early], b[early], equal_nan=True)
    assert not np.allclose(a[~early], b[~early])


def test_platt_brings_inflated_probabilities_to_the_observed_rate():
    d = frame()
    p = w.calibrate(d, "platt")
    ok = ~np.isnan(p)
    assert d.prob[ok].mean() > 0.45
    assert abs(p[ok].mean() - d.event[ok].mean()) < 0.02
