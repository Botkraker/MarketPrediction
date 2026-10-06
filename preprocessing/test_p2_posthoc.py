import numpy as np
import pandas as pd

from p2_posthoc import two_way


def test_two_way_demeaning_equals_firm_and_date_dummies():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"ticker": np.repeat(list("abcd"), 15), "session": np.tile(np.arange(15), 4)})
    d = d.sample(frac=0.8, random_state=0)                 # unbalanced, like the firm panel
    X = rng.standard_normal((len(d), 2))
    y = X @ [0.7, -1.2] + rng.standard_normal(len(d))
    dummies = np.c_[pd.get_dummies(d.ticker).to_numpy(float), pd.get_dummies(d.session).to_numpy(float)[:, 1:]]
    full = np.linalg.lstsq(np.c_[X, dummies], y, rcond=None)[0][:2]
    assert np.allclose(np.linalg.lstsq(two_way(X, d), two_way(y, d), rcond=None)[0], full, atol=1e-8)
