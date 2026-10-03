import numpy as np
import pandas as pd

from features import phantom_sessions
from h3 import residualise, walk_forward_proba


def _frame(n=700, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"session": pd.bdate_range("2016-01-04", periods=n),
                      "x": rng.normal(size=n), "fwd1": rng.normal(size=n) / 100})
    f["fwd5"] = f["fwd1"]
    return f


def test_phantom_needs_close_high_and_low_copied(tmp_path):
    p = tmp_path / "px.csv"
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"],
                  "close": [10.0, 10.0, 10.0, 11.0], "high": [10.5, 10.5, 10.2, 11.0],
                  "low": [9.5, 9.5, 9.9, 10.0]}).to_csv(p, index=False)
    # 01-02 copies all three (phantom); 01-03 is flat on close but has its own range
    assert list(phantom_sessions(p).strftime("%Y-%m-%d")) == ["2020-01-02"]


def test_a_future_leaking_feature_collapses_log_loss():
    """The harness can see a signal when one exists: feeding the answer in drives
    log-loss far below the ~1.10 of an uninformative 3-class forecast."""
    f = _frame()
    f["leak"] = f["fwd1"]
    pred = walk_forward_proba(f, ["leak"], 1, min_train=300)
    p = pred[["p0", "p1", "p2"]].to_numpy()[np.arange(len(pred)), pred.y]
    assert -np.log(p).mean() < 0.6


def test_predictions_do_not_depend_on_future_targets():
    f = _frame()
    first = walk_forward_proba(f, ["x"], 5, min_train=300).iloc[:20]
    g = f.copy()
    g.loc[400:, "fwd5"] = 99.0          # rewrite every target after the first test block
    again = walk_forward_proba(g, ["x"], 5, min_train=300).iloc[:20]
    pd.testing.assert_frame_equal(first[["p0", "p1", "p2"]], again[["p0", "p1", "p2"]])


def test_orthogonal_residual_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    a = residualise(X, 0, [1, 2], slice(0, 100))
    Y = X.copy()
    Y[100:] *= 50                       # wildly different test rows
    b = residualise(Y, 0, [1, 2], slice(0, 100))
    np.testing.assert_allclose(a[:100, 0], b[:100, 0])
