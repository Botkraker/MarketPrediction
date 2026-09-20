import numpy as np
import pandas as pd

from baseline import evaluate, walk_forward


def _series(n=700, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.0045, n)
    return pd.DataFrame({
        "session": pd.bdate_range("2014-01-01", periods=n),
        "ret_lag0": ret,
        "ret_lag1": np.roll(ret, 1),
        "ret_lag2": np.roll(ret, 2),
        "ret_lag3": np.roll(ret, 3),
        "abs_ret_lag0": np.abs(ret),
        "abs_ret_lag1": np.abs(np.roll(ret, 1)),
        "log_headlines_lag0": rng.normal(2.5, 0.4, n),
        "log_headlines_lag1": rng.normal(2.5, 0.4, n),
        "ret_next": ret,
    }).iloc[3:].reset_index(drop=True)


def test_walk_forward_never_predicts_the_training_region():
    frame = _series()
    preds = walk_forward(frame, ["ret_lag0"], min_train=500, refit_every=20)
    # first prediction is for row 500, not row 0 -- the first 500 are train-only
    assert len(preds) == len(frame.dropna(subset=["ret_lag0", "ret_next"])) - 500
    assert preds.session.min() > frame.session.iloc[499]


def test_a_future_leaking_feature_scores_near_perfect():
    """Guard on the guard: if the walk-forward were leaking, a feature that IS
    the answer would not be needed to score ~1.0. This asserts the harness can
    still detect signal, so a ~0.55 score means 'no signal', not 'broken code'."""
    frame = _series()
    frame["cheat"] = frame["ret_next"]          # deliberate leak
    preds = walk_forward(frame, ["cheat"], min_train=500, refit_every=20,
                         kind="regress")
    assert (preds.y_pred == preds.y_true).mean() > 0.95


def test_constant_baseline_uses_only_training_majority():
    frame = _series()
    preds = walk_forward(frame, ["ret_lag0"], min_train=500, refit_every=20)
    assert set(preds.y_constant.unique()) <= {0, 1}


def test_evaluate_reports_lift_against_the_constant():
    preds = pd.DataFrame({
        "y_true":     [1, 1, 0, 1],
        "y_pred":     [1, 1, 0, 0],
        "y_constant": [1, 1, 1, 1],
        "session":    pd.bdate_range("2024-01-01", periods=4),
        "ret_next":   [0.01, 0.01, -0.01, 0.01],
    })
    scores = evaluate(preds)
    assert scores["accuracy"] == 0.75
    assert scores["constant_baseline"] == 0.75
    assert scores["lift_over_constant"] == 0.0
    assert scores["beats_constant"] is False
