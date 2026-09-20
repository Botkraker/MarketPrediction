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
        "score":      [0.8, 0.7, 0.2, 0.4],   # evaluate() needs a ranking score for AUC
        "y_constant": [1, 1, 1, 1],
        "session":    pd.bdate_range("2024-01-01", periods=4),
        "ret_next":   [0.01, 0.01, -0.01, 0.01],
    })
    scores = evaluate(preds)
    assert scores["accuracy"] == 0.75
    assert scores["constant_baseline"] == 0.75
    assert scores["lift_over_constant"] == 0.0
    assert scores["beats_constant"] is False


def test_embargo_removes_sessions_adjacent_to_the_prediction():
    """Blueprint 6.3: 5-session embargo between train and test. With ret
    autocorrelation at +0.263 the adjacent sessions are the most correlated with
    the target, so training on them is the closest thing to leakage left here."""
    import numpy as np
    from baseline import EMBARGO_SESSIONS, walk_forward

    assert EMBARGO_SESSIONS == 5

    # A feature that is pure noise EXCEPT on the 3 sessions before each target.
    # With embargo=0 the model can exploit it; with embargo=5 it cannot.
    rng = np.random.default_rng(0)
    n = 900
    ret = rng.normal(0.0005, 0.0045, n)
    frame = pd.DataFrame({
        "session": pd.bdate_range("2014-01-01", periods=n),
        "ret_next": np.roll(ret, -1),
        "ret_lag0": ret,
        "ret_lag1": np.roll(ret, 1),
        "ret_lag2": np.roll(ret, 2),
    }).iloc[3:-1].reset_index(drop=True)

    without = walk_forward(frame, ["ret_lag0"], min_train=400, embargo=0)
    with_emb = walk_forward(frame, ["ret_lag0"], min_train=400, embargo=5)
    assert len(without) == len(with_emb)          # same predictions, different fits
    assert without.session.equals(with_emb.session)


def test_classify_stores_a_probability_not_a_class_label():
    """ROC-AUC on a 0/1 class label is meaningless -- it needs a ranking score."""
    from baseline import walk_forward
    frame = _series()
    preds = walk_forward(frame, ["ret_lag0"], min_train=500, kind="classify")
    assert preds.score.between(0, 1).all()
    assert preds.score.nunique() > 2               # not just {0, 1}


def test_evaluate_reports_the_blueprint_headline_metrics():
    """Blueprint 6.2: AUC primary; balanced accuracy and MCC replace raw accuracy."""
    from baseline import evaluate
    preds = pd.DataFrame({
        "y_true":     [1, 1, 0, 0, 1, 0],
        "y_pred":     [1, 1, 0, 1, 1, 0],
        "score":      [0.9, 0.8, 0.2, 0.6, 0.7, 0.1],
        "y_constant": [1, 1, 1, 1, 1, 1],
        "session":    pd.bdate_range("2024-01-01", periods=6),
        "ret_next":   [0.01, 0.01, -0.01, -0.01, 0.01, -0.01],
    })
    s = evaluate(preds)
    for metric in ("roc_auc", "balanced_accuracy", "mcc", "brier"):
        assert metric in s, metric
    assert 0.0 <= s["roc_auc"] <= 1.0
    assert -1.0 <= s["mcc"] <= 1.0
    assert s["roc_auc"] > 0.5                      # this toy model ranks correctly
