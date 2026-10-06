import numpy as np
import pandas as pd
import pytest

from baseline import walk_forward as h1_walk_forward
from bench import block_permute, compare, residualiser, seal, surrogate, walk_forward
from h3 import walk_forward_proba


def _frame(n=700, seed=0, start="2016-01-04"):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"session": pd.bdate_range(start, periods=n),
                      "x": rng.normal(size=n), "c": rng.normal(size=n)})
    f["fwd1"] = 0.002 * f.x + rng.normal(size=n) / 100
    f["ret_next"] = f["fwd1"]
    return f


def test_seal_drops_2024_and_blanks_targets_that_would_read_it():
    f = _frame(start="2023-12-01")
    s = seal(f, "p1", forward={"fwd1": 1})
    assert s.session.max() < pd.Timestamp("2024-01-01")
    assert np.isnan(s.fwd1.iloc[-1]) and s.fwd1.iloc[:-1].notna().all()


def test_seal_refuses_confirm_without_the_prereg_tag():
    with pytest.raises(SystemExit):
        seal(_frame(), "never-tagged-phase", confirm=True)


def test_terciles_reproduce_h3_exactly():
    f = _frame()
    a = walk_forward(f, ["x"], "fwd1", 1, "terciles", min_train=300)
    b = walk_forward_proba(f, ["x"], 1, min_train=300)
    assert (a.y == b.y).all() and np.allclose(a[["p0", "p1", "p2"]], b[["p0", "p1", "p2"]])


def test_regress_reproduces_the_h1_walk_forward_exactly():
    f = _frame()
    a = walk_forward(f, ["x"], "ret_next", min_train=300)
    b = h1_walk_forward(f, ["x"], min_train=300, kind="regress")
    assert np.allclose(a.pred, b.score)


def test_future_rows_do_not_change_past_predictions():
    """Loophole register (ADR §9): every transform is fitted in-fold, so rewriting the
    future, controls included, must leave earlier predictions untouched."""
    f = _frame()
    tf = residualiser("x", ["c"])
    first = walk_forward(f, ["x"], "fwd1", transform=tf, min_train=300).iloc[:100]
    g = f.copy()
    g.loc[420:, ["x", "c", "fwd1"]] = 99.0
    again = walk_forward(g, ["x"], "fwd1", transform=tf, min_train=300).iloc[:100]
    assert np.allclose(first.pred, again.pred)


def test_clark_west_finds_a_strong_planted_signal():
    f = _frame()
    f["fwd1"] = f.fwd1 + 0.01 * f.c
    base = walk_forward(f, ["x"], "fwd1", min_train=300)
    res = compare(walk_forward(f, ["x", "c"], "fwd1", min_train=300), base, 1, "mse", n_boot=0)
    assert res["cw_stat"] > 5


def test_surrogate_keeps_values_and_drift_but_not_timing():
    """The matched-noise null since run 1: as slow-moving as the news, unrelated in time."""
    rng = np.random.default_rng(1)
    x = np.zeros(1000)
    for t in range(1, 1000):
        x[t] = 0.95 * x[t - 1] + rng.normal()
    block = np.c_[x, (rng.random(1000) < 0.3).astype(float)]
    s = surrogate(block, seed=7)
    assert np.array_equal(np.sort(s, 0), np.sort(block, 0))       # same values; 0/1 stays 0/1
    lag1 = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    assert abs(lag1(s[:, 0]) - lag1(x)) < 0.05                    # drifts as slowly
    assert abs(np.corrcoef(s[:, 0], x)[0, 1]) < 0.5               # but on its own timing


def test_noise_reading_is_widened_by_the_comparisons_bootstrap_se():
    """G0 run 2: the surrogate spread alone was too narrow a yardstick for log-loss."""
    f = _frame()
    base = walk_forward(f, ["x"], "fwd1", 1, "terciles", min_train=300)
    r = compare(walk_forward(f, ["x", "c"], "fwd1", 1, "terciles", min_train=300), base, 1,
                "logloss", np.array([0.0, 0.002, 0.004]))
    assert np.isclose(r["z_noise"], (r["d"] - 0.002) / np.hypot(0.002, r["se_boot"]))


def test_block_permute_moves_whole_blocks_and_keeps_every_value():
    v = np.arange(100.0)
    w = block_permute(v, 20, np.random.default_rng(0))
    assert sorted(w) == list(v)
    assert all(np.all(np.diff(w[i:i + 20]) == 1) for i in range(0, 100, 20))
