"""The sensitivity grid must be able to tell stable from fragile."""
import json

import numpy as np
import pandas as pd
import pytest

from sensitivity import MIN_TRAIN_GRID, REFIT_GRID, run


def _frame(n=1100, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.0045, n)
    if signal:                                  # plant real autocorrelation
        for i in range(1, n):
            r[i] += signal * r[i - 1]
    f = pd.DataFrame({"session": pd.bdate_range("2014-01-01", periods=n),
                      "ret_next": np.roll(r, -1)})
    f["ret_lag0"] = r
    for lag in (1, 2):
        f[f"ret_lag{lag}"] = np.roll(r, lag)
    return f.iloc[3:-1].reset_index(drop=True)


@pytest.fixture
def small_grid(monkeypatch):
    """The real grid is 16 walk-forwards over 3k sessions -- ~80s, too slow to sit
    in the suite. Behaviour is identical at 2x2, so only the coverage test below
    uses the real constants."""
    import sensitivity
    monkeypatch.setattr(sensitivity, "MIN_TRAIN_GRID", (300, 600))
    monkeypatch.setattr(sensitivity, "REFIT_GRID", (5, 60))


def _written(tmp_path, **kwargs):
    """run() reads a parquet path, so exercise that path rather than bypassing it."""
    path = tmp_path / "features.parquet"
    _frame(**kwargs).to_parquet(path, index=False)
    return path


def test_grid_covers_every_configuration(tmp_path, small_grid):
    r = run(_written(tmp_path), tmp_path / "s.json")
    import sensitivity
    assert len(r["grid"]) == len(sensitivity.MIN_TRAIN_GRID) * len(sensitivity.REFIT_GRID)
    assert {row["min_train"] for row in r["grid"]} == set(sensitivity.MIN_TRAIN_GRID)


def test_real_grid_constants_are_sane():
    """The shipped grid must bracket the chosen setting, not merely include it."""
    assert 500 in MIN_TRAIN_GRID and 20 in REFIT_GRID
    assert min(MIN_TRAIN_GRID) < 500 < max(MIN_TRAIN_GRID)
    assert min(REFIT_GRID) < 20 < max(REFIT_GRID)


def test_a_strong_planted_signal_is_stable_across_the_grid(tmp_path, small_grid):
    r = run(_written(tmp_path, signal=0.45, seed=1), tmp_path / "s.json")
    assert r["verdict"]["beats_constant_everywhere"]
    assert r["verdict"]["stable"]


def test_pure_noise_is_not_reported_as_stable(tmp_path, small_grid):
    r = run(_written(tmp_path, signal=0.0, seed=2), tmp_path / "s.json")
    assert not r["verdict"]["stable"]


def test_smaller_min_train_yields_more_predictions(tmp_path, small_grid):
    r = run(_written(tmp_path), tmp_path / "s.json")
    by_min = {row["min_train"]: row["n"] for row in r["grid"]}
    assert by_min[min(by_min)] > by_min[max(by_min)]


def test_results_are_written_and_flagged_exploratory(tmp_path, small_grid):
    out = tmp_path / "s.json"
    run(_written(tmp_path), out)
    saved = json.loads(out.read_text())
    assert "not Holm-corrected" in saved["note"] or "NOT Holm-corrected" in saved["note"]
    assert "outcome-selection" in saved["note"]
