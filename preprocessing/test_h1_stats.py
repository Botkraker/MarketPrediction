import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import h1_stats as hs


def _preds(n=600, edge=0.0, seed=0, start="2019-01-01"):
    """Paired prediction frames. The arm's score carries `edge` of the truth."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    session = pd.bdate_range(start, periods=n)
    base = pd.DataFrame({"session": session, "y_true": y, "score": rng.normal(0, 1, n)})
    arm = base.assign(score=base.score + edge * (2 * y - 1) + rng.normal(0, 0.5, n))
    return arm, base


def test_rank_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, (5, 80))
    s = rng.integers(0, 6, (5, 80)).astype(float)          # many ties
    expected = [roc_auc_score(y[i], s[i]) for i in range(5)]
    assert np.allclose(hs.auc_rows(y, s), expected)
    assert np.isnan(hs.auc_rows(np.ones((1, 10), int), s[:1, :10]))[0]


def test_block_indices_are_contiguous_and_wrap():
    idx = hs.block_indices(50, 20, 7, np.random.default_rng(0))
    assert idx.shape == (7, 50)
    steps = np.diff(idx[:, :20], axis=1) % 50
    assert (steps == 1).all()                               # one unbroken block


def test_real_edge_gives_ci_above_zero():
    arm, base = _preds(edge=0.8)
    r = hs.delta_auc(arm, base, n_boot=500)
    assert r["delta_auc"] > 0.1 and r["ci95"][0] > 0 and r["p_bootstrap"] < 0.01


def test_identical_arms_are_a_null():
    _, base = _preds()
    r = hs.delta_auc(base, base, n_boot=300)
    assert r["delta_auc"] == 0 and r["p_bootstrap"] == 1.0


def test_misaligned_arms_are_refused():
    arm, base = _preds()
    with pytest.raises(ValueError, match="aligned"):
        hs.delta_auc(arm.iloc[1:], base.iloc[:-1])


def test_per_block_uses_quarters_and_skips_thin_ones():
    arm, base = _preds(n=300, edge=0.8)
    r = hs.per_block(arm, base)
    assert r["n_blocks"] >= 3 and all("Q" in b["block"] for b in r["blocks"])
    assert all(b["n"] >= hs.MIN_BLOCK_N for b in r["blocks"])
    assert r["share_positive"] == 1.0


def test_covid_split_partitions_2020():
    arm, base = _preds(n=800, edge=0.5, start="2019-06-03")
    r = hs.covid_split(arm, base, n_boot=200)
    in_2020 = pd.to_datetime(arm.session).dt.year.eq(2020).sum()
    assert r["only_2020"]["n"] == in_2020
    assert r["only_2020"]["n"] + r["excluding_2020"]["n"] == len(arm)


def test_purged_kfold_never_trains_on_the_test_fold_or_its_embargo(monkeypatch):
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(X[:, 0].copy())
            return self

        def predict(self, X):
            return X[:, 0]

    monkeypatch.setattr(hs, "make_model", lambda kind: Spy())
    frame = pd.DataFrame({"i": np.arange(100.0),
                          "ret_next": np.random.default_rng(0).normal(0, 1, 100)})
    r = hs.purged_kfold(frame, ["i"], k=5, embargo=5)
    assert len(r["fold_auc"]) == 5
    for fold, trained in zip(np.array_split(np.arange(100), 5), seen):
        banned = set(range(max(0, fold[0] - 5), min(100, fold[-1] + 6)))
        assert banned.isdisjoint(trained.astype(int))


@pytest.mark.parametrize("passes,expected", [
    ({"ex_price": True, "orthogonal": True}, "GO"),
    ({"ex_price": True}, "PARTIAL"),
    ({"all": True}, "LAUNDERING"),
    ({"placebo": True}, "PLACEBO"),
    ({}, "NOT SUPPORTED"),
])
def test_verdict_ladder(passes, expected):
    assert expected in hs.verdict(passes)


def test_a_pass_needs_both_significance_and_block_consistency(monkeypatch):
    arm, base = _preds(n=700, edge=0.8)
    frame = pd.DataFrame({"x": np.zeros(10), "ret_next": np.zeros(10)})
    monkeypatch.setattr(hs, "purged_kfold", lambda *a, **k: {"mean_auc": 0.5, "k": 5})
    ok = hs.analyse({"ex_price": arm, "orthogonal": arm}, base, frame,
                    {"ex_price": ["x"], "orthogonal": ["x"]}, ["x"])
    assert ok["arms"]["ex_price"]["passes"] and ok["verdict"].startswith("GO")

    monkeypatch.setattr(hs, "BLOCK_SHARE_REQUIRED", 1.01)   # impossible consistency bar
    blocked = hs.analyse({"ex_price": arm}, base, frame, {"ex_price": ["x"]}, ["x"])
    assert not blocked["arms"]["ex_price"]["passes"]
