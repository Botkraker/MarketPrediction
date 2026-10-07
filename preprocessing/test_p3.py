import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import bench
import p3
from p3 import bh, screen, walk


def synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-01", "2018-12-31")
    u = pd.concat([pd.DataFrame({"ticker": t, "session": days, "abs_r": rng.gamma(2, 0.005, len(days)),
                                 "abs_r_m": rng.gamma(2, 0.003, len(days))}) for t in ("AAA", "BBB", "CCC")],
                  ignore_index=True)
    u["har5"], u["har22"] = u.abs_r, u.abs_r
    u["next_session"] = u.groupby("ticker").session.shift(-1)
    u["y"] = 0.5 * u.abs_r + rng.gamma(2, 0.004, len(u))
    ev = u[rng.random(len(u)) < 0.3].dropna(subset=["next_session"]).reset_index(names="uidx")
    vocab = np.array(["hausse", "baisse", "resultat", "contrat", "perte", "accord"])
    ev["toks"] = [frozenset(rng.choice(vocab, 2, replace=False)) for _ in range(len(ev))]
    ev["neg_share"] = rng.random(len(ev))
    return u, ev


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert bh(p, 0.05).tolist() == [True, True] + [False] * 8


def test_screen_keeps_frequent_stable_words_and_finds_the_planted_one():
    n = 400
    quarter = np.repeat(np.arange(8), 50)
    label = np.tile([0.0, 1.0], n // 2)                                  # base rate exactly 1/2
    planted = (label == 1) & (np.arange(n) % 5 != 0)                      # only on 'big' events
    common = np.ones(n)                                                   # share = base rate
    rare = np.arange(n) < 20                                              # < 30 events
    two_quarters = quarter < 2                                            # >= 30 events, 2 quarters
    X = sparse.csr_matrix(np.c_[planted, common, rare, two_quarters].astype(float))
    big, small, eligible = screen(X, np.ones(n, bool), label, quarter)
    assert big.tolist() == [0] and small.tolist() == []
    assert 2 not in eligible and 3 not in eligible


def test_walk_never_uses_the_future():
    """ADR loophole register: changing every row from 2018 on must not change any forecast
    or score made before those rows' targets were observed."""
    u, ev = synthetic()
    late = pd.Timestamp("2018-01-01")
    t1, c1 = walk(u, ev, range(2016, 2019))
    bump = lambda d: d.assign(y=np.where(d.session >= late, 3 * d.y + 0.01, d.y))
    t2, c2 = walk(bump(u), bump(ev), range(2016, 2019))
    cols = ["ticker", "session", "score", "f0", "v3", "lex", "both"]
    pd.testing.assert_frame_equal(t1[cols], t2[cols])
    assert c1 == c2 and not t1.y.equals(t2.y)


def test_clark_west_is_zero_when_the_word_list_is_empty():
    """Controls run 1: an empty list must not 'beat' anything (ADR T2)."""
    t = pd.DataFrame({"session": pd.bdate_range("2017-01-02", periods=60).repeat(2),
                      "y": np.linspace(0.01, 0.02, 120), "f0": 0.015, "v3": 0.016, "lex": 0.015, "both": 0.016})
    assert p3.clark_west(t, "v3", "both")["z"] == 0.0


def test_confirm_needs_the_tag_and_the_tagged_code(monkeypatch):
    monkeypatch.setattr(bench, "tag_exists", lambda tag: False)
    with pytest.raises(SystemExit):
        p3.data(confirm=True)
    monkeypatch.setattr(bench, "tag_exists", lambda tag: True)
    monkeypatch.setattr(p3, "code_matches_tag", lambda tag=p3.TAG: False)
    with pytest.raises(SystemExit):
        p3.data(confirm=True)
