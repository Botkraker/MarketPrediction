import numpy as np
import pandas as pd

from p1 import novelty


def test_novelty_learns_from_earlier_sessions_only():
    """Rule 2 for the language model: a session is scored by the sessions before it,
    never by itself or by anything later."""
    cal = pd.bdate_range("2016-01-04", periods=3)
    sessions = pd.Series(cal[[0, 1, 1, 2]])
    tokens = pd.Series([["la", "bourse", "monte"], ["la", "bourse", "monte"],
                        ["la", "bourse", "monte"], ["un", "mot", "neuf"]])
    n = novelty(tokens, sessions, cal)
    assert np.isnan(n[0])                                 # nothing before the first session
    assert n[1] == n[2]                                   # its own session never trains it
    early = novelty(tokens.iloc[:3], sessions.iloc[:3], cal)
    assert np.allclose(n.iloc[1:3], early.iloc[1:3])      # later sessions change nothing
    assert n[3] > n[1]                                    # unseen words surprise more


def test_novelty_window_forgets_old_sessions():
    cal = pd.bdate_range("2016-01-04", periods=3)
    sessions = pd.Series(cal[[0, 1, 2]])
    tokens = pd.Series([["mot", "rare"], ["autre", "chose"], ["mot", "rare"]])
    short = novelty(tokens, sessions, cal, window=1)
    long = novelty(tokens, sessions, cal, window=2)
    assert short[2] > long[2]                             # beyond the window it is new again
