import sys

import pandas as pd
import pytest

import bench
import f


def test_frozen_code_reads_the_joined_prices_and_everything_is_restored(tmp_path):
    """Only ALL_DATA.csv is redirected (with usecols honoured); the window constants and the
    reader come back afterwards."""
    frame = pd.DataFrame({"Ticker": ["X"], "Date": ["2023-01-02"], "Close": [1.0]})
    other = tmp_path / "other.csv"
    other.write_text("a\n1\n")
    before = (bench.FIRM_SEAL, bench.FIRM_END, pd.read_csv)
    with f.frozen_code_on("2023-01-01", "2026-09-15", frame=frame):
        assert pd.read_csv("x/ALL_DATA.csv", usecols=["Ticker", "Close"]).columns.tolist() == ["Ticker", "Close"]
        assert pd.read_csv(other).a.tolist() == [1]
        assert (bench.FIRM_SEAL, bench.FIRM_END) == ("2023-01-01", "2026-09-15")
    assert (bench.FIRM_SEAL, bench.FIRM_END, pd.read_csv) == before


def test_confirm_needs_the_tag(monkeypatch):
    monkeypatch.setattr(bench, "tag_exists", lambda tag: False)
    monkeypatch.setattr(sys, "argv", ["f.py", "--confirm"])
    with pytest.raises(SystemExit):
        f.main()


def test_gap_logs_are_finite_when_a_trade_sits_on_a_missing_calendar_day(monkeypatch):
    """A gap of 0 sessions (a trade on a day the calendar lacks) crashed the first --confirm
    with log(0); it is clipped at 1 session (PREREG_F §9, amendment a1)."""
    import numpy as np
    import p2
    monkeypatch.setattr(p2, "panel", lambda *a, **k: pd.DataFrame({"gap": [0.0, 1.0, np.nan],
                                                                   "back_gap": [1.0, 0.0, 2.0]}))
    with f.frozen_code_on("2023-01-01", "2026-09-15", gaps=True, frame=pd.DataFrame()):
        out = p2.panel()
    assert np.isfinite(out[f.GAPS].iloc[:2].to_numpy()).all()
    assert out.log_gap.isna().iloc[2]
