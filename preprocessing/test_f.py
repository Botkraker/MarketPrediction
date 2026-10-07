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
