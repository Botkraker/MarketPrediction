import pandas as pd
import pytest

import bench
import p2t


def test_flags_split_news_by_publication_time():
    """Before the 14:10 close (or on a non-trade day before the trade) the session itself can
    react: 'pre'; on the trade date at or after the close: 'post'; no time: 'untimed'."""
    trades = {"AAA": pd.DatetimeIndex(["2020-01-06", "2020-01-07"])}
    news = pd.DataFrame({"ticker": "AAA",
                         "day": pd.to_datetime(["2020-01-04", "2020-01-06", "2020-01-06", "2020-01-07", "2020-01-07"]),
                         "published_at": pd.to_datetime(["2020-01-04 10:00", "2020-01-06 09:30", "2020-01-06 15:00",
                                                         None, "2020-01-07 14:10"])})
    f = p2t.flags3(news, trades).set_index("session")
    assert f.loc["2020-01-06", ["pre", "post", "untimed"]].tolist() == [1.0, 1.0, 0.0]
    assert f.loc["2020-01-07", ["pre", "post", "untimed"]].tolist() == [0.0, 1.0, 1.0]


def test_confirm_needs_the_tag_and_the_tagged_code(monkeypatch):
    monkeypatch.setattr(bench, "tag_exists", lambda tag: False)
    with pytest.raises(SystemExit):
        p2t.data(confirm=True)
    monkeypatch.setattr(bench, "tag_exists", lambda tag: True)
    monkeypatch.setattr(p2t, "code_matches_tag", lambda tag: False)
    with pytest.raises(SystemExit):
        p2t.data(confirm=True)
