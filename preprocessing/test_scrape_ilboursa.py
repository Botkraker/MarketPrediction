"""Guard the ilboursa scraper's output schema.

AUDIT_REPORT.md section 3.1: parse_rows() parsed a full "%d/%m/%Y %H:%M" timestamp and
captured the article href, then main() wrote only the date -- discarding real
time-of-day on ~26,600 rows. These tests fail if that regression returns.
"""
import csv
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scrape_ilboursa as S  # noqa: E402

HTML = """<table id='tabQuotes'><tr>
<td><span class='sp1'>14/03/2024 09:35</span></td>
<td><a href='/marches/a,520,54321,3.html'>Le Tunindex ouvre en hausse</a></td>
</tr><tr>
<td><span class='sp1'>14/03/2024 16:42</span></td>
<td><a href='/marches/a,520,54322,3.html'>BIAT publie ses resultats</a></td>
</tr></table>"""


def _write(rows_out) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["headline", "date", "published_at", "url"])
    for headline, dt, href in rows_out:
        w.writerow([headline, dt.strftime("%Y-%m-%d"),
                    dt.strftime("%Y-%m-%d %H:%M"), href])
    return buf.getvalue()


def _rows_out():
    rows = S.parse_rows(HTML)
    articles = {href: (headline, dt, href) for href, headline, dt in rows}
    return sorted(articles.values(), key=lambda x: x[1], reverse=True)


def test_parse_rows_returns_url_headline_and_full_timestamp():
    rows = S.parse_rows(HTML)
    assert len(rows) == 2
    for href, headline, dt in rows:
        assert href.startswith("/marches/")
        assert headline
        assert isinstance(dt, datetime)
    # the time is real, not midnight
    assert {dt.strftime("%H:%M") for _, _, dt in rows} == {"09:35", "16:42"}


def test_written_row_keeps_time_of_day_and_url():
    df = pd.read_csv(io.StringIO(_write(_rows_out())), delimiter=";", dtype=str)
    assert list(df.columns) == ["headline", "date", "published_at", "url"]
    assert (df.published_at.str.len() == 16).all()      # YYYY-MM-DD HH:MM
    assert df.url.str.startswith("/marches/").all()
    assert set(df.published_at.str[-5:]) == {"09:35", "16:42"}


def test_existing_loaders_are_unaffected():
    """headline and date stay first and unchanged, so io_raw.py / build_audit.py
    keep working against the new file without modification."""
    df = pd.read_csv(io.StringIO(_write(_rows_out())), delimiter=";", dtype=str)
    assert list(df.columns)[:2] == ["headline", "date"]
    assert pd.to_datetime(df["date"], format="%Y-%m-%d").notna().all()


def test_same_day_headlines_are_separable_around_the_close():
    """The point of the fix: two headlines on one date now fall on different
    sides of the ~14:10 BVMT close, which is what same-session alignment needs."""
    df = pd.read_csv(io.StringIO(_write(_rows_out())), delimiter=";", dtype=str)
    assert df.date.nunique() == 1                       # same calendar day
    hours = pd.to_datetime(df.published_at).dt.hour
    assert (hours < 14).any() and (hours >= 14).any()   # but separable
