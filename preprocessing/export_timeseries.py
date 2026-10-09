"""Export the project's data as plain time series (CSV) for reading outside the code.

    python3 preprocessing/export_timeseries.py   -> data/curated/export/

daily.csv         one row per Tunindex session (phantom sessions dropped): close, return,
                  and the headlines that session can react to (dated before it), with
                  their counts and mean sentiment.
headlines.csv     one row per canonical headline: date, ilboursa publication time where
                  known, outlet, text, v3 sentiment (label, class probabilities, tone =
                  p_positive - p_negative), price-report flag, listed companies named,
                  and the session it maps to (the first session after its date) with
                  that session's Tunindex close and return.
stock_prices.csv  daily closes per listed company: ALL_DATA to 2022-12-30, ilboursa (D2)
                  from 2023, joined as in the ADR-002 replication (f.prices).

Unscored headlines (no earlier labels to train on, mostly before 2016) keep a blank score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import f
import h3
import p2t

OUT = h3.CURATED / "export"


def publication_times(s: pd.DataFrame) -> pd.Series:
    """ilboursa publication time per canonical row (as p2t.timed_news joins them)."""
    raw = pd.read_csv(h3.ROOT / "data" / "raw" / "ilboursa_headlines.csv", sep=";", encoding="utf-8-sig",
                      dtype=str, keep_default_na=False)
    d1 = pd.read_csv(p2t.D1, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d1 = d1.assign(key=d1.headline.str.strip() + "|" + d1.date).drop_duplicates("key").set_index("key")
    il = (s.source == "ilboursa").to_numpy()
    n = s.row_id[il].str.split("::").str[1].astype(int).to_numpy()
    key = pd.Series(index=s.index, dtype=object)
    key[il] = raw.headline.str.strip().to_numpy()[n] + "|" + raw.date.to_numpy()[n]
    return key.map(d1.published_at)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    idx = h3.index_frame()[["session", "close", "ret"]].rename(columns={"close": "tunindex_close", "ret": "tunindex_return"})

    s = pd.read_parquet(h3.SCORED_V3)
    low = s.headline_clean.astype(str).str.lower()
    names = pd.Series("", index=s.index)
    for ticker, pat in h3.issuer_patterns().items():
        hit = low.str.contains(pat)
        names[hit] = names[hit] + ";" + ticker
    day = pd.to_datetime(s.published_date)
    news = pd.DataFrame({
        "date": day.dt.date, "published_at": publication_times(s), "source": s.source, "lang": s.lang,
        "headline": s.headline_clean, "sentiment": s.sent_label.replace("", np.nan),
        "tone": s.p_positive - s.p_negative, "p_negative": s.p_negative, "p_neutral": s.p_neutral,
        "p_positive": s.p_positive,
        "price_report": low.str.contains(h3.PRICE_REPORT_PATTERN, regex=True),
        "companies": names.str.lstrip(";").replace("", np.nan),
        "session": h3.map_to_next_session(day.dt.normalize(), h3.calendar()),
    })
    news.loc[(day < h3.calendar()[0]).to_numpy(), "session"] = pd.NaT   # before the index data: no session
    news = news.merge(idx, on="session", how="left").sort_values(["date", "source"]).reset_index(drop=True)
    news.to_csv(OUT / "headlines.csv", index=False)

    g = news.groupby("session")
    daily = idx.merge(pd.DataFrame({
        "n_headlines": g.size(), "n_scored": g.sentiment.count(), "mean_tone": g.tone.mean(),
        "n_positive": g.sentiment.apply(lambda x: (x == "positive").sum()),
        "n_neutral": g.sentiment.apply(lambda x: (x == "neutral").sum()),
        "n_negative": g.sentiment.apply(lambda x: (x == "negative").sum()),
        "n_price_reports": g.price_report.sum(), "n_naming_a_company": g.companies.count(),
    }).reset_index(), on="session", how="left")
    counts = [c for c in daily.columns if c.startswith("n_")]
    daily[counts] = daily[counts].fillna(0).astype(int)
    daily.rename(columns={"session": "date"}).to_csv(OUT / "daily.csv", index=False)

    px = f.prices()[["Ticker", "Date", "Close", "Volume"]].drop_duplicates(["Ticker", "Date"], keep="last")
    px.sort_values(["Date", "Ticker"]).to_csv(OUT / "stock_prices.csv", index=False)
    print({p.name: sum(1 for _ in open(p)) - 1 for p in sorted(OUT.glob("*.csv"))}, "rows ->", OUT)


if __name__ == "__main__":
    main()
