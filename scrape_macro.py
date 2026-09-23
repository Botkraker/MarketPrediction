"""Fetch the F1 candidate series tested by preprocessing/macro_controls.py.

  eurtnd   BCT interbank average EUR/TND, one POST per weekday to the archive
           behind bct.gov.tn/bct/siteprod/cours.jsp. Paced at ~1 request/second.
           Resumable: dates already in the output are skipped. A date whose
           page reports a different day (holiday, no fixing) is recorded as
           missing, not filled.
  stoxx50  Euro Stoxx 50 daily closes from Yahoo Finance's chart API (^STOXX50E).

Output: data/raw/macro/eurtnd_bct_daily.csv, data/raw/macro/eurostoxx50_yahoo_daily.csv
"""

import argparse
import csv
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

MACRO = Path(__file__).resolve().parent / "data" / "raw" / "macro"
BCT_URL = "https://www.bct.gov.tn/bct/siteprod/cours_archiv.jsp"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5ESTOXX50E"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; Tunindex sentiment study)"}
# First table on the page is the interbank average; its EUR row comes first.
EUR_ROW = re.compile(r"EURO\s+EUR\s+1\s+([\d]+,[\d]+)")
DAY = re.compile(r"Journ[ée]e du\s+(\d{2}/\d{2}/\d{4})")


def parse_bct(html: str, wanted: date) -> float | None:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    day, rate = DAY.search(text), EUR_ROW.search(text)
    if not day or not rate:
        return None
    if datetime.strptime(day.group(1), "%d/%m/%Y").date() != wanted:
        return None                      # the archive answered for another day
    return float(rate.group(1).replace(",", "."))


def fetch_eurtnd(start: str, end: str, pause: float = 1.0) -> Path:
    out = MACRO / "eurtnd_bct_daily.csv"
    done = set(pd.read_csv(out)["date"]) if out.exists() else set()
    session = requests.Session()
    session.headers.update(HEADERS)
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["date", "eur_tnd"])
        for day in pd.bdate_range(start, end):
            key = day.date().isoformat()
            if key in done:
                continue
            for attempt in range(3):
                try:
                    r = session.post(BCT_URL, data={"input": key, "langue": ""}, timeout=30)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"{key}: failed 3 times, left for the next run")
                continue
            rate = parse_bct(r.text, day.date())
            writer.writerow([key, "" if rate is None else rate])
            handle.flush()
            time.sleep(pause)
    return out


def fetch_stoxx50(start: str, end: str) -> Path:
    to_epoch = lambda d: int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())
    r = requests.get(YAHOO_URL, headers=HEADERS, timeout=60,
                     params={"period1": to_epoch(start), "period2": to_epoch(end),
                             "interval": "1d"})
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    frame = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
                  .tz_convert(result["meta"]["exchangeTimezoneName"]).date,
        "stoxx50_close": result["indicators"]["quote"][0]["close"]})
    frame = frame.dropna().drop_duplicates("date", keep="last")
    out = MACRO / "eurostoxx50_yahoo_daily.csv"
    frame.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", choices=("eurtnd", "stoxx50"))
    parser.add_argument("--start", default="2013-12-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()
    MACRO.mkdir(parents=True, exist_ok=True)
    path = (fetch_eurtnd if args.series == "eurtnd" else fetch_stoxx50)(args.start, args.end)
    frame = pd.read_csv(path)
    print(f"{path.name}: {len(frame)} rows, {frame.iloc[:, 1].notna().sum()} with a value, "
          f"{frame.date.min()} .. {frame.date.max()}")


if __name__ == "__main__":
    main()
