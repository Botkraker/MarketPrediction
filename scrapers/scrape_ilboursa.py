"""Scrape ilboursa market news with full timestamps (ADR-001, track D1).

Resumable: one JSON file per requested date in --cache-dir. Ctrl-C is safe; rerun to continue.

    python3 scrapers/scrape_ilboursa.py                      # 2014-01-01 -> today
    python3 scrapers/scrape_ilboursa.py --start 2016-12-31 --stop 2016-01-01 --workers 4

Output (';', utf-8-sig): headline;date;published_at;url;article_id;has_quote
  headline, date      unchanged, so io_raw.py / build_audit.py keep working
  published_at        YYYY-MM-DD HH:MM, Tunis time, as shown in the list
  url                 site path (domain stripped)
  article_id          site's sequential id; out-of-order ids flag back-dated timestamps
  has_quote           1 if the row shows a listed security's quote cell. The quote VALUES are
                      live at scrape time and are deliberately NOT stored (look-ahead).
Side file <output>.saturated.txt: dates whose whole result window fell on that same day,
so earlier articles of that day may be missing.
"""
import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import certifi
import requests
from lxml import html as lhtml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore
    truststore.inject_into_ssl()
    HAS_TRUSTSTORE = True
except ImportError:
    HAS_TRUSTSTORE = False

URL = "https://www.ilboursa.com/marches/actualites_bourse_tunis"
OUTPUT_FILE = "ilboursa_headlines.csv"
CACHE_DIR = "ilboursa_cache"
STOP_DATE = date(2014, 1, 1)
MAX_WORKERS = 6
TIMEOUT = 25
MAX_ATTEMPTS = 5
DELAY = (0.3, 1.0)

QUOTE_RE = re.compile(r"[+-]?\d+,\d+\s*%")
ID_RES = (re.compile(r"_(\d+)/?$"), re.compile(r"a,\d+,(\d+),"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Origin": "https://www.ilboursa.com",
    "Referer": URL,
}

thread_local = threading.local()
print_lock = threading.Lock()


class BadPage(Exception):
    pass


def log(msg):
    with print_lock:
        print(msg, flush=True)


# ------------------------------------------------------------------ session
def new_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    if not HAS_TRUSTSTORE:
        s.verify = certifi.where()
    return s


def refresh_token(s):
    r = s.get(URL, timeout=TIMEOUT)
    r.raise_for_status()
    token = lhtml.fromstring(r.content).xpath("//input[@name='__RequestVerificationToken']/@value")
    if not token:
        raise RuntimeError("Anti-forgery token not found")
    return token[0]


def reset_session():
    thread_local.session = new_session()
    thread_local.token = refresh_token(thread_local.session)


def get_session_and_token():
    if not hasattr(thread_local, "session"):
        reset_session()
    return thread_local.session, thread_local.token


# ------------------------------------------------------------------ parsing
def normalise_href(href):
    href = (href or "").strip()
    if href.startswith("http"):
        p = urlparse(href)
        return p.path + (f"?{p.query}" if p.query else "")
    return href


def article_id(href):
    for rx in ID_RES:
        m = rx.search(href)
        if m:
            return m.group(1)
    return ""


def parse_rows_full(content):
    tree = lhtml.fromstring(content)
    out = []
    for tr in tree.xpath("//table[@id='tabQuotes']//tr"):
        span = tr.xpath(".//span[@class='sp1']/text()")
        link = tr.xpath(".//a[@href]")
        if not span or not link:
            continue
        try:
            dt = datetime.strptime(span[0].strip(), "%d/%m/%Y %H:%M")
        except ValueError:
            continue
        a = link[0]
        rest = " ".join(tr.xpath(".//text()[not(ancestor::a)]"))
        out.append({"href": normalise_href(a.get("href")),
                    "headline": " ".join(a.text_content().split()),
                    "dt": dt,
                    "has_quote": int(bool(QUOTE_RE.search(rest)))})
    return out


def parse_rows(content):
    """(href, headline, datetime) triples; kept for preprocessing/test_scrape_ilboursa.py."""
    return [(r["href"], r["headline"], r["dt"]) for r in parse_rows_full(content)]


# ------------------------------------------------------------------ fetching
def fetch_date(d):
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            time.sleep(random.uniform(*DELAY))
            s, token = get_session_and_token()
            data = {"dateActu": d.strftime("%Y-%m-%d"), "__Invariant": "dateActu",
                    "__RequestVerificationToken": token}
            r = s.post(URL, data=data, timeout=TIMEOUT)
            if r.status_code in (400, 403):
                raise BadPage(f"HTTP {r.status_code}")
            r.raise_for_status()
            rows = parse_rows_full(r.content)
            if not rows and b"tabQuotes" not in r.content:
                raise BadPage("news table missing")
            if (rows and d < date.today() - timedelta(days=30)
                    and max(x["dt"].date() for x in rows) > d + timedelta(days=7)):
                raise BadPage("server ignored the requested date")
            return rows
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            try:
                reset_session()
            except Exception:
                pass
    raise RuntimeError(f"{d}: {last_err}")


# ------------------------------------------------------------------ cache
def cache_path(cache, d):
    return cache / f"{d.isoformat()}.json"


def save(cache, d, rows):
    payload = {
        "requested": d.isoformat(),
        "saturated": bool(rows) and min(r["dt"].date() for r in rows) >= d,
        "rows": [[r["href"], r["headline"], r["dt"].strftime("%Y-%m-%d %H:%M"), r["has_quote"]]
                 for r in rows],
    }
    tmp = cache_path(cache, d).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, cache_path(cache, d))


def build_dates(start, stop):
    dates, d = [], start
    while d >= stop:
        dates.append(d)
        d -= timedelta(days=1)
    return dates


def write_csv(cache, dates, output, stop):
    articles, conflicts, saturated = {}, 0, []
    for d in dates:
        p = cache_path(cache, d)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload["saturated"]:
            saturated.append(payload["requested"])
        for href, headline, ts, has_quote in payload["rows"]:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            if dt.date() < stop:
                continue
            if href in articles and articles[href][1] != dt:
                conflicts += 1
                continue
            articles.setdefault(href, (headline, dt, href, has_quote))

    rows_out = sorted(articles.values(), key=lambda x: x[1], reverse=True)
    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["headline", "date", "published_at", "url", "article_id", "has_quote"])
        for headline, dt, href, has_quote in rows_out:
            w.writerow([headline, dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d %H:%M"),
                        href, article_id(href), has_quote])
    Path(f"{output}.saturated.txt").write_text("\n".join(sorted(saturated)), encoding="utf-8")
    return len(rows_out), conflicts, len(saturated)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--start", type=date.fromisoformat, default=date.today())
    ap.add_argument("--stop", type=date.fromisoformat, default=STOP_DATE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--refresh-days", type=int, default=3,
                    help="re-fetch the most recent N days even if cached (default 3)")
    ap.add_argument("--csv-only", action="store_true", help="rebuild the CSV from the cache, no requests")
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dates = build_dates(args.start, args.stop)
    recent = date.today() - timedelta(days=args.refresh_days)
    todo = [] if args.csv_only else [d for d in dates if d >= recent or not cache_path(cache, d).exists()]
    log(f"Dates in range: {len(dates)} | to fetch: {len(todo)} | cache: {cache.resolve()}")

    failed = []
    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {ex.submit(fetch_date, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                save(cache, d, fut.result())
            except Exception as e:
                failed.append(d)
                log(f"Failed {e}")
            if i % 100 == 0 or i == len(todo):
                log(f"{i}/{len(todo)} dates done, {len(failed)} failed, {time.time() - t0:.0f}s")
    except KeyboardInterrupt:
        log("Interrupted. Finished dates are cached; rerun the same command to resume.")
        ex.shutdown(wait=False, cancel_futures=True)
        return
    ex.shutdown()

    for d in sorted(failed):
        try:
            save(cache, d, fetch_date(d))
            failed.remove(d)
        except Exception as e:
            log(f"Failed again {e}")

    n, conflicts, n_sat = write_csv(cache, dates, args.output, args.stop)
    log(f"Saved {n} headlines to {args.output} in {time.time() - t0:.0f}s")
    log(f"Timestamp conflicts (same url, different time): {conflicts}")
    log(f"Saturated dates (day may be incomplete): {n_sat} -> {args.output}.saturated.txt")
    if failed:
        log(f"Dates still failing (rerun to retry): {[str(d) for d in sorted(failed)]}")


if __name__ == "__main__":
    main()