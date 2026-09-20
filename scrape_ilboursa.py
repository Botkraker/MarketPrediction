import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

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
START_DATE = date.today()
STOP_DATE = date(2014, 1, 1)
STEP_DAYS = 1
MAX_WORKERS = 20
TIMEOUT = 25
MAX_ATTEMPTS = 4

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


def new_session():
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    if not HAS_TRUSTSTORE:
        s.verify = certifi.where()
    return s


def refresh_token(s):
    r = s.get(URL, timeout=TIMEOUT)
    r.raise_for_status()
    tree = lhtml.fromstring(r.content)
    token = tree.xpath("//input[@name='__RequestVerificationToken']/@value")
    if not token:
        raise RuntimeError("Anti-forgery token not found")
    return token[0]


def get_session_and_token():
    if not hasattr(thread_local, "session"):
        thread_local.session = new_session()
        thread_local.token = refresh_token(thread_local.session)
    return thread_local.session, thread_local.token


def reset_session():
    thread_local.session = new_session()
    thread_local.token = refresh_token(thread_local.session)


def parse_rows(content):
    tree = lhtml.fromstring(content)
    rows = []
    for tr in tree.xpath("//table[@id='tabQuotes']//tr"):
        span = tr.xpath(".//span[@class='sp1']/text()")
        link = tr.xpath(".//a[@href]")
        if not span or not link:
            continue
        raw_dt = span[0].strip()
        try:
            dt = datetime.strptime(raw_dt, "%d/%m/%Y %H:%M")
        except ValueError:
            continue
        headline = " ".join(link[0].text_content().split())
        href = link[0].get("href")
        rows.append((href, headline, dt))
    return rows


def fetch_date(d):
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            s, token = get_session_and_token()
            data = {
                "dateActu": d.strftime("%Y-%m-%d"),
                "__Invariant": "dateActu",
                "__RequestVerificationToken": token,
            }
            r = s.post(URL, data=data, timeout=TIMEOUT)
            if r.status_code in (400, 403):
                reset_session()
                continue
            r.raise_for_status()
            return parse_rows(r.content)
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
            try:
                reset_session()
            except Exception:
                pass
    raise RuntimeError(f"{d}: {last_err}")


def build_dates():
    dates = []
    d = START_DATE
    while d >= STOP_DATE:
        dates.append(d)
        d -= timedelta(days=STEP_DAYS)
    return dates


def main():
    start = time.time()
    dates = build_dates()
    print(f"Dates to query: {len(dates)}")

    articles = {}
    failed = []
    lock = threading.Lock()

    def collect(rows):
        with lock:
            for href, headline, dt in rows:
                if dt.date() < STOP_DATE:
                    continue
                # Keep href in the VALUE too: articles.values() below would
                # otherwise drop the URL that is sitting right here as the key.
                articles[href] = (headline, dt, href)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_date, d): d for d in dates}
        done = 0
        for future in as_completed(futures):
            d = futures[future]
            try:
                collect(future.result())
            except Exception as e:
                failed.append(d)
                with print_lock:
                    print(f"Failed {e}")
            done += 1
            if done % 100 == 0:
                with print_lock:
                    print(f"{done}/{len(dates)} dates done, {len(articles)} unique headlines")

    for d in list(failed):
        try:
            thread_local.__dict__.clear()
            collect(fetch_date(d))
            failed.remove(d)
        except Exception as e:
            print(f"Failed again {e}")

    rows_out = sorted(articles.values(), key=lambda x: x[1], reverse=True)

    # parse_rows() already parses a full "%d/%m/%Y %H:%M" timestamp and captures
    # the article href. Earlier revisions computed both and then wrote only the
    # date, discarding real time-of-day on ~26.6k rows (AUDIT_REPORT.md section 3.1).
    # `headline` and `date` are kept first and unchanged so existing loaders
    # (preprocessing/io_raw.py, audit/build_audit.py) are unaffected; the new
    # columns are additive.
    #
    # Why time-of-day matters: headlines currently carry no clock time, so
    # features.py must assume a headline dated D may only predict sessions
    # STRICTLY after D -- a published-at-18:00 headline would otherwise "predict"
    # a close that already happened. With real timestamps, same-session alignment
    # becomes possible for this source (BVMT closes ~14:10 Tunis time).
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["headline", "date", "published_at", "url"])
        for headline, dt, href in rows_out:
            writer.writerow([headline, dt.strftime("%Y-%m-%d"),
                             dt.strftime("%Y-%m-%d %H:%M"), href])

    print(f"Saved {len(rows_out)} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        print(f"Dates still failing: {sorted(str(d) for d in failed)}")


if __name__ == "__main__":
    main()