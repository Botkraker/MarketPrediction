import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import certifi
import requests
from lxml import html as lhtml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_URL = "https://www.tap.info.tn/fr/portail%20-%20economie"
OUTPUT_FILE = "tap_headlines.csv"
MAX_WORKERS = 30
TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

TAP_DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")

thread_local = threading.local()
print_lock = threading.Lock()


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update(HEADERS)
        try:
            import truststore  # noqa: F401
        except ImportError:
            s.verify = certifi.where()
        thread_local.session = s
    return thread_local.session


def page_url(pg):
    return f"{BASE_URL}?pg={pg}"


def fetch(pg):
    r = get_session().get(page_url(pg), timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def get_last_page(tree):
    last = 1
    for href in tree.xpath("//@href"):
        m = re.search(r"[?&]pg=(\d+)", str(href))
        if m:
            last = max(last, int(m.group(1)))
    for t in tree.xpath("//text()"):
        m = re.search(r"sur\s+(\d+)|of\s+(\d+)", t)
        if m:
            n = int(m.group(1) or m.group(2))
            if n > last:
                last = n
    return last


def extract_tap_date(text):
    m = TAP_DATE_RE.search(text)
    if not m:
        return ""
    day, month, year = m.group(1), m.group(2), m.group(3)
    return f"{year}-{month}-{day}"


def parse_page(pg, content=None):
    if content is None:
        content = fetch(pg)
    tree = lhtml.fromstring(content)
    headline_tds = tree.xpath("//td[@class='NewsItemHeadline']")
    rows = []
    for td in headline_tds:
        a = td.find("a")
        if a is None:
            continue
        headline = " ".join(a.text_content().split())
        if not headline:
            continue

        date = ""
        tr = td.xpath("ancestor::tr[1]")
        sib = tr[0].getnext() if tr else None
        while sib is not None:
            if sib.xpath(".//td[@class='NewsItemHeadline']"):
                break
            text_tds = sib.xpath(".//td[@class='NewsItemText']")
            if text_tds:
                date = extract_tap_date(text_tds[0].text_content())
                break
            sib = sib.getnext()

        rows.append((headline, date))
    return rows


def main():
    start = time.time()
    content = fetch(1)
    tree = lhtml.fromstring(content)
    last_page = get_last_page(tree)
    first_rows = parse_page(1, content)
    print(f"Pages detected: {last_page}")

    results = {1: first_rows}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_page, p): p for p in range(2, last_page + 1)}
        done = 0
        for future in as_completed(futures):
            p = futures[future]
            try:
                results[p] = future.result()
            except Exception as e:
                failed.append(p)
                with print_lock:
                    print(f"Page {p} failed: {e}")
            done += 1
            if done % 50 == 0:
                with print_lock:
                    print(f"{done}/{last_page - 1} done")

    for p in list(failed):
        try:
            results[p] = parse_page(p)
            failed.remove(p)
        except Exception as e:
            print(f"Page {p} failed again: {e}")

    seen = set()
    rows_out = []
    for p in sorted(results):
        for headline, date in results[p]:
            if headline in seen:
                continue
            seen.add(headline)
            rows_out.append((headline, date))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["headline", "date"])
        writer.writerows(rows_out)

    print(f"Saved {len(rows_out)} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        print(f"Still failing pages: {sorted(failed)}")


if __name__ == "__main__":
    main()