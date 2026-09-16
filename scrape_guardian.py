import csv
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

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

BASE_URL = "https://www.theguardian.com/world/tunisia"
OUTPUT_FILE = "guardian_tunisia_headlines.csv"
MAX_WORKERS = 10
TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}
URL_DATE_RE = re.compile(r"/(\d{4})/([a-z]{3})/(\d{2})/")
RESULTS_RE = re.compile(r"About\s+([\d,]+)\s+results")

thread_local = threading.local()
print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        s.mount("https://", adapter)
        s.headers.update(HEADERS)
        if not HAS_TRUSTSTORE:
            s.verify = certifi.where()
        thread_local.session = s
    return thread_local.session


def page_url(page):
    return BASE_URL if page == 1 else f"{BASE_URL}?page={page}"


def fetch(page):
    r = get_session().get(page_url(page), timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def date_from_url(href):
    m = URL_DATE_RE.search(href)
    if m and m.group(2) in MONTHS:
        return f"{m.group(1)}-{MONTHS[m.group(2)]}-{m.group(3)}"
    return ""


def find_card_date(a):
    node = a
    for _ in range(4):
        node = node.getparent()
        if node is None:
            break
        dt = node.xpath(".//time/@datetime")
        if dt:
            return dt[0][:10]
    return date_from_url(a.get("href", ""))


def parse_html(content):
    tree = lhtml.fromstring(content)
    rows = []
    seen = set()
    for a in tree.xpath("//main//a[@aria-label and @href]"):
        href = urljoin("https://www.theguardian.com", a.get("href"))
        if not URL_DATE_RE.search(href) or href in seen:
            continue
        headline = " ".join(a.get("aria-label").split())
        if not headline:
            continue
        seen.add(href)
        rows.append((href, headline, find_card_date(a)))
    return tree, rows


def parse_page(page):
    content = fetch(page)
    if content is None:
        return []
    _, rows = parse_html(content)
    return rows


def get_last_page(tree, per_page):
    last = 1
    for href in tree.xpath("//a[contains(@href,'?page=')]/@href"):
        m = re.search(r"[?&]page=(\d+)", href)
        if m:
            last = max(last, int(m.group(1)))
    text = " ".join(tree.xpath("//main//span/text()"))
    m = RESULTS_RE.search(text)
    if m and per_page:
        total = int(m.group(1).replace(",", ""))
        last = max(last, math.ceil(total / per_page))
    return last


def main():
    start = time.time()
    content = fetch(1)
    tree, first_rows = parse_html(content)
    last_page = get_last_page(tree, len(first_rows) or 20)
    log(f"Page 1: {len(first_rows)} headlines, estimated pages: {last_page}")

    results = {1: first_rows}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_page, p): p for p in range(2, last_page + 1)}
        done = 0
        for future in as_completed(futures):
            page = futures[future]
            try:
                results[page] = future.result()
            except Exception as e:
                failed.append(page)
                log(f"Page {page} failed: {e}")
            done += 1
            if done % 10 == 0:
                log(f"{done}/{last_page - 1} pages done")

    for page in list(failed):
        try:
            results[page] = parse_page(page)
            failed.remove(page)
        except Exception as e:
            log(f"Page {page} failed again: {e}")

    seen = set()
    rows_out = []
    for page in sorted(results):
        for href, headline, date in results[page]:
            if href in seen:
                continue
            seen.add(href)
            rows_out.append((headline, date))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["headline", "date"])
        writer.writerows(rows_out)

    log(f"Saved {len(rows_out)} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        log(f"Pages still failing: {sorted(failed)}")


if __name__ == "__main__":
    main()
