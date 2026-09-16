import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

BASE_URL = "https://kapitalis.com/tunisie/category/economie/"
OUTPUT_FILE = "kapitalis_headlines.csv"
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

DATE_IN_URL = re.compile(r"/tunisie/(\d{4})/(\d{2})/(\d{2})/")
PAGE_RE = re.compile(r"/category/economie/page/(\d+)/?")
thread_local = threading.local()
print_lock = threading.Lock()


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        s.mount("https://", adapter)
        s.headers.update(HEADERS)
        if not HAS_TRUSTSTORE:
            s.verify = certifi.where()
        thread_local.session = s
    return thread_local.session


def page_url(page):
    return BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"


def fetch(page):
    r = get_session().get(page_url(page), timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def parse_html(content):
    tree = lhtml.fromstring(content)
    rows = []
    seen = set()
    for a in tree.xpath("//h2/a[@href]"):
        href = a.get("href")
        m = DATE_IN_URL.search(href)
        if not m or href in seen:
            continue
        seen.add(href)
        headline = " ".join(a.text_content().split())
        if headline:
            rows.append((href, headline, f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    return tree, rows


def parse_page(page):
    _, rows = parse_html(fetch(page))
    return rows


def get_last_page():
    tree, rows = parse_html(fetch(1))
    last = 1
    for href in tree.xpath("//a[contains(@href,'/category/economie/page/')]/@href"):
        m = PAGE_RE.search(href)
        if m:
            last = max(last, int(m.group(1)))
    return last, rows


def main():
    start = time.time()
    last_page, page1_rows = get_last_page()
    print(f"Pages to scrape: {last_page}")

    results = {1: page1_rows}
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
                with print_lock:
                    print(f"Page {page} failed: {e}")
            done += 1
            if done % 100 == 0:
                with print_lock:
                    print(f"{done}/{last_page - 1} pages done")

    for page in list(failed):
        try:
            results[page] = parse_page(page)
            failed.remove(page)
        except Exception as e:
            print(f"Page {page} failed again: {e}")

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

    print(f"Saved {len(rows_out)} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        print(f"Pages still failing: {sorted(failed)}")


if __name__ == "__main__":
    main()
