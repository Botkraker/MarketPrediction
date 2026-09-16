import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from lxml import html as lhtml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.lapresse.tn/category/economie/"
OUTPUT_FILE = "headlines.csv"
MAX_WORKERS = 40
TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

DATE_IN_URL = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/")
LAST_PAGE_RE = re.compile(r"/category/economie/page/(\d+)/?")
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
        s.mount("http://", adapter)
        s.headers.update(HEADERS)
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
    for a in tree.xpath("//h3/a[@href]"):
        href = a.get("href")
        m = DATE_IN_URL.search(href)
        if not m or href in seen:
            continue
        seen.add(href)
        headline = " ".join(a.text_content().split())
        rows.append((headline, f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    return tree, rows


def parse_page(page, content=None):
    if content is None:
        content = fetch(page)
    _, rows = parse_html(content)
    return rows


def get_last_page():
    content = fetch(1)
    tree, first_rows = parse_html(content)
    last = 1
    for href in tree.xpath("//a[contains(@href,'/category/economie/page/')]/@href"):
        m = LAST_PAGE_RE.search(href)
        if m:
            last = max(last, int(m.group(1)))
    return last, first_rows


def main():
    start = time.time()
    last_page, page1_rows = get_last_page()
    with print_lock:
        print(f"Pages to scrape: {last_page}")

    results = {1: page1_rows}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(parse_page, p): p for p in range(2, last_page + 1)
        }
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
            if done % 200 == 0:
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
        for headline, date in results[page]:
            key = (headline, date)
            if key in seen:
                continue
            seen.add(key)
            rows_out.append((headline, date))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["headline", "date"])
        writer.writerows(rows_out)

    print(f"Saved {len(rows_out)} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        print(f"Pages still failing: {sorted(failed)}")


if __name__ == "__main__":
    main()