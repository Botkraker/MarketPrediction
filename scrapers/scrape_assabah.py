import csv
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

BASE_URL = "https://assabah.ma/category/%D8%AD%D9%88%D8%A7%D8%AF%D8%AB/"
LAST_PAGE = 5630
OUTPUT_FILE = "assabah_headlines.csv"
MAX_WORKERS = 30
TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,fr;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

TITLE_XPATH = "//h2[contains(concat(' ', normalize-space(@class), ' '), ' post-title ')]"
DATE_XPATH = ".//*[contains(concat(' ', normalize-space(@class), ' '), ' date ')]"

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


def find_date(h2):
    node = h2
    for _ in range(6):
        node = node.getparent()
        if node is None:
            break
        dates = node.xpath(DATE_XPATH)
        if dates:
            return " ".join(dates[0].text_content().split())
    return ""


def parse_page(page):
    tree = lhtml.fromstring(fetch(page))
    rows = []
    for h2 in tree.xpath(TITLE_XPATH):
        a = h2.find(".//a")
        node = a if a is not None else h2
        headline = " ".join(node.text_content().split())
        if not headline:
            continue
        href = a.get("href") if a is not None else headline
        rows.append((href, headline, find_date(h2)))
    return rows


def main():
    start = time.time()
    print(f"Pages to scrape: {LAST_PAGE}")

    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parse_page, p): p for p in range(1, LAST_PAGE + 1)}
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
                    print(f"{done}/{LAST_PAGE} pages done")

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
