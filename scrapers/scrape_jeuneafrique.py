import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests
from lxml import html as lhtml

BASE_URL = "https://www.jeuneafrique.com/rubriques/economie/"
LAST_PAGE = 145
OUTPUT_FILE = "jeuneafrique_headlines.csv"
MAX_WORKERS = 12
TIMEOUT = 30
MAX_ATTEMPTS = 5

ARTICLE_RE = re.compile(r"jeuneafrique\.com/\d{5,}/")
FR_MONTHS = {
    "janvier": "01", "fevrier": "02", "mars": "03", "avril": "04",
    "mai": "05", "juin": "06", "juillet": "07", "aout": "08",
    "septembre": "09", "octobre": "10", "novembre": "11", "decembre": "12",
}
FR_DATE_RE = re.compile(r"(\d{1,2})\s+([a-zA-Zéèêûôàç]+)\s+(\d{4})")
ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
DMY_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")

thread_local = threading.local()
print_lock = threading.Lock()


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session(impersonate="chrome")
        s.headers.update({"Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8"})
        thread_local.session = s
    return thread_local.session


def page_url(page):
    return BASE_URL if page == 1 else f"{BASE_URL}page/{page}/"


def fetch(page):
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = get_session().get(page_url(page), timeout=TIMEOUT)
            if r.status_code == 200:
                return r.content
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(last_err)


def normalize_date(text):
    text = text.strip()
    m = ISO_RE.search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = DMY_RE.search(text)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = FR_DATE_RE.search(text)
    if m:
        mon = m.group(2).lower().translate(str.maketrans("éèêûôàç", "eeeuoac"))
        if mon in FR_MONTHS:
            return f"{m.group(3)}-{FR_MONTHS[mon]}-{m.group(1).zfill(2)}"
    return text


def find_date(node):
    for _ in range(6):
        node = node.getparent()
        if node is None:
            break
        t = node.xpath(".//time")
        if t:
            return normalize_date(t[0].get("datetime") or t[0].text_content())
        d = node.xpath(".//*[contains(@class,'date')]")
        if d:
            return normalize_date(d[0].text_content())
    return ""


def parse_page(page):
    tree = lhtml.fromstring(fetch(page))
    rows = []
    seen = set()
    for a in tree.xpath("//*[self::h2 or self::h3 or self::h4]//a[@href]"):
        href = a.get("href")
        if not ARTICLE_RE.search(href) or href in seen:
            continue
        headline = " ".join(a.text_content().split())
        if not headline:
            continue
        seen.add(href)
        heading = a.xpath("ancestor::*[self::h2 or self::h3 or self::h4][1]")[0]
        rows.append((href, headline, find_date(heading)))
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
                if not results[page]:
                    with print_lock:
                        print(f"Page {page}: 0 headlines")
            except Exception as e:
                failed.append(page)
                with print_lock:
                    print(f"Page {page} failed: {e}")
            done += 1
            if done % 20 == 0:
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
