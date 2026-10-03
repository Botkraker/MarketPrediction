import base64
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import certifi
import requests

try:
    import truststore
    truststore.inject_into_ssl()
    HAS_TRUSTSTORE = True
except ImportError:
    HAS_TRUSTSTORE = False

SECTION_ID = "/section/business/economy"
OUTPUT_FILE = "nyt_economy_headlines.csv"
PROGRESS_FILE = "nyt_economy_progress.json"
API_URL = "https://samizdat-graphql.nytimes.com/graphql/v2"
SHA256_HASH = "6da5485c8197b41186a79abcbf90455cb10389be8509b5f7fe479f3611ed794c"
NYT_TOKEN = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAs+/oUCTBmD/cLdmcecrnBMHiU/pxQCn2DDyaPKUOXxi4p0uUSZQzsuq1pJ1m5z1i0YGPd1U1OeGHAChWtqoxC7bFMCXcwnE1oyui9G1uobgpm1GdhtwkR7ta7akVTcsF8zxiXx7DNXIPd2nIJFH83rmkZueKrC4JVaNzjvD+Z03piLn5bHWU6+w+rA+kyJtGgZNTXKyPh6EC6o5N+rknNMG5+CdTq35p8f99WjFawSvYgP9V64kgckbTbtdJ6YhVP58TnuYgr12urtwnIqWP9KSJ1e5vmgf3tunMqWNm6+AnsqNj8mCLdCuc5cEB74CwUeQcP2HQQmbCddBy2y0mEwIDAQAB"
)

PAGE_SIZE_TRY = 100
MAX_WORKERS = 8
MIN_INTERVAL = 0.3
TIMEOUT = 30
MAX_ATTEMPTS = 6
COOLDOWN = 60

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "nyt-app-type": "project-vi",
    "nyt-app-version": "0.0.5",
    "nyt-token": NYT_TOKEN,
    "origin": "https://www.nytimes.com",
    "referer": "https://www.nytimes.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
    ),
}

thread_local = threading.local()
print_lock = threading.Lock()
rate_lock = threading.Lock()
state_lock = threading.Lock()
next_allowed = 0.0
blocked_until = 0.0


def log(msg):
    with print_lock:
        print(msg, flush=True)


def wait_turn():
    global next_allowed
    while True:
        now = time.time()
        with rate_lock:
            if now < blocked_until:
                sleep_for = blocked_until - now
            elif now < next_allowed:
                sleep_for = next_allowed - now
            else:
                next_allowed = now + MIN_INTERVAL + random.uniform(0, MIN_INTERVAL)
                return
        time.sleep(min(sleep_for, 5))


def block_all(seconds, reason):
    global blocked_until
    with rate_lock:
        until = time.time() + seconds
        if until > blocked_until:
            blocked_until = until
            log(f"{reason}: pausing all threads for {seconds}s")


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        if not HAS_TRUSTSTORE:
            s.verify = certifi.where()
        thread_local.session = s
    return thread_local.session


def cursor_for_offset(offset):
    if offset <= 0:
        return None
    return base64.b64encode(f"arrayconnection:{offset - 1}".encode()).decode()


def build_params(offset, page_size):
    variables = {
        "id": SECTION_ID,
        "first": page_size,
        "exclusionMode": "NONE",
        "isFetchMore": offset > 0,
        "isEspanol": False,
        "highlightsListUri": "nyt://per/personalized-list/__null__",
        "highlightsListFirst": 0,
        "hasHighlightsList": False,
        "collectionQuery": {"sort": "newest", "text": ""},
    }
    cursor = cursor_for_offset(offset)
    if cursor:
        variables["cursor"] = cursor
    extensions = {"persistedQuery": {"version": 1, "sha256Hash": SHA256_HASH}}
    return {
        "operationName": "CollectionsQuery",
        "variables": json.dumps(variables, separators=(",", ":")),
        "extensions": json.dumps(extensions, separators=(",", ":")),
    }


def fetch_stream(offset, page_size):
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        wait_turn()
        try:
            r = get_session().get(API_URL, params=build_params(offset, page_size), timeout=TIMEOUT)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            try:
                return data["data"]["legacyCollection"]["collectionsPage"]["stream"]
            except (KeyError, TypeError):
                last_err = f"unexpected response: {str(data)[:200]}"
                time.sleep(2 * (attempt + 1))
                continue
        last_err = f"HTTP {r.status_code}"
        if r.status_code in (403, 429):
            block_all(COOLDOWN * (attempt + 1), f"{r.status_code} at offset {offset}")
        else:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(last_err)


def parse_edges(stream):
    rows = []
    for edge in stream.get("edges") or []:
        node = edge.get("node") or {}
        url = node.get("url")
        headline = ((node.get("headline") or {}).get("default") or "").strip()
        date = (node.get("firstPublished") or "")[:10]
        if url and headline:
            rows.append([url, " ".join(headline.split()), date])
    return rows


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("page_size"), {int(k): v for k, v in data.get("results", {}).items()}
    return None, {}


def save_progress(page_size, results):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"page_size": page_size, "results": results}, f, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)


def write_csv(results):
    seen = set()
    rows_out = []
    for offset in sorted(results):
        for url, headline, date in results[offset]:
            if url in seen:
                continue
            seen.add(url)
            rows_out.append((headline, date))
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["headline", "date"])
        writer.writerows(rows_out)
    return len(rows_out)


def main():
    start = time.time()
    page_size, results = load_progress()

    first = fetch_stream(0, PAGE_SIZE_TRY)
    total = int(first.get("totalCount") or 0)
    first_rows = parse_edges(first)
    if page_size is None:
        page_size = len(first_rows) or 10
        results = {}
    results[0] = first_rows
    log(f"Total items reported: {total}, page size: {page_size}")

    offsets = [o for o in range(page_size, total, page_size) if o not in results]
    log(f"Remaining pages: {len(offsets)}")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_stream, o, page_size): o for o in offsets}
        done = 0
        for future in as_completed(futures):
            offset = futures[future]
            try:
                rows = parse_edges(future.result())
                with state_lock:
                    results[offset] = rows
                    if done % 10 == 0:
                        save_progress(page_size, results)
            except Exception as e:
                failed.append(offset)
                log(f"Offset {offset} failed: {e}")
            done += 1
            if done % 20 == 0:
                log(f"{done}/{len(offsets)} pages done")

    save_progress(page_size, results)
    count = write_csv(results)
    log(f"Saved {count} headlines to {OUTPUT_FILE} in {time.time() - start:.1f}s")
    if failed:
        log(f"Offsets still failing: {sorted(failed)} (run again, finished pages are skipped)")


if __name__ == "__main__":
    main()
