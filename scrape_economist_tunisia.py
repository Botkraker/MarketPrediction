import csv
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from curl_cffi import requests

# Paste from DevTools > Network > search.json request > Request Headers
COOKIE = "economist_has_visited_app_before=true; blaize_session=054ce560-1bc8-463b-9ccd-43f7340a9704; user_geo_country=TN; cf_clearance=gapN56DMs6QJC0Dvmykf1KbZG4fMLjnLiMrfiweWE00-1789567041-1.2.1.1-F4kDbifByTAj3XCffQvvK5B.H4m_YHGFGLJoPVWEIHdxeVeaoZsq04pUzCSYoaPwHkWDKY3TBzDUbayzytFY23tINpzdu44DnZWQt2SE2Sbz6l3FgEdudUgKD10pyLWAtgJM7.cUuhK1dLysMzbbYXEhcYRpmL2i3cF0y4mzZAL82iigLb1F01svs6DIOOwKVnnhSSimeWBUQ2q3Op4keZqqy8UGe77EdjhqO3yGVaR978Tv1bcrPAZhdw4hhinjF19IL9WEeBmRoDWZZna4tyn5BXy5QiVT6LVLgNAx03sUHhUwC6mYMM9cTcT96Sl0RElVsBGusUN1aqIM_9oGcfTE8bSZ.xpUZpZ4D6bF6gwtC0BClbb7fXwPsKLu3Qh6zcvpRYONBLR_P6cf97K0FymA6CzwBSlD3CDXjWyQUxblR5Ne1OnMn2ykEfkgdfykGz9GRIZ8_sMr_y_v2xQ.PAfk4cEGXcK1uePv7JXy9GiDB7a45L.ZRGpQkHzyQhkZOr1xuGTCeikFBbX6LCZQDA; __cf_bm=dsy3C29A6J09JHae0YoZ98OUdLOaRyTJa.auIn.vHUc-1789567041.0545824-1.0.1.1-ngxoanBm.ZbHpzAPqeRiGZS_6.CYVk0Lq6Xti54dqEEEj9btQZB0E9ERGqvOYeo7Te6B8UHOPFP00iMlMmRypNlzPmyaxM3.R_B38OMpcUZ6ChGsbY9c1ltjliRAMBMY; _cfuvid=sfKOY2aIHUNC1IgZVPDg9.8mgG6G7EhOUErUEah0Bsg-1789567041.8314705-1.0.1.1-bsiQBMvXQ45KuDbfK1ESZxq86O7uBAOXNUO8IX7qcDE; uspv=0; utag_topics_content_id=; utag_topics_id=; utag_topics_label=; utag_topics_probability=; utag_main__sn=4; utag_main_ses_id=1789563451375%3Bexp-session; landingURL=https://www.economist.com/search?q=tunisia&nav_source=masthead&page=2; landingScreen=search.2; datadome=9aunzWcHTZJ_YoMdCKUZGbgVMKpzZLbH21d~ZiI0nPa2O8PMmQhHxEsgFnLxO98BGStSyv52Wywvlv9lIESjRgZA3i1GtkojOy0u6quiOEJ~cwlF7dAn2CsqoopFVk4S; utag_main__se=2%3Bexp-session; utag_main__ss=0%3Bexp-session; utag_main__st=1789565739220%3Bexp-session; utag_main__pn=2%3Bexp-session"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
BUILD_ID = "yihqehR7R26h98_7ZJZ7y"

# Search caps each query at 200 results, so several queries widen coverage
QUERIES = [
    "tunisia",
    "tunisian",
    "tunisia economy",
    "tunisia imf",
    "tunisia debt",
    "tunisia tourism",
    "tunisia inflation",
    "tunisia bank",
    "tunisia trade",
    "tunis",
]

OUTPUT_ALL = "economist_tunisia_all.csv"
OUTPUT_ECON = "economist_tunisia_economy.csv"
MAX_WORKERS = 4
MIN_INTERVAL = 0.8
TIMEOUT = 30
MAX_ATTEMPTS = 4

SITE = "https://www.economist.com"
URL_DATE_RE = re.compile(r"economist\.com/([a-z0-9-]+)/(\d{4})/(\d{2})/(\d{2})/")
RUBRIC_DATE_RE = re.compile(r"^\s*([A-Z][a-z]{2,8}\.? \d{1,2}, \d{4})")
BUILD_ID_RE = re.compile(r'"buildId"\s*:\s*"([^"]+)"')
ECON_SECTIONS = {"finance-and-economics", "business", "graphic-detail", "economic-and-financial-indicators"}
ECON_KEYWORDS = re.compile(
    r"econom|gdp|inflation|imf|debt|default|bail|loan|budget|fiscal|deficit|subsid|"
    r"bank|currency|dinar|interest rate|tax|trade|export|import|invest|tourism|tourist|"
    r"phosphate|olive|oil|gas|energy|wheat|bread|price|wage|jobs|unemploy|poverty|"
    r"growth|recession|market|business|financ|reform|austerity|sovereign|credit",
    re.IGNORECASE,
)

thread_local = threading.local()
print_lock = threading.Lock()
rate_lock = threading.Lock()
build_lock = threading.Lock()
next_allowed = 0.0
challenge_hit = threading.Event()
build_id = BUILD_ID


def log(msg):
    with print_lock:
        print(msg, flush=True)


def wait_turn():
    global next_allowed
    while True:
        now = time.time()
        with rate_lock:
            if now >= next_allowed:
                next_allowed = now + MIN_INTERVAL + random.uniform(0, MIN_INTERVAL)
                return
            sleep_for = next_allowed - now
        time.sleep(sleep_for)


def get_session():
    if not hasattr(thread_local, "session"):
        s = requests.Session(impersonate="chrome")
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Cookie": COOKIE,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{SITE}/search?q=tunisia",
            "x-nextjs-data": "1",
        })
        thread_local.session = s
    return thread_local.session


def is_challenge(r):
    return r.status_code == 403 or "Just a moment..." in r.text[:2000]


def refresh_build_id(old):
    global build_id
    with build_lock:
        if build_id != old:
            return build_id
        r = get_session().get(f"{SITE}/search", params={"q": "tunisia"}, timeout=TIMEOUT)
        if is_challenge(r):
            challenge_hit.set()
            raise RuntimeError("Cloudflare challenge, cookie expired")
        m = BUILD_ID_RE.search(r.text)
        if not m:
            raise RuntimeError("Could not find new buildId")
        build_id = m.group(1)
        log(f"buildId changed, now using {build_id}")
        return build_id


def fetch_json(query, page):
    params = {"q": query, "nav_source": "masthead", "page": page}
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        if challenge_hit.is_set():
            raise RuntimeError("Cloudflare challenge, cookie expired")
        wait_turn()
        current = build_id
        url = f"{SITE}/_next/data/{current}/search.json"
        try:
            r = get_session().get(url, params=params, timeout=TIMEOUT)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            continue
        if is_challenge(r):
            challenge_hit.set()
            raise RuntimeError("Cloudflare challenge, cookie expired")
        if r.status_code == 404:
            refresh_build_id(current)
            continue
        if r.status_code == 200:
            try:
                return r.json().get("pageProps", {})
            except Exception as e:
                last_err = f"bad JSON: {e}"
        else:
            last_err = f"HTTP {r.status_code}"
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(last_err)


def parse_date(url, rubric):
    m = URL_DATE_RE.search(url or "")
    if m:
        return f"{m.group(2)}-{m.group(3)}-{m.group(4)}"
    m = RUBRIC_DATE_RE.match(rubric or "")
    if m:
        raw = m.group(1).replace(".", "")
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return ""


def parse_results(props):
    rows = []
    for item in props.get("searchResults") or []:
        headline = " ".join((item.get("headline") or "").split())
        url = item.get("url") or ""
        rubric = item.get("rubric") or ""
        if not headline or not url:
            continue
        m = URL_DATE_RE.search(url)
        rows.append({
            "url": url,
            "headline": headline,
            "date": parse_date(url, rubric),
            "rubric": rubric,
            "section": m.group(1) if m else "",
        })
    return rows


def scrape(query, page):
    return query, page, parse_results(fetch_json(query, page))


def is_economic(row):
    if row["section"] in ECON_SECTIONS:
        return True
    return bool(ECON_KEYWORDS.search(f"{row['headline']} {row['rubric']}"))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["headline", "date"])
        for r in rows:
            writer.writerow([r["headline"], r["date"]])


def main():
    if "PASTE" in COOKIE or "PASTE" in USER_AGENT:
        log("Fill in COOKIE and USER_AGENT at the top of the script first.")
        return

    start = time.time()
    articles = {}
    tasks = []

    for query in QUERIES:
        try:
            props = fetch_json(query, 1)
        except Exception as e:
            log(f"[{query}] page 1 failed: {e}")
            if challenge_hit.is_set():
                break
            continue
        page_count = int(props.get("pageCount") or 1)
        rows = parse_results(props)
        for r in rows:
            articles.setdefault(r["url"], r)
        log(f"[{query}] {props.get('resultCount')} results, {page_count} pages")
        tasks.extend((query, p) for p in range(2, page_count + 1))

    if not challenge_hit.is_set() and tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(scrape, q, p) for q, p in tasks]
            done = 0
            for future in as_completed(futures):
                try:
                    _, _, rows = future.result()
                    for r in rows:
                        articles.setdefault(r["url"], r)
                except Exception as e:
                    log(f"Failed: {e}")
                done += 1
                if done % 20 == 0:
                    log(f"{done}/{len(tasks)} pages done, {len(articles)} unique articles")

    all_rows = sorted(articles.values(), key=lambda r: r["date"], reverse=True)
    econ_rows = [r for r in all_rows if is_economic(r)]

    write_csv(OUTPUT_ALL, all_rows)
    write_csv(OUTPUT_ECON, econ_rows)
    log(f"Saved {len(all_rows)} articles to {OUTPUT_ALL}")
    log(f"Saved {len(econ_rows)} economic articles to {OUTPUT_ECON}")
    if challenge_hit.is_set():
        log("Stopped early: Cloudflare cookie expired. Copy a fresh cookie from the browser and run again.")
    log(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()