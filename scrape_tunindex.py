"""
scrape_tunindex.py
Scrape Tunindex (BVMT, ilboursa code PX1) daily OHLC + volume from 2010 to today,
working around the 3-month-per-download limit by looping in 85-day chunks.

Two methods:
  scrape_requests()  fast, no browser. Resubmits the ASP.NET form per chunk.
  scrape_selenium()  fallback that drives Chrome (use if requests returns empty).

Install:
  pip install requests beautifulsoup4 pandas python-dateutil
  # selenium fallback only:
  pip install selenium webdriver-manager

Run:
  python scrape_tunindex.py
Output:
  tunindex_2010_today.csv  with columns: date, open, high, low, close, volume
"""

import io
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
from dateutil.relativedelta import relativedelta

TICKER = "PX1"
PAGE = f"https://www.ilboursa.com/marches/download/{TICKER}"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Referer": PAGE,
}
CHUNK_DAYS = 85          # stay under the 3-month cap
SLEEP_SEC = 1.2          # be polite between requests
START = "01/01/2010"     # DD/MM/YYYY
END = None               # None = today


def _chunks(start, end):
    cur = start
    while cur < end:
        stop = min(cur + relativedelta(days=+CHUNK_DAYS), end)
        yield cur, stop
        cur = stop + relativedelta(days=+1)


def _clean(frames):
    if not frames:
        raise RuntimeError("No data scraped. Try the selenium fallback.")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"ouverture": "open", "haut": "high", "bas": "low",
                            "cloture": "close", "volume": "volume",
                            "date": "date", "symbole": "symbol"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                     .str.replace("\u202f", "", regex=False)
                     .str.replace(" ", "", regex=False)
                     .str.replace(",", ".", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["date", "close"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"]
            if c in df.columns]
    return df[keep]


# ----------------------------------------------------------------------
# METHOD 1: requests (resubmit the form, viewstate handled automatically)
# ----------------------------------------------------------------------
def scrape_requests(start=START, end=END, out_csv="tunindex_2010_today.csv"):
    import requests
    from bs4 import BeautifulSoup

    start_dt = datetime.strptime(start, "%d/%m/%Y")
    end_dt = datetime.now() if end is None else datetime.strptime(end, "%d/%m/%Y")

    s = requests.Session()
    s.headers.update(HEADERS)
    r = s.get(PAGE, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    form = soup.find("form")
    action = urljoin(PAGE, form.get("action") or "")

    def base_payload():
        p = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            if name:
                p[name] = inp.get("value", "")
        for btn in form.find_all(["button", "input"]):
            if (btn.name == "button" or btn.get("type") == "submit"):
                if btn.get("name"):
                    p[btn["name"]] = btn.get("value", "")
        return p

    frames = []
    for i, (a, b) in enumerate(_chunks(start_dt, end_dt), 1):
        payload = base_payload()
        # the page's JS sets these two ids to YYYY-MM-DD
        for key in list(payload):
            low = key.lower()
            if "dtfrom" in low or "from" in low:
                payload[key] = a.strftime("%Y-%m-%d")
            if "dtto" in low or key.lower().endswith("to"):
                payload[key] = b.strftime("%Y-%m-%d")
        payload.setdefault("dtFrom", a.strftime("%Y-%m-%d"))
        payload.setdefault("dtTo", b.strftime("%Y-%m-%d"))

        resp = s.post(action, data=payload, timeout=60)
        ctype = resp.headers.get("Content-Type", "")
        cdisp = resp.headers.get("Content-Disposition", "")
        text = resp.text

        looks_csv = ("csv" in ctype.lower() or "attachment" in cdisp.lower()
                     or text[:200].count(";") >= 3)
        if looks_csv and text.strip():
            try:
                part = pd.read_csv(io.StringIO(text), sep=";")
                if len(part):
                    frames.append(part)
                    print(f"chunk {i}: {a:%Y-%m-%d} to {b:%Y-%m-%d}  "
                          f"{len(part)} rows")
                else:
                    print(f"chunk {i}: {a:%Y-%m-%d} to {b:%Y-%m-%d}  empty")
            except Exception as e:
                print(f"chunk {i}: parse failed ({e})")
        else:
            print(f"chunk {i}: {a:%Y-%m-%d} to {b:%Y-%m-%d}  "
                  f"no CSV (ctype={ctype!r}). If all chunks look like this, "
                  f"use scrape_selenium().")
        time.sleep(SLEEP_SEC)

    df = _clean(frames)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv} "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    return df


# ----------------------------------------------------------------------
# METHOD 2: selenium fallback (drives the real form)
# ----------------------------------------------------------------------
def scrape_selenium(start=START, end=END, out_csv="tunindex_2010_today.csv",
                    headless=True):
    import os, glob
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    start_dt = datetime.strptime(start, "%d/%m/%Y")
    end_dt = datetime.now() if end is None else datetime.strptime(end, "%d/%m/%Y")

    dl = os.path.abspath("_px1_dl")
    os.makedirs(dl, exist_ok=True)
    for f in glob.glob(os.path.join(dl, "*.csv")):
        os.remove(f)

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_experimental_option("prefs", {
        "download.default_directory": dl,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    })
    driver = webdriver.Chrome(options=opts)
    frames = []
    try:
        driver.get(PAGE)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "dtFrom")))
        for i, (a, b) in enumerate(_chunks(start_dt, end_dt), 1):
            raw = os.path.join(dl, f"cotations_{TICKER}.csv")
            if os.path.exists(raw):
                os.remove(raw)
            driver.execute_script(
                f"document.getElementById('dtFrom').value='{a:%Y-%m-%d}';")
            driver.execute_script(
                f"document.getElementById('dtTo').value='{b:%Y-%m-%d}';")
            driver.find_element(
                By.XPATH, "//button[contains(@class,'btnR')]").click()
            waited = 0
            while not os.path.exists(raw) and waited < 15:
                time.sleep(1)
                waited += 1
            if os.path.exists(raw):
                try:
                    part = pd.read_csv(raw, sep=";")
                    if len(part):
                        frames.append(part)
                        print(f"chunk {i}: {a:%Y-%m-%d} to {b:%Y-%m-%d}  "
                              f"{len(part)} rows")
                except Exception:
                    pass
            else:
                print(f"chunk {i}: {a:%Y-%m-%d} to {b:%Y-%m-%d}  empty")
            time.sleep(SLEEP_SEC)
    finally:
        driver.quit()

    df = _clean(frames)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv} "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    return df


if __name__ == "__main__":
    try:
        scrape_requests()
    except Exception as e:
        print(f"requests method failed ({e}); trying selenium fallback...")
        scrape_selenium()
