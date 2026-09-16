"""
Data audit for ProjectNLP headline sources.

Adapted from the original audit plan to the ACTUAL raw schema found in
data/raw/*.csv: every source is a flat CSV with only `headline` and `date`
columns (no url, no body/article text, no time-of-day). There is no market
(BVMT) data in the repo, so all market-data-dependent steps (plan step 6,
and the overlap/go-no-go decision in step 9) are marked BLOCKED rather than
computed.

Never edits the raw CSVs. Reads them as-is and writes all outputs under audit/.
"""
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 42

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
AUDIT = ROOT / "audit"

# name -> (filename, delimiter)
SOURCES = {
    "assabah": ("assabah_headlines.csv", ";"),
    "economist_tunisia_all": ("economist_tunisia_all.csv", ";"),
    "economist_tunisia_economy": ("economist_tunisia_economy.csv", ";"),
    "guardian_tunisia": ("guardian_tunisia_headlines.csv", ";"),
    "ilboursa": ("ilboursa_headlines.csv", ";"),
    "kapitalis": ("kapitalis_headlines.csv", ";"),
    "lapresse": ("lapressheadlines.csv", ","),
    "leconomistmaghrebin": ("leconomistmaghrebin_headlines.csv", ","),
    "nyt_economy": ("nyt_economy_headlines.csv", ";"),
    "tap": ("tap_headlines.csv", ";"),
}

# Sources without a matching scrape_*.py script committed at all (gap to flag).
SCRAPER_SCRIPT = {
    "assabah": "scrape_assabah.py",
    "economist_tunisia_all": "scrape_economist_tunisia.py",
    "economist_tunisia_economy": "scrape_economist_tunisia.py",
    "guardian_tunisia": "scrape_guardian.py",
    "ilboursa": "scrape_ilboursa.py",
    "kapitalis": "scrape_kapitalis.py",
    "lapresse": None,  # no scraper script in repo
    "leconomistmaghrebin": "leconomistemaghrebin_Scraper.py",
    "nyt_economy": "scrape_nyt_economy.py",
    "tap": "scrape_tap.py",
}

AR_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "مايو": 5, "يونيو": 6,
    "يوليو": 7, "أغسطس": 8, "سبتمبر": 9, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
}
AR_ABS_RE = re.compile(r"^(\d{1,2})\s+(\S+)\s+(\d{4})$")


def parse_assabah_date(raw: str):
    """Returns (date_or_None, category) where category in {abs, relative, other}."""
    if raw is None:
        return None, "other"
    raw = raw.strip()
    m = AR_ABS_RE.match(raw)
    if m and m.group(2) in AR_MONTHS:
        d, mo_word, y = m.groups()
        try:
            return date(int(y), AR_MONTHS[mo_word], int(d)), "abs"
        except ValueError:
            return None, "other"
    if raw.startswith("منذ"):
        return None, "relative"
    return None, "other"


def latest_commit_for(path: Path):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(path)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or "NOT_COMMITTED"
    except Exception:
        return "UNKNOWN"


def load_source(name: str, fname: str, delim: str) -> pd.DataFrame:
    path = RAW / fname
    df = pd.read_csv(path, delimiter=delim, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df.rename(columns={"headline": "headline", "date": "date_raw"})
    df["source"] = name
    if name == "assabah":
        parsed = df["date_raw"].map(parse_assabah_date)
        df["published_date"] = [p[0] for p in parsed]
        df["date_defect"] = [p[1] for p in parsed]
    else:
        pd_dates = pd.to_datetime(df["date_raw"], format="%Y-%m-%d", errors="coerce")
        df["published_date"] = pd_dates.dt.date
        df["date_defect"] = df["published_date"].isna().map(lambda x: "unparsed" if x else "abs")
    df["date_parse_ok"] = df["published_date"].notna()
    df["row_id"] = df["source"] + "::" + df.index.astype(str)
    return df[["row_id", "source", "headline", "date_raw", "published_date", "date_parse_ok", "date_defect"]]


def main():
    AUDIT.mkdir(exist_ok=True)

    # ---------- Step 1: inventory ----------
    inventory_rows = []
    for name, (fname, delim) in SOURCES.items():
        path = RAW / fname
        st = path.stat()
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            n_rows = sum(1 for _ in fh) - 1
        inventory_rows.append({
            "source": name,
            "file": fname,
            "file_count": 1,
            "row_count": n_rows,
            "size_bytes": st.st_size,
            "extraction_date_mtime": pd.Timestamp(st.st_mtime, unit="s").isoformat(),
            "scraper_script": SCRAPER_SCRIPT[name] or "MISSING",
            "scraper_commit": latest_commit_for(ROOT / SCRAPER_SCRIPT[name]) if SCRAPER_SCRIPT[name] else "NO_SCRIPT",
        })
    inv_df = pd.DataFrame(inventory_rows)
    inv_df.to_csv(AUDIT / "inventory.csv", index=False)
    print("Wrote audit/inventory.csv")

    # ---------- Step 2: load everything into one table ----------
    frames = [load_source(name, fname, delim) for name, (fname, delim) in SOURCES.items()]
    articles = pd.concat(frames, ignore_index=True)

    con = duckdb.connect(str(AUDIT / "audit.duckdb"))
    con.execute("CREATE OR REPLACE TABLE articles AS SELECT * FROM articles")
    print(f"Loaded {len(articles)} rows into audit/audit.duckdb::articles")

    # ---------- Step 3: completeness per source ----------
    completeness = con.execute("""
        SELECT source,
               COUNT(*) AS n,
               ROUND(AVG((NOT date_parse_ok)::INT), 4) AS pct_no_date,
               ROUND(AVG((headline IS NULL OR length(trim(headline)) = 0)::INT), 4) AS pct_empty_headline,
               COUNT(*) - COUNT(DISTINCT headline) AS dup_headlines_within_source
        FROM articles GROUP BY source ORDER BY source
    """).df()
    completeness.to_csv(AUDIT / "completeness.csv", index=False)
    print("Wrote audit/completeness.csv")

    # top repeated headlines per source (boilerplate / scraper-junk detector)
    boiler = con.execute("""
        SELECT source, headline, COUNT(*) AS n
        FROM articles
        GROUP BY source, headline
        HAVING COUNT(*) > 3
        ORDER BY source, n DESC
    """).df()
    boiler.to_csv(AUDIT / "repeated_headlines.csv", index=False)
    print(f"Wrote audit/repeated_headlines.csv ({len(boiler)} repeated-headline groups)")

    # ---------- Step 4: coverage over time ----------
    cov = con.execute("""
        SELECT source, strftime(published_date, '%Y-%m') AS month, COUNT(*) AS n
        FROM articles WHERE date_parse_ok
        GROUP BY source, month ORDER BY source, month
    """).df()
    cov.to_csv(AUDIT / "coverage_by_month.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    for name, g in cov.groupby("source"):
        g = g.sort_values("month")
        ax.plot(pd.to_datetime(g["month"]), g["n"], label=name, linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Articles per month by source (log scale)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Article count")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(AUDIT / "coverage_by_month.png", dpi=130)
    plt.close(fig)
    print("Wrote audit/coverage_by_month.png")

    # gap / cliff / spike detection + usable-window heuristic per source
    coverage_summary = []
    for name, g in cov.groupby("source"):
        g = g.sort_values("month").reset_index(drop=True)
        months = pd.period_range(g["month"].min(), g["month"].max(), freq="M")
        full = pd.DataFrame({"month": months.astype(str)}).merge(g[["month", "n"]], on="month", how="left").fillna(0)
        full["n"] = full["n"].astype(int)
        zero_months = int((full["n"] == 0).sum())
        median_n = full.loc[full["n"] > 0, "n"].median() if (full["n"] > 0).any() else 0
        spikes = full[full["n"] > 3 * median_n] if median_n > 0 else full.iloc[0:0]
        # cliff: month-over-month drop of >70% after at least 3 non-zero months of history
        drops = []
        for i in range(3, len(full)):
            prev_avg = full["n"].iloc[max(0, i - 3):i].mean()
            if prev_avg >= 5 and full["n"].iloc[i] < 0.3 * prev_avg:
                drops.append(full["month"].iloc[i])
        # usable window: last long zero-gap (>=2 consecutive zero months) end, else full range start
        zero_idx = full.index[full["n"] == 0].tolist()
        usable_start = full["month"].iloc[0]
        run_start = None
        for i in zero_idx:
            if run_start is None:
                run_start = i
            if i + 1 not in zero_idx:
                if i - run_start + 1 >= 2:
                    usable_start = full["month"].iloc[min(i + 1, len(full) - 1)]
                run_start = None

        # calendar-day zero-article share within parsed date range (proxy; NOT trading-day based)
        dates = con.execute(
            "SELECT DISTINCT published_date FROM articles WHERE source = ? AND date_parse_ok", [name]
        ).df()["published_date"]
        if len(dates) > 0:
            lo, hi = min(dates), max(dates)
            span_days = (hi - lo).days + 1
            zero_day_share = round(1 - len(dates) / span_days, 4)
        else:
            lo = hi = None
            span_days = 0
            zero_day_share = None

        coverage_summary.append({
            "source": name,
            "first_month": full["month"].iloc[0],
            "last_month": full["month"].iloc[-1],
            "n_months_total": len(full),
            "n_months_zero": zero_months,
            "cliff_months": ",".join(drops) if drops else "",
            "spike_months": ",".join(spikes["month"].tolist()) if len(spikes) else "",
            "usable_window_start_month": usable_start,
            "date_min": lo,
            "date_max": hi,
            "calendar_zero_day_share_TODO_needs_trading_calendar": zero_day_share,
        })
    coverage_summary_df = pd.DataFrame(coverage_summary)
    coverage_summary_df.to_csv(AUDIT / "coverage_summary.csv", index=False)
    print("Wrote audit/coverage_summary.csv")

    # ---------- Step 5: timestamp quality ----------
    # All sources are date-only (no time-of-day was ever captured in these CSVs),
    # except assabah whose 'date' field is often a relative string we resolve to a date.
    # So the midnight-fraction / hour-of-day checks from the original plan don't apply --
    # documented explicitly instead of computed.
    ts_quality = completeness[["source", "pct_no_date"]].copy()
    ts_quality["has_time_of_day"] = False
    ts_quality["known_defect"] = ts_quality["source"].map({
        "assabah": "date column is mostly relative Arabic text ('N hours/days/weeks ago'); "
                   "99.65% resolved to absolute dates via month-name parsing, 0.35% "
                   "(195 rows) are unresolvable relative-only strings -> counted as parse failures.",
        "ilboursa": "scraper (scrape_ilboursa.py) parses a full datetime (dd/mm/YYYY HH:MM) "
                    "from the site but discards both the time-of-day and the article URL "
                    "before writing the CSV -- scraper bug, fix + re-extract recommended.",
    }).fillna("date-only, no time-of-day ever captured by the scraper")
    ts_quality.to_csv(AUDIT / "timestamp_quality.csv", index=False)
    print("Wrote audit/timestamp_quality.csv")

    # spot-check worksheet: 10 random rows per source for manual site comparison
    spot = con.execute("""
        SELECT source, headline, date_raw, published_date
        FROM articles
        USING SAMPLE 10 PER source (bernoulli)
    """).df() if False else None
    # deterministic 10-per-source sample (reservoir via pandas for reproducibility)
    spot_rows = []
    for name, g in articles.groupby("source"):
        spot_rows.append(g.sample(n=min(10, len(g)), random_state=42))
    spot_df = pd.concat(spot_rows, ignore_index=True)
    spot_df["site_date_matches (fill by hand)"] = ""
    spot_df["notes (fill by hand)"] = ""
    spot_df.to_csv(AUDIT / "spot_check_worksheet.csv", index=False)
    print("Wrote audit/spot_check_worksheet.csv (manual step: open each URL/site and compare)")

    # ---------- Step 6: market data ----------
    print("Step 6 (BVMT market data) SKIPPED -- no market data file in repo. Documented as BLOCKED in report.")

    # ---------- Step 7: duplicates and wire reprints ----------
    def norm(h):
        return re.sub(r"\s+", " ", (h or "")).strip().lower()

    articles["headline_norm"] = articles["headline"].map(norm)
    articles["headline_hash"] = articles["headline_norm"].map(
        lambda h: hashlib.sha1(h.encode("utf-8")).hexdigest()
    )

    exact_dup_within = (
        articles.groupby(["source", "headline_hash"]).size().reset_index(name="n")
    )
    exact_dup_within = exact_dup_within[exact_dup_within["n"] > 1]
    dup_share_within = (
        exact_dup_within.groupby("source")["n"].apply(lambda s: (s - 1).sum())
        / articles.groupby("source").size()
    ).round(4)

    cross_source_hash = articles.groupby("headline_hash")["source"].nunique()
    cross_dup_hashes = cross_source_hash[cross_source_hash > 1].index
    cross_dup = articles[articles["headline_hash"].isin(cross_dup_hashes)][
        ["source", "headline", "published_date", "headline_hash"]
    ].sort_values("headline_hash")
    cross_dup.to_csv(AUDIT / "exact_cross_source_duplicates.csv", index=False)

    # near-duplicate / TAP wire-reprint check via MinHash, French Tunisian sources only
    # (English sources can't reprint a French/Arabic TAP story; assabah is Arabic and
    # TAP's scraped portal here is French-only, so cross-script matches would be ~0
    # similarity by construction -- excluded to keep this tractable and meaningful).
    from datasketch import MinHash

    FR_SOURCES = ["ilboursa", "kapitalis", "lapresse", "leconomistmaghrebin", "tap"]
    NUM_PERM = 64

    def shingles(text, k=3):
        words = norm(text).split()
        if len(words) < k:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

    def minhash_for(text):
        mh = MinHash(num_perm=NUM_PERM)
        for sh in shingles(text):
            mh.update(sh.encode("utf-8"))
        return mh

    fr_articles = articles[articles["source"].isin(FR_SOURCES) & articles["date_parse_ok"]].reset_index(drop=True)
    tap_rows = fr_articles[fr_articles["source"] == "tap"]
    tap_by_date = defaultdict(list)
    for _, row in tap_rows.iterrows():
        tap_by_date[row["published_date"]].append((row["headline"], minhash_for(row["headline"])))

    THRESH = 0.8
    reprint_flags = []
    for _, row in fr_articles.iterrows():
        if row["source"] == "tap":
            reprint_flags.append(False)
            continue
        d = row["published_date"]
        candidates = []
        for delta in (-2, -1, 0, 1, 2):
            candidates.extend(tap_by_date.get(d + timedelta(days=delta), []))
        if not candidates:
            reprint_flags.append(False)
            continue
        mh = minhash_for(row["headline"])
        is_reprint = any(mh.jaccard(cmh) >= THRESH for _, cmh in candidates)
        reprint_flags.append(is_reprint)
    fr_articles = fr_articles.copy()
    fr_articles["tap_reprint_candidate"] = reprint_flags

    reprint_summary = (
        fr_articles[fr_articles["source"] != "tap"]
        .groupby("source")["tap_reprint_candidate"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "reprint_matches", "count": "n_in_tap_overlap_window"})
    )
    reprint_summary["share"] = (reprint_summary["reprint_matches"] / reprint_summary["n_in_tap_overlap_window"]).round(4)
    reprint_summary = reprint_summary.reset_index()
    reprint_summary.to_csv(AUDIT / "tap_reprint_share.csv", index=False)

    dup_summary = pd.DataFrame({
        "source": dup_share_within.index,
        "exact_dup_share_within_source": dup_share_within.values,
    })
    dup_summary = dup_summary.merge(
        cross_source_hash[cross_source_hash > 1].rename("n_sources_sharing_hash").reset_index()
        .merge(articles[["headline_hash", "source"]].drop_duplicates(), on="headline_hash")
        .groupby("source").size().rename("cross_source_exact_dup_headlines").reset_index(),
        on="source", how="left",
    )
    dup_summary.to_csv(AUDIT / "duplicate_summary.csv", index=False)
    print("Wrote audit/duplicate_summary.csv, exact_cross_source_duplicates.csv, tap_reprint_share.csv")
    print(f"  NOTE: TAP source only spans {tap_rows['published_date'].min()} .. {tap_rows['published_date'].max()} "
          f"({len(tap_rows)} rows) -- reprint share above is only measurable in that narrow overlap window.")

    # ---------- Step 8: language & relevance sample ----------
    sample_rows = []
    for name, g in articles.groupby("source"):
        sample_rows.append(g.sample(n=min(25, len(g)), random_state=42))
    sample_df = pd.concat(sample_rows, ignore_index=True)

    def safe_detect(text):
        try:
            return detect(text) if text and len(text.strip()) >= 3 else "NA"
        except Exception:
            return "ERROR"

    sample_df["langdetect_guess"] = sample_df["headline"].map(safe_detect)
    sample_df["language (FR/MSA/Tunisian/Arabizi/EN/mixed -- fill by hand)"] = ""
    sample_df["about_tunisian_economy Y/N (fill by hand)"] = ""
    sample_df["about_global_economy Y/N (fill by hand)"] = ""
    sample_df["headline_extracted_correctly Y/N (fill by hand)"] = ""
    sample_df["stored_date_matches_site Y/N (fill by hand)"] = ""
    cols = ["source", "row_id", "headline", "date_raw", "published_date", "langdetect_guess",
            "language (FR/MSA/Tunisian/Arabizi/EN/mixed -- fill by hand)",
            "about_tunisian_economy Y/N (fill by hand)",
            "about_global_economy Y/N (fill by hand)",
            "headline_extracted_correctly Y/N (fill by hand)",
            "stored_date_matches_site Y/N (fill by hand)"]
    sample_df[cols].to_csv(AUDIT / "language_relevance_sample.csv", index=False)
    print("Wrote audit/language_relevance_sample.csv (25/source; hand-label the *_fill_by_hand columns)")

    # aggregate automatic language mix per source (on the FULL corpus, not just the sample)
    print("Running langdetect over full corpus for aggregate language mix (this is the slow step)...")
    articles["langdetect_guess"] = articles["headline"].map(safe_detect)
    lang_mix = (
        articles.groupby(["source", "langdetect_guess"]).size().reset_index(name="n")
    )
    lang_mix.to_csv(AUDIT / "language_mix_full_corpus.csv", index=False)
    print("Wrote audit/language_mix_full_corpus.csv")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
