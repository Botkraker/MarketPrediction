# Data Audit Report — raw-v1

Scope: the 10 scraper outputs currently in `data/raw/`. Produced by `audit/build_audit.py`
(deterministic, re-runnable; outputs live alongside this report in `audit/`). Nothing in
`data/raw/` was edited — every defect below is documented for a scraper fix + re-extract,
per the "never patch raw files by hand" rule.

## 0. How this audit differs from the original plan

The plan this was scoped against assumed per-source JSON with `url, title, body, published_at`
(with time-of-day) plus a BVMT market-data table. The actual raw data is different in three
ways that were confirmed with you before running this:

1. **Headline-only corpus, by design.** Every `data/raw/*.csv` has exactly two columns,
   `headline` and `date` — no article body, no URL. So every body/url-dependent check from the
   original plan (paywall/teaser detection via body length, URL-based dedup, MinHash on
   title+body, "full text extracted correctly") either doesn't apply or was adapted to run on
   the headline text only, which is a weaker signal — flagged inline below wherever it matters.
2. **No time-of-day, audited as-is.** Every source's `date` field is date-only (or worse — see
   §2). Timestamp-quality checks from the plan (midnight fraction, hour-of-day histogram) are
   not computed since they'd be vacuous; documented as a known limitation per source instead,
   to be fixed in a follow-up scraper pass.
3. **No market data.** There is no BVMT price file in the repo. Step 6 (missing sessions, price
   sanity checks) and the step 9 overlap-window / go-no-go decision are **BLOCKED** — not
   computed — pending a market-data source.

## 1. Inventory (step 1)

See `audit/inventory.csv` for the full machine-readable table (file, row count, size,
extraction mtime, scraper script, scraper commit). Two gaps found:

- **No scraper script for `lapresse`** (`lapressheadlines.csv` exists in `data/raw/` but no
  `scrape_lapresse.py` is in the repo) — can't be re-run or reproduced from source.
  `inventory.csv` records this source's `scraper_commit` as `NO_SCRIPT`.
- Every other scraper is already committed at `c1b7b3b` ("first init"), which is what
  `inventory.csv`'s `scraper_commit` column resolves to (computed live via `git log` per
  script, so it stays correct if scrapers are updated later).

## 2. Per-source completeness, coverage, timestamps, duplicates

| source | rows | usable date range | median articles/active day | calendar zero-day share* | dup share (exact, within-source) | cross-source exact-dup headlines | timestamp quality | status |
|---|---:|---|---:|---:|---:|---:|---|---|
| assabah | 56,292 | 2010-10 → 2026-08 | 12.0 | 0.19 | 6.5% | 0 | date field is 99.65% relative/absolute Arabic text resolved via month-name parsing; 0.35% (195 rows) unresolvable relative-only strings, counted as parse failures | **keep, fix scraper first** — see defects below |
| economist_tunisia_all | 1,197 | 1997-04 → 2026-09, but usable window starts **2020-05** (large pre-2020 gaps, see coverage plot) | 1.0 | 0.93 | 2.3% | 739 (100% overlap with `economist_tunisia_economy`) | date-only; 8 rows (0.67%) empty date string | **keep, shorter window (≥2020-05)** |
| economist_tunisia_economy | 746 | same as above | 1.0 | 0.95 | 0.9% | 739 (**= all 746 rows minus 7**, strict subset of `economist_tunisia_all`) | same | **drop / merge** — this file is a redundant derived subset (see §4) |
| guardian_tunisia | 1,204 | 2004-10 → 2026-04, usable window unstable (70 zero-months, many scattered gaps — this is a low-volume topical search, not a full feed) | 1.0 | 0.91 | 0.7% | 0 | date-only, 0% missing | **keep with a shorter/most-recent window**, treat as sparse/topical supplement, not a feed |
| ilboursa | 26,596 | 2014-01 → 2026-09 | 7.0 | 0.17 | 2.1% | 142 | date-only **but the scraper bug below throws away real time-of-day and the article URL that it already parses** | **keep, fix scraper first** |
| kapitalis | 18,000 | 2015-05 → 2026-09 | 4.0 | 0.05 | 0.2% | 49 | date-only, 0% missing | **keep** |
| lapresse | 6,741 | 2019-04 → 2026-09, usable window starts **2024-12** (7 zero-months before that, then a cliff/recovery pattern) | 2.0 | 0.51 | 10.7% | 26 | date-only, 0% missing | **keep, shorter window (≥2024-12)** |
| leconomistmaghrebin | 24,698 | 2013-12 → 2026-09 | 5.0 | 0.09 | 0.4% | 125 | date-only, 0% missing | **keep** |
| nyt_economy | 1,100 | 2024-10 → 2026-09 | 2.0 | 0.33 | 2.0% | 0 | date-only, 0% missing | **keep** (short by construction — recent topical scrape) |
| tap | 283 | 2026-08-16 → 2026-09-15 (**one month**) | 9.0 | 0.0 | 0% | 0 | date-only, 0% missing | **keep, but window is far too short to be useful as a wire-reprint reference** — re-scrape with a wider historical range (the scraper has no date-paging logic, unlike e.g. `scrape_ilboursa.py`) |

\* Calendar-day (not trading-day) share of days with zero articles within each source's own
parsed min–max date range. This is a proxy pending the BVMT trading calendar (see §0.3) — it
is **not** the plan's trading-day-based metric.

Full numbers: `completeness.csv`, `coverage_summary.csv`, `coverage_by_month.csv`,
`coverage_by_month.png` (log-scale, one line per source — gaps/cliffs/spikes are visible
directly), `duplicate_summary.csv`.

## 3. Scraper defects found (fix + re-extract, do not hand-patch)

1. **`scrape_ilboursa.py` drops real data it already has.** `parse_rows()` parses a full
   `datetime` (`%d/%m/%Y %H:%M`) and captures the article `href`, but `main()` only writes
   `[headline, dt.strftime("%Y-%m-%d")]` to the CSV — the time-of-day and the URL are computed
   and then thrown away. This is the single highest-value fix: re-extracting adds real
   timestamps and enables URL-based dedup for this source at zero extra scraping cost.
2. **`scrape_assabah.py`'s date field is not always a real date.** 195 rows (0.35%) are
   unresolvable relative strings (`"منذ 9 ساعات"` / `"منذ أسبوعين"`, i.e. "N hours/weeks ago")
   with no captured scrape timestamp to resolve them against. The other 99.65% are absolute
   Arabic-month-name dates and parse fine. Fix: capture an absolute date at scrape time (the
   site clearly has one, since older items already render as absolute dates) instead of
   relative text.
3. **`scrape_assabah.py` sometimes captures a section/rubric label instead of an article
   headline.** The single most common "headline" value is `مختصرات` ("briefs" — a column
   name), appearing **1,252 times** (2.2% of all rows). Other repeats — `أخبار الحوادث`,
   `شريط الأحداث`, `خارج الحدود`, `متفرقات`, `دليل حقوقك` — are the same pattern: recurring
   section titles, not per-article headlines. See `repeated_headlines.csv`.
4. **`lapresse` shows the identical pattern**: top repeated "headlines" are rubric/column names
   — `Express` (251×), `Des faits et des chiffres` (135×), `Ils ont dit` (77×),
   `Kiosque international` (67×), `EXPRESS` (47×, inconsistent casing with the first) — not
   real article titles. Its scraper script isn't in the repo to fix directly (see §1), but
   whatever produced this file has the same section-label-instead-of-headline bug.
5. **No scraper script for `lapresse`**, and **no scraper was under git version control at
   all** before this commit — both are reproducibility gaps, not data-quality bugs per se.

`ilboursa`'s and `leconomistmaghrebin`'s own repeated headlines (`repeated_headlines.csv`) look
like genuinely recurring generic titles for real weekly recap articles (e.g. "La Bourse de
Tunis débute la semaine dans le rouge") rather than extraction bugs — lower priority.

## 4. `economist_tunisia_economy` is a redundant subset

Checked directly: every one of `economist_tunisia_economy`'s 746 rows (740 with non-empty
headlines) is byte-identical to a row in `economist_tunisia_all`. It is a topic-filtered view
of the same scrape, not an independent source. Recommendation: stop producing this file; if an
"economy-only" view is needed downstream, derive it with a keyword/topic filter over
`economist_tunisia_all` instead of a second scrape that silently duplicates 100% of its rows
into the corpus.

## 5. Duplicates and wire reprints (step 7)

- **Exact duplicates** (`hash(headline)`): within-source rates range from 0% (tap) to 10.7%
  (lapresse, driven by the rubric-label bug in §3.4) — see `duplicate_summary.csv`.
- **Cross-source exact duplicates** (`exact_cross_source_duplicates.csv`): real overlap exists
  between the Tunisian French-language outlets — `ilboursa`↔`leconomistmaghrebin` (80
  identical headlines), `ilboursa`↔`kapitalis` (28), `kapitalis`↔`leconomistmaghrebin` (15),
  `lapresse`↔`leconomistmaghrebin` (12), `ilboursa`↔`lapresse` (12) — consistent with wire
  syndication (verbatim headline + presumably verbatim body) that isn't attributable to our
  `tap` scrape specifically, since `tap` only covers one month (§2).
- **MinHash near-duplicate / TAP-reprint check** (`tap_reprint_share.csv`): ran on headline
  text only (word-3-gram shingles, 64 permutations, Jaccard ≥0.8) restricted to the French
  Tunisian outlets vs. `tap` within a ±2-day window (English sources and `assabah`'s Arabic
  script excluded as structurally unmatchable against `tap`'s French-only portal). **Result:
  0% matches for every source**, despite the exact-hash check above finding real cross-source
  duplication. Conclusion: **headline-only near-dup detection is too weak for wire-reprint
  measurement** — outlets that reprint the same wire body frequently write their own headline
  for it, so body text is needed for this check to mean anything. This step should be re-run
  once article bodies are captured (see §0.1); until then, the exact-hash cross-source numbers
  above are the more reliable (if narrower) signal.

## 6. Language mix (automatic, step 8)

`langdetect` run over the full corpus per source (`language_mix_full_corpus.csv`); a 25-row/
source manual-labeling template with the automatic guess attached is in
`language_relevance_sample.csv` — **the `*_fill_by_hand` columns are intentionally blank and
need your (or a human labeler's) judgment**, per the plan's own point that this comparison
should be human vs. automatic, not automatic vs. itself.

| source | dominant auto-detected language | notes |
|---|---|---|
| assabah | 99.6% Arabic | `langdetect` can't distinguish MSA / Tunisian dialect / Arabizi — that distinction needs the manual pass |
| tap, ilboursa, kapitalis, leconomistmaghrebin | 93–100% French | consistent with francophone Tunisian business press |
| lapresse | 93.4% French, 3.9% "ca" (Catalan) | the Catalan tag is very likely a `langdetect` misfire on short French headlines — check in the manual pass |
| economist_tunisia_all/economy, guardian, nyt | 88–99% English | expected |

## 7. Spot-check worksheet (step 5)

`spot_check_worksheet.csv` has 10 random rows per source for manual comparison against the
live site (stored date vs. displayed date, and — since there's no URL for most sources — you'll
need to search the headline on-site). This step needs a human with a browser; not automated
here.

## 8. Decisions (step 9)

- **Overlap window / go-no-go (market data)**: **BLOCKED.** No BVMT data in the repo. Cannot
  compute the training/test-block overlap window until a market-data source is identified and
  ingested (see §0.3). This is the biggest open item before any modeling work can start.
- **Thin-source rule** ("drop/merge if >50% of trading days have no article" — proxied here by
  calendar days, pending the real trading calendar): by the calendar-day proxy, no source
  exceeds 50% zero-days except none currently — `economist_tunisia_all` (93%) and
  `economist_tunisia_economy` (95%) look alarming but are **explained by 1 article/active-day
  median over a 29-year range**, i.e. this is a low-volume topical feed, not a broken scraper;
  same shape for `guardian_tunisia` (91%) and `nyt_economy` (33%, expected — a short recent
  window). None of these should be dropped on this rule; they should just be scoped to their
  usable windows (§2).
- **Source registry update**: mark `economist_tunisia_economy` as `drop` (see §4), `tap` as
  `keep — re-scrape with wider date range before use`, `lapresse` as
  `keep — scraper script missing, headline-extraction bug to fix` (§3.4), all others `keep`
  with the usable windows in §2's table.
- **Tag**: after this report and its outputs are committed, tag the snapshot `raw-v1` per the
  plan (done as part of this same commit — see the commit message for the tag).

## 9. Explicit blocked/skipped items (for transparency)

- Step 6 (BVMT missing sessions, high<low checks, stale-price runs, 20-date cross-check):
  **not run** — no market data file exists in the repo.
- Trading-day-based zero-news-day share and the overlap-window computation in step 9: **not
  run** — same reason; calendar-day proxies were substituted and labeled as such throughout.
- Live-site spot checks (step 5) and hand-labeling (step 8): **template generated, not filled
  in** — these require a human with a browser and judgment calls the audit shouldn't fabricate.
