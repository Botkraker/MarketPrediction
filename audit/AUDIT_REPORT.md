# Data Audit Report — raw-v1

**Revision 2026-09-30:** §8i added — the fine-tuned CamemBERT and the translation step
now have producing scripts and reproduce the committed outputs within seed noise; the
2016–18 sentiment is shown to come from under-trained yearly models; a per-headline
English series is built; the first per-outlet H2 run is reported; and seven data
defects found by the time-series EDA are recorded. §8h's provenance warning is
narrowed accordingly (see the note at its head).

**Revision 2026-09-28:** §8h added — gold set v2, three-annotator agreement with a
150-row human anchor, the sentiment-instrument bake-off, three macro controls, and the
H1 re-run. §8h is flagged **PROVISIONAL**: its artifacts arrived as a data-only commit
with no producing script, so it is the one section here that `build_audit.py` cannot
regenerate.

**Revision 2026-09-20:** §6b added (wrong-country check — `assabah` is Moroccan, excluded);
§8 and §9 corrected, then §8b added once step 6 was actually implemented and run. Everything else is the original raw-v1 text.

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

1. **`scrape_ilboursa.py` drops real data it already has.** — **FIXED 2026-09-20 (re-scrape pending).**
   `parse_rows()` returned `(href, headline, dt)` and `collect()` keyed the dict by
   `href`, but `articles.values()` discarded the key and `main()` wrote only
   `[headline, date]`. The scraper now writes `headline;date;published_at;url`;
   `headline` and `date` stay first and unchanged so `preprocessing/io_raw.py` and
   `audit/build_audit.py` are unaffected (both select columns by name). Guarded by
   `preprocessing/test_scrape_ilboursa.py`.
   **The 26,596 rows currently in `data/raw/ilboursa_headlines.csv` are still
   date-only** — the fix takes effect on the next extraction. Until then
   `features.py` must keep the conservative rule (news dated D predicts sessions
   strictly after D). Once re-scraped, ilboursa headlines can be split around the
   ~14:10 Tunis close and aligned to the same session, recovering intraday signal
   on 35% of the corpus. Original finding follows. `parse_rows()` parses a full
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

## 6b. Topical geography — wrong-country check (added 2026-09-20)

Added after `preprocessing/relevance.py` showed `assabah` losing 99.87% of its rows to the
relevance filter. The cause was not the filter: **`scrape_assabah.py` targets the wrong
country.** Output: `topical_geography.csv` (per-source, regenerated on every audit run).

`BASE_URL` in `scrape_assabah.py:19` is `https://assabah.ma/category/%D8%AD%D9%88%D8%A7%D8%AF%D8%AB/`
— `.ma` is Morocco's ccTLD, and the category decodes to `حوادث` ("accidents/crime").
The intended source is Tunisia's `assabah.com.tn`. The masthead الصباح / "Assabah" belongs to
a well-known daily in **both** countries, which is how the wrong domain was reached.

| source | n | tunisia share | top other country | its share | flagged |
|---|---:|---:|---|---:|---|
| **assabah** | 56,292 | **0.0005** | morocco | **0.2107** | **True** |
| economist_tunisia_all | 1,197 | 0.0326 | morocco | 0.0017 | False |
| economist_tunisia_economy | 746 | 0.0241 | morocco | 0.0013 | False |
| guardian_tunisia | 1,204 | 0.6030 | algeria | 0.0058 | False |
| ilboursa | 26,596 | 0.1891 | morocco | 0.0459 | False |
| kapitalis | 18,000 | 0.5189 | morocco | 0.0130 | False |
| lapresse | 6,741 | 0.4142 | morocco | 0.0178 | False |
| leconomistmaghrebin | 24,698 | 0.2797 | morocco | 0.0128 | False |
| nyt_economy | 1,100 | 0.0000 | morocco | 0.0000 | False |
| tap | 283 | 0.1378 | morocco | 0.0106 | False |

Flag rule: `top_other_n > tunisia_n AND top_other_n >= 0.05 * n`. Both conditions are required
so that a source is not flagged on incidental foreign coverage. Exactly one source fires.

**Corroborating markers** (counts over all 56,292 `assabah` headlines). Morocco uses the
dirham, Tunisia the dinar:

| marker | hits |
|---|---:|
| درهم (dirham, MA currency) | 175 |
| دينار (dinar, TN currency) | **0** |
| مراكش (Marrakech) | 1,621 |
| طنجة (Tangier) | 1,630 |
| أكادير (Agadir) | 546 |
| صفاقس (Sfax, TN 2nd city) | **0** |
| قرطاج (Carthage) | **0** |
| بنزرت (Bizerte) | **0** |
| الملك ("the King") | 239 |

Zero dinar mentions, zero Sfax/Carthage/Bizerte, and 239 references to a monarch in a
republic that has had none since 1957. The `langdetect` pass in §6 reported 99.6% Arabic for
this source and passed it — **language is not provenance**, which is precisely the gap this
check closes.

### Known blind spots of this check

- **Domestic sources under-report their own country.** `ilboursa` shows 18.9% Tunisia against
  4.6% Morocco (1,220 headlines). This is *not* a second wrong-country case: a Tunisian
  financial outlet rarely names Tunisia in a headline while routinely covering Maghreb
  markets. The 5% floor plus the dominance condition is what keeps it unflagged.
- **Uninformative for non-geographic sources.** `nyt_economy` scores 0.0 on every country. It
  is the `global_linked` source, relevant via Fed / oil / ECB keywords, never via geography.
  Read its row as "not applicable", not as "clean".
- Gazetteer matching is substring-based and coarse by design. It answers "is this the wrong
  country entirely", not "is this on-topic".

### Action taken

`assabah` is excluded from the preprocessing pipeline by removing its key from
`config.SOURCE_WINDOWS` (`clean.py` iterates that dict). **Nothing was deleted**:
`data/raw/assabah_headlines.csv`, `scrape_assabah.py` and `io_raw.SOURCES` are all retained so
this audit stays reproducible and this section's evidence can be regenerated. `funnel.csv` now
records `assabah, raw=56292, cleaned=0, relevant=0, canonical=0` — an explicit exclusion, not a
silent absence. Canonical corpus: 42,713 → **42,645** (−68; no other source changed).

Re-add the `SOURCE_WINDOWS` key once `scrape_assabah.py` is retargeted to `assabah.com.tn`.
The Arabic relative-date parsing (`منذ 9 ساعات`) is already written and reusable.

## 7. Spot-check worksheet (step 5)

`spot_check_worksheet.csv` has 10 random rows per source for manual comparison against the
live site (stored date vs. displayed date, and — since there's no URL for most sources — you'll
need to search the headline on-site). This step needs a human with a browser; not automated
here.

## 8. Decisions (step 9)

- **Overlap window / go-no-go (market data)**: ~~**BLOCKED.**~~ **SUPERSEDED 2026-09-20.**
  BVMT data has since been added to the repo (commits `690dc1f`, `2ae3352`) and *is* inventoried
  by this audit: `bvmt/ALL_DATA.csv` (187,987 OHLCV rows), `bvmt/tunindex_2010_today.csv`
  (4,167 sessions, 2010→2026), plus ticker and market-cap tables. See `inventory.csv`.
  Two follow-ups remain, both in `build_audit.py`, neither yet fixed:
  1. **Step 6 is hardcoded to skip** (`build_audit.py:313` prints "no market data file in repo"
     unconditionally). Missing sessions, high<low and stale-price checks are therefore still
     uncomputed even though the inputs are present. The module docstring (lines 6-7) repeats
     the same stale claim.
  2. `BVMT_FILES` inventory rows hardcode `scraper_script = "MISSING (no scraper for BVMT data
     in repo)"` (`build_audit.py:160`) although `scrape_tunindex.py` exists in the repo.
  The trading calendar for news alignment should be derived from `tunindex_2010_today.csv`.
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

## 8b. Market data — step 6, now computed (added 2026-09-20)

Step 6 previously printed "SKIPPED — no market data file in repo" unconditionally, long after
the BVMT files landed. Fixed: it now runs whenever `bvmt/tunindex_2010_today.csv` is present.
Outputs: `market_data_quality.csv`, `trading_calendar.csv`, `market_missing_weekdays.csv`.

| check | value | reading |
|---|---:|---|
| sessions | 4,167 | 2010-01-04 → 2026-09-16 |
| unparseable dates | 0 | clean |
| duplicate sessions | 0 | clean |
| weekdays absent in range | 191 / 4,358 (4.4%) | ≈11/yr — consistent with Tunisian public holidays, **not scrape gaps** |
| high < low | 0 | clean |
| close outside [low, high] | 0 | clean |
| **open outside [low, high]** | **69** | **see below** |
| non-positive close | 0 | clean |
| zero-volume sessions | 56 | flagged, not investigated |
| stale close runs | 20 | consecutive identical closes |
| null OHLCV cells | 0 | clean |

### The `open` column is not an opening price

**1,384 of 4,167 rows (33.2%) have `open` exactly equal to the previous session's `close`**,
and 68 of the 69 range violations are exactly those rows. Example:

```
2010-01-04  open 4291.72  high 4320.55  low 4295.02  close 4320.31   <- open < low
2010-01-05  open 4320.31  ...                                        <- = 2010-01-04 close
2010-01-06  open 4404.72  ...                        close 4404.72   <- = 2010-01-05 close
```

The 69 out-of-range cases are only the detectable tip; the other ~1,315 carried-forward opens
fall inside `[low, high]` by coincidence and are invisible to a range check.

**Consequence for feature engineering:** an intraday return computed as `(close - open)/open`
silently becomes a close-to-close return for a third of sessions — two different quantities in
one column. **Use close-to-close returns.** Do not use `open` as a price feature without
establishing its provenance in the upstream source.

### Trading calendar

`trading_calendar.csv` is the 4,167-session list. News→trading-day alignment must map each
headline to the next session *in this file*, not to the next weekday — 191 weekdays are not
sessions. Headlines on non-session days accumulate to the following session.

## 8c. Corpus/market characteristics that constrain modelling (added 2026-09-20)

Measured while building `preprocessing/features.py` and `preprocessing/baseline.py`.
These are data properties, not results, but each one forecloses a modelling choice.

### Usable study window starts 2014, not 2010

Median relevant headlines per covered day, by year: **1** for 2010-2013, then 6-15
from 2014 onward. A daily news feature built on one headline per day is noise.
`features.py` therefore defaults to `--start 2014-01-01`: 3,179 sessions,
**100% of them with at least one headline**, median 12/session.

Separately, the 2,932-row sentiment gold set spans 2,058 distinct days at a median
of **1 labelled headline per day**. It is stratified for classifier training
(source x language x relevance x year), NOT for time-series construction, and must
not be used to build daily sentiment. Daily features require scoring the full
42,645-headline corpus with a trained model.

### News volume does not move trading volume

| relationship | r | verdict |
|---|---:|---|
| news volume vs \|return\| | +0.095 *** | as expected |
| news volume vs high-low range | +0.140 *** | as expected |
| **news volume vs trading volume** | **+0.018 ns** | **contradicts the developed-market norm** |

Checked and not explained by: the 55 zero-volume sessions (excluding them gives
r=+0.007), or by source mix (all eight sources are individually flat against
volume, including `ilboursa`, the dedicated financial outlet). BVMT volume has
skew **12.5**, with the top 1% of sessions carrying 9.3% of all volume — it is
driven by episodic block trades, not by news-reading flow. **Volume-based
features are not advisable on this market.**

### Return autocorrelation IS directionally exploitable (corrected 2026-09-20)

> **CORRECTION.** This section previously stated the opposite, on the basis of a
> lag-offset bug in `features.py`. Row *i* is session *i* and the target
> `ret_next[i]` is session *i+1*, but lags were counted from the ROW, so
> `ret_lag1[i] = ret[i-1]`. The most recent AVAILABLE return — `ret[i]`, known the
> moment session *i* closes — was in no feature set. The baseline was predicting
> session *i+1* from data ending at *i-1*. Corrected by adding `ret_lag0`; lags are
> now counted from the target and a regression test enforces it
> (`test_features.py::test_most_recent_available_return_is_exposed_as_ret_lag0`).

| feature set | accuracy | OOS R² | McNemar vs always-up |
|---|---:|---:|---:|
| `ret_lag1..3` (buggy) | 0.5564 | +0.0068 | p = 0.418 **ns** |
| `ret_lag0..2` (correct) | **0.5736** | **+0.0645** | **p = 0.028 SIG** |

`corr(ret_t-1, ret_t) = +0.263` and `corr(|ret|, |ret|_{t-1}) = +0.387`, both far
above a liquid-market norm, consistent with thin trading and partial adjustment
(only 20 stale closes, so not an artifact). Unlike the earlier claim, this **does**
translate into directional predictability.

### Superseded text follows (retained for the record)

### Return autocorrelation is real but not directionally exploitable

`corr(ret_t-1, ret_t) = +0.263` and `corr(|ret|_t-1, |ret|_t) = +0.387` — both
far above a liquid-market norm of ~0, and consistent with thin trading and partial
adjustment (only 20 stale closes, so not an artifact).

It does **not** translate into directional predictability. Walk-forward
out-of-sample R^2 on `ret_lag1..3` is **+0.0068** (in-sample +0.0135), and the
best directional accuracy is 0.5564 against an always-up constant of 0.5493 —
McNemar **p = 0.418**. Autocorrelation on the return *level* is swamped by noise
once reduced to a *sign*.

### The pre-registered bar for H1

Always-up constant: **0.5493** over 2,678 walk-forward predictions. No price-only
feature set beats it significantly. Established before any sentiment score exists,
so the H1 comparison is against a bar fixed in advance.

Two incidental findings: regressing the return and taking its sign beats
classifying direction in every feature set (classify ~0.545 vs regress ~0.556),
because a logistic fit on a drift-dominated binary label collapses to "always up"
(it predicted up 77-92% of the time). And adding news *counts* to the price
features makes accuracy slightly worse (0.5538 vs 0.5564) — counting headlines
without reading them carries no signal, which is the control H1 needs.

## 8d. Two threats to the validity of any sentiment result (added 2026-09-20)

Both found by inspecting the LLM annotations and the corpus. Neither is a data-quality
defect in the ordinary sense; both would invalidate an H1 claim if left uncontrolled.

### 1. The annotation prompt does not define the task (construct validity)

The system prompt in `llm_annotate.py` is, in full: a one-line role statement plus
four lines of JSON formatting rules. It never defines what `positive` means, never
says when to use `neutral`, never anchors sentiment to *market impact* rather than
tone, never explains the `relevance_tag` it passes in, and gives no examples to
anchor a five-point ordinal scale.

The resulting labels measure vocabulary, not expected market effect:

| headline | label | model's stated reason |
|---|---|---|
| "un code de l'environnement est **en cours d'élaboration**" | positive | "development of environmental regulations suggests progress" |
| "Les billets et monnaies en circulation **dépassent** 24 milliards de dinars" | positive | restates the headline |
| "Quatre priorités politiques clés en matière d'investissement" | positive | "highlights economic focus" |

A draft regulation scores positive; currency in circulation rising scores positive
(it is neutral at best, inflationary at worst). `neutral` is used almost only for
off-topic items (including a **weather forecast**), questions, and literal
"unchanged" reports — i.e. it means "no sentiment word found", not "market-neutral".

Distribution: positive 58.8%, neutral 9.4%. For financial headlines neutral should
dominate. **Consequence:** a null H1 would be uninterpretable — indistinguishable
from "we measured the wrong construct".

**Action taken.** `llm_annotate.py` now carries versioned prompts. `PROMPT_V1` is kept
byte-for-byte (a test asserts it) so the original 3,000 labels remain attributable;
`PROMPT_V2` is the default for new runs. v2 states the question as expected market
impact on Tunisian listed companies, makes `neutral` the explicit default with the
reasoning that "a label of neutral is a correct and informative answer, not a failure
to decide", defines all five labels by mechanism, names the observed traps (rising
inflation/debt/money supply despite the word *hausse*; draft laws; `inchangé`;
questions; topic-is-not-sentiment), and carries ten worked examples drawn from the
corpus but **excluded from the gold set** so they cannot leak. Every label now records
`annotator_N_prompt`.

**v1 and v2 labels are not comparable and must never be pooled.** The 3,000 existing
labels are v1. Re-annotation under v2 produces a separate annotator column.

**Open question for the smoke test:** v1 gave a 9.4% neutral share. If v2 does not move
that substantially upward on a 50-row trial, the limit is model capability, not prompt
wording, and the remedy is a larger model rather than more instruction.

### 2. Price-report headlines launder momentum into the sentiment channel

**4,244 of 42,645 relevant headlines (10.0%)** report the index's own move:
"Le Tunindex termine sur une note stable (+0,08%)". Concentrated in the three
largest sources — ilboursa 11.9%, kapitalis 11.2%, leconomistmaghrebin 9.3%.

Measured on the 2,356 such headlines carrying an explicit direction word:

| check | r | sign agreement |
|---|---:|---:|
| direction word vs **that day's** return | **+0.519** (p=2e-110) | **84.3%** |
| same feature vs the **next session's** return | +0.133 (p=5e-08) | 56.4% |

These headlines *are* the return, in words. Mapped forward one session as a
feature they still predict at r=0.133 — because return autocorrelation is +0.263
(section 8c). For scale:

| predictor | directional accuracy |
|---|---:|
| always-up constant | 0.5493 |
| best price-only model | 0.5564 |
| **yesterday's price report read as text** | **0.5640** |

> **CORRECTION 2026-09-20.** The comparison above used the buggy baseline
> (§8c). With `ret_lag0` restored the momentum model reaches **0.5736** and beats
> the price-report feature (0.5654 under the tightened v2 regex). The dramatic
> framing — "a no-news feature beats the momentum model" — was an artifact of a
> handicapped control.
>
> **The confound itself is still real and still needs controlling.** Under the v2
> pattern, price-report direction words match the same-day return at **r = +0.629,
> 92.9% sign agreement** (up from 0.519/84.3%, because the pattern is now more
> precise). Sentiment built on them re-encodes `ret_t`, which is exactly what
> `ret_lag0` now controls for. The three-arm design stands; only the motivating
> number changed.

An uncontrolled sentiment model could still report "sentiment improves prediction"
while measuring autocorrelation, which is why the placebo arm exists.

**Control, not removal.** A market report is real news; dropping it by default
would be an unjustified editorial choice. `config.PRICE_REPORT_PATTERN` flags them
and `features.py` emits `n_price_reports`, `n_non_price` and `price_report_share`.
H1 must be reported three ways: all headlines, excluding price reports, and price
reports only — the last as a placebo. If sentiment only works with them included,
it is momentum.

## 8e. Adversarial review, 2026-09-20 — findings and dispositions

An independent adversarial review of the modelling chain was commissioned and its
claims were independently recomputed before being accepted. Two documented findings
were overturned by it; two of its own claims did not survive recomputation. All
numbers below were recomputed from the repo's data.

| # | finding | status |
|---|---|---|
| R1 | Lag-offset bug: the most recent available return was in no feature set | **CONFIRMED — fixed.** §8c, §8d rewritten |
| R2 | Sentiment classifier fitted on a random split spanning 2005-2026 | **CONFIRMED — fixed** |
| R3a | No inter-annotator agreement exists | **CONFIRMED — still open** (needs LM Studio) |
| R3b | Label scale is "not monotone" against returns | **PARTLY REJECTED** — see below |
| R3c | Classifier not distinguishable from a constant | **CONFIRMED — still open** |
| S1 | `PRICE_REPORT_PATTERN` mis-specified; a test certified the bug | **CONFIRMED — fixed** |
| S2 | Placebo arm has no data on ~27% of sessions; harness would crash | **CONFIRMED — fixed** |
| S3 | Underpowered; no power analysis | **CONFIRMED — fixed** (MDE now reported) |
| S4 | Multiple comparisons uncorrected | **CONFIRMED — fixed** (Holm) |
| S5 | No Arabic in the study | **CONFIRMED** — framing corrected |
| S6 | Relevance filter unvalidated, wrong-country false positives | **CONFIRMED — fixed**, plus a larger false-negative mode the review missed |
| M3 | Placebo necessary but not sufficient | **CONFIRMED — fixed** (orthogonalised arm) |
| M6 | H2 would rank coverage length, not contribution | **CONFIRMED — fixed** |

### R3b — partly rejected

The review reported `neutral > positive > very_positive` on **same-day** returns and
concluded the ordinal scale is non-monotone. Recomputed against the **next-session**
return (what H1 actually predicts):

```
very_negative  -4.6 bp  n=86      negative  -1.1 bp  n=668
neutral        +3.6 bp  n=273     positive  +6.1 bp  n=1724
very_positive  +2.5 bp  n=176
```

Monotone for four of five levels; only `very_positive` inverts. That inversion is
**not significant**: Welch p = 0.402, bootstrap 95% CI on (very_positive − positive)
= [−12.3, +4.5] bp, which includes zero. With n=176 the extreme is unpowered, not
inverted. The equal-spacing assumption in `score_corpus.LABEL_SCORE` is therefore
**unvalidated, not contradicted** — a weaker but still real criticism.

What does survive: row-level Spearman(label, next return) = **+0.0517** (p=0.005),
and collapsing to three classes gives **+0.0544**. The five-point scale is not
earning its granularity. `score_corpus.monotonicity_check()` now reports per-label
counts and flags under-populated levels.

### S3 — the design cannot detect the effect it is looking for

Computed by `hypothesis_tests.minimum_detectable_effect()` on the real run:

```
discordant pairs            853
SE of accuracy difference   0.0109
MDE at 80% power            3.06 pp
typical published effect    1-2 pp
adequately_powered          False
```

**"H1 NOT SUPPORTED" is the guaranteed outcome of this design whether or not H1 is
true.** Any null result must be reported as *inconclusive*, not as evidence of
absence. A continuous-outcome test (OOS R² / Diebold-Mariano on returns rather than
signs) would use the magnitude information the sign test discards and should be
added before any claim is made.

### S6 — fixed on both sides, and the larger side was missed

*False positives.* Generic keywords (`inflation`, `croissance`, `growth`, `gdp`)
matched foreign-country headlines: "L'inflation au Maroc recule de 0,6%", "Why
Indians are unhappy about 7.8% economic growth". A country-negation rule
(`preprocessing/relevance_geo.py`) now blocks a generic match when a headline names
a foreign country and no Tunisian entity. Wrong-country rate inside `tunisia_econ`:
**2.80% → 0.00%**; 1,111 rows retagged.

*False negatives — larger, and not in the review.* The keyword lists are TOPIC
words, so a headline about a listed company carries none of them. **7,215 rows
naming a BVMT issuer were being discarded as `other`** — "Le bénéfice net de Land'Or
bondit de 80 %", "MPBS : Le bénéfice semestriel recule de 4%", "Carthage Cement
confirme la consolidation". These are the most index-relevant headlines in the
corpus. `config.KEYWORDS_ISSUERS` now loads the 79 issuer names (≥4 chars) from
`data/raw/bvmt/sotcks_list.csv`, so the list cannot drift from the actual
constituents. Short tickers (AB, BT, CC, SAH) are excluded — they match inside
ordinary words. Country negation still applies on top.

Net corpus effect: relevant 43,184 → **47,784**; canonical **42,645 → 46,013**
(−789 wrong-country, ~+4,400 issuer recoveries, −1,185 absorbed by the template-key
dedup in M5).

Still unvalidated: 3,186 rows (8.25%) match a generic keyword and name no country at
all. Only hand labels settle those. `preprocessing/relevance_validation.py worksheet`
generates a stratified sheet; `audit/language_relevance_sample.csv` remains unfilled.

### S5 — there is no Arabic in this study

`data/curated/03_dedup.parquet` relevant canonical rows: **fr 41,597 / en 1,048 /
ar 0.** The only Arabic source was `assabah`, correctly excluded as Moroccan (§6b),
which also removed all 68 Arabic gold rows. "French/Arabic news sentiment" is not
supportable; the corpus is French with ~2.4% English.

Related: `lang` is **assigned per source** from `config.SOURCE_LANG`
(`relevance.py`), not detected per headline. Every "by language" figure is therefore
a by-source-group figure wearing a language label, and should be described that way.

### M5 — near-duplicate leakage: fixed, but it is NOT the cause of the overfit gap

`dedup.py` clustered by SHA-1 of the raw normalised headline, so template-driven
financial headlines differing only in a number or date formed separate clusters and
could land on opposite sides of the split.

Fixed by hashing a **template key** instead: numbers and percentages masked to `#`
(with the sign preserved), French weekday/month words masked to `@`, punctuation
folded. Deterministic, no threshold, no similarity pass.

Measured on the 47,784-row relevant corpus: clusters **47,198 → 46,013** (−2.51%),
806 merged groups absorbing 1,991 exact clusters, and **2,183 near-identical pairs
that exact hashing had placed in different clusters** are now together. ilboursa is
worst affected at 6.71% of rows, as expected for a template-driven outlet.

**Two merges were deliberately rejected.** A token-sorted key would have collapsed
100 further groups, 98 benign — but two were semantic opposites:

```
"Fitch révise la perspective de la Tunisie de négative à stable"
"Fitch révise la perspective de la Tunisie de stable à négative"

"le dinar s'apprécie vis-à-vis du dollar et se déprécie face à l'euro"
"le dinar se déprécie vis-à-vis du dollar et s'apprécie face à l'euro"
```

A 2% error rate landing exactly on the sentiment-bearing cases is not worth 100
extra merges, so clause order stays significant. The same reasoning preserves the
`+`/`-` sign: stripping it would have merged `à -0,36%` with `à +0,21%`.

**Correcting the premise.** The review offered this as "a plausible contributor to
the 0.893 train / 0.6059 validation gap". Measured against the frozen split, it is
not: **1 of 439 validation rows (0.23%) and 0 of 979 evaluation rows** share a
cluster with a training row, because `gold.py` already samples one row per exact
cluster. M5 moves validation accuracy by at most 0.23pp of a 28.7pp gap.

The overfit is a **capacity/sample-size problem**, not leakage: 1,514 training rows
against a 60k-feature char-ngram space over 5 ordinal classes, held-out accuracy
1.6pp above the majority floor, macro-F1 0.464. M5 should be reported as a
correctness guard — and as a prerequisite for any gold-set expansion, where
collisions grow roughly quadratically — not as the explanation.

### M1 — pre-registration: timestamp fixed, estimator choice still tainted

`hypothesis_tests.py` claims in its docstring to have been written before any
sentiment score existed. **Partly resolved 2026-09-20:** the harness is committed at
`7ed9c0c` and tagged **`prereg-h1-v1`**, so the claim now rests on git history rather
than a file mtime. That is verifiable by an editor.

What is **not** resolved: Worse, `kind="regress"` is the default *because* §8c had
already compared regress against classify across 8 configurations on the same data
H1 is tested on. **The estimator was selected on the outcome.** `min_train=500` and
`refit_every=20` are likewise unjustified free parameters that set the test window.
Disposition: the tag fixes the timestamp, but `kind="regress"` remains
outcome-selected and must be declared as such in the paper — or re-run with
`kind` fixed in advance on data not used for the comparison. `min_train` and
`refit_every` still need a sensitivity analysis.

### M1 follow-up — the momentum result is NOT robust to the test-window start

`preprocessing/sensitivity.py` grids `min_train` × `refit_every` (16 configurations,
`sensitivity_results.json`).

**Accuracy is stable; significance is not.**

| min_train | n | accuracy | constant | lift | McNemar |
|---:|---:|---:|---:|---:|---:|
| 300 | 2,878 | 0.5716 | 0.5431 | +0.0285 | **0.008** |
| 500 | 2,678 | 0.5736 | 0.5493 | +0.0243 | **0.028** |
| 750 | 2,428 | 0.5725 | 0.5540 | +0.0185 | 0.111 |
| 1000 | 2,178 | 0.5721 | 0.5579 | +0.0142 | 0.249 |

Accuracy varies only over 0.5712–0.5750 across all 16 settings and `refit_every`
barely matters (≤0.003 within a `min_train` level). But the result is significant in
**8 of 16 configurations, all at `min_train ≤ 500`**.

The mechanism is in the `constant` column: a later start captures more of the 2025
bull run, so the always-up baseline climbs 0.5410 → 0.5579 while model accuracy
holds. The lift is eaten by the drift, not by the model getting worse.

**This qualifies §8c.** "Return autocorrelation is directionally exploitable
(p = 0.028)" is true at `min_train=500` and false at 750. The honest statement is:
the momentum edge is real in magnitude and stable in accuracy, but its statistical
significance depends on where the test window starts, because the constant baseline
is non-stationary. Report the grid, not the single favourable cell.

This is also why the continuous test matters — OOS R² rises monotonically with
`min_train` (0.058 → 0.068) even as the sign test loses significance, which is the
sign test discarding magnitude exactly as §8e/S3 describes.

### M4 — the gold sample does not match the corpus it scores

`gold.py` takes exactly one row per `source × relevance_tag × year` stratum before
filling at random, which over-weights thin source-years (tap: 6 rows; economist: 15)
relative to their corpus share. The classifier's training prior therefore does not
match the corpus prior, and a daily `sent_mean` is partly driven by *which sources
published that day*. Additionally `split.py` stratifies the evaluation split by
`adjudicated_label`, guaranteeing the held-out label distribution matches training by
construction — so 0.6059 is optimistic relative to deployment.

## 8f. Annotation pilot: the first inter-annotator agreement (2026-09-20)

R3a said no reliability estimate exists and none could be produced without a second
annotator. That was treated as blocked on a local LM Studio server, which turns out
not to be installed on this machine at all. The block was wrong in a simpler way:
**an LLM was available the whole time.** Annotator 2 below is Claude (Opus 5), a
different model family from qwen2.5-7b, which is what a cross-family agreement
statistic requires.

60 headlines, stratified at 12 per annotator-1 label. Artifacts:
`data/curated/annotation_pilot_v2.csv`, `..._agreement.json`.

| metric | value |
|---|---:|
| raw agreement | 48.3% |
| Fleiss κ (nominal) | 0.328 — "fair" |
| **quadratic-weighted Cohen κ** | **0.659 — "substantial"** |
| directional agreement (neg/neutral/pos) | 68.3% |

**The gap between 0.328 and 0.659 is the finding.** The annotators disagree on the
exact level and agree on the direction, which is precisely what an ordinal-weighted
statistic is built to show and what a nominal one hides. Any paper reporting only
Fleiss on this scale would understate agreement by half.

### The disagreements are systematic, not random

Confusion is off-diagonal by exactly one level in a consistent direction:
qwen `very_negative` → Claude `negative` (7), qwen `positive` → Claude `neutral` (7),
qwen `very_positive` → Claude `positive` (5). 17 of 60 moved toward neutral.

The large disagreements reproduce §8d's documented v1 failure mode exactly:

| headline | qwen (v1) | Claude (v2) |
|---|---|---|
| "BCT : refinancement des banques à son **plus haut historique**" | very_positive | **negative** |
| "Le FMI accorde à **l'Ukraine** un plan d'aide de 15,6 Mds$" | very_negative | neutral |
| "signature de la déclaration « The last mile to get a job »" | very_positive | neutral |
| "BNP Paribas réalise un bénéfice record de 11 Mds€" | very_positive | neutral |
| "BIAT sacrée « Meilleure banque sur le marché de change »" | very_positive | neutral |

The first is the clearest case: record bank refinancing means banks cannot fund
themselves — liquidity stress. v1 read "plus haut historique" as a growth phrase and
scored it very_positive. The second scores IMF aid *to Ukraine* as very_negative
*for Tunisia*. Both are independently checkable in the pilot CSV.

### What this pilot does NOT establish

- **n = 60**, stratified to over-sample rare extremes, so raw agreement is deflated
  relative to a natural sample. κ is chance-corrected and less affected.
- **Annotator 2 authored PROMPT_V2.** Applying a rubric one wrote is a genuine
  conflict of interest. The individual disagreements are checkable, but this is not
  a substitute for an independent annotator.
- **The two annotators used different prompts** (v1 vs v2), so this κ measures
  prompt-version difference *confounded with* model difference and cannot separate
  them. To isolate the model, run both under v2.
- The full 2,932-row gold set remains single-annotator v1. Nothing here changes it.

### What it does establish

The agreement machinery works end to end, an ordinal-weighted κ is the right
headline statistic for this scale, and v1's labels carry a **measurable, systematic**
bias rather than random noise — which is consistent with §8e's finding that adding
v1-derived sentiment makes the return forecast significantly worse.

## 8g. Conformance to the architecture blueprint (v1.1) — 2026-09-20

Checked against `Tunindex_Sentiment_Pipeline_Architecture_Blueprint.docx`. Three
items were not gaps but **violations of an explicit instruction**; two are fixed.

### Fixed: §6.2 — raw accuracy was the headline metric, which the blueprint forbids

> "Balanced accuracy, MCC — robust to class imbalance; **replaces raw accuracy as
> headline**." Primary metric: **ROC-AUC**.

Every figure reported before this point (the 0.5736 H1 bar, the 0.5493 constant, the
whole sensitivity grid) was raw accuracy. At a 54.9% base rate that metric is nearly
blind, which is exactly why the blueprint rules it out. `baseline.evaluate()` now
reports ROC-AUC, balanced accuracy, MCC and Brier alongside it.

| feature set | AUC | balAcc | MCC | Brier | acc | const |
|---|---:|---:|---:|---:|---:|---:|
| momentum (ret_lag0) | 0.5880 | 0.5565 | 0.1211 | 0.2897 | 0.5743 | 0.5493 |
| momentum3 | 0.5943 | 0.5571 | 0.1219 | 0.2866 | 0.5743 | 0.5493 |
| momentum3_vol | **0.5979** | **0.5720** | **0.1501** | 0.2848 | 0.5859 | 0.5493 |
| momentum3_news | 0.5954 | 0.5619 | 0.1303 | 0.2861 | 0.5773 | 0.5493 |

**AUC ≈ 0.59 against a 0.5 null is a clearer statement than "0.574 versus 0.549".**
The blueprint's metric choice was correct and the accuracy framing was obscuring the
result in both directions.

A second bug surfaced here: for `kind="classify"` the stored score was the predicted
class, not a probability, so an AUC computed on it would have been meaningless.
`walk_forward` now uses `predict_proba` for classification.

### Fixed: §6.3 — no embargo between train and test

The blueprint requires a **5-session embargo**; `baseline.walk_forward` trained on
`X[:i]` to predict session `i`, gap zero. With `corr(ret_t-1, ret_t) = +0.263` the
adjacent sessions are the ones most correlated with the target.

`EMBARGO_SESSIONS = 5` is now the default. Measured cost:

| embargo | 0 | 1 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| AUC | 0.5943 | 0.5942 | **0.5943** | 0.5940 | 0.5942 |

Essentially free, because the model refits every 20 sessions. **That is a result,
not a non-finding:** the momentum edge is not an artifact of training on sessions
adjacent to the target.

### §5.2 F1 — the energy channel is MEASURED and immaterial (2026-09-20)

Before building F1, the premise was tested. Brent daily spot (EIA series RBRTE,
9,087 observations 1987-2026, saved to `data/raw/macro/brent_eia_daily.csv`) matched
to 97.5% of trading sessions:

| relationship | r | |
|---|---:|---|
| Brent move vs same-session Tunindex return | +0.0021 | ns |
| Brent move vs next-session return | +0.0203 | ns |
| Brent move vs \|return\| | +0.0070 | ns |
| R² of Brent alone on next return | **0.00041** | vs 0.06910 for `ret_lag0` — 168× weaker |

Consistent with the `global_linked` news evidence: international headline counts show
no relationship with Tunindex returns or volatility (r ≈ +0.02, ns) while domestic
counts do (r = +0.095 for volatility, p < 0.001). Tunisia's market responds to
domestic news flow and not to international flow — expected for a closed, illiquid
frontier market driven by episodic block trades (§8c).

**Disposition: a measured deviation, not a gap.** H1 is reported against F0, and
§5.2's energy control is documented as tested and immaterial for this market. That
is defensible; an unexplained omission would not be.

**Also a power argument.** The design is underpowered at MDE 3.06pp (§8e/S3). Adding
controls with R² = 0.0004 spends degrees of freedom and makes a true effect *harder*
to detect. Conforming to F1 mechanically would have hurt the study.

### Still untested in F1

Brent is one of three F1 components. **EUR/TND and European index returns were not
tested** — FRED's CSV endpoint timed out repeatedly, Yahoo rate-limits unauthenticated
requests, Stooq serves a JavaScript challenge, and the ECB does not publish a TND
reference rate. The EU takes roughly 70% of Tunisian exports and tourism, so the
demand channel is the more plausible of the two remaining and should be tested before
F1 is closed out. BCT (`bct.gov.tn`) responds and is the authoritative source for
both the dinar and the policy rate.

### Original text: the F1 rung is missing entirely

The feature ladder is F0 (price) → **F1 (+ macro/FX: TND rates, Brent, European
index returns)** → F2 (+ local sentiment) → F3 (+ international sentiment). There is
**no macro or FX data in this repo at all**, so every H1 test so far compares
sentiment against **F0**, not F1 as the blueprint specifies. If Brent or EUR/TND
explains what sentiment appeared to explain, the current design cannot tell.
This is the largest remaining structural gap.

### Other outstanding blueprint items

| § | requirement | status |
|---|---|---|
| 5.2 F0 | lagged returns 1–5, rolling vol, **volume change**, **day-of-week** | partial: 3 lags, no day-of-week, no volume change |
| 5.2 F2 | local sentiment **split FR / AR** | FR only — no Arabic corpus (§8e/S5) |
| 5.2 F3 | international sentiment as its own rung | tagged `global_linked`, not a separate feature set |
| 6.1 | initial training ≈3 years, test blocks of 3 months | `min_train=500` (~2y), refit every 20 sessions |
| 6.2 | block-bootstrap CIs on AUC | absent |
| 6.3 | purged, embargoed k-fold as a diagnostic | absent |
| 6.3 | per-block metrics over time; COVID-2020 examined separately | `by_year` only, no shock analysis |
| 3.1 §6 | MinHash LSH deduplication | deviated to a template key — measured and justified (M5) |
| 3.1 §3 | language ID **per paragraph** | assigned per source (§8e/S5) |
| 1.4 | source registry with [Confirmed] / [To be verified] | not in that form; §1-2 cover the same ground |

**Note on §6.1.** The blueprint's "initial training ≈3 years" is ~750 sessions, and
the sensitivity grid (M1 follow-up) shows the price-only result loses significance at
`min_train=750`. Conforming to the blueprint on this point produces a null on the
sign test. The AUC and the Diebold-Mariano test are the metrics that survive it.

### Where the work exceeds the blueprint

No price-report confound control, orthogonalised arm, Holm correction, power
analysis, Diebold-Mariano test, wrong-country provenance check or parameter
sensitivity grid appears in the blueprint. Each came out of the audit or the
adversarial review, and several caught real defects the blueprint would not have.

## 8h. Gold set v2, the sentiment instrument, and macro controls (2026-09-28)

### Provenance warning — read this before citing anything below

> **Update 2026-09-30 (§8i):** two of the orphaned families now have producing scripts —
> the CamemBERT fine-tune (`preprocessing/camembert_clf.py`, §8i.1) and the translation
> step (`preprocessing/translate_headlines.py`, §8i.4). Both reproduce the committed
> figures within seed noise, not bit-for-bit. Everything else below is still
> provisional in the sense stated here.

Every artifact in this section arrived in commit `f00134c` as a **data-only** change:
`.vscode/settings.json` and `data.dvc`, 42 new files under `data/curated/`, and **no
code**. The metadata records Windows paths (`C:\Users\LENOVO\...`), so the work was done
off this repository and only its outputs were shipped.

**No script in this repository regenerates any of it.** That breaks the rule stated in
§0 — every section here must be reproducible by `python3 audit/build_audit.py` — and so
this section is recorded as **PROVISIONAL**. The figures below are read from the
committed metadata files, not recomputed. Restoring the producing scripts is milestone
M1 in `PRD.md` and blocks every claim downstream of it.

### 8h.1 The v2 gold set

`data/curated/sentiment_gold_v2_full.csv`, 2,930 rows, sampled from a 2,870-row eligible
pool stratified by source × publication year, one row per exact duplicate cluster.

Exclusions applied at sampling:

| Excluded | n | Why |
|---|---|---|
| `not_in_study` | 68 | The Moroccan `assabah::*` rows |
| `annotation_pilot` | 60 | Already used in §8f |
| `prompt_v2_example` | 2 | Worked examples inside `PROMPT_V2`, excluded so they cannot leak |

This **resolves the decision left open on 2026-09-20**. The 68 Moroccan rows that had
leaked into the v1 gold set as its entire `ar` stratum are excluded from v2 by the
eligible-pool filter rather than deleted, which is the cheapest fix that destroys
nothing and was the one recommended.

**The §8d prompt gate is passed.** v1 produced 9.4% `neutral`; v2 produces **52.8%**
(1,546 of 2,930). The gate was explicit: if v2 failed to move the neutral share
substantially upward, the binding constraint was model capability rather than prompt
wording. It moved by 43 percentage points. **The wording was the constraint.**

Final label distribution: neutral 1,546 · positive 916 · negative 439 ·
very_negative 27 · very_positive 2.

The two extreme classes are now so thin that the 5-point scale cannot be estimated.
This is recorded, not fixed: the 3-class collapse loses nothing measurable (§8e, R3b).

### 8h.2 Three annotators, and the first human labels

| Annotator | Model | Rows |
|---|---|---|
| 1 | `qwen2.5-7b-instruct-1m` | 2,930 |
| 2 | `ministral-8b-instruct-2410` | 2,930 |
| 3 | **human** | 150 |

| Pair | n | raw | κ nominal | κ linear | **κ quadratic** |
|---|---|---|---|---|---|
| qwen vs ministral | 1,930 | 0.703 | 0.460 | 0.515 | **0.597** moderate |
| qwen vs human | 150 | 0.713 | 0.545 | 0.608 | **0.689** substantial |
| ministral vs human | 150 | 0.713 | 0.464 | 0.507 | **0.571** moderate |

Fleiss nominal across all three: 0.486, "moderate"; 85 of 150 rows unanimous.

The §8f pattern **reproduces at scale**: the nominal statistic is roughly two-thirds of
the quadratic one in every pair. Disagreement is about *level*, agreement about
*direction*. Report the ordinal-weighted figure; nominal Fleiss understates a gold set
whose scale is ordinal.

Note the ranking: **qwen agrees with the human more than ministral does**, and more than
the two models agree with each other. The smaller model is not the weaker annotator here.

This also supersedes the §8f caveat that the two annotators had used different prompts,
confounding model with prompt. All three v2 annotators work from `PROMPT_V2`.

### 8h.3 Adjudication composition

| Method | n | Share |
|---|---|---|
| majority of the two models | 1,970 | 67.2% |
| **tiebreak by annotator 1 (qwen)** | 810 | **27.6%** |
| human label taken directly | 150 | 5.1% |

**27.6% of the gold set is qwen's opinion with a majority-vote label on it.** Where the
two models split, qwen decides. This is a disclosed weakness of the instrument and must
appear in any write-up of it; it is the single largest reason the labels remain
provisional.

### 8h.4 Frozen split

2,339 train / 441 validation / **150 evaluation**, seed 20260919, fingerprint
`id_sha256 = 5c429fd8aa2e…`.

The evaluation split **is** the human-labelled subset. That is a deliberate design
change from v1's random stratification: the instrument is now graded against humans
rather than against other models. It is the only ground truth in this project and must
never be trained on.

Language: 2,287 fr / 52 en train; 141 fr / 9 en evaluation. Still no Arabic (§8e, S5).

### 8h.5 Model bake-off

Held-out validation, 3-class, n=441, majority floor 0.524.

| Model | QWK | Accuracy | macro F1 | Cost |
|---|---|---|---|---|
| TF-IDF + logistic (the §8e bar) | 0.561 | 0.698 | 0.670 | — |
| FinBERT zero-shot, **raw French** | **0.021** | 0.517 | 0.249 | — |
| FinBERT zero-shot, translated | 0.503 | 0.626 | 0.611 | + translation |
| FinBERT + trained head, translated | 0.604 | 0.669 | 0.664 | + translation |
| XLM-R fine-tuned | 0.589 | 0.705 | 0.698 | 108 min |
| **CamemBERT fine-tuned** | **0.708** | 0.735 | 0.732 | **4.9 min** |

On the human evaluation split (n=150, floor 0.567): CamemBERT QWK **0.635**,
accuracy 0.660, MCC-equivalent κ 0.488, per-seed QWK 0.586 / 0.644 / 0.593.

Three findings worth carrying into the write-up:

1. **FinBERT on raw French is at chance** — κ = 0.0005, macro F1 0.249. A
   finance-domain English encoder reads French headlines as noise. Translation recovers
   it entirely (0.021 → 0.503 zero-shot), so the translation step, not the domain
   pre-training, is doing the work.
2. **Monolingual beats multilingual on a 97%-French corpus.** CamemBERT 0.708 against
   XLM-R 0.589, at **22× less training time**.
3. **The TF-IDF ceiling was capacity, not labels.** §8e left this open: if a transformer
   only matched TF-IDF, the binding constraint was label quality and no GPU would fix
   it. CamemBERT clears it by 0.147 QWK. The labels were good enough to learn from; the
   linear model was not good enough to learn them.

Ensembling TF-IDF with the transformer helps accuracy marginally (0.748) and **hurts**
QWK (0.692). Report the single model.

### 8h.6 Macro controls — three tested, three rejected

Extending the §8g method (next-session OLS, Newey-West 5 lags, from 2014-01-01,
`enters_f1` iff p < 0.05) to the two channels §8g left untested.

| Candidate | Coverage | next-session r | R² | p (HAC) | Enters F1 |
|---|---|---|---|---|---|
| Brent spot (EIA) | 100% | +0.024 | 0.0006 | 0.433 | no |
| EuroStoxx 50 | 100% | +0.062 | 0.0038 | 0.072 | no |
| EUR/TND (BCT) | 100% | +0.004 | 0.00002 | 0.810 | no |

Against `ret_lag0` at R² = 0.0691, the strongest candidate is **18× weaker** and misses
significance. The EU demand channel — §8g's more plausible remaining hypothesis, since
the EU takes ~70% of exports and tourism — comes closest (p = 0.072) but does not clear
the bar, and only on the same session (p = 0.008), which is not usable for prediction.

All three do show a weak link to **volatility**: |next return| p = 0.022 (Brent) and
0.020 (EuroStoxx). Magnitude, not direction. Consistent with §8c.

**Blueprint §5.2's F1 rung is now measured on all three channels and declined on all
three.** This is a documented deviation supported by evidence, not a gap. The power
argument in §8g applies with more force: at MDE 3.2pp, spending degrees of freedom on
R² ≈ 0.0006 controls makes a true effect harder to detect.

### 8h.7 H1 re-run

The baseline **improved** — `vol_chg_lag0` and four day-of-week dummies were added,
closing two of §8g's outstanding F0 gaps:

| | AUC | balAcc | MCC | Brier | Accuracy |
|---|---|---|---|---|---|
| price-only (F0) | **0.6074** | 0.5675 | 0.141 | 0.2801 | 0.5818 |
| + sentiment, `all` | 0.6016 | 0.5632 | 0.133 | 0.2830 | 0.5788 |
| + sentiment, `ex_price` | 0.6026 | 0.5651 | 0.137 | 0.2825 | 0.5807 |
| + sentiment, `placebo` | 0.6036 | 0.5723 | 0.151 | 0.2820 | 0.5863 |
| + sentiment, `orthogonal` | 0.6046 | 0.5609 | 0.128 | 0.2815 | 0.5758 |

**Every arm is below the baseline on AUC.** Holm-adjusted sign tests: nothing
significant (lowest adjusted p = 0.131, `orthogonal`). Diebold-Mariano on squared-error
loss of returns finds the degradation **significant** in 3 of 4 arms — `all` p = 0.039,
`ex_price` p = 0.038, `placebo` p = 0.036 — with `orthogonal` at p = 0.076. OOS R² falls
from 0.0615 to 0.053–0.057 in every arm.

New in this run: **block-bootstrap ΔAUC** (2,000 resamples, block length 20), which
§8g listed as outstanding. For `all`: ΔAUC −0.0058, CI95 [−0.0123, +0.0001],
p = 0.054, Holm 0.216.

Power is unchanged in substance: 933 discordant pairs, MDE at 80% power **3.2pp**,
`adequately_powered: false`.

**Verdict: H1 REJECTED, not merely unsupported.** The sign test is null and the
better-powered continuous test says the sentiment features make the forecast worse.

### 8h.7b Two documented figures corrected on recomputation (2026-09-28)

**The price-report share is 3.74%, not 10%.** §8d states "4,244 of 42,645 relevant
headlines (10.0%)", with ilboursa 11.9%, kapitalis 11.2%, leconomistmaghrebin 9.3%.
Recomputing `config.PRICE_REPORT_PATTERN` against the 46,013 canonical rows gives
**1,722 = 3.74%** — ilboursa 3.30%, kapitalis 4.71%, leconomistmaghrebin 4.62%, French
3.83%, English 0.00%.

The cause is a regex change, not a row-set change. Commit `f1177e0` introduced both the
v1 flat-alternation pattern and the 10% figures; commit `7ed9c0c` ("Address adversarial
methodology review") replaced the pattern with the v2 co-occurrence lookahead — §8e, S1 —
and edited the pattern while leaving the percentages in the comment above it. Running v1
on the pre-dedup, pre-issuer 42,135-row corpus reproduces 10.42% / 11.92% / 12.48% /
9.24%, which is where §8d's numbers come from.

The v2 pattern is the better one — it correctly stopped flagging bond issues, air-traffic
statistics and company earnings that merely contain `en baisse de`. But it under-flags on
brittle literals: `en hausse` misses "en petite hausse", `séance du` misses "séances", and
`démarre`/`évolue` are not in the verb list. 1,565 canonical rows carry an index token
with no move verb. **3.74% is therefore a floor, and the placebo arm is built on fewer
rows than §8d implies.** Zero English rows flagged is correct, not a gap: exactly one of
the 1,036 English canonical rows contains any index token at all.

Stale copies remain in `preprocessing/config.py:136`, `README.md` and §8d above. They are
left in place rather than silently rewritten, per the rule that the evidence chain records
corrections rather than overwriting history.

**The |return| autocorrelation +0.387 is window-specific.** It holds on the 2014+ sample
(recomputed +0.3868); on the full 4,167-session 2010+ sample it is **+0.4438**. The
signed-return figure +0.263 is robust (+0.2637 vs +0.2732 full-sample).

### 8h.8 What this does and does not establish

**Does:** the v2 prompt fixed the neutral collapse; a fine-tuned monolingual encoder is
a materially better instrument than TF-IDF; agreement is moderate-to-substantial on the
ordinal scale with a human anchor; three macro channels are measured and declined.

**Does not:** that news sentiment cannot predict Tunindex. What is rejected is that
*these daily aggregates, built from this instrument, over this window* add anything to
price lags. Three confounds remain live — 27.6% of labels are a qwen tiebreak, the
daily aggregation discards intra-day ordering, and the design is underpowered on the
sign test.

**Unresolved and blocking:** `hypothesis_results.json` does not record which scored
corpus fed it, and `features.py:46` still defaults to `04_scored.parquet` — the **v1**
file dated 2026-09-20 — while `04_scored_v2.parquet` and `04_scored_v2_camembert.parquet`
sit beside it. Until provenance is recorded, §8h.7 cannot be attributed to a specific
instrument.

## 8i. Instrument and translation reproduced; headline series; per-outlet H2 (2026-09-30)

Every figure in this section comes from a **committed script**, named in each
subsection, and was recomputed on 2026-09-29/30. None of it is regenerated by
`build_audit.py`; the named script is the reproducibility path. Figures credited to
`TsEDA.ipynb` come from the executed time-series EDA notebook at the repo root.

### 8i.1 The CamemBERT fine-tune, reproduced (`preprocessing/camembert_clf.py`)

The recipe is taken from the committed metadata: `almanach/camembert-base`, 3 classes
(`very_*` collapsed into their neighbour), 5 epochs, 3 seeds with probabilities
averaged, 15% of the training rows held out as a dev set. The metadata does **not**
record learning rate, batch size, max length or weight decay; those were set to
2e-5 / 16 / 64 / 0.01 and are written into every output as guesses.

Trained on the 2,339-row train split, graded on the 150-row **human** evaluation split:

| | Original (off-repo) | Reproduced |
|---|---|---|
| QWK | 0.6349 | **0.6178** |
| Accuracy | 0.660 | 0.687 |
| Nominal κ | 0.4876 | 0.4961 |
| Per-seed QWK | 0.586 / 0.644 / 0.593 | 0.589 / 0.617 / 0.544 |
| Majority floor | 0.567 | 0.567 |

The 0.017 QWK gap is inside the seed spread. The reproduced file's `macro_f1`
(0.406) averages over all five `LABELS`, two of them empty after the collapse, so it
is **not** comparable with the original's 0.669. Output:
`data/curated/finetune/camembert_3class_evaluation_repro.json`.

### 8i.2 The yearly models before 2019 are under-trained — 2016–18 is not a measurement

`score_corpus.py --model camembert --mode expanding --min-train 200` reruns the
leakage-free yearly refit: 39,692 of 46,013 headlines scored, 6,321 unscored.
Output: `04_scored_v2_camembert_repro.parquet`.

| Year scored | Gold rows the model saw | negative | positive | neutral |
|---|---|---|---|---|
| 2016 | 298 | 0% | 0% | **100%** |
| 2017 | 441 | 0% | 18% | 82% |
| 2018 | 582 | 0% | 41% | 59% |
| 2019 | 724 | 11% | 32% | 57% |
| 2020–26 | 893–2,128 | 9–22% | 29–35% | remainder |

Below ~700 training rows the model never predicts `negative`. From 2019 the
reproduction agrees with the committed `04_scored_v2_camembert.parquet` on 81–86% of
labels per year (74% overall), which is seed-level agreement.

The committed scores fail in the same years by a different route. On price-report
headlines — which restate the move of the session they describe, so a working
instrument must agree with that move's sign — `TsEDA.ipynb` §B1 finds sign agreement
**0.468** (n=348) in 2016–18 against **0.877** (n=725) from 2019, and a placebo
correlation with `ret_lag1` of ~0.000 against **+0.491**. The 2019 boundary was first
chosen **post hoc** from that table; it is now also motivated independently, as the
first year whose model saw more than 700 gold rows.

**Consequence for §8h.7.** About 28% of H1's 2,678 walk-forward predictions fall in
2016–18, where the sentiment features are noise. H1 has **not** been rerun. Pending
decision: rescore with `--min-train` ≈ 700 (sentiment starts 2019) and rerun H1 on
that window, declared as a post-hoc restriction.

### 8i.3 The saved model and the static fill — descriptive only

`camembert_clf.py --save data/models/camembert_3class` trains on train + validation
(2,780 rows, the 150 evaluation rows asserted absent) and saves 3 seeds (1.3 GB,
DVC). `--score-corpus` scores every relevant headline with it →
`04_scored_v2_camembert_static.parquet`, metadata `leakage_free: false`. That model
saw gold labels up to 2026 and scores ~2,780 headlines in-sample. It exists only to
fill the pre-2019 rows of the headline series (§8i.5), tagged
`sent_source = saved_model`. **It never enters a result.**

### 8i.4 The translation step, reproduced (`preprocessing/translate_headlines.py`)

`finbert_head_results.json` records `translator_revision: c4aed37b…`, which matches
the Hugging Face commit history of **`Helsinki-NLP/opus-mt-fr-en`**; the script pins
that revision. Local GPU, no API.

| Check | Result |
|---|---|
| `--check-gold` vs the committed gold translations | **2,835 / 2,857 identical (99.23%)**; the rest are small wording variants |
| Full corpus | 46,013 rows: 44,977 translated, 1,036 English passed through; 41 min on a GTX 1650 |
| Digits preserved (ordinals excluded) | 99.88% of French rows |
| Degenerate outputs (loops, truncation after `\|`) | 15 detected and retranslated by segment; 1 still flagged `translation_suspect` (a false positive) |

Known limits: French decimal commas are kept verbatim ("un dividende de 4,500 dinars"
= 4.5 dinars reads in English as four thousand five hundred); and `normalize.py` leaves
U+FFFC in 3 `headline_clean` values — an upstream defect, not fixed because it would
move dedup counts. Incremental: the output is its own cache, keyed on model revision.

### 8i.5 The per-headline series (`preprocessing/build_headline_series.py`)

`data/curated/tunindex_headline_series.csv`: 45,706 rows (one per headline), 3,179
sessions, 2014-01-02..2026-09-16. English `headline`, `source`, original `lang`,
CamemBERT `sent_label`/`sent_score`, `sent_source`, and the session's `close`, `ret`,
`ret_next`. Alignment is `map_to_next_session(side="right")`, unchanged.
`sent_source`: `yearly_refit` 30,858 rows (2019+, leakage-free), `saved_model` 14,848
(before 2019, look-ahead). Asserts on every run: headlines per session equal
`n_headlines` in `tunindex_timeseries.csv`; every row has English text; no
`yearly_refit` headline is scored by a model fitted after its publication.

### 8i.6 First per-outlet H2 run (`preprocessing/outlet_ranking.py`)

2019+ only, `yearly_refit` rows only: 1,927 sessions, 1,426 walk-forward predictions
(`min_train` 500, `refit_every` 20, embargo 5). Each outlet enters as its daily mean
score plus a `has_<outlet>` indicator. Tested only with ≥ 100 sessions of news **and**
≥ 10% non-neutral headlines.

Linear model (regress, take the sign); price-only baseline AUC **0.5981**; all outlets
together 0.5969, DM p = 0.57.

| Outlet | Sessions | In-sample weight (HAC t) | ΔAUC add-one | DM p (Holm) | LOO ΔAUC |
|---|---|---|---|---|---|
| leconomistmaghrebin | 1,822 | +0.044 (2.23) | +0.0024 | 1.00 | +0.0036 |
| ilboursa | 1,913 | −0.042 (−2.02) | +0.0013 | 1.00 | +0.0020 |
| lapresse | 448 | +0.089 (2.41) | −0.0022 | 1.00 | −0.0006 |
| kapitalis | 1,839 | +0.004 (0.23) | −0.0042 | **0.036, worse** | −0.0046 |

- **Price reports removed:** the ranking is unchanged; kapitalis is still the worst but
  Holm p = 0.10.
- **Gradient boosting** (`baseline.walk_forward(kind="gbm")`, new): baseline AUC
  0.5695, below the linear model's; no outlet is significant.
- **Not tested:** economist_tunisia_all (89 sessions), tap (20).
- **Not testable — the instrument is blind on English:** in 2019+ CamemBERT scores
  Guardian 162/162, NYT 207/210 and Economist 95/97 headlines `neutral`. Their
  sentiment column is a constant.

**Reading.** No outlet's sentiment adds out of sample, and in-sample weights of
|t| ≈ 2 do not survive the walk-forward. With 1,426 predictions this is
**INCONCLUSIVE**, not evidence that outlets are interchangeable. The one surviving
result is negative: kapitalis degrades the forecast. H2's international-vs-domestic
comparison cannot be run until the English outlets are scored by an English-capable
instrument.

### 8i.7 Data defects found by the time-series EDA (`TsEDA.ipynb`) — recorded, none fixed

1. **Phantom sessions.** The 19 zero-return sessions copy the previous row's close,
   high and low. 12 of them have zero volume, most on Tunisian public holidays, so
   `audit/trading_calendar.csv` still keeps some holidays as sessions, and `ret_next = 0`
   on the row before each is a fake target. (Supersedes "19 sessions are exactly flat"
   in `architecture.md` §8.)
2. **Zero volume is often missing data.** Two contiguous blocks, 2021-09-01..30 and
   2026-04-14..05-06, show zero volume with normal returns. `vol_chg_lag0`, a
   price-only baseline feature, carries a fill value on 73 sessions.
3. **Look-ahead in the orthogonal arm.** `features.py:127–135` fits the residualisation
   coefficients on the whole frame, test period included (4 coefficients).
4. **`sent_resid_price_only_lag1`** is non-zero on 1,507 of 1,508 sessions where the
   placebo has no data: the 0-filled input was residualised. H1's orthogonal arm uses
   `sent_resid_lag1` and is unaffected.
5. **Brent returns on filled sessions** are non-zero on 76 of 80 forward-filled rows,
   74 of them repeating the previous return. (Supersedes "mechanically 0.0" in
   `architecture.md` §8; the macro builder has no committed script.)
6. **Sentiment blackout.** No sentiment before 2016-01-04; the first 500-session
   training window lies entirely inside it, so `has_sent_lag1` acts as a 2016-onward
   dummy in every expanding window.
7. **Weekday effect.** Next-session up-share is 0.609 on Fridays against 0.499 on
   Tuesdays (χ² p ≈ 0.001, uncorrected), partly the 2025 drift. Exploratory.

### 8i.8 What is still orphaned

**Closed in this revision:** the fine-tuned CamemBERT (code exists; the committed
`04_scored_v2_camembert.parquet` is matched within seed noise, not regenerated
bit-for-bit) and the translation step (99.23% identical).

**Still orphaned:** the v2 gold-set construction, the XLM-R fine-tune, the FinBERT
head, the macro controls, the three `sent_resid_*_lag1` columns, and the build that
produced the committed `daily_features.parquet`.

## 9. Explicit blocked/skipped items (for transparency)

- Step 6 (BVMT missing sessions, high<low checks, stale-price runs, 20-date cross-check):
  **still not run — but no longer for the stated reason.** Market data now exists and is
  inventoried; the skip at `build_audit.py:313` is unconditional. See §8. This is now a code
  fix, not a data gap.
- Trading-day-based zero-news-day share and the overlap-window computation in step 9: **not
  run** — same reason; calendar-day proxies were substituted and labeled as such throughout.
- Live-site spot checks (step 5) and hand-labeling (step 8): **template generated, not filled
  in** — these require a human with a browser and judgment calls the audit shouldn't fabricate.
