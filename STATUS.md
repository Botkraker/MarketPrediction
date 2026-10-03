# STATUS — H3 (direction and crash)

Phase 0 recon, 2026-10-03. No code changed. Every number below was recomputed on
2026-10-03 unless it cites an AUDIT_REPORT section.

## Decisions taken at the 2026-10-03 interview

| Question | Decision |
|---|---|
| New data | None. Only data already in the repo. |
| New model weights | **None — CamemBERT only.** XLM-R, mDeBERTa and FinBERT are out of scope. |
| Primary target | **3-class direction, next session (h=1).** h=5 and h=20 direction and crash are secondary families. |
| Firm-level panel | Crash events only. Index-level target for direction. |
| Macro (`data/raw/macro/`) | Robustness specification only, never in the primary baseline. |
| Instrument ground truth | Headline metric on the 150 human rows. Per-outlet and per-year breakdown on gold v2, labelled "LLM agreement", not truth. |
| Known defects | Fix the orthogonal-arm look-ahead and the phantom sessions **before** any H3 model runs. |
| Process | Waterfall. One phase at a time, approval at each checkpoint before the next starts. |
| Compute | **Google Colab, not this PC.** Free accounts, resumable jobs, checkpoints in a shared Google Drive folder; parallel jobs across accounts when there are several models to run. |
| Reports | Terminal plus this file. |

## Inventory

### Prices

| File | Rows | Range | Notes |
|---|---:|---|---|
| `data/raw/bvmt/tunindex_2010_today.csv` | 4,167 sessions | 2010-01-04 → 2026-09-16 | `open` is not an opening price (rule 1); 19 phantom sessions (§8i.7) |
| `data/raw/bvmt/ALL_DATA.csv` | 187,987 | 2010-01-04 → **2022-12-30** | `Ticker, Date, Open, High, Low, Close, Volume`; 88 tickers, 57 (2010) → 83 (2019–22) per year |
| `data/raw/bvmt/sotcks_list.csv`, `stocks_market_cap.csv` | — | — | issuer names (feed `KEYWORDS_ISSUERS`) and market caps |

**The firm-level file stops at 2022-12-30.** Sentiment is usable only from 2019 on (§8i.2),
so the firm-level crash panel with sentiment covers **2019–2022, about four years**, and
none of the 2023–26 period. Breadth and turnover features can't be built after 2022.

### Headlines (per-headline series, by session year)

| source | 2014 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ilboursa | 1437 | 1566 | 1475 | 1453 | 1372 | 1331 | 1487 | 1340 | 1438 | 1285 | 1124 | 1167 | 790 |
| kapitalis | 0 | 660 | 777 | 745 | 1139 | 1481 | 1285 | 778 | 1338 | 1254 | 997 | 794 | 582 |
| leconomistmaghrebin | 981 | 1164 | 618 | 566 | 602 | 534 | 711 | 702 | 1092 | 1319 | 1600 | 1451 | 961 |
| lapresse | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 133 | 1765 | 1578 |
| guardian_tunisia | 28 | 174 | 25 | 35 | 23 | 22 | 17 | 34 | 12 | 52 | 17 | 7 | 1 |
| nyt_economy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 99 | 96 |
| economist_tunisia_all | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 13 | 12 | 11 | 8 | 22 | 29 |
| tap | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 80 |

45,706 rows. `sent_source`: 30,858 `yearly_refit` (2019+, usable) / 14,848 `saved_model`
(pre-2019, descriptive only). Every timestamp is **date-only**. The ilboursa time-of-day fix
needs a re-scrape, which is new data and therefore out of scope.

### Labels

| Set | Rows | Use |
|---|---:|---|
| gold v2 train | 2,339 | fine-tuning; 27.6% of labels are a qwen tiebreak (§8h.3) |
| gold v2 validation | 441 | model selection |
| gold v2 evaluation = **human** | **150** | the only ground truth; never trained on |

Classes after the 3-class collapse: neutral 52.8%, positive ~31%, negative ~16%. English: 52
train / 9 evaluation rows.

### Code and environment

- `python3 -m pytest preprocessing`: **117 passed** (2026-10-03).
- Harness to reuse: `baseline.walk_forward` (expanding window, refit every 20, embargo 5;
  kinds `regress` / `classify` / `gbm`), `baseline.evaluate`,
  `hypothesis_tests.{holm, diebold_mariano, paired_test, minimum_detectable_effect}`,
  `outlet_ranking.panel`, `features.map_to_next_session`.
- Not installed locally: `lightgbm`, `hmmlearn`, `arch` (GARCH). These are software, not
  data: install with pip on Colab.

## What is done, open and broken

**Done:** corpus (46,013 canonical); gold v2 with human anchor; CamemBERT instrument
reproduced (human QWK 0.618); translation reproduced; H1 run, REJECTED (§8h.7); first H2
per-outlet run, INCONCLUSIVE (§8i.6); macro controls tested and declined (§8h.6).

**Broken (fix before H3, as agreed):**
1. Orthogonal-arm look-ahead: `features.py:127–135` fits the residual on the whole sample.
2. 19 phantom sessions, 12 of them zero-volume holidays, are still in
   `audit/trading_calendar.csv`. They put a fake `ret_next = 0` on the session before each.

**Broken, found today:**
3. **The 2026-09-30 data is not versioned.** `data.dvc` was last committed in `f00134c`
   (2026-09-28); `dvc status` reports `data` modified. The headline series, translations,
   repro scores and `data/models/camembert_3class` exist only on this disk. Colab can't
   reproduce anything until they are `dvc add`-ed and pushed (or copied to the shared
   Drive folder).

**Open, not blocking H3:** zero-volume blocks that are really missing data (2021-09,
2026-04/05) feed `vol_chg_lag0`; `sent_resid_price_only_lag1` residualises 0-filled inputs;
orphaned artifacts (gold v2 build, XLM-R, FinBERT head, macro builder,
`daily_features.parquet` build); `features.py` defaults to the v1 scored file; stale 10%
price-report figures in `config.py:136`, README and §8d; no dependency manifest.

## Conflicts between the H3 plan and existing rules

1. **Rule 4 says never classify direction.** On a drift-dominated label, logistic regression
   collapsed to "always up". The prereg must handle this explicitly. Proposal: the primary
   estimator regresses the return and maps it to drop/flat/rise with the dead-zone
   thresholds. Class-weighted multinomial logistic is a secondary specification and is
   reported against it.
2. **Phase 2 asks for XLM-R/mDeBERTa/FinBERT; the interview says CamemBERT only.** The
   English-outlet half of H2 stays untestable. Phase 2 becomes: calibrated CamemBERT
   probabilities, yearly refit at `--min-train` ≈ 700, relevance tagging from text.
3. **Topic shares via BERTopic need a sentence-embedding model.** Use CamemBERT mean-pooled
   embeddings or TF-IDF + NMF instead; no new weights.
4. **h=20 needs embargo ≥ 20.** `walk_forward` drops `embargo` rows before the prediction
   point, which is correct only while the target horizon ≤ embargo. With h=20 and
   embargo 5, training targets would overlap the test target.

## Ranked plan (waterfall, one checkpoint each)

1. **Phase 1, pre-registration.** `audit/PREREG_H3.md`, committed and tagged before any
   model runs. Count crash events for each definition on the index (2014+, 2019+) and on
   the 2019–22 firm panel, then pick k for a 5–10% event rate. Fix the dead-zone
   thresholds from the training window only.
2. **Housekeeping, before any model.** Fix the phantom sessions and the orthogonal
   look-ahead, each as its own commit with a test. `dvc add data` and commit
   `data.dvc`. Set up the shared Drive folder and a resumable Colab runner.
3. **Phase 2, instrument (Colab).** Human-150 grading with a bootstrap CI; gold-v2
   per-outlet/per-year F1; rescore 2019+ with probabilities.
4. **Phase 3, features.** Derived only; the timing check reduces to confirming
   `side="right"`, since no timestamps exist.
5. **Phase 4–5, models and evaluation (Colab, parallel by model × horizon × window
   start).**
6. **Phase 6, deliverables.**

## Out of scope under the no-new-data constraint

English-capable instrument (Guardian/NYT/Economist), so the H2 international comparison;
firm-level prices after 2022-12-30; intraday timestamps (ilboursa re-scrape); any Arabic
source; additional human labels.
