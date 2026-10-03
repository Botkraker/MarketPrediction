# Architecture Essentials

Everything here is load-bearing. Violating any of it produces a number that looks
fine and is wrong. Full detail in [architecture.md](architecture.md).

---

## The nine rules

**1. Returns are close-to-close.**
`open` is exactly the previous session's close in 1,384 of 4,167 rows (33.2%), so
`(close - open)/open` is a close-to-close return for a third of sessions and an
intraday return for the rest — two quantities in one column. `features.load_prices`
drops `open` entirely. Do not add it back.

**2. News dated D predicts sessions strictly after D.**
Headlines are date-only for 65% of the corpus; same-day mapping would leak.
`map_to_next_session` uses `searchsorted(side="right")` against
`audit/trading_calendar.csv` — **not** the next weekday. 191 weekdays in range are
holidays, not sessions.

**3. Lags are counted from the target, never from the row.**
`ret_lag0` is the most recent *available* return, known the moment session *i* closes.
Counting from the row once excluded it entirely and made the baseline predict session
*i+1* from data ending at *i−1*. Guarded by
`test_features.py::test_most_recent_available_return_is_exposed_as_ret_lag0`.

**4. Regress the return and take the sign. Never classify direction.**
Logistic on a drift-dominated binary label collapses to "always up" — it predicts up
77–92% of the time. Regression ~0.556 vs classification ~0.545 on every feature set.

**5. ROC-AUC is the headline metric.**
Balanced accuracy and MCC replace raw accuracy. At a 54.9% base rate accuracy is nearly
blind: "AUC 0.607 against a 0.5 null" says far more than "0.582 vs 0.549". For the
sentiment classifier, report **QWK**, not accuracy — 53% of labels are one class.

**6. Scoring is leakage-free or it is not a result — and before 2019 it is not a result either.**
`score_corpus.py --model camembert --mode expanding` refits on gold rows strictly
before each year. 6,321 headlines have no prior labels and stay **unscored**. Never
impute them. `--mode static` exists only to quantify the difference.
The 2016–18 yearly models saw 298–582 gold rows and collapse (2016: 100% neutral;
2017–18: no `negative` at all), so **sentiment is usable from 2019 only**. The saved
model in `data/models/camembert_3class` saw 2026 labels: its scores
(`sent_source == "saved_model"` in the headline series) fill pre-2019 rows for
description and **never enter a result**. AUDIT_REPORT §8i.2–8i.3.

**7. H1 is reported in four arms, always.**
`all`, `ex_price`, `placebo` (price reports only), `orthogonal` (residualised on
`ret_lag0`, `ret_lag1`, `log_headlines_lag0`). Price reports restate the index's own
move and reach 0.5640 directional accuracy with zero news content. If sentiment only
works with them in, it is momentum. Holm-Bonferroni across the four.

**The share is 3.74%, not the 10% quoted elsewhere.** 1,722 of 46,013 canonical rows
under the current v2 `PRICE_REPORT_PATTERN` (ilboursa 3.30%, kapitalis 4.71%,
leconomistmaghrebin 4.62%, English 0.00% — those outlets never publish BVMT price
reports). The 10% / 11.9 / 11.2 / 9.3 figures still in `config.py:136`, `README.md` and
`AUDIT_REPORT.md` §8d are **v1-regex numbers on a pre-dedup, pre-issuer 42,645-row
corpus**; commit `7ed9c0c` replaced the pattern and left the percentages. v2 also
under-flags on brittle literals (`en hausse` misses "en petite hausse", `séance du`
misses "séances"), so 3.74% is a floor.

**8. A null on the sign test is INCONCLUSIVE, not evidence of absence.**
MDE at 80% power is **3.2pp**; published effects are 1–2pp. Always report
Diebold-Mariano alongside McNemar. When the two disagree, that disagreement is the
finding — do not resolve it by choosing one.

**9. Never hand-patch `data/`.**
A defect gets a scraper or pipeline fix and a re-extract. `data/` is DVC-tracked and
every file in it must be regenerable by a committed script.

---

## Numbers you will be asked for

| | |
|---|---|
| Canonical relevant headlines | 46,013 (8 sources) |
| Trading sessions | 4,167 · 2010-01-04 → 2026-09-16 |
| Gold set v2 | 2,930 rows · 2,339 / 441 / 150 |
| Evaluation split | the 150 **human**-labelled rows |
| Best sentiment instrument | CamemBERT · val QWK 0.708 · human-eval QWK 0.635 (off-repo) / **0.618 reproduced** by `camembert_clf.py` |
| Usable sentiment window | **2019-01 onward** (first yearly model with > 700 gold rows) |
| Price-only baseline | **AUC 0.6074** · balAcc 0.5675 · MCC 0.141 |
| Always-up constant | 0.5493 over 2,678 predictions |
| Return autocorrelation | +0.263 (\|ret\| +0.387) — **2014+ window**; full 2010+ sample is +0.273 / +0.444 |
| Tests | 117, all passing |
| Analysis frame | `data/curated/tunindex_timeseries.csv` · 3,179 × 65 · 2014-01-02..2026-09-16 |
| Price reports | 3.74% of canonical rows (1,722 / 46,013) |
| Headlines scored | 39,692 of 46,013 · 6,321 unscored, never imputed |
| Per-headline series | `data/curated/tunindex_headline_series.csv` · 45,706 rows · English text · `sent_source` 30,858 yearly_refit / 14,848 saved_model |
| Translation | opus-mt-fr-en @ `c4aed37b` · reproduces gold translations 99.23% |
| H2, per outlet (2019+) | no outlet adds out of sample; kapitalis degrades (Holm p 0.036); English outlets untestable — **INCONCLUSIVE** |

---

## Traps that already cost a rewrite

**The relevance filter is correct — do not "fix" it.** It rejected 99.87% of the
Moroccan Assabah corpus, which was the right call. It also *was* wrong in the other
direction: 7,215 rows naming a listed BVMT issuer were dropped as `other` before
`KEYWORDS_ISSUERS` landed.

**Claims here have failed recomputation in both directions.** Two review findings were
wrong, and on 2026-09-28 a fresh check found the documented 10% price-report share was a
stale v1-regex number (true value 3.74%) while a reported "leakage bug" in the `_lag1`
sentiment columns turned out to be correct code. Verify before acting on any documented
finding, including the ones in these files.

**A tenth rule, learned the hard way: a `_lag1` suffix is not a `.shift(1)`.** The
one-session lag is already performed by `map_to_next_session(side="right")`, so the
suffix names the offset from the *target*. Empirical proof:
`sent_mean_price_only_lag1` correlates +0.227 with `ret_lag1` and only +0.039 with
`ret_lag0` — a leaking feature would put that 0.227 in the `ret_lag0` column. Adding a
shift would make every sentiment feature two sessions stale.

**CamemBERT is blind on English.** It scores Guardian 162/162, NYT 207/210 and
Economist 95/97 headlines `neutral` (2019+). Any English-outlet sentiment from it is a
constant, so no fr-vs-en or international-vs-domestic comparison can be run on it.

**Some "sessions" are not sessions.** The 19 zero-return rows copy the previous
close/high/low; 12 are zero-volume holidays still in `trading_calendar.csv`, and the row
before each carries a fake `ret_next = 0`. Zero volume in 2021-09 and 2026-04/05 is
missing data, not zero trading. And `features.py:127–135` fits the orthogonal arm's
residual on the whole sample, test period included. AUDIT_REPORT §8i.7 — recorded,
not yet fixed.

---

## Before you run anything

```bash
python3 -m pytest preprocessing   # 117 must pass. `python` is not on PATH.
```

Run from the repo root. Nothing takes a relative path from elsewhere.
