# H1 pre-registration: amendment a1

**Amends:** `prereg-h1-v1`, the harness at commit `7ed9c0c`. That tag is referenced in the README and in AUDIT_REPORT M1. It is **not present** in this clone or on `origin`, so it should be pushed from wherever it was created.
**Date:** 2026-09-23
**Status:** written and committed **before any H1 run on v2 scores**. The commit that adds this file, tagged `prereg-h1-v1-a1`, fixes everything below.

## Disclosure: what had been seen when this was written

- **H1 has been run once, on v1 labels** (2026-09-21, `hypothesis_results.json`). That was the pipeline demonstration on the single-annotator PROMPT_V1 labels, which AUDIT_REPORT §8d found did not measure sentiment. Its interpretation line was "H1 REJECTED": the DM test found sentiment made forecasts worse in all three arms. The author of this amendment had seen that result.
- The price-only baseline results were also known: §8c, and the M1 sensitivity grid.
- **H1 has never been run on any v2 score**, neither the TF-IDF scores in `04_scored_v2.parquet` nor the CamemBERT ones. `daily_features.parquet` predates both.
- `kind="regress"` stays outcome-selected, as M1 already declares. This amendment does not change it.

## A. Scorer

H1 is run on **fine-tuned CamemBERT** (`almanach/camembert-base`, 3 classes, 5 epochs, 3 seeds averaged). The model is refit each year on gold labels dated strictly before that year (`score_corpus.py --scorer camembert --ft-epochs 5`). The scored file is `data/curated/04_scored_v2_camembert.parquet`, and `features.py` now reads it by default.

Selection is documented in AUDIT_REPORT §8i. The candidates were compared on the LLM-adjudicated validation split. The 150-row human evaluation split was then spent once, on this model: QWK 0.635 [0.524, 0.732].

The project owner chose **no TF-IDF fallback** for the thin 2016–2018 windows, where CamemBERT's out-of-time QWK falls below TF-IDF's. That weakness is left in and made visible through the per-block check in §D.

## B. F0 completed (blueprint §5.2)

The H1 baseline was `ret_lag0, ret_lag1, ret_lag2`. It is now `baseline.F0_FEATURES`, which adds:

| feature | definition |
|---|---|
| `vol_chg_lag0` | log volume of session *i* minus that of *i−1*. Volume 0 (56 sessions, impossible for the index) counts as missing. A missing change is filled with 0. The change is clipped to ±3 log units (21 sessions). |
| `dow_next_mon` … `dow_next_thu` | day of week of the **target** session *i+1*, taken from the exchange calendar and so known in advance. Friday is the reference level. |

Every sentiment arm contains F0 plus its own sentiment columns, so the paired comparison stays F0 against F0 plus sentiment.

## C. F1: candidate controls (blueprint §5.2)

`macro_controls.py` fixes one rule for every candidate. A candidate enters the baseline if its day-*D* move (as-of joined, so it is known before session *D+1* opens) predicts the next-session Tunindex return at **p < 0.05, OLS with Newey-West (5 lags), 2014 onward**.

| candidate | next-session r | p (HAC) | enters F1 |
|---|---|---|---|
| Brent (EIA RBRTE) | +0.024 | 0.433 | no |
| EUR/TND | *pending data* | | |
| European index returns | *pending data* | | |

The two pending candidates are tested with this script and rule before H1 runs, and their rows are filled in by a follow-up commit that changes nothing else. If a candidate's data cannot be obtained, H1 is reported against F0 and that gap is stated.

## D. Statistics and the go/no-go (blueprint §6.1–6.3)

`h1_stats.py`. The primary metric becomes **ΔAUC = AUC(arm) − AUC(F0)** over the paired walk-forward predictions.

| item | fixed value |
|---|---|
| CI | paired circular moving-block bootstrap, blocks of 20 sessions, 2,000 resamples, seed 20260923, percentile 95% CI and two-sided p |
| multiplicity | Holm across the four arms (`all`, `ex_price`, `placebo`, `orthogonal`) |
| per-block | ΔAUC in each calendar quarter with at least 30 sessions and both classes present |
| COVID | ΔAUC in 2020 alone, and ΔAUC with 2020 removed. Reported only; not part of the decision. |
| purged k-fold | 5 contiguous folds, 5-session purge on each side. A diagnostic only; it trains on the future. |

**An arm passes** if and only if its Holm-adjusted bootstrap p is below 0.05, its ΔAUC is above 0, **and** its ΔAUC is above 0 in at least ⅔ of the scorable quarters.

**Verdict:** this mirrors the existing ladder in `_interpret`.

| outcome | verdict |
|---|---|
| `ex_price` and `orthogonal` both pass | **GO**: H1 supported |
| `ex_price` passes, `orthogonal` fails | PARTIAL |
| `all` passes, `ex_price` fails | NO-GO: momentum laundering |
| `placebo` passes, `ex_price` fails | NO-GO: placebo fired |
| none of the above | NO-GO: not supported |

A null is reported as a null. McNemar, Diebold–Mariano and the power calculation are still computed and reported, as secondary results.

## E. Leak fixed in the orthogonal arm

`features.py` residualised sentiment on `ret_lag0, ret_lag1, log_headlines_lag0` using a beta estimated on the **full sample**, so every residual depended on future sessions. The beta is now fit on an expanding window of rows strictly before each session (`expanding_residual`), after at least 60 rows of history. A test asserts that changing a later row leaves every earlier residual unchanged.

## Unchanged

- `kind="regress"`, `min_train=500`, `refit_every=20`, and the 5-session embargo.
- The four arms and the price-report regex.
- The next-session news alignment rule.
- The `sensitivity.py` grid, which now runs on F0.

## What would violate this amendment

Any change to the constants in `h1_stats.py`, `F0_FEATURES`, the F1 rule, the scorer, or the arms, made after an H1 result on v2 scores has been seen. If such a change is ever needed, it goes in a further numbered amendment with its own disclosure.
