# Pre-registration — P1: news flow and the size of the next index move

Written 2026-10-06, **before any P1 news feature has been compared with any P1 target**.
It is frozen by the git tag `prereg-p1-v1`. Anything decided after that tag is labelled
**post hoc** in AUDIT_REPORT §P1. Plan: ADR-001 §8, P1 card. Open choices were settled at
the owner's interview of 2026-10-06 (STATUS.md). Code: `preprocessing/p1.py` on the
harness `preprocessing/bench.py`.

What was known when this was written:
- H1–H3 (§8h.7, §8i.6, §H3) and P0 (§P0): the harness, its controls and the MDE table
  approved at G0.
- P1's own controls (§9 below), which use planted, shuffled or stale data, never the real
  news-target link.
- Descriptive ranges of the P1 features.
- §8c: headline counts move with the same session's |return| and range. That is a
  contemporaneous association, not a forecast, and it is why P1 exists.

## 1. Hypothesis

**P1.** Adding three news-flow features to a price-only volatility model lowers the
out-of-sample loss of the next-session variance forecast (ADR-001: a necessary-condition
test for the content models of P3 to P6).

## 2. Data

| Item | Fixed choice |
|---|---|
| Prices | `data/raw/bvmt/tunindex_2010_today.csv`: high, low, close. `open` is never read (rule 1). |
| Calendar | `audit/trading_calendar.csv` minus the 19 phantom sessions (`features.phantom_sessions`) |
| Headlines | `data/curated/04_scored_v3_camembert.parquet`: every canonical relevant headline, scored or not. Every headline from 2016 on carries a v3 label. |
| Headline → session | `features.map_to_next_session`: a headline dated D enters the first session strictly after D (rule 2). News on row t predicts session t+1. |
| Design window | sessions 2016-01-04 → 2023-12-29. Targets that would read a session from 2024 are blanked. |
| Confirmation window | every session from 2024-01-01 to the last one. **One run**, after this tag, with `--confirm`. The model walks forward from 2016 and only sessions ≥ 2024-01-01 are scored. |

## 3. Target

- **Primary:** next-session Parkinson variance, (ln H/L)² / (4 ln 2).
- A session with high == low has no variance to score. It is **not a target**, and the
  count is reported (owner): 3 in the design window.
- **Secondary:** the next squared return (the ADR's |return|, squared so that a variance
  loss applies), and the sum of the next 5 squared returns (h = 5). Zero values are not
  targets.

## 4. Baseline

HAR on log range, built with a 1-basis-point floor on ln H/L:
- the last session,
- the mean over the last 5 sessions,
- the mean over the last 22 sessions.

It also has the last session's |return|, weekday dummies of the target session (Mon–Thu)
and the calendar days to the target session. The model regresses the log target; the
forecast is exp(fit) × a Duan smearing factor, both fitted on the training slice.

## 5. News block (three features; ADR P1 card)

Per arm, from the headlines mapped to the row's session, and 0 on a session without any:

| feature | definition |
|---|---|
| `log_n` | log(1 + number of headlines) |
| `neg_share` | share labelled `negative` by the v3 instrument |
| `novelty` | mean over the session's headlines of the surprise per token, in bits, under a word-bigram model of **every** headline of the previous 250 calendar sessions (never the session itself) |

The novelty model:
- p(w | v) = λ c(v, w) / c(v) + (1 − λ) p₁(w) when v was seen, else p₁(w).
- p₁(w) = (c(w) + 1) / (N + V + 1), with λ = 0.5 fixed.
- Tokens come from `dedup._template_key`: lower case, numbers and date words masked.
- It is NaN while its window is empty (never from 2016 on).

Two of the three features are text-derived, within the limit of 2.

**Arms (rule 6):** `ex_price` (price reports removed, `config.PRICE_REPORT_PATTERN`)
**decides G1** (owner). `all` and `price_only` are run and reported.

## 6. Harness

`bench.walk_forward`, kind `logvar`:
- expanding window, `min_train = 500` sessions, refit every 20, embargo max(5, h);
- scaler and smearing factor fitted on the training slice;
- no tuning.

## 7. Primary test and decision rule

- **Statistic.** D = mean QLIKE(treatment) − mean QLIKE(baseline) on identical sessions,
  `ex_price` arm, where QLIKE = y/ŷ − ln(y/ŷ) − 1.
- **Null.** 100 phase-randomised surrogates of the 3-feature block (same values, same
  drift, random timing) give μ₀ and sd₀.
- **Scale.** √(sd₀² + SE²), where SE comes from 2,000 circular block-bootstrap resamples
  (block 20) of the per-session loss differential. This is the yardstick validated at
  G0 (§P0).
- **Effect.** D − μ₀, with a 95% interval of effect ± 1.96 × scale.
- **SESOI.** 3% of the baseline's mean QLIKE in the same run (owner).

| interval | verdict |
|---|---|
| entirely below 0 | **NEWS IMPROVES** |
| entirely above 0 | NEWS WORSE THAN NOISE |
| lower end above −SESOI | **NO EFFECT WORTH HAVING** |
| otherwise | **INCONCLUSIVE**, reported with its MDE |

**G1 (ADR).** The confirmation run's `ex_price` verdict decides:
- **NEWS IMPROVES:** P3 adds a volatility label.
- **Any other verdict:** P3's index arm becomes exploratory only.

The design run is reported and does not decide G1. AUC is not used.

## 8. Secondary families (reported, never decisive)

| Family | Tests | Correction |
|---|---|---|
| S1 | each `ex_price` feature alone, primary rule | Holm across 3 |
| S2 | MSE on log variance, `ex_price` block, Clark-West | none |
| S3 | secondary targets (next squared return; next 5 squared returns), `ex_price` block, primary rule | Holm across 2 |
| S4 | arms `all` and `price_only`, primary rule | none; descriptive |

## 9. Controls, run before this tag

`bench.controls` on the decisive arm, design window, with G0's bands: the planted signal
(planted on the log scale) must be found in 72–88 of 100 runs; shuffled labels and stale
news must give at most 9 false finds each.

Results (`data/curated/p1_controls.json`, 2026-10-06, 1,493 design predictions):

| control | result | band |
|---|---|---|
| planted signal found | 76 of 100 (median statistic 2.68, target 2.8; spread 1.02 SE) | 72–88, pass |
| shuffled labels, false finds | 0 of 100 | ≤ 9, pass |
| stale news, false finds | 1 of 100 (1 better, 0 worse) | ≤ 9, pass |

For P1's own block, the estimation cost of 3 surrogate columns is 0.00104 QLIKE and the
design-window MDE is 0.0053.

## 10. Power

MDE approved at G0 (§P0), computed with H3's news block as a stand-in: 0.0054 in the design
window (1.9% of the baseline's loss) and 0.0081 in the confirmation window (2.8%). The run
reports the MDE of P1's own block.

## 11. Deviations

None at the time of writing.

## 12. Outputs

`data/curated/p1_controls.json`, `p1_design.json`, and `p1_confirm.json` (written once;
the script refuses to overwrite it). AUDIT_REPORT §P1 reports every arm, every family,
every null with its MDE, and every deviation from this document.
