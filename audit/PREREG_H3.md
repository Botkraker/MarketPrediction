# Pre-registration — H3: news and next-session direction; news and firm drawdowns

Written 2026-10-03, **before any H3 model has been fitted**. Frozen by the git tag
`prereg-h3-v1`. Anything decided after that tag is labelled **post hoc** in
AUDIT_REPORT §H3. Decisions come from the 2026-10-03 interview (STATUS.md).

What was known when this was written: H1 (§8h.7, REJECTED) and H2 (§8i.6, INCONCLUSIVE),
both on the v2 scores; the v3 instrument (§8j); event counts and class shares from
prices alone (§5 below). No H3 target has been compared with any news feature.

## 1. Hypotheses

- **H3a (primary).** Adding news tone to a price-only model improves the out-of-sample
  probability forecast of next-session Tunindex direction (drop / flat / rise).
- **H3b (secondary).** The same at horizons of 5 and 20 sessions.
- **H3c (secondary).** Adding firm-level and market news tone improves the
  out-of-sample forecast of large firm-level drawdowns.

## 2. Data

| Item | Fixed choice |
|---|---|
| Prices | `data/raw/bvmt/tunindex_2010_today.csv`, close-to-close only (rule 1) |
| Calendar | `audit/trading_calendar.csv` **minus the 19 phantom sessions** (rows whose close, high and low copy the previous session; §8i.7). Returns are recomputed across the removed rows. |
| Sentiment | `data/curated/04_scored_v3_camembert.parquet` (§8j), yearly refit, `--min-train 700`. Headlines with `sent_label == ""` are excluded, never imputed. |
| Headline → session | `features.map_to_next_session(side="right")`, unchanged: a headline dated D enters sessions strictly after D (rule 2). The feature at row *i* predicts session *i+1*. |
| Index window | first session 2016-01-04; last row with a defined target |
| Firm panel (H3c) | `data/raw/bvmt/ALL_DATA.csv`, exact duplicate (Ticker, Date) rows dropped (2,095 pairs, identical close), each ticker reindexed to the calendar with close carried forward on non-trading days; 2016-01-04 → 2022-12-30 (the file ends there) |
| Issuer matching | issuer names from `sotcks_list.csv`, same rule as `config.KEYWORDS_ISSUERS` (name ≥ 4 chars, stopwords excluded), case-insensitive whole-word match; a headline naming two firms counts for both |

## 3. Fixes made before any model runs

1. Phantom sessions removed from the calendar (above), with a test.
2. Orthogonal arm: residualisation coefficients are fitted **on the training slice
   only**, at every refit (was whole-sample, `features.py:127–135`), with a test.

## 4. Harness (all families)

Expanding window, `min_train = 500` sessions, refit every 20 sessions, embargo
`max(5, h)` sessions, where h is the target horizon. Features standardised on the
training slice. No hyperparameter tuning: values below are fixed.

## 5. Targets (event counts computed from prices only, 2016+)

**Direction, horizon h ∈ {1, 5, 20}.** Forward return r = close[t+h]/close[t] − 1.
Classes: drop if r < q₁, rise if r > q₂, flat otherwise, where q₁, q₂ are the 33.3rd
and 66.7th percentiles of r **in the training slice at that refit** (no look-ahead).
Whole-period reference values, not used: h=1 −0.09% / +0.18%; h=5 −0.15% / +0.61%;
h=20 −0.17% / +2.01%.

**Firm drawdown (H3c).** Event = the minimum close over the next 20 sessions falls more
than **k = 12%** below today's close. Rule: k is the smallest of {8, 10, 12, 15}% whose
event share on the 2016–2022 reindexed panel lies in 5–10%. Shares: 17.2 / 11.4 / **7.9**
/ 4.8%. Firm-sessions in a ticker's first 40 sessions are excluded.

**Index crash: descriptive only.** A 3% forward 20-session drawdown hits 8.6% of
sessions but only 29 separate episodes (4%: 15). That is below the 30-event floor, so no
test is run on it.

## 6. Features

**F0, price-only baseline (index):** `ret_lag0`, `ret_lag1`, `ret_lag2`,
`dow_next_{mon,tue,wed,thu}`, realised volatility (std of the last 20 returns), and
drawdown from the 250-session maximum close. `vol_chg_lag0` is **excluded**: it carries
a fill value on 73 sessions where zero volume is missing data (§8i.7).

**News, compact composite (added to F0):**
- `tone` = mean over the session's scored headlines of `p_positive − p_negative`
  (0 when none),
- `has_news` = 1 if at least one scored headline,
- `log_n` = log(1 + number of scored headlines).

**Four arms (rule 7), each F0 + news built from:** `all` headlines (**primary**),
`ex_price` (price reports removed, `config.PRICE_REPORT_PATTERN`), `placebo` (price
reports only), `orthogonal` (`tone` residualised on `ret_lag0`, `ret_lag1`, `log_n`,
fitted on the training slice).

**Firm panel (H3c).** Baseline: firm return lags 0–2, firm 20-session realised
volatility, firm drawdown from its 60-session maximum, index `ret_lag0`, index realised
volatility. Treatment adds: firm tone (mean `p_positive − p_negative` of headlines
naming the firm over the last 20 sessions, 0 if none), `has_firm_news` over the same 20
sessions, and the index `tone` of the `all` arm.

## 7. Models

- **Direction:** multinomial logistic regression, L2, C = 1.0.
  **Logged exception to rule 4** ("never classify direction"). That rule came from a
  binary label dominated by upward drift, on which logistic collapsed to "always up".
  Tercile classes are balanced by construction, which removes that cause. Collapse is
  still checked: the share of each predicted class is reported.
- **Firm drawdown:** pooled logistic regression, L2, C = 1.0, `class_weight="balanced"`.

## 8. Metrics and tests

**Direction (H3a, H3b).**
- **Primary statistic:** mean per-session log-loss difference, treatment − baseline,
  on identical sessions. Test: moving-block bootstrap, 2,000 resamples, block length
  max(20, 2h). Two-sided p-value and 95% CI.
- **Also reported, not deciding:** macro one-vs-rest ROC-AUC difference with the same
  bootstrap CI, a Diebold-Mariano test on the log-loss differential (Newey-West HAC),
  balanced accuracy, multiclass Brier score, MCC, and predicted-class shares.
- **Power:** the minimum detectable log-loss difference at 80% power
  (2.8 × bootstrap SE) is stated with every null.

**Firm drawdown (H3c).** Primary: PR-AUC difference, treatment − baseline, with a
**time-block** bootstrap (20-session blocks of dates; all firms on a date resampled
together, because events cluster across firms). Also: recall at the alarm threshold
that gives a 5% alarm rate on the training slice, median lead time in sessions,
calibration plot, log-loss difference.

## 9. Families, correction, decision rules

| Family | Tests | Correction |
|---|---|---|
| **P (primary)** | H3a, four arms | Holm across 4 |
| S1 | H3b, `all` arm, h = 5 and h = 20 | Holm across 2 |
| S2 | H3c, one test | none |

**H3a "signal found"** if and only if both hold:
1. the `all` arm lowers log-loss, Holm p < 0.05, and
2. the `ex_price` arm lowers log-loss, Holm p < 0.05.

- If only `all` (and/or `placebo`) pass, the reading is **momentum restated as text, not
  news**.
- If no arm passes, the result is **INCONCLUSIVE**, reported with its minimum detectable
  effect. It is not evidence of absence.
- AUC alone never establishes a signal.
- A significant **increase** in log-loss is reported as "news degrades the forecast",
  as in H1.

S1 and S2 are read the same way, at their own corrections, and are labelled secondary.

## 10. Pre-specified robustness (reported, not corrected, never decisive)

1. H3a `all` arm restricted to 2019+ predictions. Before 2019 the instrument rests on
   Haiku labels (§8j); this shows how much the result depends on them.
2. H3a `all` arm on the v2 scores, 2019+, for comparability with H1/H2.

## 11. Deviations from the original H3 plan, decided before any result

| Plan item | Decision (2026-10-03) |
|---|---|
| XLM-R / mDeBERTa / FinBERT re-scoring | Out of scope: no new model weights. English outlets remain untestable. |
| Window-start sensitivity 2017–2020 | Dropped by the user |
| Long/flat strategy, Sharpe after costs | Dropped by the user |
| Constrained LightGBM | Dropped by the user (H2's GBM underperformed linear) |
| HMM regime model, crash on index | Not run: firm panel chosen as the crash target; index crash descriptive only |
| Topic shares, surprise, EWMA, dispersion | Not in any confirmatory family. May be explored afterwards, labelled exploratory. |
| Labels | +1,800 Claude Haiku labels (gold v3, §8j), approved before this document |

## 12. Outputs

`data/curated/h3_results.json`; AUDIT_REPORT §H3, which includes every null and every
deviation from this document.
