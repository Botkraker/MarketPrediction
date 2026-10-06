# Pre-registration — P2: firm news vs no news

Written 2026-10-06, **before any P2 coefficient has been estimated with the real news flags
on the real targets**. It is frozen by the git tag `prereg-p2-v1`. Anything decided after
that tag is labelled **post hoc** in AUDIT_REPORT §P2. Plan: ADR-001 §8, P2 card. Open
choices were settled at the owner's interview of 2026-10-06 (STATUS.md). Code:
`preprocessing/p2.py` on the harness `preprocessing/bench.py`.

What was known when this was written:
- H1–H3 (§8h.7, §8i.6, §H3), P0 (§P0, with the MDE table approved at G0) and P1 (§P1:
  news flow adds nothing to the next index swing).
- The interview probe, built on random news flags only:
  - noise spreads of c;
  - 947 design trade days whose previous trade was more than 5 sessions back (188 of them
    news days);
  - 24 of 3,844 issuer headlines that read like a report of the stock's own move;
  - a volatility-arm MDE.
- Prices only: on consecutive sessions in 2016–2020, 17 drops below −10% and 4 rises
  above +10%. The drops sit mostly in April–July. That pattern fits unadjusted dividend or
  bonus-share ex-dates; it is not verified, because the repo has no corporate-actions file.
- P2's own controls (§9), runs 1 and 2. They use planted, shuffled or stale data, never the
  real news against the real targets.
- One property of the real news flags, revealed by run 1. With targets shuffled inside each
  firm (so each firm keeps its level of volatility), the real flags "found" smaller next
  moves after news in 30 of 100 runs, all with the same sign. So firms with more news are
  calmer than the others. That is a between-firm fact. The owner's fix (firm intercepts in
  the volatility model) takes it out of the test; the within-firm effect P2 tests stays
  unseen.

## 1. Hypotheses

**P2a, drift vs reversal.** On a firm's trade day, how much of the move continues over the
next 1 and 5 trades depends on whether there was issuer news. Term c ≠ 0. The ADR's
expectation is c > 0: moves with news continue, moves without news reverse.

**P2b, news-day volatility.** Within a firm, the next trade's absolute abnormal return is
larger after a news day than the firm's recent volatility predicts. Term g ≠ 0.

## 2. Data

| Item | Fixed choice |
|---|---|
| Firm prices | `data/raw/bvmt/ALL_DATA.csv`: `Ticker`, `Date`, `Close` only; `Open`, `High` and `Low` are never read. Duplicate (ticker, date) rows are dropped, keeping the last (rule 11). |
| Index | `data/raw/bvmt/tunindex_2010_today.csv`, `close` |
| Calendar | `audit/trading_calendar.csv` minus the 19 phantom sessions (`h3.calendar`) |
| Headlines | `data/curated/04_scored_v3_camembert.parquet`: every canonical relevant headline, scored or not |
| Issuer match | `h3.issuer_patterns`: the issuer's name (at least 4 characters, minus `config.ISSUER_STOPWORDS`) as a whole word in the lower-cased headline |
| Price reports | `config.PRICE_REPORT_PATTERN` |
| Design window | firm trade days 2016-01-04 → 2020-12-31. Every firm or index price from 2021-01-01 is blank, so targets that would read one are missing. |
| Confirmation window | 2021-01-01 → 2022-12-30 (firm prices end there). **One run**, after this tag, with `--confirm`. It estimates on confirmation-window units only. |

## 3. Units

One unit is a trade day t of a firm that has an issuer pattern. It is kept only if all of
the following hold:
- the firm's **previous** trade is at most 5 index sessions before t, and its **next** trade
  at most 5 after (owner: the card's forward rule, applied backward as well, per ADR T7);
- **no one-trade move beyond ±10%** in r_t or in the target window t+1 … t+k. Those are
  presumed unadjusted corporate actions; the owner delegated this choice under the ADR's
  guards;
- the test's target and controls exist.

Every drop is counted by year and by firm in the output (ADR loophole register).

## 4. Variables

| Variable | Definition |
|---|---|
| r_t | close_t / close at the firm's previous trade − 1 (trade-to-trade) |
| r_m,t | the Tunindex over the same interval |
| news_t | 1 if at least one issuer headline of the arm is dated after the previous trade and up to t, else 0 |
| AR_k,t | (close at the firm's k-th next trade / close_t − 1) − (Tunindex over the same dates), k = 1, 5 |
| Dimson AR | the same, with β × the index term. β is set per firm and year (see below). |
| HAR | \|r_t\|, and the mean \|r\| over the last 5 and the last 22 trades (t included); \|r_m,t\| |

The Dimson β:
- The firm's session returns (close carried over sessions with no trade) are regressed on
  the index returns of the previous, the same and the next session.
- β is the sum of the three slopes, fitted on the 250 sessions before 1 January. The last
  session's lead stays inside that window, so only the past is used.
- Moves beyond ±10% are left out of the fit.
- A firm with fewer than 60 trades in the window gets β = 1.

## 5. Models (OLS)

| Test | Model | Tested term |
|---|---|---|
| c1 | AR_1,t = a + c·r_t·news_t + b·r_t + d·news_t + e·r_m,t | c |
| c5 | AR_5,t, same right-hand side | c |
| vol | \|AR_1,t\| = a_firm + g·news_t + h₁\|r_t\| + h₅·mean₅\|r\| + h₂₂·mean₂₂\|r\| + h_m\|r_m,t\| | g |

- The firm intercepts in `vol` are removed by demeaning within each firm. They were added
  by the owner after controls run 1.
- Arms (rule 6): `ex_price` (price reports removed) decides; `all` and `price_only` are
  reported.
- The news flag is the only text-derived variable.

## 6. Primary test and decision rule

- **Null.** Draw 100 sets of random flags, iid at the arm's news rate on the same units
  (seeds 20261005 + 60,000 + s). Their mean estimate is μ₀ (owner: centre on the random-flag
  mean).
- **Uncertainty.** SE is the standard deviation of the estimate over 2,000 circular block
  bootstrap resamples (seed 20261005):
  - each block is 20 consecutive dates, and every firm on a date moves with it (card);
  - `vol` keeps the full-sample firm means in each resample.
- **Statistic.** z = (estimate − μ₀) / SE, two-sided.
  - 95% interval: (estimate − μ₀) ± 1.96 SE.
  - MDE = 2.8 SE.
  - Holm across the three primary tests.
- **SESOI (owner).**
  - c1: 0.20;
  - c5: 0.40;
  - vol: 10% of the mean |AR_1| over the vol units of the same run.

Verdict per test, applied in this order:

| condition | verdict |
|---|---|
| the 95% interval lies inside (−SESOI, +SESOI) | **NO EFFECT WORTH HAVING**, even if it excludes 0 |
| Holm p < 0.05 | **CHANNEL**; the direction is reported (drift or more reversal with news; bigger or smaller next move) |
| otherwise | **INCONCLUSIVE**, reported with its MDE |

**G2 (ADR).** The confirmation run's `ex_price` verdicts decide:
- **any CHANNEL:** G2 passes, and P3 targets that channel. If there is more than one, the
  owner picks at the G2 review.
- **all three NO EFFECT WORTH HAVING:** Exit the content route. Write up the nulls with
  their MDE, then W or D1 (ADR §3).
- **otherwise:** the owner decides at the G2 review.

The design run is reported and does not decide G2. AUC is not used.

## 7. Secondary families (reported, never decisive)

| Family | What | Correction |
|---|---|---|
| S1 | Dimson-β AR, `ex_price`, the three tests | Holm across 3 |
| S2 | arms `all` and `price_only`, the three tests | Holm within the arm; descriptive |
| S3 | the moves beyond ±10% kept in | Holm across 3; descriptive |
| S4 | leave-one-firm-out: the range of (estimate − μ₀) with each firm left out (no bootstrap) | descriptive |
| — | b, d, e and the HAR coefficients | reported |

## 8. Known limits

- Headlines carry a date only. A headline dated on the move day may come after the 14:10
  close (ADR §2), so part of c can be the first reaction rather than drift. Either way the
  news is known before the next trade.
- Prices are not adjusted for corporate actions. Ex-dates beyond ±10% are excluded; smaller
  dividend ex-dates remain.
- Issuers are matched by name, so short tickers are not matched (config).
- The confirmation window is 2021–2022 only: firm prices end 2022-12-30, and D2 is not
  approved.
- The `vol` bootstrap does not re-estimate the firm means per resample. The controls below
  include that approximation.

## 9. Controls, run before this tag

`p2.py --controls`, `ex_price` arm, design window, G0's bands:
- the planted coefficient (on random flags) must be found in 72–88 of 100 runs;
- targets block-permuted within each firm (20 trades) and stale news (every headline moved
  250 or more sessions back on the design calendar) must each give at most 9 false finds.

**Run 1 (2026-10-06): failed.** Output: `data/curated/p2_controls_run1.json`, copied before
the rerun.

| test | planted found | permuted, false finds | stale, false finds |
|---|---|---|---|
| c1 | 78 | 9 | 4 |
| c5 | 82 | 4 | 6 |
| vol (one intercept) | 75 | **30**, all negative | **15**, all negative |

Owner, 2026-10-06: add firm intercepts to the volatility model only, keep the c models as
the card writes them, and rerun every control with the same bands.

**Run 2 (2026-10-06): pass.** Output: `data/curated/p2_controls.json`.

| test | units (news days) | planted found (median statistic) | permuted, false finds | stale, false finds |
|---|---|---|---|---|
| c1 | 68,126 (2,940) | 78 (2.66) | 9 (+5 / −4) | 4 (+3 / −1) |
| c5 | 67,292 (2,897) | 82 (2.91) | 4 (+2 / −2) | 6 (+0 / −6) |
| vol | 68,028 (2,933) | 75 (2.86) | 5 (+2 / −3) | 5 (+2 / −3) |

## 10. Power

- **Planted effect at a median statistic near 2.8**, the empirical MDE (design window).
- **Confirmation window:** that effect scaled by √(design dates / 505 confirmation sessions).
- **Holm's first step** needs 1.156 times more.

| test | design MDE | projected confirmation MDE | SESOI |
|---|---|---|---|
| c1 | 0.071 | 0.112 | 0.20 |
| c5 | 0.180 | 0.282 | 0.40 |
| vol | 0.00063 (4.9% of mean \|AR_1\| 0.01301) | 0.00099 (7.6%) | 0.00130 (10%) |

For comparison, the G0 table (clustered SE, before the unit rules) had c1 at 0.083 / 0.131
and c5 at 0.263 / 0.411. Each run reports the MDE of its own units.

## 11. Deviations

None at the time of writing.

## 12. Outputs

- `data/curated/p2_controls_run1.json` and `p2_controls.json`;
- `p2_design.json`;
- `p2_confirm.json`, written once (the script refuses to overwrite it).

AUDIT_REPORT §P2 reports every arm, every family, every null with its MDE, the drops by year
and firm, and every deviation from this document.
