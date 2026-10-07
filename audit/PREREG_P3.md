# Pre-registration — P3a: a market-labelled word list for firm events

Written 2026-10-06, **before any word list has been fitted on the real targets with the
real headlines**.
- It is frozen by the annotated git tag `prereg-p3-v1`. The confirmation run refuses to
  start unless `preprocessing/` is identical to that tag.
- Anything decided after the tag is labelled **post hoc** in AUDIT_REPORT §P3.
- Plan: ADR-001 §8, P3 card, aimed at the channel G2 confirmed (AUDIT_REPORT §P2).
- Owner decisions: STATUS.md, "P3 decisions".
- Code: `preprocessing/p3.py`, on `p2.py` and `bench.py`.

What was known when this was written:
- **Earlier phases.**
  - H1–H3 and P1: no sentiment or index-level effect.
  - P2: after a day with issuer news, a firm's next trade moves more than its HAR predicts.
    Drift was below its SESOI, and the post-hoc checks are in §P2.
- **The interview probe** (headline text and trade dates only):
  - firm events per year;
  - eligible words per training window: none if words are learned from 2016 only, 117 at
    the first refit if learned from 2014;
  - 3–4 corporate-action words among the eligible ones.
- **P3's own controls (§9) and power curve (§10).** They use planted words on shuffled
  targets, shuffled targets, and headlines moved 250 or more sessions. They never pair
  the real words with the real targets.

## 1. Hypothesis

**P3a.** Within firm events, the content of the issuer headlines adds to the forecast of
the next trade's absolute abnormal return, beyond:
- the firm's own volatility (F0);
- v3's negative share.

Content here means a word list that price reactions select. ADR §7: "B wins only if the
market reacts to content that positive/negative tone misses."

## 2. Data

| Item | Fixed choice |
|---|---|
| Firm prices | `ALL_DATA.csv`: `Ticker`, `Date`, `Close`; `Open`, `High` and `Low` are never read. Duplicates are dropped (keep last). **Phantom sessions are dropped** (data rule 2; P2 kept them, AUDIT_REPORT §P2). |
| Index | Tunindex `close`, phantom sessions dropped |
| Calendar | `h3.calendar()` (phantom-free) |
| Headlines | `04_scored_v3_camembert.parquet`: `headline_clean`, `published_date`, `sent_label`. Issuers matched by `h3.issuer_patterns`; price reports (`config.PRICE_REPORT_PATTERN`) excluded. |
| Words learned from | 2014-01-01 (owner) |
| Model comparisons | events from 2016-01-01 (v3 is valid from 2016, §8j) |
| Design | forecasts for 2017–2020 events. Every price from 2021-01-01 is blank. |
| Confirmation | forecasts for 2021–2022 events. **One run** with `--confirm` after the tag: the walk runs through 2016–2022 and only 2021–22 events are scored. P2's sealed run read 2021–22 firm prices once, for a different question (whether there was news). No word list or forecast has been fitted there. |

## 3. Units

A firm event is a firm trade day with at least one issuer headline (price reports
excluded) that is also one of P2's volatility units:
- the previous and the next trade are within 5 sessions;
- no one-trade move beyond ±10% in the move or the next trade;
- the target and the HAR terms exist.

A headline belongs to the first trade on or after its date. When that trade is not a unit,
the headline is dropped, never moved.

## 4. Variables

| Variable | Definition |
|---|---|
| y | \|AR_1\|: the next trade's market-adjusted return, in absolute value |
| F0 | forecast of the in-fold firm model (see below) |
| neg_share | share of the event's headlines that v3 labels negative |
| words | per headline: issuer names become "issuer"; corporate-action words (`p2_posthoc.CORPORATE`) are removed; `dedup._template_key` lower-cases and masks numbers and date words. Unigrams and bigrams. The event's document is the union over its headlines, counted once (binary). |

The firm model is an OLS of y on |r_t|, the mean |r| over the last 5 and the last 22
trades, and |r_m,t|, with firm intercepts. A firm unseen in training gets the mean
intercept. It is fitted on all units whose next trade comes before the refit date.

## 5. The word list (card)

On each 1 January τ, from 2016 to the last year:
1. Fit the firm model on the units whose next trade is before τ.
2. Label each training event (next trade before τ) by y − F0. The top tercile is "big"
   and the bottom tercile "small"; the middle is dropped, for the screen only.
3. Keep the eligible words: those in at least 30 training events and at least 6 distinct
   quarters.
4. Test each eligible word with a two-sided binomial test: its share of "big" among the
   labelled events against the overall share. Benjamini–Hochberg at 10%.
5. Score each event as (n_big − n_small) / (n_big + n_small + 1), over the selected words
   it contains.

Each event of year y gets its F0 and its score from the fits of 1 January of y, never from
later data.

## 6. Forecasts and the primary test

**Models.** Refit every 1 January on 2016+ events whose next trade comes before it. Four
OLS models of y:
- (F0);
- (F0, neg_share);
- (F0, score);
- (F0, score, neg_share).

A word score that is 0 for every training event is left out, so the model is exactly the
smaller one (controls run 2).

**Primary test (decides G3).** Clark-West of F0 + v3 + words against F0 + v3.
- Per event: f = (y − ŷ_v3)² − [(y − ŷ_both)² − (ŷ_v3 − ŷ_both)²]. The gain is the mean
  of f.
- SE: standard deviation of that mean over 2,000 circular bootstraps of 20-date blocks.
  All firms of a date move together; seed 20261005.
- z = gain / SE, set to 0 when the two forecasts coincide.
- 95% interval: gain ± 1.96 SE. MDE = 2.8 SE.

**SESOI (owner):** 2.5% of the F0 + v3 MSE in the same run.

Verdict, applied in this order:

| condition | verdict |
|---|---|
| the interval lies inside (−SESOI, +SESOI) | **NO GAIN WORTH HAVING** |
| lower end > 0 | **WORDS ADD TO V3** |
| upper end < 0 | WORDS HURT |
| otherwise | **INCONCLUSIVE**, reported with its MDE |

There is one decisive test, so G3 needs no Holm correction (3b is exploratory, since G1
did not pass).

## 7. G3 (ADR)

G3 passes only if all three hold:
- the design verdict is WORDS ADD TO V3;
- the sealed verdict is WORDS ADD TO V3;
- the words are stable: those selected at the sealed run's last refit recur in at least
  50% of its refits (median selection frequency).

Otherwise: **Exit** (ADR G3).

## 8. Secondary (reported, never decisive)

- **Head-to-head:** D = MSE(F0 + words) − MSE(F0 + v3), same bootstrap. Known bias: an
  empty list skips the coefficient v3 must estimate (controls run 1), so D leans toward
  the words when they carry nothing.
- **Clark-West:** F0 + words vs F0, and F0 + v3 vs F0.
- **Robustness arms** (Clark-West, words added to v3):
  - corporate-action words kept;
  - issuer names unmasked;
  - an L2 logistic on the screened words (C = 1);
  - DMR: one Poisson GLM per eligible word on the residual, with offset the log of the
    event's eligible-word count.
- **Diagnostics (card):**
  - selection frequency of every word, and Jaccard between consecutive refits;
  - the gain with each test year left out;
  - a contemporaneous check: Spearman between the score and the same-session |r|.
    Validity only, never a result.
  - a price-report placebo: issuer price-report headlines, 2016–2020, count only. Too few
    for the screen, so not estimable.

## 9. Controls, run before this tag (G0's bands)

`p3.py --controls`. Planted word: the target is shuffled within each firm in 20-trade
blocks, the word is added to 10% of events, and their y is raised. Shuffled targets: same
shuffle, no plant. Stale news: every headline moved back 250 or more sessions on the
2014–2020 calendar (circular).

**Run 1 (`p3_controls_run1.json`): decisive test = head-to-head.**

| control | result |
|---|---|
| planted | 78, pass |
| shuffled | 6, pass |
| stale | **16, all "words better": fail** |

- Diagnosis, on stale data only: in 16 of 20 stale runs the screen selected no word. So
  "F0 + words" was just F0, while F0 + v3 paid for a coefficient on a meaningless feature
  (ADR T2).
- Firm intercepts did not help (3 vs 3 false finds).
- Owner: the decisive test becomes the Clark-West add-on.

**Run 2 (`p3_controls_run2.json`): Clark-West.**

| control | result |
|---|---|
| planted | 70, fail |
| shuffled | 10, fail |
| stale | 17, fail |

- Cause: with an empty list, the two forecasts differed only by rounding (about 1e-18),
  so z was noise over noise. The median SE was 6e-20.
- Fix: a score that is 0 in every training row is left out of the model.

**Run 3 (`p3_controls.json`).**

| control | result |
|---|---|
| planted | **70 of 100, fail by 2** (median statistic 2.79) |
| shuffled | 4 (2 add / 2 hurt), pass; empty list in 59 runs |
| stale | 2 (0 add / 2 hurt), pass; empty list in 85 runs |

- Diagnosis (same seeds): z varies with a spread of 1.59 across planted runs.
  - The screen kept the planted word in all 5 refits in only 39 runs, and 38 of those
    found it.
  - It kept the word in 2 refits or fewer in 11 runs, and 1 of those found it.
- **Owner:** accept this as a measured power loss from the card's screen, and correct the
  MDE (§10). Deviation from the G0 band, logged in §12.

## 10. Power (`p3.py --power`, `p3_power.json`)

The planted control at 1.0 to 2.0 times the calibrated effect, same seeds:

| effect (× calibrated) | 1.0 | 1.15 | 1.3 | 1.5 | 1.75 | 2.0 |
|---|---|---|---|---|---|---|
| found, of 100 | 70 | 91 | 95 | 98 | 99 | 99 |
| median gain, share of the MSE | 0.87% | 1.29% | 1.89% | 2.75% | 3.78% | 5.03% |

- **MDE at 80% power:** at most 1.29% in design, and about 2.14% projected for the sealed
  window (× √(2,302 / 837)).
- **SESOI:** 2.5% sits above both.
- The negative controls are conservative: most null runs select no word and score exactly
  0.

## 11. Known limits

- Headlines carry a date only, as in P2.
- Corporate-action words are removed by a pattern, which can miss some.
- Prices are unadjusted; the ±10% screen is the same as P2's.
- At small effects the screen drops real words (the power loss in §9–§10).
- P2 has already read the 2021–22 firm prices once, for a different question.
- v3's labels trace back to LLM-labelled training rows (ADR T6). v3 is the rival, not the
  tested feature, so any look-ahead in v3 makes G3 harder, not easier.

## 12. Deviations

- The planted control missed the G0 band: 70 against a floor of 72. The owner accepted it
  (§9).
- None other at the time of writing.

## 13. Outputs

- `data/curated/p3_controls_run1.json`, `p3_controls_run2.json`, `p3_controls.json`;
- `p3_power.json`;
- `p3_design.json`;
- `p3_confirm.json`, written once.

AUDIT_REPORT §P3 reports every family and arm, every null with its MDE, the stability
diagnostics and every deviation from this document.
