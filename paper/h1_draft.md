# Does French-language news sentiment predict the Tunisian stock index? A pre-registered null

*Draft, 2026-09-23. Methods and results for H1. Every number is taken from `audit/AUDIT_REPORT.md` (§8h–§8j) and `audit/PREREG_H1_AMENDMENT_A1.md`, which give the provenance. The introduction and related work are still to be written.*

## Abstract (draft)

We test whether the daily sentiment of French-language financial headlines improves next-session prediction of the Tunindex, the main index of the Bourse de Tunis, beyond a model that uses only prices.

- **Corpus.** 46,013 deduplicated, relevant headlines from eight outlets, 2014–2026.
- **Scorer.** A fine-tuned CamemBERT, trained on a 2,930-headline gold set labelled by two local LLM annotators and anchored by 150 human labels. It reaches quadratic-weighted κ 0.635 against the human labels.
- **Protocol.** The analysis was pre-registered, with one amendment committed before the run. It uses walk-forward evaluation with a five-session embargo, a price-report placebo arm, and a momentum-orthogonalised arm.
- **Result.** Sentiment does not improve out-of-sample AUC. All four arms score slightly below the price-only baseline (ΔAUC −0.003 to −0.006; 95% block-bootstrap intervals include zero), and none improves in more than 44% of quarterly test blocks. A Diebold–Mariano test finds that sentiment slightly worsens return forecasts.

The result is a well-controlled null for a low-liquidity frontier market. Its limits are that headlines carry no timestamps and that there is a single human annotator.

## 1. Data

**Headlines.** Eight outlets were scraped: French, plus 1,048 English headlines. Cleaning, a relevance filter with country negation and listed-issuer matching, and template-key deduplication leave **46,013 canonical relevant headlines** (from 47,784 relevant rows). Language is assigned per source, not detected per headline.

Two scraped sources are excluded. One is a strict subset of another outlet. The other, an Arabic outlet, turned out to be the Moroccan *Assabah* rather than the Tunisian one: it has 11,860 mentions of Morocco against 27 of Tunisia. The corpus therefore contains no Arabic, and H2, the bilingual hypothesis, is not tested here.

**Prices.** Daily Tunindex closes come from the BVMT, 2010–2026. Returns are close-to-close. The `open` field equals the previous close on about a third of rows, so it is not used. The index rises on 54.2% of sessions, so an "always up" rule already reaches that accuracy. All accuracy figures are reported against a walk-forward majority baseline for this reason.

**Alignment.** Headlines carry a date but no time. A headline dated *D* is only allowed to predict sessions strictly after *D*. News on non-trading days is carried forward to the next session. This rule rules out any leak of same-day information, at the cost of discarding any genuine same-day signal (§5).

**Macro series.** These were used to test F1 controls (§3.2):
- Brent spot from the EIA;
- the Central Bank of Tunisia's daily interbank EUR/TND average, 3,206 fixings;
- Euro Stoxx 50 closes, 3,209.

## 2. Sentiment measurement

### 2.1 Gold labels

The labelling rubric (PROMPT_V2) asks for the headline's **expected impact on the Tunisian market**, on a five-point scale that is collapsed to three classes. An earlier rubric left the task undefined. Under it the same model labelled 61% of headlines positive and 8% neutral; under PROMPT_V2 the figures are 31% and 54%. That earlier set was discarded.

**Sample.** 2,930 headlines were labelled by two local models from different families: qwen2.5-7b and ministral-8b. Between the two models, quadratic κ is 0.577 on the core 1,000 rows and 0.597 on the extension. Both miss the blueprint's 0.6 target.

**Human anchor.** A human labelled a blind subset of 150 headlines. Agreement with the human is κ 0.689 for qwen and 0.571 for ministral.

**Adjudication.** The final label comes from:
- the human, on the 150 anchor rows;
- model agreement, on 1,970 rows;
- a qwen tiebreak, on 810 rows (28%).

**Splits.** Train 2,339, validation 441 (both LLM-adjudicated), and **evaluation = the 150 human rows**, which were used exactly once.

### 2.2 Scorer selection

All candidates were trained on the train split and compared on validation. No choice was tuned on validation:
- TF-IDF uses fixed settings.
- FinBERT's head regularisation is chosen by cross-validation inside train.
- The fine-tuned encoders' epoch count is chosen on a 15% dev fold carved out of train, and each averages three seeds.

| scorer | QWK | macro-F1 |
|---|---:|---:|
| TF-IDF character n-grams | 0.561 | 0.670 |
| FinBERT head, French → English translation | 0.604 | 0.664 |
| TF-IDF + FinBERT, averaged | 0.625 | 0.697 |
| XLM-R base, fine-tuned | 0.589 | 0.698 |
| **CamemBERT base, fine-tuned** | **0.708** | 0.732 |

CamemBERT improves on the best translation-based scorer by +0.083 QWK (paired bootstrap 95% CI [+0.018, +0.151]). Translation loses phrases that carry sentiment: *sans entrain* ("without enthusiasm") comes out as "without training". On the human evaluation split, CamemBERT reaches **QWK 0.635 [0.524, 0.732]**. That lies between the two LLM annotators it learned from. Its main error is reading neutral headlines as positive.

### 2.3 Scoring the corpus without leakage

The scorer is refit at the start of each calendar year, on gold labels dated strictly before that year, and then scores only that year's headlines. This covers 39,692 of the 46,013 headlines. The years 2014–2015 have too few earlier labels to score.

The price of this protocol is weak early models. Checked out of time, CamemBERT beats TF-IDF in every year from 2019 (κ 0.60–0.64 against 0.33–0.48). It loses in 2017–2018, when it was fine-tuned on only about 400–600 labels. That weakness was left in, not patched with a second scorer. Section 4 shows it does not drive the result.

## 3. Design

### 3.1 Target, model and baseline

- **Target:** the next session's return. H1's model regresses the return and takes its sign. That estimator was chosen after comparing it with classification on the same data, and it is declared as outcome-selected.
- **Walk-forward:** an expanding window with 500 initial sessions, a refit every 20 sessions, and a five-session embargo. That gives 2,678 paired predictions, 2016–2026.
- **Baseline F0:** the three most recent closed-session returns, the change in log volume, and the target session's day of week.

### 3.2 F1 controls tested, not assumed

A candidate control enters the baseline if its day-*D* move predicts the next session's return at *p* < 0.05, using OLS with Newey–West errors. This rule was fixed before testing.

| candidate | *r* with next-session return | *p* | enters |
|---|---:|---:|---|
| Brent | +0.024 | 0.43 | no |
| EUR/TND | +0.004 | 0.81 | no |
| Euro Stoxx 50 | +0.062 | 0.072 | no, a near miss |

H1 is therefore tested against F0.

### 3.3 Sentiment arms

Each arm adds the day's mean sentiment, its positive and negative shares, and a news-present indicator to F0.

- **all:** every scored headline.
- **ex_price:** without headlines that restate the index's own move, about 10% of the corpus.
- **placebo:** those price-report headlines only. If this arm "works", sentiment is carrying momentum, not news.
- **orthogonal:** sentiment residualised on recent returns and headline volume. The residual is fitted only on earlier sessions.

### 3.4 Decision rule (pre-registered)

The primary metric is ΔAUC, the arm's AUC minus F0's, computed on the paired predictions.

- **Inference:** a circular block bootstrap (20-session blocks, 2,000 resamples), with Holm correction across the four arms.
- **An arm passes** if its Holm-adjusted *p* is below 0.05 and ΔAUC is positive in at least two-thirds of the quarterly test blocks.
- **H1 is supported** only if both ex_price and orthogonal pass.

Secondary tests are McNemar on directional accuracy and Diebold–Mariano on squared return errors. The analysis also reports ΔAUC for 2020 alone, and a purged five-fold diagnostic.

## 4. Results

| arm | AUC | ΔAUC | 95% CI | quarters with ΔAUC > 0 |
|---|---:|---:|---|---:|
| F0 (baseline) | 0.6074 | — | — | — |
| all | 0.6016 | −0.0058 | [−0.0123, +0.0001] | 42% |
| ex_price | 0.6026 | −0.0048 | [−0.0113, +0.0009] | 30% |
| placebo | 0.6036 | −0.0038 | [−0.0097, +0.0017] | 40% |
| orthogonal | 0.6046 | −0.0028 | [−0.0065, +0.0001] | 44% |

The Holm-adjusted *p* is 0.216 for every arm. **No arm passes, and H1 is not supported.** The purged five-fold diagnostic agrees: F0 scores 0.597 and the arms 0.593–0.598. Removing 2020 changes nothing.

The secondary tests point the same way:
- Directional accuracy does not differ from F0 in any arm (McNemar *p* 0.31–0.86).
- The Diebold–Mariano test finds return forecasts slightly *worse* with sentiment in the all, ex_price and placebo arms (*p* ≈ 0.04; out-of-sample R² 0.053–0.056 against 0.0615 for F0).

The design's minimum detectable effect in directional accuracy is 3.2 percentage points. That exceeds the 1–2 point effects typical of published studies, which is why ΔAUC was made the primary test.

**The weak early scorer does not explain the null.** In the ex_price arm, the quarters from 2019 onward, where the scorer is at its strongest, are also mostly negative or zero.

**The baseline is not trivial.** F0 beats the walk-forward majority rule in all 16 settings of a sensitivity grid over the training-window start and refit interval, and does so significantly in 12 of them. Sentiment had to clear a real bar.

## 5. Limitations

1. **No timestamps.** A headline dated *D* can only predict *D+1*. If headline sentiment is priced within the same session, this design cannot see it by construction. One outlet now records timestamps, which makes a same-day test possible. It would require a new pre-registered amendment.
2. **Label quality.** The two model annotators miss the κ 0.6 target. 28% of labels are one model's tiebreak. A single human anchors the evaluation, so human–human reliability is unmeasured.
3. **Scorer ceiling.** κ 0.635 against the human leaves room for noise to hide a small effect. The result speaks to sentiment *as measured by this scorer*.
4. **French only.** There is no Arabic source, so the bilingual hypothesis H2 is untested.
5. **The estimator was outcome-selected.** Regression was chosen over classification on the same data, as declared. Every other constant was fixed before the run.

## 6. Reproducibility

- **Code and history:** the protocol is tagged `prereg-h1-v1-a1`, and every step has a script under `preprocessing/`.
- **Data:** versioned with DVC.
- **Model revisions:** recorded in the scoring metadata.
- **Pinned seeds:** splits, fine-tuning, and the bootstrap (20260923).

*To be written: introduction, related work (Ibrahim, Khan & Kaplan 2025, the reference design), and discussion.*
