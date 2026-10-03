# PRD — Tunindex News-Sentiment Pipeline

**Status:** active research, pre-publication · **Owner:** Yassine Ben Sassi · **Last revised:** 2026-09-30

---

## 1. Problem

Does French-language news sentiment predict the next session of the Tunisian stock
index (Tunindex, BVMT) better than price history alone?

The question has been answered for liquid emerging markets. Ibrahim, Khan & Kaplan
(2025, *Borsa Istanbul Review* 25) report that it does for Borsa Istanbul, and that
international outlets carry more signal than domestic ones. Nobody has tested either
claim on a **frontier** market — thinner, less liquid, and covered by a press that is
small enough to scrape in full.

Tunisia is a useful test case precisely because it breaks the assumptions: return
autocorrelation is +0.263 (|return| +0.387), trading volume is episodic rather than
news-driven, and the entire relevant press is ~46k headlines rather than millions.

## 2. Hypotheses

| | Claim | Test |
|---|---|---|
| **H1** | Sentiment features add predictive power over lagged price and volume, under walk-forward validation. | Paired comparison against the price-only baseline, four arms, Holm-corrected. |
| **H2** | The Türkiye source ranking (international > domestic) does not transfer to Tunisia. | Per-source ablation on the final model, restricted to the common coverage window. |

H2 has a **first per-outlet run** (2026-09-30, AUDIT_REPORT §8i.6) but not an answer.
Over 2019+ no outlet's sentiment adds to the price-only model out of sample, and
kapitalis makes it worse — **INCONCLUSIVE** at 1,426 predictions. The international
side cannot be tested yet: the corpus has no Arabic, the NYT, Guardian and Economist
total ~1,000 rows against ~45,000 French, and CamemBERT scores almost every English
headline `neutral`. An international-vs-domestic ranking needs an English-capable
instrument first (FinBERT on the originals, or on `headlines_en.parquet`).

## 3. Success criteria

A result is reportable only when **all** of these hold.

**Statistical**
- Primary metric is **ROC-AUC**; balanced accuracy and MCC replace raw accuracy as
  headline figures. At a 54.9% base rate, accuracy is nearly blind.
- H1 must beat the price-only baseline (**AUC 0.6074**, balanced accuracy 0.5675,
  MCC 0.141 over 2,678 walk-forward predictions) under a **paired** test, not a bare
  accuracy comparison.
- Significance survives **Holm-Bonferroni** across the four arms.
- Both the sign test (McNemar) and the continuous test (Diebold-Mariano on
  squared-error loss of returns, Newey-West HAC) are reported. Disagreement between
  them is itself a finding and must be stated, not resolved by picking one.
- Robustness: the `min_train` × `refit_every` grid is reported in full, never a
  single favourable cell.

**Instrumental**
- The sentiment scorer is graded on the **human** evaluation split (150 rows) before
  its scores are treated as a measurement.
- Inter-annotator agreement is reported as **ordinal-weighted** Cohen's κ; nominal
  Fleiss roughly halves it and understates a disagreement that is about level, not
  direction.
- Scoring is leakage-free (`--mode expanding`). Headlines with no prior gold labels
  stay **unscored**, never imputed.
- Sentiment enters a result only from **2019**, the first year whose yearly model saw
  more than 700 gold rows. Saved-model scores (`sent_source = saved_model`) are
  descriptive fill and never enter a result.

**Honesty**
- The design is underpowered on the sign test: MDE at 80% power is **3.2pp** against
  published effects of 1–2pp. A null from the sign test alone is **INCONCLUSIVE**, and
  must be written that way — never as evidence of absence.
- Every editorial decision traces to a numbered section of `audit/AUDIT_REPORT.md`,
  and every section is regenerable by `python3 audit/build_audit.py`.

## 4. Scope

**In scope**
- Headline-level French and English text from 8 Tunisian and international outlets.
- Daily Tunindex OHLCV, 2010-01-04 → 2026-09-16, 4,167 sessions.
- Next-session directional and continuous prediction.
- A fine-tuned encoder as the sentiment instrument.

**Explicitly out of scope**
- Intraday prediction. Headlines are date-only for 65% of the corpus.
- Article bodies. Headlines only.
- Trading strategy or backtest P&L. The model's edge already dies at ~10bps costs;
  this is a predictability study, not a strategy.
- Volume features. News volume vs trading volume is r=+0.018, ns, flat across all
  8 sources individually.
- Arabic, until a genuine Tunisian Arabic outlet is scraped.

## 5. Current status — 2026-09-30

**Done**
- Corpus: 46,013 canonical relevant headlines, 8 sources, audited end to end.
- Analysis frame: `data/curated/tunindex_timeseries.csv`, 3,179 sessions × 65 columns
  (price, counts, sentiment from both scorers, three macro series), built and verified
  by `preprocessing/build_timeseries.py`. Dictionary in `architecture.md` §8.
- Gold set v2: 2,930 rows, prompt v2, three annotators (qwen2.5-7b, ministral-8b,
  and 150 human rows). Neutral share 9.4% → 52.8%, which was the explicit gate.
- Frozen split 2,339 / 441 / 150; evaluation **is** the human-labelled subset.
- Sentiment instrument: CamemBERT fine-tuned, validation QWK **0.708**, human
  evaluation QWK 0.635. Beats TF-IDF (0.561) and XLM-R (0.589) decisively; FinBERT
  on raw French is at chance (κ=0.0005) and needs translation to function.
- Macro controls tested and all three rejected on evidence: Brent, EuroStoxx 50,
  EUR/TND. None enters F1.
- H1 harness runs, four arms, both tests, block-bootstrap ΔAUC CIs.
- **Added 2026-09-30:**
  - CamemBERT fine-tune reproduced by `camembert_clf.py`: human-eval QWK 0.618 against
    the off-repo 0.635, within seed noise. Saved model in `data/models/camembert_3class`.
  - Translation reproduced by `translate_headlines.py` (opus-mt-fr-en, pinned
    revision): 99.23% identical to the committed gold translations; whole corpus
    translated.
  - `tunindex_headline_series.csv`: one row per headline, English text, outlet,
    CamemBERT label, Tunindex values, and a `sent_source` column separating leak-free
    from filled rows.
  - First per-outlet H2 run (`outlet_ranking.py`), plus a gradient-boosting variant of
    the walk-forward (`kind="gbm"`, weaker than linear).
  - Time-series EDA (`TsEDA.ipynb`): found the pre-2019 instrument collapse and seven
    data defects (AUDIT_REPORT §8i.7).

**Current H1 verdict: REJECTED.** Every sentiment arm scores *below* the price-only
baseline. Diebold-Mariano finds the degradation significant in 3 of 4 arms
(p = 0.036–0.039). This is not evidence that news sentiment cannot predict Tunindex —
it is evidence that *these* daily aggregates carry no signal the price lags do not.
**Caveat found 2026-09-30:** ~28% of the 2,678 predictions fall in 2016–18, where the
sentiment instrument is degenerate. The verdict stands until H1 is rerun on 2019+.

**Not done**
- Orphaned artifacts remain the blocking defect (`architecture.md` §7). *Closed:* five
  feature columns (2026-09-28), the CamemBERT fine-tune and the translation step
  (2026-09-30). *Still orphaned:* v2 gold-set construction, XLM-R, the FinBERT head,
  the macro controls, `sent_resid_*_lag1`, and the build of the committed
  `daily_features.parquet`.
- **H1 on 2019+:** rescore with `--min-train` ≈ 700, rebuild features, rerun the four
  arms, declared as a post-hoc restriction.
- **Seven EDA defects** unfixed, including look-ahead in the orthogonal arm
  (`features.py:127–135`) and phantom holiday sessions (AUDIT_REPORT §8i.7).
- An English-capable instrument for the international outlets (H2).
- SHAP / feature attribution.
- Committing this revision: the new scripts, and a DVC push of the new data and the
  saved model.

## 6. Milestones

| # | Milestone | Exit condition |
|---|---|---|
| M1 | Reproducibility restored | Every file in `data/curated/` is regenerable from a committed script. *In progress: 5 of 8 orphan columns closed; CamemBERT and translation closed.* |
| M2 | Instrument graded | CamemBERT scored on the human evaluation split, with CIs, and the label ceiling stated. *Point estimate done (QWK 0.618 reproduced); CIs and ceiling outstanding; pre-2019 collapse documented.* |
| M3 | H1 final | Four arms, both tests, Holm, sensitivity grid, block-bootstrap CIs, **on 2019+**, EDA defects fixed. |
| M4 | H2 or formal withdrawal | *First per-outlet run done (§8i.6, French outlets only).* Next: an English-capable instrument, then either a fr-vs-en / international-vs-domestic ranking with the imbalance declared, or formal withdrawal. |
| M5 | Paper-ready | Every claim traces to a regenerable audit section. |

## 7. Risks

| Risk | Impact | Standing mitigation |
|---|---|---|
| Design underpowered on the sign test | A true 1–2pp effect is undetectable | Report DM alongside; call nulls inconclusive |
| Labels are LLM-generated | Reviewers discount the instrument | 150 human rows anchor it; ordinal κ reported; qwen breaks 27.6% of ties and that is disclosed |
| Price-report headlines launder momentum | H1 "works" for the wrong reason | Four arms; placebo and orthogonal arms are mandatory |
| Unreproducible artifacts | Result cannot be defended | M1 blocks everything downstream |
| Corpus is 97% French | H2 unanswerable | Declared as a limitation, not a finding |
| Instrument collapses before 2019 | Pre-2019 sentiment is noise | Result window starts 2019; saved-model fill tagged and excluded |
| CamemBERT blind on English | International outlets have no sentiment | Score them with an English-capable model before any H2 ranking |
| Phantom sessions / missing volume | Fake targets and baseline features | Recorded in §8i.7; fix before M3 |
