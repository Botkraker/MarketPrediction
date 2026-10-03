# Does financial news predict the Tunisian stock market?

A replication of a Borsa Istanbul news-sentiment study on a frontier market: the
Tunis Stock Exchange (BVMT) and its main index, Tunindex. The project scrapes about
twelve years of Tunisian financial headlines (2014–2026), builds a sentiment instrument
for French-language financial news, and tests whether that sentiment improves market
forecasts beyond what price history already gives.

**Short answer: it does not.** Across three pre-registered tests, adding news sentiment
to a price-only model never improves the out-of-sample forecast. Price history on its
own does carry information, including a usable three-week warning for large drops in
individual stocks.

## Background: the Turkish study

Ibrahim, Khan & Kaplan (2025, *Borsa Istanbul Review* 25) combined FinBERT sentiment
with ensemble machine learning and SHAP explanations on Borsa Istanbul. They report that
news sentiment improves prediction of the Turkish market, and that international outlets
carry more signal than domestic ones.

Both findings come from a liquid emerging market. This project asks whether they hold in
a frontier market, where conditions are different:

- **Thin trading.** Daily Tunindex returns have an autocorrelation of +0.26, far above
  liquid-market levels, so yesterday's move already predicts part of today's.
- **Episodic volume.** Trading volume is driven by occasional block trades, not by news
  flow (correlation between news volume and trading volume: +0.02).
- **A small, French-language press.** The whole relevant press fits in about 46,000
  headlines, mostly in French, which rules out English financial models used as they are.

## Research questions and results

| | Question | Result |
|---|---|---|
| **H1** | Does daily news sentiment improve next-session up/down prediction over price history? | **Rejected.** Every sentiment variant scores below the price-only baseline (ROC-AUC 0.607). A Diebold-Mariano test finds the loss significant in 3 of 4 variants. |
| **H2** | Does the Turkish ranking of sources (international above domestic) carry over? | **Inconclusive.** No single outlet adds out of sample; one (Kapitalis) makes forecasts worse. The English outlets could not be tested (see Limitations). |
| **H3** | Does news improve a three-way forecast (drop / flat / rise) at 1, 5 or 20 sessions, or warn of firm-level crashes? | **Inconclusive at every horizon.** News slightly worsens the next-session forecast, but not significantly after correction. For crashes, news adds nothing to a price-only early-warning model. |

### H3 in numbers

H3 was pre-registered ([audit/PREREG_H3.md](audit/PREREG_H3.md), git tag
`prereg-h3-v1`) before any model was fitted.

| Test | Price-only | + news | Difference [95% CI] |
|---|---|---|---|
| Next session, log-loss (lower is better) | 1.0790 | 1.0818 | +0.0028 [−0.0003, +0.0060] |
| 5 sessions, log-loss | | | +0.0016 [−0.004, +0.008] |
| 20 sessions, log-loss | | | −0.0055 [−0.017, +0.005] |
| Firm drop > 12% within 20 sessions, PR-AUC | 0.179 | 0.182 | +0.0025 [−0.003, +0.008] |

None of these differences is significant. Each null is reported with the smallest effect
the test could have detected, so they are inconclusive results, not proof that news is
useless. The price-only crash model flags drops at 2.2 times the base rate, with a median
of 15 trading days' warning.

## Data

| Source | Language | Headlines (canonical, relevant) |
|---|---|---|
| ilboursa, Kapitalis, L'Economiste Maghrébin, La Presse, TAP | French | 44,977 |
| The Guardian, New York Times, The Economist | English | 1,036 |
| Tunindex daily prices, 2010–2026 | | 4,167 sessions |
| Per-stock prices, 88 listed firms, 2010–2022 | | 187,987 rows |

Headlines were scraped by the scripts in `scrapers/`. The data itself is versioned with
DVC on DagsHub, not stored in git.

Two data problems shaped the design:

- **The Arabic source was the wrong newspaper.** The scraped "Assabah" was the Moroccan
  paper, not the Tunisian one. It was detected by a geography check and excluded, so the
  corpus has no Arabic.
- **The index file's `open` column is unreliable.** In a third of rows it repeats the
  previous close, so all returns are close-to-close.

The full evidence chain is in [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md).

## Method

1. **Corpus.** Clean, filter for relevance (Tunisian economy, listed issuers, global
   links), and remove near-duplicates with a template key that keeps signs and clause
   order.
2. **Gold labels.** 4,730 headlines labelled for expected market impact:
   - 2,930 by two local LLMs (qwen2.5-7b, ministral-8b), adjudicated;
   - 150 by a human, kept aside as the only ground truth;
   - 1,800 by Claude Haiku, added to give the pre-2019 years enough training data.

   Haiku was first checked against the human labels, where it agreed better than either
   local model (quadratic κ 0.756 vs 0.676).
3. **Sentiment instrument.** CamemBERT fine-tuned on the gold labels (3 classes,
   3 seeds). On the 150 human headlines it reaches quadratic κ 0.680. Every year is scored
   by a model trained only on earlier labels, so no score uses future information.
4. **Forecasting.** Expanding-window walk-forward, refit every 20 sessions, with an
   embargo of at least the forecast horizon. News published on day D only predicts
   sessions after D.
5. **Testing.** Paired comparisons on identical sessions:
   - block-bootstrap confidence intervals;
   - Diebold-Mariano tests;
   - Holm correction within each family of tests.

   Price-report headlines ("Tunindex closes up 0.3%") restate the market's own move, so
   every test is also run without them and on them alone.

## Repository layout

```
scrapers/          one scraper per outlet, plus Tunindex prices
preprocessing/     corpus, labels, sentiment model, features, hypothesis tests, unit tests
audit/             data audit, pre-registration, AUDIT_REPORT.md (the full evidence chain)
colab/             GPU notebook: CamemBERT training and yearly scoring, resumable
docs/              PRD.md (scope and success criteria), architecture.md (pipeline reference)
TsEDA.ipynb        time-series exploration of the analysis frame
```

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull                                   # restores data/ from DagsHub
python3 -m pytest preprocessing            # 121 tests
```

Main steps, run from the repository root:

```bash
# corpus
python3 preprocessing/clean.py && python3 preprocessing/relevance.py && python3 preprocessing/dedup.py

# sentiment instrument (GPU; on Colab use colab/h3_camembert_v3.ipynb)
python3 preprocessing/camembert_clf.py --gold data/curated/sentiment_gold_v3.csv \
    --split data/curated/sentiment_gold_v3_split.csv --out data/curated/finetune/eval.json
python3 preprocessing/score_corpus.py --model camembert --min-train 700 \
    --gold data/curated/sentiment_gold_v3.csv --split data/curated/sentiment_gold_v3_split.csv \
    --output data/curated/04_scored_v3_camembert.parquet \
    --metadata data/curated/04_scored_v3_camembert_metadata.json

# hypotheses
python3 preprocessing/hypothesis_tests.py  # H1
python3 preprocessing/outlet_ranking.py    # H2
python3 preprocessing/h3.py                # H3, about 4 minutes on CPU
```

The data audit regenerates with `python3 audit/build_audit.py`.

## Limitations

- **No Arabic, and English outlets are untested.** CamemBERT reads French only and scores
  almost every English headline neutral, so the international-versus-domestic comparison
  from the Turkish study could not be run. The obvious next step is an English-capable
  model for the roughly 1,000 English headlines.
- **Date-only timestamps.** Headlines carry no time of day, so news is only allowed to
  predict the following session. Any same-day effect is invisible by design.
- **Statistical power.** About 2,160 out-of-sample sessions. Effects smaller than the
  stated minimum detectable effects would not show up.
- **LLM-labelled training data.** Most gold labels come from language models; 1,800 come
  from a model that may know how events turned out. Dates and sources were hidden from it,
  and results are unchanged when the period it labelled is excluded.
- **Firm prices stop in 2022**, which limits the crash panel to 2016–2022.

## Citation

Ibrahim, Khan & Kaplan (2025). *Borsa Istanbul Review*, 25.

## Author

Yassine Ben Sassi
