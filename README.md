<h1 align="center">Tunindex News</h1>

<p align="center">
  <em>Does the financial press move the Tunis Stock Exchange? We read 46,000 headlines to find out.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-111111?style=flat-square" alt="Python 3.12">
  <img src="https://img.shields.io/badge/tests-153%20passing-111111?style=flat-square" alt="153 tests passing">
  <img src="https://img.shields.io/badge/every%20test-pre--registered-111111?style=flat-square" alt="Every test pre-registered">
  <img src="https://img.shields.io/badge/data-DVC%20on%20DagsHub-111111?style=flat-square" alt="Data on DagsHub via DVC">
  <img src="https://img.shields.io/badge/language-French%20news-111111?style=flat-square" alt="French news">
</p>

A 2025 study of Borsa Istanbul found that news sentiment helps predict the Turkish market.
This project asks the same question about Tunisia, a much thinner market where trading comes
in bursts and almost all financial news is written in French. If you already know recent
prices, does the news tell you anything more?

## Short answer

What the news says doesn't help. When it comes out does.

When a listed company is in the news after the market closes at 14:10, its next trade moves
more than its recent volatility would predict. The extra movement is about 40% of an average
move. We found it in 2016-2020, confirmed it in 2021-2022, and confirmed it again on
2023-2026 prices whose returns nobody had looked at before the test was written down and
frozen. News
published during trading hours shows no effect we could detect.

| Period | News after the close | News before the close |
|---|---|---|
| 2016-2020 (where the test was designed) | +34% [+25%, +43%] | +4% [−2%, +10%] |
| 2021-2022 (held back, first check) | +47% [+27%, +68%] | +9%, unclear |
| 2023-2026 (new prices, second check) | **+43% [+30%, +56%]** | +6% [−5%, +18%] |

Each number is how much bigger the next move is, as a share of an average move, after
allowing for the stock's own recent swings. Brackets are 95% intervals. Before running
anything we agreed that an effect under 10% would not be worth reporting.

The tone of the news is a different story. Sentiment scores from a French language model
never improved a forecast of the index at any horizon we tried, and a word list chosen by
how prices actually reacted did no better for individual stocks.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/h3_effects_dark.svg">
    <img src="docs/assets/h3_effects.svg" width="860" alt="Effect of adding news sentiment on four forecasts, each with a 95% interval. Next session: slightly worse with news. 5 sessions: about zero. 20 sessions: slightly better, with a wide interval. Firm crash warning: slightly better, with an interval that includes zero. Every interval crosses zero.">
  </picture>
</p>

Each dot shows how much a forecast changes when sentiment is added, with its 95% interval.
Every interval crosses zero, and every estimate is smaller than the effect the test could
have detected (the grey tick).

Prices on their own are useful. A model that sees only past prices flags stocks about to
fall more than 12% within 20 trading days at 2 to 2.5 times the base rate, roughly three
weeks ahead. That also held on the 2023-2026 prices.

The full write-up is in [docs/REPORT.md](docs/REPORT.md).

## Turkey and Tunisia side by side

| | Borsa Istanbul (Ibrahim, Khan & Kaplan, 2025) | Tunis Stock Exchange (this project) |
|---|---|---|
| Market | liquid emerging market | thin frontier market, return autocorrelation +0.26 |
| News | domestic and international outlets | about 46,000 headlines, 98% in French |
| Sentiment model | FinBERT (English finance) | CamemBERT fine-tuned on French market headlines |
| Does sentiment help? | yes | no measurable gain |
| Do international outlets matter more? | yes | could not be tested: too few English headlines, and the French model reads English as neutral |
| What else matters | not tested | timing: company news after the close is followed by a bigger next move |

## How the tests were run

Every test was written down and tagged in git before we looked at the data that would judge
it. Each one was built on an early period and then run once on a later period kept aside
for that purpose. Before a result counted, the code had to recover a fake signal planted in
the data and had to ignore news shifted to the wrong dates. Each test also fixed in advance
the smallest effect worth caring about, so a "no" could be told apart from "too little data
to say".

The plan and every decision along the way are recorded in [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md),
with the pre-registrations next to it in [audit/](audit/).

<details>
<summary><strong>Every test and its result</strong></summary>

<br>

| Question | Result | Details |
|---|---|---|
| Does daily sentiment improve next-day up/down prediction of the index? | No. Every sentiment version scored below the price-only model (ROC-AUC 0.607). | AUDIT §8h.7 |
| Do some outlets help more than others? | Unclear. No French outlet helped; the English ones had too few headlines to test. | AUDIT §8i.6 |
| Does sentiment help a drop / flat / rise forecast at 1, 5 or 20 days? | Unclear at every horizon. All intervals cross zero. | AUDIT §H3 |
| Does the amount of news help forecast index volatility? | No effect worth having, in both periods. | AUDIT §P1 |
| Do company news days move that company's next trade more? | Yes, in all three periods (+12%, +18%, +15.5%). | AUDIT §P2, §F2.4 |
| Is that because of news published after the close? | Yes (table above). | AUDIT §P2t, §F2.4 |
| Does a word list learned from price reactions add anything? | No gain worth having. | AUDIT §P3 |
| Can past prices alone warn of a big drop in a stock? | Yes, at 2 to 2.5 times the base rate. | AUDIT §W, §F2.4 |

</details>

## Data

| Source | Language | Size |
|---|---|---|
| ilboursa, Kapitalis, L'Economiste Maghrébin, La Presse, TAP | French | 44,977 headlines |
| The Guardian, New York Times, The Economist | English | 1,036 headlines |
| Tunindex daily prices, 2010-2026 | | 4,167 sessions |
| Stock prices, 88 listed companies, 2010-2022 | | 187,987 rows |
| Stock prices from ilboursa, 82 companies, July 2022 to September 2026 | | 59,980 rows |

Only ilboursa publishes the time of each headline, which is what made the after-close test
possible.

Two problems turned up along the way. The scraped "Assabah" was the Moroccan newspaper, not
the Tunisian one, so it was dropped, which leaves no Arabic source. And 19 days in the index
file are copies of the previous day (mostly public holidays), so they are removed.

The data is versioned with DVC on DagsHub rather than stored in git.

<details>
<summary><strong>How the sentiment labels were made</strong></summary>

<br>

Each headline was labelled for its likely effect on Tunisian listed shares, which is not the
same as whether it sounds positive.

- 2,930 headlines were labelled by two local language models (qwen2.5-7b and ministral-8b),
  and their disagreements were resolved.
- 150 were labelled by a person. These are used only to grade the models, never to train
  them.
- 1,800 headlines from 2014-2018 were labelled by a third language model, so that the early
  yearly models had enough examples. It was graded first on the 150 human labels and agreed
  with them more closely than either local model (quadratic κ 0.756 against 0.676). It saw
  neither the date nor the outlet.

A CamemBERT model trained on these labels scores every headline. Each year's scores come
from a model trained only on earlier labels.

</details>

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull                          # data from DagsHub
python3 -m pytest preprocessing   # 153 tests
python3 preprocessing/h3.py       # the sentiment forecasts, about 4 minutes on a CPU
OMP_NUM_THREADS=1 python3 preprocessing/f.py --check   # the company-news test, about 3 minutes
```

<details>
<summary><strong>Full pipeline</strong></summary>

<br>

```bash
# corpus
python3 preprocessing/clean.py && python3 preprocessing/relevance.py && python3 preprocessing/dedup.py

# sentiment model (GPU; on Colab use colab/h3_camembert_v3.ipynb, which resumes after a disconnect)
python3 preprocessing/camembert_clf.py --gold data/curated/sentiment_gold_v3.csv \
    --split data/curated/sentiment_gold_v3_split.csv --out data/curated/finetune/eval.json
python3 preprocessing/score_corpus.py --model camembert --min-train 700 \
    --gold data/curated/sentiment_gold_v3.csv --split data/curated/sentiment_gold_v3_split.csv \
    --output data/curated/04_scored_v3_camembert.parquet \
    --metadata data/curated/04_scored_v3_camembert_metadata.json

# hypotheses
python3 preprocessing/hypothesis_tests.py   # H1
python3 preprocessing/outlet_ranking.py     # H2
python3 preprocessing/h3.py                 # H3

# audit and figures
python3 audit/build_audit.py
python3 docs/make_figures.py
```

</details>

A few rules hold everywhere, because breaking any of them gives a result that looks fine and
is wrong:

- News dated on day D only predicts sessions after D, unless its publication time is known.
- Each year's sentiment comes from a model trained only on earlier labels.
- Returns are close to close. In a third of rows the index file's `open` column repeats the
  previous close.
- The 150 human labels are used for grading only.
- Every comparison uses the same trading sessions for both models, with block-bootstrap
  intervals and a correction for running several tests.

## Repository

```
scrapers/        one scraper per outlet, plus Tunindex and stock prices
preprocessing/   corpus, labels, sentiment model, tests of each question, unit tests
audit/           data audit, pre-registrations, AUDIT_REPORT.md
colab/           GPU notebook for training and scoring
docs/            final report, PRD, pipeline reference, README figure
TsEDA.ipynb      time-series exploration
```

## Limits

The English outlets could not be tested and there is no Arabic source, so the comparison
the Turkish study cared most about could not be made. Most training labels come from
language models, so the sentiment result is a result for this particular scorer.

The timing result leans on ilboursa. It is the only outlet with publication times, it
carries about 71% of company news, and it also supplied the 2023-2026 stock prices. Stock
prices are not adjusted for dividends; one-day moves beyond ±10% are filtered out instead.
News published during trading hours was never measured precisely enough to rule out a small
effect. More in [docs/REPORT.md](docs/REPORT.md), section 6.

## Reference

Ibrahim, Khan & Kaplan (2025). *Borsa Istanbul Review*, 25.

<p align="center"><sub>Yassine Ben Sassi</sub></p>
