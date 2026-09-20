# Tunindex Sentiment Pipeline

Data collection and preprocessing for testing whether French and Arabic news sentiment improves next-session prediction of the Tunisian stock index (Tunindex, BVMT) beyond a price-only baseline.

The design follows the architecture blueprint (v1.1, September 2026), which adapts the FinBERT + ensemble ML + SHAP approach of Ibrahim, Khan & Kaplan (2025, *Borsa Istanbul Review* 25) to a bilingual, low-liquidity frontier market. The blueprint frames two hypotheses:

- **H1:** sentiment features add predictive power over lagged price, volume and macro controls under walk-forward validation.
- **H2:** the source ranking found for Turkiye (international outlets dominate) does not automatically transfer to Tunisia.

The audience is NLP and quantitative finance students and researchers working on frontier-market text. Only the data and preprocessing stages exist in this repo. No sentiment model, forecasting model or result is included yet, and none is claimed.

## Features

- Nine headline scrapers (French, Arabic and English outlets) plus a Tunindex OHLC scraper: `scrape_*.py`, `leconomistemaghrebin_Scraper.py`.
- Reproducible data audit of the raw headline CSVs, including a **wrong-country provenance check**: `audit/build_audit.py`, findings in [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md).
- Preprocessing pipeline: cleaning and source windows, bilingual relevance filter, headline deduplication, funnel counts.
- Sentiment gold-set tooling: stratified sampling, local LLM pre-annotation, adjudication, frozen train/validation/evaluation split.
- Unit tests for normalisation, relevance, dedup, gold set, annotation and split logic.
- Data versioned with DVC.

## Tech stack

Python, pandas, requests, lxml, curl_cffi, DuckDB, matplotlib, langdetect, fastText (`lid.176.ftz` model), pytest, DVC.

## Architecture

```mermaid
flowchart LR
    A[scrape_*.py] --> B[data/raw/*.csv]
    B --> C[preprocessing: clean, relevance, dedup]
    C --> D[data/curated/*.parquet]
    D --> E[gold.py sampling]
    E --> F[llm_annotate.py + adjudicate]
    F --> G[split.py]
    B --> H[audit/build_audit.py]
```

### Active sources

Eight of the ten scraped sources feed the pipeline, yielding **42,645 canonical relevant
headlines** (see `data/curated/funnel.csv`). Two are excluded, both recorded in the funnel with
`cleaned=0` rather than silently dropped:

| excluded source | reason |
|---|---|
| `economist_tunisia_economy` | 100%-overlapping subset of `economist_tunisia_all` (AUDIT_REPORT §4) |
| `assabah` | **wrong country** — `scrape_assabah.py` targets `assabah.ma` (Morocco), crime section, not Tunisia's `assabah.com.tn`. 11,860 Morocco mentions vs 27 Tunisia; 175 dirham vs 0 dinar. AUDIT_REPORT §6b |

Exclusion is a single key in `config.SOURCE_WINDOWS`. Raw files and scrapers are **retained**,
never deleted, so the audit stays reproducible.

Because `assabah` was the only Arabic source, the corpus is currently **French-dominant**
(~97% fr, plus en); the bilingual comparison in H2 is on hold until an Arabic outlet is
scraped. This is a known limitation, not a finding.

Later stages in the blueprint (language-routed sentiment scoring, daily features aligned to the BVMT calendar, walk-forward models, SHAP) are not implemented here.

## Prerequisites

- Python 3 <!-- TODO: minimum Python version -->
- Git and DVC
- Access to the DVC remote configured in [.dvc/config](.dvc/config) <!-- TODO: how collaborators obtain DagShub credentials -->
- Optional, for `llm_annotate.py`: a local LM Studio server on `localhost:1234` serving `qwen2.5-7b-instruct-1m`
- Optional, for the Tunindex Selenium fallback: Chrome

## Installation

```bash
git clone <repo-url>  # TODO: repository URL
cd ProjectNLP
python -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow requests certifi lxml curl_cffi python-dateutil duckdb matplotlib langdetect pytest dvc
dvc pull
```

There is no `requirements.txt` or `pyproject.toml`. The package list above comes from the imports in the code. `pyarrow` is needed for the `.parquet` files, and `truststore` and `selenium` are optional.

`dvc pull` restores `data/` (tracked by [data.dvc](data.dvc)) and the fastText model tracked by [audit/models/lid.176.ftz.dvc](audit/models/lid.176.ftz.dvc).

## Configuration

No environment variables are read by the code. Settings live in files:

| Setting | Where | Default | Description |
|---|---|---|---|
| `SOURCE_WINDOWS` | [preprocessing/config.py](preprocessing/config.py) | per source | Date window kept for each source |
| `BOILERPLATE_TITLES` | preprocessing/config.py | per source | Rubric labels dropped as non-headlines |
| `FASTTEXT_CONFIDENCE_THRESHOLD` | preprocessing/config.py | `0.5` | Placeholder, not yet tuned |
| `ARABIZI_MIN_WORDS` | preprocessing/config.py | `2` | Words needed to flag a headline as Arabizi |
| Relevance keyword lists | preprocessing/config.py | first-pass | Not yet validated against hand labels |
| `--endpoint`, `--model` | `llm_annotate.py` flags | `http://localhost:1234/api/v1/chat`, `qwen2.5-7b-instruct-1m` | Local annotation server |

## Usage

Scrapers write their CSV to the current directory. Each is run as a script:

```bash
python scrape_tap.py
python scrape_tunindex.py
```

The preprocessing scripts read `data/raw/` and write `data/curated/`. Run them in order from the repo root:

```bash
python preprocessing/clean.py
python preprocessing/relevance.py
python preprocessing/dedup.py
python preprocessing/funnel.py
```

Build and annotate the sentiment gold set (defaults are set in each script):

```bash
python preprocessing/gold.py --target 3000
python preprocessing/llm_annotate.py
python preprocessing/adjudicate_from_annotator.py
python preprocessing/split.py
```

Rebuild the audit outputs in `audit/`:

```bash
python audit/build_audit.py
```

## Project structure

```text
.
├── scrape_*.py                  # one scraper per source, plus Tunindex OHLC
├── leconomistemaghrebin_Scraper.py
├── preprocessing/               # cleaning, relevance, dedup, gold set, split, tests
├── audit/                       # raw-data audit script, report and CSV outputs
├── data/                        # DVC-tracked: raw/, curated/, per-ticker CSVs
├── data.dvc                     # DVC pointer for data/
├── .dvc/                        # DVC configuration
└── main.ipynb                   # scratch notebook
```

## Testing

```bash
python -m pytest preprocessing
```

3 of 28 tests currently fail (`test_gold`, `test_adjudicate_from_annotator`,
`test_llm_annotate`). Pre-existing and unrelated to source exclusion — verified identical
before and after that change.

## Contributing

<!-- TODO: contribution guidelines -->

## License

<!-- TODO: no LICENSE file in the repo; choose a license -->

## Author

<!-- TODO: author and contact -->
