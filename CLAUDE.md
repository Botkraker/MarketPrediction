# CLAUDE.md

Entry point for Claude Code in this repository.
General working rules live in [AGENTS.md](AGENTS.md) and are not repeated here.

---

## Before anything else

Read [architecture-essentials.md](architecture-essentials.md). Nine rules, ~90 lines.
Every one of them is a way to produce a number that looks fine and is wrong.
Always interview me to get better information and context about the project

Then, depending on the task:

| Task | Read |
|---|---|
| Changing a pipeline stage | `architecture.md` §3–6 |
| Analysing the daily series | `architecture.md` §8 — the data dictionary; §8b for the per-headline series |
| Touching statistics or a claim | `PRD.md` §3, `AGENTS.md` "Statistical conduct" |
| Anything that will be written up | `audit/AUDIT_REPORT.md` §8e, §8h and §8i |

## Update — 2026-10-03

H3 (direction + crash) is in progress; plan and decisions in `STATUS.md`. **Gold v3**
(`sentiment_gold_v3.csv`, +1,800 Claude-Haiku labels for 2014–18, `gold_haiku.py`) lifts
CamemBERT to human-eval QWK **0.680** and makes sentiment usable from **2016**:
`04_scored_v3_camembert.parquet` (with class probabilities). AUDIT_REPORT §8j. Heavy jobs
run on Colab (`colab/h3_camembert_v3.ipynb`, resumable per-year cache), not this PC.
The 2019-only statements below describe the v2 scores.

## State — 2026-09-30

Corpus and instrument are done. **H1 has a verdict and it is REJECTED**: every
sentiment arm scores below the price-only baseline (AUC 0.6074), with
Diebold-Mariano significant-worse in 3 of 4 arms. **Caveat found 2026-09-30:** the
yearly CamemBERT models for 2016–18 saw 298–582 gold rows and collapse (2016 calls
everything neutral), and ~28% of H1's predictions sit in those years. **Sentiment is
usable from 2019 only.** H1 has not been rerun on that window.

**H2 has a first per-outlet run** (`outlet_ranking.py`, 2019+): no outlet adds out of
sample, kapitalis degrades the forecast (Holm p 0.036), and the English outlets are
untestable because CamemBERT scores almost every English headline neutral.
INCONCLUSIVE. AUDIT_REPORT §8i.6.

The blocking problem is still reproducibility, but it shrank. **Closed 2026-09-30:**
the CamemBERT fine-tune (`preprocessing/camembert_clf.py`, human-eval QWK 0.618 vs the
off-repo 0.635) and the translation step (`preprocessing/translate_headlines.py`,
opus-mt-fr-en @ `c4aed37b`, 99.23% identical to the committed gold translations).
**Still orphaned:** v2 gold-set construction, XLM-R, the FinBERT head, the macro
controls, the three `sent_resid_*_lag1` columns, and the build of the committed
`daily_features.parquet`. Full list: `architecture.md` §7.

The EDA (`TsEDA.ipynb`) recorded seven data defects, none fixed — including
**look-ahead in the orthogonal arm** (`features.py:127–135` fits on the whole sample)
and 19 phantom sessions (12 zero-volume holidays) in the trading calendar.
AUDIT_REPORT §8i.7.

**Frames:**
- `data/curated/tunindex_timeseries.csv` — the session-level analysis frame, 3,179 × 65.
- `data/curated/tunindex_headline_series.csv` — one row per headline (45,706): English
  text, outlet, CamemBERT label, `sent_source`, Tunindex values. `architecture.md` §8b.
- `daily_features.parquet` stays the modelling frame `hypothesis_tests.py` consumes.

**Uncommitted as of 2026-09-30:** everything from this revision — the four new scripts,
the `score_corpus.py` / `baseline.py` changes, `TsEDA.ipynb`, the docs, and the new data
and saved model (needs `dvc push`).

## Critical path

1. **Commit this revision** and `dvc push` (new scored files, translations, headline
   series, `data/models/camembert_3class`).
2. **Rerun H1 on 2019+.** Rescore with `--min-train` ≈ 700, rebuild features with
   `--scored` pointing at the new file, run the four arms. Declare the window post hoc.
   Fix the orthogonal-arm look-ahead and the phantom sessions first.
3. **Restore the remaining orphans** — above.
4. **Grade the instrument** on the human evaluation split with CIs, and state the label
   ceiling. Point estimate done (0.618).
5. **H2:** score the English outlets with an English-capable model (FinBERT on the
   originals or on `headlines_en.parquet`), then rank, or withdraw H2.

## Subagents

Spawn these only when the work is genuinely parallel and independent. A task that is
merely large is not a reason to fan out — each agent starts cold and re-derives context
this session already has.

| Agent | Owns | Never |
|---|---|---|
| **reproducer** | Reconstructing missing scripts from `data/curated/` metadata; one script per artifact family | Changes a number. If output differs from the committed artifact, it reports the gap. |
| **auditor** | `audit/build_audit.py` and `AUDIT_REPORT.md`; every section regenerable | Edits `preprocessing/` |
| **statistician** | `hypothesis_tests.py`, `sensitivity.py`, `baseline.py`; power, corrections, both tests | Touches the corpus or the instrument |
| **instrument** | Gold set, annotation, agreement, split, sentiment models | Touches `features.py` or anything downstream |
| **janitor** | Dead files, naming, dependency manifest, README drift | Deletes anything without explicit approval |

Contract for all of them: read `architecture-essentials.md` first, run
`python3 -m pytest preprocessing` before reporting done, and state plainly what was not
finished. Findings go into `AUDIT_REPORT.md`, not into a summary message.

## Project-specific reflexes

- `python3`, never `python` — it is not on PATH here.
- Run from the repo root.
- Numbers in these docs were true on 2026-09-30. If one matters to a decision,
  recompute it rather than quoting it.
- The 150-row human evaluation split is the only ground truth in this project. Protect
  it: never train on it, never let a prompt example leak into it.
- `features.py` still defaults to the **v1** `04_scored.parquet`. Pass
  `--scored data/curated/04_scored_v2_camembert.parquet` or you will silently rebuild
  the feature frame on scores its own metadata calls "NOT a sentiment measurement".
- A `_lag1` suffix on a sentiment column does **not** mean `.shift(1)` was applied, and
  must not. The lag is performed upstream by `map_to_next_session(side="right")`; the
  suffix names the offset from the target. Adding a shift would make features two
  sessions stale and break arm alignment. Verified: `architecture.md` §8.
- **Filter `sent_source == "yearly_refit"`** (2019+) before any predictive use of the
  headline series. `saved_model` rows were scored by a model that saw 2026 labels.
- Never score the historical corpus with `data/models/camembert_3class` for a result —
  it exists for new headlines and the labelled descriptive fill.
- CamemBERT scores French. English outlets need a different instrument.
- One GPU job at a time: the GTX 1650 has 4 GB. CamemBERT scoring ~3 h, translation
  ~40 min, saved-model corpus scoring ~40 min. Run them in the background.
- When a finding and a document disagree, the document is stale. Fix the document in the
  same commit.
