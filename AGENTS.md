# AGENTS.md

Working rules for any coding agent in this repository. Tool-agnostic.
Claude-specific entry points are in [CLAUDE.md](CLAUDE.md).

---

## What this repo is

A research pipeline targeting publication. **Methodological rigor is the binding
constraint, not speed.** A fast wrong number is worse than no number: it gets written
into a draft and defended.

Read [architecture-essentials.md](architecture-essentials.md) before touching anything
under `preprocessing/` or `audit/`. It is 90 lines and every one is load-bearing.

## Read order

1. `architecture-essentials.md` — the nine rules
2. `PRD.md` — hypotheses and what counts as a result
3. `architecture.md` — pipeline detail, when you need a specific stage
4. `audit/AUDIT_REPORT.md` — the evidence chain; §8e holds the adversarial review,
   §8i the reproduced instrument, the per-outlet H2 run and the open EDA defects

## Commands

```bash
python3 -m pytest preprocessing        # 117 tests, all must pass
python3 audit/build_audit.py           # regenerate the evidence chain
python3 preprocessing/<stage>.py       # run from repo root, always
```

`python` is **not** on PATH in this environment. Use `python3`.

GPU stages (`camembert_clf.py`, `score_corpus.py --model camembert`,
`translate_headlines.py`) take 40 min to 3 h on the local GTX 1650. Run them in the
background and never two at once — a 4 GB card holds one.

## Hard rules

**Never hand-edit `data/`.** It is DVC-tracked. A defect gets a script fix and a
re-extract. If you cannot regenerate a file with a committed script, it does not
belong in a result.

**Never delete a raw source or a scraper.** Sources are excluded by removing a key from
`config.SOURCE_WINDOWS`; the raw file, the scraper and the `io_raw.SOURCES` entry all
stay so the audit remains reproducible against the evidence that justified the
exclusion. Excluded sources appear in the funnel with `cleaned=0`, never silently
absent.

**Every editorial decision traces to a numbered section of `audit/AUDIT_REPORT.md`,
and every section is regenerable by `build_audit.py`.** A finding that lives only in a
code comment, a JSON metadata field or a chat log does not exist. Two claims went stale
exactly this way. When a decision changes, update the report **in the same commit**.

**Verify before you act on a documented finding — including one in these files.** An
adversarial review of this pipeline produced claims that did not survive recomputation,
in both directions. Recompute, then act.

**No result without its guard.** Non-trivial logic leaves one runnable check behind.
`test_baseline.test_a_future_leaking_feature_scores_near_perfect` feeds the answer in as
a feature and asserts >0.95 — that is what proves ~0.55 means "no signal" rather than
"broken harness". New estimators need the equivalent.

## Statistical conduct

These are not style preferences. Getting them wrong invalidates the paper.

- Report **ROC-AUC** as the headline, never raw accuracy. QWK for the classifier.
- Report the **full** sensitivity grid, never a single favourable cell. The momentum
  result is significant in only 8 of 16 configurations and that must be visible.
- Report **both** the sign test and Diebold-Mariano. Disagreement is a finding.
- A null on an underpowered test is **INCONCLUSIVE**. Write that word.
- Holm-Bonferroni across the four H1 arms. `baseline.py`'s 8 configurations are
  **exploratory** and must be labelled so.
- **Sentiment enters a result only from 2019, and only as `sent_source ==
  "yearly_refit"`.** Earlier yearly models are under-trained, and the saved model saw
  2026 labels. The 2019 boundary was first seen post hoc — say so wherever it is used.
- Holm across outlets for any per-outlet (H2) claim. An outlet whose sentiment is a
  constant (CamemBERT on English) is reported as untestable, never as a null.
- Declare outcome-selected choices. The default estimator (`kind="regress"`) was chosen
  after comparing it against classification on the same data H1 is tested on. That is
  disclosed, not hidden.

## Code conventions

- Match the surrounding file. These modules carry long "why" docstrings that explain
  the methodological reason the file exists — keep that density when you edit them.
- Editorial decisions live in `config.py`, not inline in the stage that uses them.
- Stages do not mutate their input. Each writes a new numbered artifact.
- Flag, do not drop. `dedup.py` marks duplicates and keeps every row; `relevance.py`
  tags rather than filters. The decision to exclude happens once, visibly, in config.
- Prefer deterministic over tuned. Template-key dedup was chosen over a similarity
  threshold precisely because it has no knob to overfit.

## Git

- Branch before committing; do not commit to `master` directly unless asked.
- Commit the report update with the code change that motivated it.
- Two remotes: `origin` (GitHub) and `dagshub` (which also hosts the DVC remote).
  **Push both**, and run `dvc push` — DagsHub holding stale `.dvc` pointers means
  nothing there is reproducible.
- Do not commit the credential embedded in the `dagshub` remote URL to anything shared.

## Scope discipline

Do the task asked. If you find a real problem with it, say so in a sentence or two and
then finish the work under stated assumptions. Do not silently widen scope, and do not
narrow it — if part is blocked, complete everything else and say explicitly what was
left out and why.

Destructive actions — deleting files, rewriting `data/`, force-pushing — are confirmed
with the user first, every time, even when a previous deletion was approved.
