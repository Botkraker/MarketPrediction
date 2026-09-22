# Human anchor annotation — instructions

You are labelling 150 headlines in `data/curated/sentiment_gold_v2_human.csv`.
Fill in `human_label`; `human_notes` is optional but useful where the call is hard.

**Why this exists.** Both LLM annotators score against `PROMPT_V2`, and that prompt
was written by one of the candidate annotator families. Two models agreeing on a
rubric they were both handed is not evidence the rubric is right. Your labels are
the only outside measurement in the loop, and they are what decides whether the
models' heavy use of `neutral` is correct caution or over-correction.

**Before you start:** do not look at `sentiment_gold_v2.csv` (model labels) or
`sentiment_gold_v2_v1labels.csv` (the old v1 labels). The worksheet is deliberately
blind and shuffled out of date order.

## The question

For each headline, answer one question: would a Tunisian equity investor, reading
only this headline, become more or less optimistic about the near-term value of
Tunisian listed companies?

You are **not** judging whether the news is pleasant, whether the topic sounds
economic, whether it describes progress, or whether the words are upbeat. Judge
expected **market impact** only.

## The labels

| label | meaning |
|---|---|
| `very_negative` | large, direct hit to earnings, solvency or the cost of capital: sovereign downgrade, banking crisis, default, major devaluation, sharp index fall |
| `negative` | clear adverse effect of ordinary size: rising inflation, a rate hike, falling profits, widening deficit, a strike at a listed firm |
| `neutral` | no clear directional implication, or effects plausibly offset: unchanged policy, procedural and announcement news, off-topic items |
| `positive` | clear favourable effect of ordinary size: falling inflation, a rate cut, rising profits, new financing secured, improved outlook |
| `very_positive` | large, direct improvement: major investment inflow, sovereign upgrade, debt relief, sharp index rise, record results at a large listed company |

`neutral` is the default and should be common. Choose a directional label only when
you can state which way prices should move and why. Reserve `very_negative` and
`very_positive` for genuinely large or systemic items; they should be rare.

Foreign news is neutral unless it plausibly transmits to Tunisia (oil, EU demand,
Fed or ECB rates, remittances, tourism, grain). Another country's domestic affairs
are neutral.

## Traps

- "increases" / "hausse" is not automatically positive — rising inflation, debt,
  unemployment and money supply are negative or neutral.
- A draft law, plan or project "under preparation" has not happened yet: neutral.
- A headline that merely concerns the economy is not positive. Topic is not sentiment.
- "maintains" / "inchangé" means no change: neutral.
- A question ("Will X happen?") states no fact: neutral.
- A headline reporting the index's own past move is labelled by the direction it
  reports. Note that `grignote` and `s'effrite` are small moves, but they still have
  a direction.

## When you are done

```bash
python preprocessing/merge_human.py
python preprocessing/agreement.py --input data/curated/sentiment_gold_v2.csv
```

`merge_human.py` writes your labels as `annotator_3` (annotators 1-2 are qwen2.5-7b and ministral-8b), leaving the other 850 rows
blank. That is intended: `agreement.py` compares each pair of annotators on the rows
that pair both labelled, so a partial column does not shrink the model-vs-model
statistic. Fleiss' kappa still uses only rows every annotator labelled, and the JSON
reports how many those are.

**Read quadratic-weighted Cohen's kappa as the headline**, not Fleiss. The labels are
ordinal, and a nominal statistic treats `very_negative` vs `positive` as no worse
than `negative` vs `neutral`. The pilot in AUDIT_REPORT.md section 8f scored 0.328
nominal against 0.659 quadratic on the same 60 rows.
