# Pre-registration — P2t: P2's news-day volatility channel, split by publication time (D1)

Written 2026-10-07, **before any coefficient has been estimated with the real timed flags
on the real targets**.
- It is frozen by the annotated git tag `prereg-p2t-v1`. The sealed run refuses to start
  unless `preprocessing/` is identical to that tag.
- Plan: ADR-001 §8 (P2 card, track D1) and §10 ("revisit at G2 if D1 is approved").
- Owner decisions: STATUS.md, "P2t decisions".
- Code: `preprocessing/p2t.py`, on `p2.py`.

What was known when this was written:
- **P2.** After a day with issuer news, a firm's next trade moves more than its HAR
  predicts: design +0.00158, sealed +0.00222 (AUDIT_REPORT §P2).
- **The D1 audit** (times only):
  - all 17,265 canonical ilboursa headlines have a publication time;
  - 67.6% were published on a weekday before 14:10;
  - there is no per-day cap on the listing;
  - 0.42% of article ids are back-dated;
  - 70.8% of the 2016–22 firm news is ilboursa, and 71.2% of that is pre-close.
- **P2t's own controls (§7).** They use random flags, shuffled targets or stale headlines,
  never the real timed flags against the real targets.
- **The 2021–22 window has been read twice before:** by P2's sealed run (news presence)
  and P3's (word content). The timing split has never been looked at there.

## 1. Question

Is the bigger next move after issuer news:
- **the first reaction** to news published after the close, which the market can only
  price at the next trade (**timing**); or
- **digestion** of news the market could already price in the same session (**pre-close**)?

## 2. Data and units

**Units.** P2's volatility units: firm trade days with trades within 5 sessions on both
sides, and no one-trade move beyond ±10% in the move or the next trade. Phantom sessions
are dropped from firm and index prices (data rule 2; P2 kept them).

**News.** Issuer headlines with price reports excluded. Each headline belongs to the first
trade on or after its date (P2 card). Each trade day gets three flags, 1 if it holds at
least one headline of that kind:

| flag | definition |
|---|---|
| **post** | timed (ilboursa, D1), published on the trade date at or after 14:10, the BVMT close (ADR §2) |
| **pre** | timed and published earlier: before 14:10 on the trade date, or on an earlier non-trade day |
| **untimed** | other outlets (no publication time) |

**Windows.**
- Design: 2016-01-04 → 2020-12-31.
- Confirmation: 2021-01-04 → 2022-12-30. **One run** with `--confirm` after the tag.

## 3. Model and decisive tests

|AR_1| = a_firm + g_pre·pre + g_post·post + g_untimed·untimed + h₁|r_t| + h₅·mean₅|r| +
h₂₂·mean₂₂|r| + h_m|r_m,t|

The firm intercepts are removed by demeaning within each firm.

**Decisive:** g_pre and g_post.
- **Effect:** each coefficient minus its mean under 100 sets of random flags at its own
  rate (seeds 20261005 + 60,000 + s). The other flags stay as they are.
- **Uncertainty:** SE from 2,000 circular bootstraps of 20-date blocks, all firms
  together.
- **Yardstick, per test** (owner, after controls runs 1–2; §7):

  | test | yardstick |
  |---|---|
  | g_pre | √(random-flag spread² + SE²), P0's rule |
  | g_post | SE alone |

- 95% interval: effect ± 1.96 × yardstick. MDE = 2.8 × yardstick.
- Holm across the two decisive tests.
- **SESOI:** 10% of the mean |AR_1| in the same run (P2's, owner).

Verdict per test, applied in this order:

| condition | verdict |
|---|---|
| the interval lies inside ±SESOI | NO EFFECT WORTH HAVING |
| Holm p < 0.05 | CHANNEL |
| otherwise | INCONCLUSIVE, reported with its MDE |

**Reading** (sealed run; the design run is reported):

| result | reading |
|---|---|
| g_pre a CHANNEL | **DIGESTION**: pre-close news is followed by a bigger next move |
| only g_post a CHANNEL | **TIMING**: the channel is the first reaction to after-close news |
| both NO EFFECT WORTH HAVING | **NEITHER** |
| otherwise | not conclusive; the owner reads it at the review |

## 4. Secondary (reported, never decisive)

- g_post − g_pre, with its bootstrap interval;
- g_untimed, with the P0 yardstick;
- the all-headlines arm (price reports included: rule 6);
- corporate-action headlines removed (P2's post-hoc pattern);
- close cut-offs at 14:00 and 14:30.

## 5. Power

The planted effect at a median statistic near 2.8 (controls run 3, design) gives the MDE.
The sealed window is projected by √(1,239 / 503) design and sealed dates.

| test | design MDE | projected sealed MDE | SESOI |
|---|---|---|---|
| g_pre | 0.00113 (8.6% of mean \|AR_1\| 0.01303) | 0.00177 (13.6%) | 0.00130 (10%) |
| g_post | 0.00140 (10.7%) | 0.00220 (16.9%) | 0.00130 (10%) |

Both sealed MDEs exceed the SESOI.
- In the sealed run, a null can only read INCONCLUSIVE, not NO EFFECT WORTH HAVING.
- A CHANNEL can still be confirmed. For scale, P2's pooled effect was 12% (design) and 18%
  (sealed) of the mean |AR_1|.

## 6. Known limits

- Only ilboursa headlines are timed. 29% of firm news is untimed and carries its own
  flag.
- The close is taken as 14:10 (ADR §2), checked at 14:00 and 14:30.
- Prices are unadjusted; the ±10% screen is P2's.
- 2021–22 has been read twice before, for related questions (above).

## 7. Controls, run before this tag (G0's bands)

Controls: planted (on all-random flags), shuffled targets (20-trade blocks within each
firm), stale headlines (dates moved 250 or more sessions back, time of day kept). Bands:
planted found 72–88; at most 9 false finds each.

Each cell shows planted found / shuffled false finds / stale false finds.

| run | g_pre | g_post | change after the run |
|---|---|---|---|
| 1, SE alone (`p2t_controls_run1.json`) | 79 / 4 / **10** (+3 / −7): fail by one | 84 / 6 / 5: pass | owner: widen as in P0 |
| 2, √(spread² + SE²) for both (`p2t_controls_run2.json`) | 85 / 1 / 0: pass | **91** / 0 / 1: over-powered | owner: each test keeps the yardstick that passed all its own controls |
| 3, per-test yardstick (`p2t_controls.json`) | 85 / 1 / 0: pass | 84 / 6 / 5: pass | **all pass** |

## 8. Deviations

None at the time of writing.

## 9. Outputs

- `data/curated/p2t_controls_run1.json`, `p2t_controls_run2.json`, `p2t_controls.json`;
- `p2t_design.json`;
- `p2t_confirm.json`, written once.

AUDIT_REPORT §P2t reports every flag, arm and cut-off with its MDE, and every deviation.
