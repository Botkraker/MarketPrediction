# Pre-registration — F: P2, P2t and W once more, on firm prices from 2023 (ADR-002 F2)

Written 2026-10-07, **before any statistic has been computed on firm prices after 2022**.
- It is frozen by the annotated git tag `prereg-f-v1`. `f.py --confirm` refuses to start
  unless that tag exists and `preprocessing/` is identical to it. It writes
  `f_confirm.json` once.
- Plan: ADR-002, option B (owner, 2026-10-07). Owner decisions: STATUS.md, "ADR-002 decisions".
- Code: `preprocessing/f.py`, on the tagged `p2.py`, `p2t.py`, `w.py` and `h3.py`.

What was known when this was written:
- **ADR-001 results** (AUDIT_REPORT §P2, §P2t, §W). Each quantity is a share of the mean |AR_1|.

  | result | design | sealed (2021–22) |
  |---|---|---|
  | P2, firm news-day volatility | CHANNEL, 12% | CHANNEL, 18% |
  | P2t, after-close news | CHANNEL, +34% | CHANNEL, +47% |
  | P2t, before-close news | INCONCLUSIVE | INCONCLUSIVE |

  W: screened PR-AUC 0.119 [0.104, 0.140], 2.08× the base rate.
- **The 2021–22 window served three specs** (P2, P3, P2t). P2t was written after P2's sealed
  result. This is why ADR-002 asks for an untouched window.
- **F2.1 power check** (AUDIT_REPORT §F): headline counts only, GO.
- **D2:** 2022-H2 comparison against ALL_DATA (§7). For 2023 on, only row counts per year
  have been looked at.

## 1. Question

On firm sessions nobody has analysed, 2023-01-01 → 2026-09-15:
- After issuer news published after the close, is the next trade's move still bigger than
  its HAR predicts (P2t g_post)?
- Does P2's pooled news-day effect hold?
- Does W's price-only drawdown warning still beat the base rate?

## 2. Data and windows

**Prices.**
- ALL_DATA.csv up to 2022-12-30, then D2 from 2023-01-01.
- D2: ilboursa daily prices for the tickers that traded in 2022-H2, downloaded 2026-10-07
  by `scrapers/scrape_tunindex.py --ticker`, in `data/raw/bvmt_d2/`. 82 of 83 have a file.
- **Cash dividends are not adjusted**, in either source.
- **ilboursa back-adjusts for splits and free shares.** Six tickers had such events after
  2022 (§7): AB, DH, MPBS, SAH, SMART, STAR.
  - Their ALL_DATA prices are multiplied by the constant 2022-H2 factor (owner), computed
    in `f.prices`. This puts them on D2's basis. Every return before 2023 is unchanged, and
    the 2022 → 2023 junction carries no fake move.
  - Their split days inside the window are smooth, as ilboursa gives them.
- The ±10% one-trade screen stays P2's.

**Index.** `data/raw/bvmt/tunindex_2010_today.csv` (to 2026-09-16).

**News.** As tagged: canonical headlines to 2026-09-15, issuer patterns, ilboursa times from D1.

**Window.**
- Firm sessions 2023-01-01 → 2026-09-15 (`bench.FIRM_SEAL`, `bench.FIRM_END` swapped at run
  time).
- Earlier sessions feed lags, HAR terms, Dimson betas, W's walk-forward fits and calibrators,
  exactly as in the tagged code.
- 2026-09-15 is the last canonical headline (owner). There is no design run: every spec
  below is already fixed.

## 3. Specs, unchanged

| part | spec | code |
|---|---|---|
| P2 | PREREG_P2 at `prereg-p2-v1`: all three tests, the arms and its secondaries | `p2.run(confirm=True)` |
| P2t | PREREG_P2T at `prereg-p2t-v1`: flags, close 14:10, per-test yardsticks, secondaries | `p2t.run(confirm=True)` |
| W | AUDIT_REPORT §W: H3c's price-only model, calibrators fitted on earlier predictions only, the screen | `f.w_run`, which is `w.main`'s steps, scored from 2023-01-01 |

**How the frozen code is reached** (`f.frozen_code_on`):
- the window constants are swapped;
- every read of `ALL_DATA.csv` returns the joined prices;
- P2t's own tag guard is replaced by F's.

The tagged functions are not edited.

**Code changes in `preprocessing/` from `prereg-p2t-v1` to `prereg-f-v1`.** None touches a
tested computation, and §7's check reproduces the sealed results through the new path.
- `config.py`: comment only (price-report share).
- `features.py`: the default scored file. P2, P2t, W and h3 do not use it.
- `h3.firm_panel(end="2022-12-30")` and `w.jump_windows(end="2022-12-30")`: the literal end
  date became an argument with the same default.
- `f.py` and `test_f.py`: new.

## 4. Decisive tests and reading

**Family:** P2's volatility test, and P2t's g_post and g_pre. Holm across the three (owner).

Each test keeps its tagged estimate and yardstick:
- P2: effect minus random-flag mean, over the SE.
- P2t g_post: SE alone.
- P2t g_pre: √(random-flag spread² + SE²).

95% interval = effect ± 1.96 × yardstick.

**SESOI:** 10% of the mean |AR_1| in the same run (P2's and P2t's own).

Verdict per test, applied in this order:

| condition | verdict |
|---|---|
| the interval lies inside ±SESOI | NO EFFECT WORTH HAVING |
| Holm p (F's family) < 0.05 | CHANNEL |
| otherwise | INCONCLUSIVE, reported with its MDE |

**The main claim (g_post):**

| result | reading |
|---|---|
| CHANNEL with a positive effect | **CONFIRMED**: after-close firm news is followed by a bigger next move, on untouched data |
| NO EFFECT WORTH HAVING | **NOT REPLICATED**: the timing result becomes a design-window finding (ADR-002, Revisit) |
| otherwise | **NOT CONFIRMED, NOT REFUTED** |

**W** (owner): **REPLICATES** if the screened sample's PR-AUC lower 95% bound is above that
sample's event rate. The bootstrap is 2,000 resamples of 20-session date blocks, as in W.

## 5. Secondary (reported, never decisive)

- **P2 and P2t, everything their `run()` reports:**
  - P2: c1 and c5, the all and price-only arms, Dimson-beta targets, the run with moves over
    10%, leave one firm out;
  - P2t: g_untimed, g_post − g_pre, all headlines, corporate-action headlines removed, closes
    at 14:00 and 14:30;
  - each script's own Holm p, as tagged.
- **Trading gaps** (ADR-002; owner): log(sessions to the next trade) and log(sessions since
  the previous trade) added to the HAR terms of P2's volatility model and of P2t's model.
- **Dividend-adjusted returns:** not run. The owner chose not to collect dividends.
- **W:** the unscreened sample; Brier and log loss before and after calibration; reliability;
  alarms; lead times; every result by year.

## 6. Power (AUDIT_REPORT §F2.1)

Projected MDEs as a share of the mean |AR_1|:

| test | projected MDE | compared with |
|---|---|---|
| g_post | 16.4–20.6% (about 26% if the backtest's under-prediction repeats) | lower 95% bounds 25.1% (design) and 27.0% (sealed) |
| g_pre | 13–18% | SESOI 10% |
| P2 pooled | 8.3–8.4% | SESOI 10% |

- **g_post:** a CHANNEL is likely if the effect is near its earlier size.
- **g_pre:** the MDE is above the SESOI, so a null can only read INCONCLUSIVE.
- **P2 pooled:** the MDE is below the SESOI, so a null could read NO EFFECT WORTH HAVING.

## 7. Checks before this tag

**D2 against ALL_DATA, 2022-07-01 → 2022-12-30** (`f.py --overlap`, `f_d2_overlap.json`):
- **Pilot** (SFBT, BIAT, SMD): identical sessions, closes and volumes. The full run followed.
- **All 82 files:** the same sessions as ALL_DATA.
  - 76 have identical closes and volumes.
  - Six have closes equal to ALL_DATA's times a constant: AB 0.758, DH 0.5, MPBS 0.5,
    SAH 0.972, SMART 0.663, STAR 0.231. Each factor is constant to 1e-15 across the half-year,
    so it is a back-adjustment for a split or free-share issue after 2022.
  - Every dividend payer (SFBT, BIAT, ...) has a factor of exactly 1, so cash dividends are
    not adjusted.
- **AMI** returns 404 at ilboursa. It has no issuer pattern, so P2, P2t and W never use it.
- **HTML instead of CSV.** 15 tickers had 85-day chunks that came back as HTML. In every
  checkable case these are ranges with no trades:
  - AMV's only 2022-H2 trade is 2022-12-30; SIMPA's first is 2022-10-10.
  - Re-downloading the 14 tickers gave byte-identical files.
- **Coverage.** 81 tickers have rows from 2023 (52,562 rows); ADWYA has none.
- Five issuers have no D2 file (ELBEN, SPHAX, STEQ, TVAL, XABYT). All stopped trading by
  2020.

**The replication path on the ADR-001 sealed window** (`f.py --check`, `f_check.json`):
**passed** on 2026-10-07: `{'p2_vol': True, 'p2t': True, 'w': True, 'pass': True}`, in 2 min 20 s. It
passed again after the six tickers were rescaled.
- It runs the frozen code through `frozen_code_on` with the joined prices, on 2021-01-01 →
  2022-12-30, and W from its start to 2022-12-30.
- It must reproduce `p2_confirm.json` (vol), `p2t_confirm.json` (all three flags) and
  `w_results.json` (screened PR-AUC, Brier, CI, n) to 1e-9.
- Prices after 2022 are blank in that window. None is computed on.

**P2's and P2t's planted and negative controls** (ADR-001, G0 bands) apply unchanged: same
code, same yardsticks. They are not rerun, because rerunning them on this window would read
its prices before the tag.

## 8. Known limits

- **Prices are unadjusted.** The ±10% screen is a patch.
- **The news mix changes.** Untimed news is 42% of flagged pairs in F against 28% in
  2021–22, mostly lapresse from 2024. It has its own flag and never decides.
- **ilboursa is both the price source and the timed-news source.**
- **Only tickers that traded in 2022-H2 are covered** (81 with rows from 2023). Issuer names
  come from the 2022 stock list, so firms listed later have no news flags and are not
  included.
- **Split and free-share days** are smooth for six tickers and appear as jumps nowhere.
  Cash-dividend days still appear as jumps, and the ±10% screen is the only guard.
- **Trading is thin.** Units need trades within 5 sessions on both sides (P2).

## 9. Deviations

None at the time of writing.

## 10. Outputs

- `data/curated/f_power.json`, `f_d2_overlap.json`, `f_check.json`;
- `f_confirm.json`, written once.

AUDIT_REPORT §F reports every test, arm and secondary with its MDE, and every deviation.
