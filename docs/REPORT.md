# Does French news tell you anything about Tunisian stocks that prices don't?

Final report, 2026-10-07. Plans: ADR-001 (the pre-registered waterfall, executed, exited at
G3) and ADR-002 (finishing the project, option B). Evidence chain: [audit/AUDIT_REPORT.md](../audit/AUDIT_REPORT.md).
Every number below is cited to a section there or to a result file in `data/curated/`.

## 1. The answer

- **What the news says adds nothing measurable.**
  - Sentiment (v3 CamemBERT tone) never improved a forecast of the index: next-day
    direction, drop/flat/rise, or volatility.
  - Neither did a word list chosen by price reactions, for firm moves.
- **When firm news is published does matter.** After an issuer headline published after the
  14:10 close, that firm's next trade moves more than its own recent volatility predicts,
  by about 40% of an average move.
  - Found on 2016–20, confirmed on 2021–22, and confirmed again on 2023–26 firm prices that
    no test had used. Each confirmation was pre-registered.
  - News the market could already trade on the same day shows no effect we can detect.
- **Prices alone give a usable warning.** A price-only model flags firm drawdowns of more
  than 12% within 20 sessions at about 2–2.5 times the base rate, about 15 sessions ahead,
  with calibrated probabilities. This also held on 2023–26.

## 2. Data

| | what | source |
|---|---|---|
| Headlines | 46,013 canonical headlines from 8 outlets, about 98% French, 2014–2026-09-15 | README; AUDIT §5–§8 |
| Sentiment instrument | CamemBERT fine-tuned on LLM labels with a human-labelled anchor (v3 scores, usable from 2016) | AUDIT §8i–§8j |
| Publication times (D1) | ilboursa re-scrape: all 17,265 canonical ilboursa headlines timed; 71% of 2016–22 firm news is ilboursa | PREREG_P2T; AUDIT §P2t |
| Index | Tunindex daily closes, 2010 → 2026-09-16 (close-to-close returns only) | AUDIT §8b |
| Firm prices | ALL_DATA.csv, 88 firms, 2010–2022; D2 ilboursa download, 82 tickers, 2022-07 → 2026-09-15 | AUDIT §F2.2 |

**Firm prices are not adjusted for cash dividends.** A one-trade move beyond ±10% is
screened out as a presumed ex-date. ilboursa back-adjusts for splits and free shares, and
six tickers were put on its basis (AUDIT §F2.2).

## 3. How it was tested

The binding plan, ADR-001, is a waterfall:
- P0, a test bench with controls;
- P1, index news flow;
- P2, firm news against no news;
- P3, a market-labelled word list;
- P4–P6 (embeddings, fine-tuning, an LLM), run only if P3 passed its gate.

Every phase follows the same rules:
- **Pre-registered before its first confirmatory fit.** A document plus a git tag
  `prereg-<phase>-v1`.
- **Two windows.** A design window, then one sealed confirmation run that refuses to start
  without the tag.
- **Read against matched noise.** Random features or flags at the same rate, with a
  20-date block bootstrap.
- **Controls before any result.** The pipeline must find a planted signal (72–88 times in
  100) and make at most 9 false finds on shuffled targets and stale news (AUDIT §P0).
- **A pre-registered smallest effect of interest (SESOI).** A null counts as evidence of
  absence only when its 95% interval lies inside ±SESOI. Otherwise it is INCONCLUSIVE and
  reported with its MDE (the effect detectable with 80% power).
- **Holm correction** within each phase's decisive family.

ADR-002 added one more confirmation:
- P2, P2t and W were rerun once, with the tagged code unchanged, on firm prices from 2023
  (`prereg-f-v1-a1`, [PREREG_F](../audit/PREREG_F.md)).
- Before the tag, the same path reproduced the 2021–22 sealed results to 1e-9 (AUDIT §F2.3).

## 4. Results

### 4.1 What the headlines say: nothing measurable

| test | window | result | verdict | source |
|---|---|---|---|---|
| H1: daily tone → next-day index direction | 2014– | every sentiment variant below the price-only ROC-AUC 0.607; Diebold-Mariano: loss in 3 of 4 | rejected | AUDIT §8h.7 |
| H3: tone → drop / flat / rise, h = 1, 5, 20 | 2016– | Δ log-loss +0.0028 [−0.0003, +0.0060], MDE 0.0046 (h = 1) | inconclusive | AUDIT §H3 |
| P0: H1 rerun on v3 tone (Clark-West, 4 arms) | 2016–2026 | smallest Holm p 0.31 | inconclusive | AUDIT §P0 |
| P3: market-labelled word list beyond v3, firm volatility (Clark-West gain) | 2017–20 / 2021–22 | +1.6e-08 / +2.1e-09; MDE 1.2e-07 against SESOI 4.5e-06 | **no gain worth having** | AUDIT §P3 |

**P3 decided ADR-001's gate G3.** The content arm stops there, and P4–P6 were not run.

### 4.2 Index news flow: no effect worth having

P1 asked whether the day's news flow helps forecast the next index variance (Δ QLIKE
against noise):
- design 2018–23: −0.00069 [−0.00443, +0.00305], MDE 0.0053, SESOI 0.0086;
- sealed 2024–26: −0.00025 [−0.00395, +0.00345], MDE 0.0053, SESOI 0.0090.

Both intervals lie inside ±SESOI: **no effect worth having** (AUDIT §P1).

### 4.3 Firm news: the publication-time result

**The unit** is a firm trade day. The target |AR_1| is the size of the next trade's return
beyond the index.

**The model** compares each firm with itself: firm intercepts, plus a HAR baseline of the
firm's last move, its 5- and 22-trade average moves, and the index move. P2 adds a flag for
issuer news (price reports excluded). P2t splits that flag by publication time:
- **after close:** published on the trade date at or after 14:10;
- **before close:** published earlier;
- **untimed:** outlets without times.

Effects are shares of the mean |AR_1| in each run. The SESOI is 10%.

| test | design 2016–20 | sealed 2021–22 | **F 2023–26 (untouched)** |
|---|---|---|---|
| P2: any issuer news | +12% (CHANNEL) | +18% [+10.5%, +26%] (CHANNEL) | **+15.5% [+9.3%, +21.7%] (CHANNEL)** |
| P2t: after close | +34.1% [+25.1%, +43.1%] (CHANNEL) | +47.4% [+27.0%, +67.8%] (CHANNEL) | **+43.0% [+29.8%, +56.3%] (CHANNEL)** |
| P2t: before close | +4.3% [−1.8%, +10.4%] (INCONCLUSIVE) | +9.4% (INCONCLUSIVE) | +6.4% [−4.8%, +17.6%] (INCONCLUSIVE) |

Sources: AUDIT §P2, §P2t, §F2.4; `p2t_design.json`, `p2t_confirm.json`, `f_confirm.json`.

**What holds on 2023–26.** Every secondary leaves the after-close effect in place, between
+42% and +45% (AUDIT §F2.4):
- all headlines;
- corporate-action headlines removed;
- closes at 14:00 and 14:30;
- log trading gaps added.

P2's pooled effect stays between +14% and +17% with any one firm left out.

**What does not hold.** Drift, meaning news making the move continue rather than reverse,
is **no effect worth having** in every window (P2 c1 and c5).

**Reading.** The bigger next move is the market's first reaction to news it could not trade
on the same day. It is not slow digestion of news it already had.
- Before-close news shows no effect we can detect. In 2023–26 its MDE is 16% against a SESOI
  of 10%, so a small effect is not excluded.
- What the headline says adds nothing on top (P3).

**Which headlines arrive after the close?** Descriptive only; keyword rules in
`f.py --types`, `f_types.json`. Issuer headlines, price reports excluded:

| window | after close: earnings / dividends / other | before close: earnings / dividends / other |
|---|---|---|
| 2016–20 | 42% / 4% / 54% (734) | 47% / 4% / 49% (1,981) |
| 2021–22 | 39% / 3% / 58% (222) | 53% / 2% / 46% (648) |
| 2023–26 | 48% / 3% / 49% (444) | 40% / 2% / 58% (881) |

Earnings news is no more common after the close than before it. The timing effect is not
simply after-close news being mostly results announcements.

### 4.4 A price-only drawdown warning

H3c's pre-registered price-only model predicts a firm drop of more than 12% within 20
sessions. Platt calibrators are fitted only on earlier predictions. The results below use
the screened sample.

| | ADR-001 W (2019–22) | F (2023–26, untouched) |
|---|---|---|
| PR-AUC [95%] | 0.119 [0.104, 0.140] | 0.104 [0.089, 0.122] |
| event rate (lift) | 0.057 (2.08×) | 0.042 (2.47×) |
| Brier, raw → calibrated | 0.215 → 0.053 | 0.203 → 0.040 |
| alarms: precision / recall | 16.0% / 15.2% | 12.8% / 14.0% |
| median lead time | 15 sessions | 15 sessions |

The rule set in PREREG_F (lower bound above the event rate) is met: **W replicates**
(AUDIT §W, §F2.4).

## 5. Comparison with Ibrahim, Khan & Kaplan (2025)

| | Borsa Istanbul (Ibrahim, Khan & Kaplan, 2025) | Tunis Stock Exchange (this project) |
|---|---|---|
| Market | liquid emerging market | thin frontier market; index return autocorrelation +0.26 |
| News | domestic and international outlets | about 46,000 headlines, 98% French |
| Sentiment model | FinBERT (English finance) | CamemBERT fine-tuned on French market headlines |
| Does sentiment help? | yes | no measurable gain (H1, H3, P0, P3) |
| Do international outlets matter more? | yes | not testable: about 1,000 English headlines, which the French model reads as neutral (H2, AUDIT §8i.6) |
| What news does carry | not tested | **timing**: after-close firm news → bigger next move (P2t, F) |

The Turkish study's columns are as summarised in this repository's README; no other claim
about that paper is made here.

**Why tone may not transfer.** In a thin market, where volume comes in block trades
(AUDIT §8c), the information in a headline seems to be priced at the first trade after it
can be traded on. Its tone adds nothing to that.

## 6. Limitations

- **The sentiment instrument is LLM-labelled.** Most training labels come from language
  models; a human-labelled set anchors evaluation only (AUDIT §8i–§8j). The tone null is a
  null for this instrument.
- **One outlet carries the timing.** ilboursa is 71% of firm news and the only timed source.
  It is also the source of firm prices from 2023.
  - Untimed news rises to 42% of flagged days in 2023–26 (lapresse from 2024).
- **Unadjusted prices.** Cash dividends are not adjusted; the ±10% screen is a patch, and
  the dividend-adjusted secondary was not run (owner).
- **Thin trading.** P2's volatility units need trades within 5 sessions on both sides and no
  move beyond 10%. That keeps 44,498 of 47,304 firm trade days in 2023–26 (AUDIT §F2.4).
- **No Arabic source, and English is untestable.** H2's outlet question remains open.
- **Before-close news stays underpowered.** Its MDE was 20% in 2021–22 and 16% in 2023–26,
  against a SESOI of 10%. In 2016–20 the MDE was 9%, but the interval [−1.8%, +10.4%] just
  crossed the SESOI.

## 7. What was decided after seeing data

Each item below is recorded in its pre-registration or AUDIT section and was approved by
the owner.
- **P0:** the yardstick was widened after controls runs 1–2 (stale news beat iid noise);
  G0 was then passed (§P0).
- **P2:** firm intercepts were added to the volatility model after the first controls run.
  Post-hoc design checks left the channel in place (§P2).
- **P3:** the decisive test became Clark-West after a control showed an empty word list
  "winning" a head-to-head. The MDE comes from a power curve (§P3).
- **P2t was written after P2's sealed result.** Its yardsticks were set after controls runs
  1–2, and it reused the 2021–22 window (§P2t). **This is why ADR-002 asked for F**, and F
  removes the caveat.
- **F:**
  - six back-adjusted tickers were rescaled (owner, before the tag);
  - amendment a1 clips zero trading gaps after a crash that produced no output;
  - a2: W was run by a separate call, because the tagged runner omitted it (PREREG_F §9).

## 8. Reproduce

```bash
dvc pull                                                 # data (DagsHub)
python3 -m pytest preprocessing                          # 153 tests
OMP_NUM_THREADS=1 python3 preprocessing/f.py --check     # F's path reproduces the 2021-22 sealed results
OMP_NUM_THREADS=1 python3 preprocessing/f.py --confirm   # the F replication (refuses if f_confirm.json exists)
python3 preprocessing/f.py --types                       # after-close headlines by type
```

Each ADR-001 phase's commands are in its PREREG and AUDIT section.

**Tags:**
- pre-registrations: `prereg-h1-v1`, `prereg-h3-v1`, `prereg-p1-v1`, `prereg-p2-v1`,
  `prereg-p3-v1`, `prereg-p2t-v1`, `prereg-f-v1`, `prereg-f-v1-a1`;
- `results-v1` (the ADR-001 freeze) and `results-v2` (this report).
