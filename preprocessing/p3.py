"""ADR-001 P3a: a word list learned from firm price reactions, head to head with v3.

Plan: ADR-001 §8, P3 card, aimed at the channel G2 confirmed (AUDIT_REPORT §P2). The owner's
decisions (2026-10-06) are in STATUS.md and are fixed in audit/PREREG_P3.md before the
design run. Firm prices from 2021-01-01 stay blank until the annotated tag prereg-p3-v1
exists, the code in preprocessing/ matches it, and --confirm is given.

UNIT    a firm event: a firm trade day with >= 1 issuer headline (price reports excluded),
        under P2's unit rules (trades within 5 sessions both ways, no move beyond 10% in
        the move or the next trade), phantom sessions dropped (data rule 2).
LABEL   the next trade's |AR| minus the in-fold firm model's forecast (firm intercepts,
        HAR on |r|, |market|: P2's volatility channel). The top and bottom terciles of the
        training events train the words; the middle is dropped for the screen only.
WORDS   issuer names, numbers and date words masked; corporate-action words removed
        (owner); unigrams + bigrams, once per event; kept if in >= 30 training events and
        >= 6 quarters; two-sided binomial test of each word's share of 'big' events against
        the training share, Benjamini-Hochberg at 10%; score = (big - small) / (big + small
        + 1). Learned from 2014 (owner); refit every 1 January; each event is scored by the
        list fitted before its year.
MODELS  2016+ events, refit every 1 January: OLS of |AR| on F0 (the firm model's forecast),
        F0 + v3 negative share, F0 + word score, F0 + both. Design forecasts 2017-2020.
TEST    do the words add what v3 misses (ADR §7)? Clark-West, F0 + v3 + words vs F0 + v3,
        SE from 2,000 bootstraps of 20-date blocks (all firms of a date together) (owner,
        after controls run 1: the head-to-head favoured an empty word list, which skips the
        coefficient v3 must estimate, ADR T2). Reported: the head-to-head, words vs F0 and
        v3 vs F0 (Clark-West).

    OMP_NUM_THREADS=1 python3 preprocessing/p3.py --controls --jobs 10  # P3's own controls
    OMP_NUM_THREADS=1 python3 preprocessing/p3.py                       # design 2016-2020
    OMP_NUM_THREADS=1 python3 preprocessing/p3.py --confirm             # once, after the tag
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression

import bench
import h3
import p2
from config import PRICE_REPORT_PATTERN
from dedup import _template_key
from p2_posthoc import CORPORATE

TAG = "prereg-p3-v1"
WORDS_FROM, TEST_FROM = "2014-01-01", "2016-01-01"   # owner: words from 2014; v3 valid from 2016
MIN_DOCS, MIN_QUARTERS, FDR = 30, 6, 0.10             # card
DOCS = ("toks", "toks_corp", "toks_issuer")            # corporate words removed / kept / issuer names kept
ARMS = {"f0": ["f0"], "v3": ["f0", "neg_share"], "lex": ["f0", "score"], "both": ["f0", "score", "neg_share"]}
SESOI = 0.025                                          # owner, 2026-10-06: 2.5% of the F0 + v3 MSE
PLANT_RATE = 0.10                                      # share of events given the planted word


# ------------------------------------------------------------------ data
def tokens(text: str) -> frozenset:
    w = _template_key(text).split()
    return frozenset(w) | frozenset(f"{a} {b}" for a, b in zip(w, w[1:]))


def headlines() -> pd.DataFrame:
    """Issuer headlines, price reports excluded: one row per (headline, ticker) with its
    date, its v3 'negative' flag and three token sets (DOCS)."""
    s = pd.read_parquet(h3.SCORED_V3, columns=["headline_clean", "published_date", "sent_label"])
    s = s[~s.headline_clean.astype(str).str.contains(PRICE_REPORT_PATTERN, case=False, regex=True, na=False)]
    low = s.headline_clean.astype(str).str.lower()
    pats = h3.issuer_patterns()
    masked = low
    for pat in pats.values():
        masked = masked.str.replace(pat, " issuer ", regex=True)
    parts = [pd.DataFrame({"ticker": t, "day": pd.to_datetime(s.published_date[m]).dt.normalize(),
                           "negative": (s.sent_label[m].astype(str) == "negative").astype(float),
                           "masked": masked[m], "low": low[m]})
             for t, pat in pats.items() if (m := low.str.contains(pat)).any()]
    h = pd.concat(parts, ignore_index=True)
    drop = lambda x: re.sub(CORPORATE, " ", x)
    return h.assign(is_price_report=False, toks=h.masked.map(lambda x: tokens(drop(x))),
                    toks_corp=h.masked.map(tokens), toks_issuer=h["low"].map(lambda x: tokens(drop(x))))


def code_matches_tag(tag: str = TAG) -> bool:
    return subprocess.run(["git", "diff", "--quiet", tag, "--", "preprocessing/"], cwd=h3.ROOT).returncode == 0


def data(confirm: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The firm panel from 2014 without phantom sessions, and the issuer headlines. The
    confirmation run needs the tag and code identical to it (review, AUDIT_REPORT §P2)."""
    if confirm:
        bench.seal(pd.DataFrame({"session": pd.to_datetime([])}), "p3", confirm=True)
        if not code_matches_tag():
            raise SystemExit(f"preprocessing/ differs from {TAG}: the confirmation run uses the tagged code")
    h = headlines()
    p = p2.panel(confirm, h, start=WORDS_FROM, drop_phantoms=True)
    return p.assign(next_session=p.groupby("ticker").session.shift(-1), y=p.ar1.abs()), h


def units(p: pd.DataFrame, confirm: bool) -> pd.DataFrame:
    """P2's volatility units (AUDIT_REPORT §P2), in time order within each firm."""
    window = p.session.notna() if confirm else p.session < bench.FIRM_SEAL
    keep = (window & (p.back_gap <= p2.GAP) & (p.gap <= p2.GAP) & p.clean1
            & p[["r", "r_m", "ar1"] + p2.HAR].notna().all(axis=1))
    return p[keep].reset_index(drop=True)


def events(p: pd.DataFrame, h: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """The units with issuer news. A headline belongs to the first trade on or after its date
    (P2 card); when that trade is not a unit the headline is dropped, never moved."""
    parts = []
    for t, g in h.groupby("ticker"):
        trades = pd.DatetimeIndex(p.session[p.ticker == t])
        j = trades.searchsorted(pd.DatetimeIndex(g.day), side="left")
        parts.append(g[j < len(trades)].assign(session=trades[j[j < len(trades)]]))
    union = lambda s: frozenset().union(*s)
    agg = (pd.concat(parts).groupby(["ticker", "session"])
           .agg(neg_share=("negative", "mean"), **{c: (c, union) for c in DOCS}).reset_index())
    return u.reset_index(names="uidx").merge(agg, on=["ticker", "session"], how="inner")


# ------------------------------------------------------------------ the word list
def matrix(docs: pd.Series) -> tuple[sparse.csr_matrix, np.ndarray]:
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(docs):
        for w in s:
            rows.append(i)
            cols.append(vocab.setdefault(w, len(vocab)))
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
    return X, np.array(list(vocab), dtype=object)


def bh(p: np.ndarray, q: float = FDR) -> np.ndarray:
    """Benjamini-Hochberg: True for the p-values rejected at false-discovery rate q."""
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    keep = np.zeros(len(p), bool)
    if ok.any():
        keep[order[:np.flatnonzero(ok).max() + 1]] = True
    return keep


def screen(X, train: np.ndarray, label: np.ndarray, quarter: np.ndarray):
    """Card screen on the training events: eligible words (>= MIN_DOCS events, >= MIN_QUARTERS
    quarters), then a two-sided binomial test of their share of 'big' (label 1) among the
    labelled events against the overall share, Benjamini-Hochberg at FDR."""
    Xt = X[train]
    Q = sparse.csr_matrix((np.ones(train.sum()), (np.arange(train.sum()), quarter[train])))
    eligible = ((np.asarray(Xt.sum(0)).ravel() >= MIN_DOCS)
                & (np.asarray(((Xt.T @ Q) > 0).sum(1)).ravel() >= MIN_QUARTERS))
    lab = train & ~np.isnan(label)
    n = np.asarray(X[lab].sum(0)).ravel()
    k = np.asarray(X[lab & (label == 1)].sum(0)).ravel()
    base = float(np.mean(label[lab] == 1))
    idx = np.flatnonzero(eligible & (n > 0))
    pv = np.array([stats.binomtest(int(k[j]), int(n[j]), base).pvalue for j in idx])
    sel = idx[bh(pv)] if len(idx) else idx
    share = k[sel] / n[sel]
    return sel[share > base], sel[share < base], np.flatnonzero(eligible)


def score(method, X, big, small, eligible, train, label, resid, now) -> np.ndarray:
    """card: (big - small) / (big + small + 1). Robustness (card): an L2 logistic on the
    screened words, or DMR (one Poisson GLM per eligible word on the residual)."""
    Xn = X[now]
    if method == "card":
        nb, ns = np.asarray(Xn[:, big].sum(1)).ravel(), np.asarray(Xn[:, small].sum(1)).ravel()
        return (nb - ns) / (nb + ns + 1)
    if method == "logit":
        cols, lab = np.r_[big, small], train & ~np.isnan(label)
        if len(cols) == 0:
            return np.zeros(now.sum())
        fit = LogisticRegression(C=1.0, max_iter=2000).fit(X[lab][:, cols], label[lab])
        return fit.decision_function(Xn[:, cols])
    Xt = X[train][:, eligible]                                           # dmr
    m = np.asarray(Xt.sum(1)).ravel()
    phi = np.zeros(len(eligible))
    for j in range(len(eligible)):
        c = np.asarray(Xt[:, j].todense()).ravel()
        try:
            phi[j] = sm.GLM(c[m > 0], sm.add_constant(resid[train][m > 0]), family=sm.families.Poisson(),
                            offset=np.log(m[m > 0])).fit().params[1]
        except (np.linalg.LinAlgError, ValueError):
            phi[j] = 0.0
    mn = np.asarray(Xn[:, eligible].sum(1)).ravel()
    return (Xn[:, eligible] @ phi) / np.maximum(mn, 1)


def firm_model(rows: pd.DataFrame):
    """P2's volatility model without news: OLS of |AR| on HAR with firm intercepts. Returns
    the forecast for any rows; a firm unseen in training gets the mean intercept."""
    firm = pd.factorize(rows.ticker)[0]
    X, y = rows[p2.HAR].to_numpy(float), rows.y.to_numpy(float)
    beta = np.linalg.lstsq(p2.within(X, firm), p2.within(y, firm), rcond=None)[0]
    alpha = pd.Series(y - X @ beta).groupby(rows.ticker.to_numpy()).mean()
    return lambda d: d.ticker.map(alpha).fillna(alpha.mean()).to_numpy(float) + d[p2.HAR].to_numpy(float) @ beta


def walk(u: pd.DataFrame, ev: pd.DataFrame, years, docs: str = "toks", method: str = "card"):
    """The yearly walk-forward. On each 1 January the firm model and the word list are fitted
    on rows whose next trade is before that date; that year's events get their F0 and their
    score from them. The four models are refitted on earlier 2016+ events the same way."""
    X, vocab = matrix(ev[docs])
    quarter = pd.factorize(ev.session.dt.to_period("Q"))[0]
    ev = ev.assign(f0=np.nan, score=np.nan)
    chosen = {}
    for y in years:
        tau = pd.Timestamp(f"{y}-01-01")
        fm = firm_model(u[u.next_session < tau])
        train = (ev.next_session < tau).to_numpy()
        resid = ev.y.to_numpy(float) - fm(ev)
        lo, hi = np.quantile(resid[train], [1 / 3, 2 / 3])
        label = np.where(resid > hi, 1.0, np.where(resid < lo, 0.0, np.nan))
        big, small, eligible = screen(X, train, label, quarter)
        now = (ev.session.dt.year == y).to_numpy()
        ev.loc[now, "f0"] = fm(ev[now])
        ev.loc[now, "score"] = score(method, X, big, small, eligible, train, label, resid, now)
        chosen[y] = {"eligible": len(eligible), "big": sorted(vocab[big]), "small": sorted(vocab[small])}
    out = []
    for y in list(years)[1:]:
        tau = pd.Timestamp(f"{y}-01-01")
        tr = ev[(ev.session >= TEST_FROM) & (ev.next_session < tau)]
        te = ev[ev.session.dt.year == y]
        row = te[["ticker", "session", "y", "score", "abs_r"]].copy()
        for name, cols in ARMS.items():
            cols = [c for c in cols if tr[c].any()]     # a score that is 0 in every training row adds nothing
            b = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols]], tr.y, rcond=None)[0]
            row[name] = np.c_[np.ones(len(te)), te[cols]] @ b
        out.append(row)
    return pd.concat(out, ignore_index=True), chosen


# ------------------------------------------------------------------ tests of the forecasts
def date_boot(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """N_BOOT means over circular resamples of 20-date blocks, all firms of a date together."""
    codes, uniq = pd.factorize(dates, sort=True)
    sums, n = np.bincount(codes, weights=values), np.bincount(codes)
    W = p2.block_counts(len(uniq))
    return (W @ sums) / (W @ n)


def head_to_head(t: pd.DataFrame, a: str = "lex", b: str = "v3") -> dict:
    d = ((t.y - t[a]) ** 2 - (t.y - t[b]) ** 2).to_numpy()
    se = float(date_boot(d, t.session.to_numpy()).std(ddof=1))
    D = float(d.mean())
    z = D / se
    return {"n": len(t), "dates": int(t.session.nunique()),
            "mse": {k: float(((t.y - t[k]) ** 2).mean()) for k in ARMS},
            "d": D, "se_boot": se, "z": z, "p": float(2 * stats.norm.sf(abs(z))),
            "ci95": [D - bench.Z * se, D + bench.Z * se], "mde": 2.8 * se}


def clark_west(t: pd.DataFrame, small: str, big: str) -> dict:
    """Clark & West (2007): the bigger model's MSE gain, credited back the squared gap between
    the forecasts (estimation noise under the null), with a 20-date block-bootstrap SE.
    Positive: the bigger model adds. Identical forecasts (an empty word list): z = 0."""
    f = ((t.y - t[small]) ** 2 - ((t.y - t[big]) ** 2 - (t[small] - t[big]) ** 2)).to_numpy()
    se = float(date_boot(f, t.session.to_numpy()).std(ddof=1))
    m = float(f.mean())
    z = m / se if se > 0 else 0.0
    return {"n": len(t), "dates": int(t.session.nunique()), "gain": m, "se_boot": se, "z": z,
            "p": float(2 * stats.norm.sf(abs(z))), "ci95": [m - bench.Z * se, m + bench.Z * se],
            "mde": 2.8 * se, "mse": {k: float(((t.y - t[k]) ** 2).mean()) for k in ARMS}}


def verdict(res: dict, sesoi: float) -> str:
    lo, hi = res["ci95"]
    if -sesoi < lo and hi < sesoi:
        return "NO GAIN WORTH HAVING"
    return "WORDS ADD TO V3" if lo > 0 else "WORDS HURT" if hi < 0 else "INCONCLUSIVE"


def stability(chosen: dict) -> dict:
    """ADR G3: the words selected at the last refit must recur in >= 50% of refits."""
    years = sorted(chosen)
    sets = [set(chosen[y]["big"]) | set(chosen[y]["small"]) for y in years]
    freq = {w: sum(w in s for s in sets) / len(sets) for w in sets[-1]}
    med = float(np.median(list(freq.values()))) if freq else 0.0
    return {"refits": years, "selected_per_refit": [len(s) for s in sets], "median_frequency_last": med,
            "pass": med >= 0.5,
            "jaccard_consecutive": [len(a & b) / len(a | b) if a | b else 1.0 for a, b in zip(sets, sets[1:])],
            "frequency_last": dict(sorted(freq.items(), key=lambda kv: -kv[1]))}


def run(confirm: bool) -> dict:
    if SESOI is None:
        raise SystemExit("SESOI is not set: the pre-registration fixes it before any run")
    p, h = data(confirm)
    u = units(p, confirm)
    ev = events(p, h, u)
    years = range(2016, 2023 if confirm else 2021)
    window = lambda t: t[(t.session >= bench.FIRM_SEAL) if confirm else t.session.notna()]
    t, chosen = walk(u, ev, years)
    tt = window(t)
    primary = clark_west(tt, "v3", "both")
    sesoi = SESOI * primary["mse"]["v3"]
    primary |= {"sesoi": sesoi, "verdict": verdict(primary, sesoi)}
    out = {"mode": "confirm" if confirm else "design", "events": int(len(tt)),
           "window": f"{tt.session.min().date()}..{tt.session.max().date()}", "primary": primary,
           "secondary": {"head_to_head_words_vs_v3": head_to_head(tt), "words_vs_f0": clark_west(tt, "f0", "lex"),
                         "v3_vs_f0": clark_west(tt, "f0", "v3")},
           "stability": stability(chosen), "words": chosen,
           "robustness": {name: clark_west(window(walk(u, ev, years, docs, method)[0]), "v3", "both")
                          for name, docs, method in (("corporate_words_kept", "toks_corp", "card"),
                                                     ("issuer_names_kept", "toks_issuer", "card"),
                                                     ("l2_logistic", "toks", "logit"), ("dmr", "toks", "dmr"))},
           "leave_one_year_out": {int(y): clark_west(tt[tt.session.dt.year != y], "v3", "both")["gain"]
                                  for y in sorted(tt.session.dt.year.unique())},
           "contemporaneous_check": float(stats.spearmanr(tt.score, tt.abs_r)[0])}
    pr = p2.issuer_news()
    out["price_report_placebo"] = {
        "issuer_price_report_headlines_2016_2020":
            int((pr.is_price_report & (pr.day >= TEST_FROM) & (pr.day < bench.FIRM_SEAL)).sum()),
        "reading": "not estimable: too few events for the word screen"}
    out["g3_reading"] = g3(out, confirm)
    return out


def g3(out: dict, confirm: bool) -> str:
    if not confirm:
        return "design run: G3 is read on the confirmation run, with this design result"
    design = json.loads((h3.CURATED / "p3_design.json").read_text())
    ok = (design["primary"]["verdict"] == out["primary"]["verdict"] == "WORDS ADD TO V3"
          and out["stability"]["pass"])
    return "G3 PASSED" if ok else "G3 NOT PASSED: EXIT (ADR G3)"


# ------------------------------------------------------------------ controls (before the tag)
def permute(u: pd.DataFrame, rng) -> np.ndarray:
    """|AR| shuffled in 20-trade blocks within each firm: no link to words or v3 is left."""
    y = u.y.to_numpy(float).copy()
    for ix in u.groupby("ticker").indices.values():
        y[ix] = bench.block_permute(y[ix], 20, rng)
    return y


def stat(t: pd.DataFrame) -> tuple:
    """The decisive Clark-West statistic (z positive: the words add to v3), its SE, the
    F0 + v3 MSE and the gain."""
    r = clark_west(t, "v3", "both")
    return r["z"], r["se_boot"], r["mse"]["v3"], r["gain"]


def plant(u: pd.DataFrame, ev: pd.DataFrame, g: float, seed: int, years=range(2016, 2021)) -> tuple:
    """Positive control: shuffled targets (no real link), a planted word in PLANT_RATE of the
    events, and g standard deviations of |AR| added to those events' targets."""
    rng = np.random.default_rng(seed)
    y = permute(u, rng)
    pick = rng.random(len(ev)) < PLANT_RATE
    y[ev.uidx.to_numpy()[pick]] += g * float(u.y.std())
    e = ev.assign(y=y[ev.uidx.to_numpy()], toks=[s | {"zzplanted"} if k else s for s, k in zip(ev.toks, pick)])
    return stat(walk(u.assign(y=y), e, years)[0])


def controls(jobs: int) -> dict:
    """bench's G0 controls rebuilt for P3, design window, G0's bands. None pairs the real
    words with the real targets: the planted word sits on shuffled targets."""
    p, h = data(False)
    u = units(p, False)
    ev = events(p, h, u)
    years = range(2016, 2021)
    planted = lambda g, seed: plant(u, ev, g, seed, years)

    lo, hi = 0.02, 2.0
    for _ in range(8):
        mid = float(np.sqrt(lo * hi))
        runs = bench.pmap(lambda s: planted(mid, bench.SEED + 10_000 + s), range(bench.CALIBRATION_RUNS), jobs)
        lo, hi = (mid, hi) if np.median([r[0] for r in runs]) < 2.8 else (lo, mid)
    g = float(np.sqrt(lo * hi))
    found = np.array(bench.pmap(lambda s: planted(g, bench.SEED + 20_000 + s), range(bench.SEEDS), jobs))
    n_found = int((found[:, 0] > bench.Z).sum())
    se = float(np.median(found[:, 1]))
    out = {"events_2016_2020": int(((ev.session >= TEST_FROM) & (ev.session < bench.FIRM_SEAL)).sum()),
           "positive": {"gamma_in_sd_of_abs_ar": g, "found": n_found, "of": bench.SEEDS, "band": list(bench.POWER_BAND),
                        "median_stat": float(np.median(found[:, 0])),
                        "pass": bench.POWER_BAND[0] <= n_found <= bench.POWER_BAND[1]},
           "mde": {"se_of_gain_median": se, "mde_gain": 2.8 * se, "mse_v3_median": float(np.median(found[:, 2])),
                   "mde_share_of_mse": 2.8 * se / float(np.median(found[:, 2]))}}

    def permuted(s):
        y = permute(u, np.random.default_rng(bench.SEED + 30_000 + s))
        return stat(walk(u.assign(y=y), ev.assign(y=y[ev.uidx.to_numpy()]), years)[0])

    cal = h3.calendar()
    cal = cal[(cal >= WORDS_FROM) & (cal < bench.FIRM_SEAL)]
    hw = h[(h.day >= WORDS_FROM) & (h.day < bench.FIRM_SEAL)]
    pos = cal.searchsorted(hw.day)
    hw, pos = hw[pos < len(cal)], pos[pos < len(cal)]
    step = max(1, (len(cal) - 500) // bench.SEEDS)

    def stale(s):
        e = events(p, hw.assign(day=cal[(pos - 250 - step * s) % len(cal)]), u)
        return stat(walk(u, e, years)[0])

    for name, fn in (("permuted_labels", permuted), ("stale_news", stale)):
        res = np.array(bench.pmap(fn, range(bench.SEEDS), jobs))
        z, n = res[:, 0], int((np.abs(res[:, 0]) > bench.Z).sum())
        out[name] = {"found": n, "of": bench.SEEDS, "max": bench.FALSE_MAX, "found_words_add": int((z > bench.Z).sum()),
                     "found_words_hurt": int((z < -bench.Z).sum()), "empty_word_list_runs": int((res[:, 1] == 0).sum()),
                     "pass": n <= bench.FALSE_MAX}
    out["pass"] = all(out[k]["pass"] for k in ("positive", "permuted_labels", "stale_news"))
    return out


POWER_STEPS = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0)


def power(jobs: int) -> dict:
    """Owner, after controls run 3: the planted word was found in 70 of 100 runs (band
    72-88) because the yearly screen sometimes drops it. The MDE used for the SESOI is the
    median Clark-West gain at the smallest planted effect found in >= 80 of 100 runs (same
    seeds as the positive control), as a share of the F0 + v3 MSE."""
    p, h = data(False)
    u = units(p, False)
    ev = events(p, h, u)
    g0 = json.loads((h3.CURATED / "p3_controls.json").read_text())["positive"]["gamma_in_sd_of_abs_ar"]
    steps = []
    for k in POWER_STEPS:
        r = np.array(bench.pmap(lambda s: plant(u, ev, k * g0, bench.SEED + 20_000 + s), range(bench.SEEDS), jobs))
        steps.append({"gamma_in_sd_of_abs_ar": k * g0, "found": int((r[:, 0] > bench.Z).sum()), "of": bench.SEEDS,
                      "median_z": float(np.median(r[:, 0])), "median_gain": float(np.median(r[:, 3])),
                      "median_mse_v3": float(np.median(r[:, 2])),
                      "median_gain_share_of_mse": float(np.median(r[:, 3]) / np.median(r[:, 2]))})
    hit = next((s for s in steps if s["found"] >= 80), None)
    return {"calibrated_gamma": g0, "steps": steps,
            "mde80_share_of_mse": hit["median_gain_share_of_mse"] if hit else None,
            "mde80_gamma_in_sd_of_abs_ar": hit["gamma_in_sd_of_abs_ar"] if hit else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controls", action="store_true", help="P3's own controls, design window")
    ap.add_argument("--power", action="store_true", help="power curve of the positive control (MDE at 80%%)")
    ap.add_argument("--confirm", action="store_true", help=f"the one sealed run; needs {TAG}")
    ap.add_argument("--jobs", type=int, default=1, help="workers for the controls")
    args = ap.parse_args()
    name = ("p3_controls" if args.controls else "p3_power" if args.power
            else "p3_confirm" if args.confirm else "p3_design")
    path = h3.CURATED / f"{name}.json"
    if args.confirm and path.exists():
        raise SystemExit(f"{path.name} exists: the confirmation run is made once")
    out = (controls(args.jobs) if args.controls else power(args.jobs) if args.power else run(args.confirm))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out.get("pass", out.get("mde80_share_of_mse", out.get("g3_reading"))), f"-> {path}")


if __name__ == "__main__":
    main()
