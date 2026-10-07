"""ADR-001 P2: when a stock moves, does it matter whether there was news about it?

Plan: ADR-001 §8, P2 card. The owner's decisions (2026-10-06) are in STATUS.md and are
fixed in audit/PREREG_P2.md before the design run. Built on bench.py (gate G0,
AUDIT_REPORT §P0). Firm prices from 2021-01-01 stay blank until the tag prereg-p2-v1
exists and --confirm is given (card: design 2016-2020, confirm 2021-2022).

UNIT    a firm trade day: ALL_DATA.csv traded days, duplicates dropped. It counts only if
        the stock also traded within 5 sessions before and after it (card: after; owner:
        before too, ADR T7), and if no move in it or in its target window exceeds 10%:
        those look like unadjusted dividend or bonus-share ex-dates, and the repo has no
        corporate-actions file. Every drop is counted by year and firm.
MOVE    r = trade-to-trade return; r_m = Tunindex over the same interval.
NEWS    an issuer headline (h3.issuer_patterns) dated after the previous trade and up to
        this one. Arms: ex_price decides (card); all and price_only are reported (rule 6).
TARGET  AR_k = return over the next k trades minus the Tunindex over the same interval,
        k = 1, 5. Secondary: Dimson-beta AR.
TESTS   c1, c5  c in AR_k = a + c r.news + b r + d news + e r_m
        vol     g in |AR_1| = a_firm + g news + HAR(|r|: last trade, 5- and 22-trade means)
                + h |r_m|. Firm intercepts compare each firm with itself (owner, after the
                first controls: across firms, calm news-heavy stocks read as a news effect).
        Each estimate minus its mean over 100 sets of random flags at the arm's news rate,
        over the SE of 2,000 bootstraps of 20-date blocks, all firms together (owner).
        Holm across the three. SESOI (owner): c 0.20 (k = 1) and 0.40 (k = 5); g 10% of
        the mean |AR_1| in the same run.

    OMP_NUM_THREADS=1 python3 preprocessing/p2.py --controls --jobs 10  # P2's own controls
    OMP_NUM_THREADS=1 python3 preprocessing/p2.py                       # design 2016-2020
    OMP_NUM_THREADS=1 python3 preprocessing/p2.py --confirm             # once, after the tag
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import bench
import h3
from config import PRICE_REPORT_PATTERN
from features import DEFAULT_PRICES, phantom_sessions

GAP, JUMP, BLOCK = 5, 0.10, 20           # sessions both ways (owner); ex-date screen; card
DIMSON_WINDOW, DIMSON_MIN_TRADES = 250, 60
ARMS = ("ex_price", "all", "price_only")
TESTS = {"c1": 1, "c5": 5, "vol": 1}     # test -> trades in its target window
SESOI = {"c1": 0.20, "c5": 0.40, "vol": 0.10}   # owner, 2026-10-06; vol: share of mean |AR_1|
HAR = ["abs_r", "har5", "har22", "abs_r_m"]
TERMS = {"c": ["c", "a", "b", "d", "e"], "vol": ["g", "h_last", "h_5", "h_22", "h_market"]}


# ------------------------------------------------------------------ data
def flags(trades: pd.DatetimeIndex, days) -> np.ndarray:
    """True on each trade day with a headline dated after the previous trade and up to it."""
    j = trades.searchsorted(pd.DatetimeIndex(days), side="left")
    return np.bincount(j[j < len(trades)], minlength=len(trades)) > 0


def issuer_news() -> pd.DataFrame:
    """Every canonical relevant headline naming a BVMT issuer, scored or not: one row per
    (headline, ticker), with its date and whether it is a price report."""
    s = pd.read_parquet(h3.SCORED_V3, columns=["headline_clean", "published_date"])
    text = s.headline_clean.astype(str)
    day = pd.to_datetime(s.published_date).dt.normalize()
    price = text.str.contains(PRICE_REPORT_PATTERN, case=False, regex=True, na=False)
    low = text.str.lower()
    parts = [pd.DataFrame({"ticker": t, "day": day[m], "is_price_report": price[m]})
             for t, pat in h3.issuer_patterns().items() if (m := low.str.contains(pat)).any()]
    return pd.concat(parts, ignore_index=True)


def arm(news: pd.DataFrame, name: str) -> pd.DataFrame:
    return {"ex_price": news[~news.is_price_report], "all": news,
            "price_only": news[news.is_price_report]}[name]


def dimson_betas(close: pd.Series, idx_ret: pd.Series, cal: pd.DatetimeIndex, years) -> dict:
    """Per year: the summed slopes of the firm's session return (close carried over sessions
    without a trade) on the Tunindex return of the session before, the same one and the one
    after, over the DIMSON_WINDOW sessions before 1 January. The last session's lead stays
    inside the window, so only the past is used; moves beyond JUMP are left out. Fewer than
    DIMSON_MIN_TRADES trades in the window: beta = 1."""
    cal = cal[cal <= close.index.max()]
    fc = close.reindex(cal).ffill()
    fr = (fc / fc.shift() - 1).where(lambda v: v.abs() <= JUMP).to_numpy()
    traded = cal.isin(close.index)
    m = idx_ret.reindex(cal)
    X = np.c_[m.shift(1), m, m.shift(-1)]
    out = {}
    for y in years:
        p = cal.searchsorted(pd.Timestamp(f"{y}-01-01"))
        w = slice(max(0, p - 1 - DIMSON_WINDOW), max(0, p - 1))
        ok = ~np.isnan(X[w]).any(axis=1) & ~np.isnan(fr[w])
        if traded[w].sum() < DIMSON_MIN_TRADES or ok.sum() < DIMSON_MIN_TRADES:
            out[y] = 1.0
        else:
            coef, *_ = np.linalg.lstsq(np.c_[np.ones(ok.sum()), X[w][ok]], fr[w][ok], rcond=None)
            out[y] = float(coef[1:].sum())
    return out


def panel(confirm: bool = False, news: pd.DataFrame | None = None, start: str = h3.START,
          drop_phantoms: bool = False) -> pd.DataFrame:
    """One row per firm trade day, `start` to FIRM_END. The design run reads no price from
    FIRM_SEAL on: those trade dates stay (counted, never priced) with blank prices. The
    confirmation run reads them after bench.seal has checked the tag. drop_phantoms takes
    the phantom sessions out of the firm and index prices (data rule 2): P2 as
    pre-registered kept them (AUDIT_REPORT §P2); P3 drops them."""
    if confirm:
        bench.seal(pd.DataFrame({"session": pd.to_datetime([])}), "p2", confirm=True)
    last = pd.Timestamp(bench.FIRM_END) if confirm else pd.Timestamp(bench.FIRM_SEAL) - pd.Timedelta(days=1)
    news = issuer_news() if news is None else news
    cal = h3.calendar()
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv", usecols=["Ticker", "Date", "Close"])
    a["session"] = pd.to_datetime(a.Date).dt.normalize()
    a = a.drop_duplicates(["Ticker", "session"], keep="last").sort_values(["Ticker", "session"])
    px = pd.read_csv(DEFAULT_PRICES, encoding="utf-8-sig", usecols=["date", "close"])
    idx = pd.Series(px.close.to_numpy(float), index=pd.to_datetime(px.date).dt.normalize())
    if drop_phantoms:
        a, idx = a[~a.session.isin(phantom_sessions())], idx[~idx.index.isin(phantom_sessions())]
    idx = idx[idx.index <= last]
    v = idx.reindex(cal)
    idx_ret = v / v.shift() - 1
    years = range(pd.Timestamp(start).year, last.year + 1)
    parts = []
    for ticker in h3.issuer_patterns():
        g = a[a.Ticker == ticker]
        if len(g) < 2:
            continue
        t = pd.DatetimeIndex(g.session)
        c = np.where(t <= last, g.Close.to_numpy(float), np.nan)
        im = idx.reindex(t).to_numpy()
        pos = cal.searchsorted(t)
        f = pd.DataFrame({"ticker": ticker, "session": t, "back_gap": np.r_[np.nan, np.diff(pos)],
                          "gap": np.r_[np.diff(pos), np.nan], "r": c / np.r_[np.nan, c[:-1]] - 1,
                          "r_m": im / np.r_[np.nan, im[:-1]] - 1})
        beta = t.year.map(dimson_betas(pd.Series(c, index=t).dropna(), idx_ret, cal, years)).to_numpy(float)
        jump = np.abs(f.r.to_numpy()) > JUMP
        cum, i = np.r_[0, np.cumsum(jump)], np.arange(len(t))
        for k in (1, 5):
            gross = np.r_[c[k:], [np.nan] * k] / c - 1
            market = np.r_[im[k:], [np.nan] * k] / im - 1
            f[f"ar{k}"], f[f"ar{k}_dimson"] = gross - market, gross - beta * market
            f[f"clean{k}"] = ~jump & (cum[np.minimum(i + 1 + k, len(t))] == cum[i + 1])
        mine = news[news.ticker == ticker]
        for name in ARMS:
            f[f"news_{name}"] = flags(t, arm(mine, name).day).astype(float)
        f["abs_r"], f["abs_r_m"] = f.r.abs(), f.r_m.abs()
        f["har5"], f["har22"] = f.abs_r.rolling(5).mean(), f.abs_r.rolling(22).mean()
        parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    return p[(p.session >= start) & (p.session <= bench.FIRM_END)].reset_index(drop=True)


def rows(p: pd.DataFrame, test: str, confirm: bool, target: str | None = None,
         clean: bool = True) -> pd.DataFrame:
    """The units of one test in one window, sorted by date (the bootstrap resamples dates)."""
    k = TESTS[test]
    target = target or f"ar{k}"
    window = (p.session >= bench.FIRM_SEAL) if confirm else (p.session < bench.FIRM_SEAL)
    need = ["r", "r_m", target] + (HAR if test == "vol" else [])
    keep = window & (p.back_gap <= GAP) & (p.gap <= GAP) & p[need].notna().all(axis=1)
    if clean:
        keep &= p[f"clean{k}"]
    return p[keep].sort_values("session", kind="stable").reset_index(drop=True)


def units(p: pd.DataFrame, confirm: bool) -> dict:
    """Every firm trade day of the window and why some are not units (ADR loophole register:
    report drops by year and firm)."""
    w = p[(p.session >= bench.FIRM_SEAL) if confirm else (p.session < bench.FIRM_SEAL)]
    out = {"window": f"{w.session.min().date()}..{w.session.max().date()}",
           "firm_trade_days": len(w), "firms": int(w.ticker.nunique())}
    for name, m in (("previous_trade_over_5_sessions", ~(w.back_gap <= GAP)),
                    ("next_trade_over_5_sessions", ~(w.gap <= GAP)),
                    ("move_over_10pct_in_1_trade_window", ~w.clean1),
                    ("move_over_10pct_in_5_trade_window", ~w.clean5)):
        out[name] = {"n": int(m.sum()),
                     "by_year": {int(y): int(n) for y, n in m.groupby(w.session.dt.year).sum().items()},
                     "by_firm": {t: int(n) for t, n in m.groupby(w.ticker).sum().items() if n}}
    return out


# ------------------------------------------------------------------ the test
def design_matrix(d: pd.DataFrame, test: str, news: np.ndarray, target: str):
    """y and X of one test; the coefficient tested is column 0. The volatility model's firm
    intercepts are removed by demeaning within each firm (Frisch-Waugh)."""
    y = d[target].to_numpy(float)
    if test == "vol":
        firm = pd.factorize(d.ticker)[0]
        return within(np.abs(y), firm), within(np.c_[news, d[HAR].to_numpy(float)], firm)
    r = d.r.to_numpy(float)
    return y, np.c_[r * news, np.ones(len(d)), r, news, d.r_m.to_numpy(float)]


def within(M: np.ndarray, firm: np.ndarray) -> np.ndarray:
    """M minus its mean within each firm. ponytail: the bootstrap reuses the full-sample
    firm means; exact intercepts per resample need per-(date, firm) sums."""
    M2 = M.reshape(len(M), -1)
    n = np.bincount(firm)
    means = np.column_stack([np.bincount(firm, weights=col) / n for col in M2.T])
    return (M2 - means[firm]).reshape(M.shape)


def by_date(y: np.ndarray, X: np.ndarray, dates: np.ndarray):
    """X'X and X'y summed within each date (rows sorted by date), so the OLS on any resample
    of whole dates is one small solve."""
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    return (np.add.reduceat(X[:, :, None] * X[:, None, :], starts, axis=0),
            np.add.reduceat(X * y[:, None], starts, axis=0))


def solve(XtX: np.ndarray, Xty: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """OLS from per-date sums: every date once, or one row of date counts per resample
    (pseudo-inverse there: a resample of a thin arm can miss every news day)."""
    if w is None:
        return np.linalg.solve(XtX.sum(0), Xty.sum(0))
    k = XtX.shape[1]
    A = (w @ XtX.reshape(len(XtX), -1)).reshape(-1, k, k)
    return (np.linalg.pinv(A) @ (w @ Xty)[..., None])[..., 0]


def block_counts(n_dates: int) -> np.ndarray:
    """Date counts of N_BOOT circular resamples of 20-date blocks, all firms together."""
    rng = np.random.default_rng(bench.SEED)
    return np.array([np.bincount(h3.block_indices(n_dates, BLOCK, rng), minlength=n_dates)
                     for _ in range(bench.N_BOOT)], dtype=float)


def estimate(d, test, news, target, counts) -> tuple[np.ndarray, float]:
    """Coefficients, and the block-bootstrap SE of the one tested."""
    y, X = design_matrix(d, test, news, target)
    XtX, Xty = by_date(y, X, d.session.to_numpy())
    return solve(XtX, Xty), float(solve(XtX, Xty, counts)[:, 0].std(ddof=1))


def noise(d, test, target, rate, offset=0) -> np.ndarray:
    """The tested coefficient with the news flag replaced by random flags at the arm's rate,
    100 seeds (as bench.p2_mde): their mean centres the test (owner)."""
    out = np.empty(bench.SEEDS)
    for s in range(bench.SEEDS):
        z = (np.random.default_rng(bench.SEED + 60_000 + offset + s).random(len(d)) < rate).astype(float)
        y, X = design_matrix(d, test, z, target)
        out[s] = np.linalg.lstsq(X, y, rcond=None)[0][0]
    return out


def evaluate(d, test, news, target, null) -> dict:
    coef, se = estimate(d, test, news, target, block_counts(d.session.nunique()))
    effect = float(coef[0] - null.mean())
    z = effect / se
    return {"n": len(d), "dates": int(d.session.nunique()), "firms": int(d.ticker.nunique()),
            "news_days": int(news.sum()),
            "terms": dict(zip(TERMS["vol" if test == "vol" else "c"], coef.tolist())),
            "noise_mean": float(null.mean()), "noise_sd": float(null.std(ddof=1)),
            "effect_vs_noise": effect, "se_boot": se, "z": z, "p": float(2 * stats.norm.sf(abs(z))),
            "ci95": [effect - bench.Z * se, effect + bench.Z * se], "mde": 2.8 * se}


def read(res: dict, sesoi: dict) -> dict:
    """Holm across the tests, then the pre-registered verdict (audit/PREREG_P2.md). An
    interval inside +-SESOI is no effect worth having, even when it excludes zero."""
    for test, p in bench.holm({t: v["p"] for t, v in res.items() if "p" in v}).items():
        v, s = res[test], sesoi[test]
        lo, hi = v["ci95"]
        up = v["effect_vs_noise"] > 0
        v |= {"p_holm": p, "sesoi": s,
              "verdict": ("NO EFFECT WORTH HAVING" if -s < lo and hi < s
                          else "CHANNEL" if p < 0.05 else "INCONCLUSIVE"),
              "direction": (("bigger" if up else "smaller") + " next move after news" if test == "vol"
                            else "drift with news" if up else "more reversal with news")}
    return res


def family(p: pd.DataFrame, confirm: bool, news_arm: str = "ex_price", ar: str = "",
           clean: bool = True) -> dict:
    """The three tests on one arm and one target definition, read with Holm."""
    res, sesoi = {}, dict(SESOI)
    for test, k in TESTS.items():
        target = f"ar{k}{ar}"
        d = rows(p, test, confirm, target, clean)
        news = d[f"news_{news_arm}"].to_numpy(float)
        if not news.any():
            res[test] = {"n": len(d), "news_days": 0, "verdict": "NO NEWS DAYS"}
            continue
        res[test] = evaluate(d, test, news, target, noise(d, test, target, news.mean()))
        if test == "vol":
            res[test]["mean_abs_ar1"] = float(np.abs(d[target]).mean())
            sesoi["vol"] = SESOI["vol"] * res[test]["mean_abs_ar1"]
    return read(res, sesoi)


def leave_one_firm_out(p: pd.DataFrame, confirm: bool, primary: dict) -> dict:
    """Descriptive: the primary effect with each firm left out in turn (no bootstrap)."""
    out = {}
    for test, k in TESTS.items():
        d = rows(p, test, confirm)
        y, X = design_matrix(d, test, d.news_ex_price.to_numpy(float), f"ar{k}")
        tick = d.ticker.to_numpy()
        eff = {t: float(np.linalg.lstsq(X[tick != t], y[tick != t], rcond=None)[0][0]
                        - primary[test]["noise_mean"]) for t in np.unique(tick)}
        lo, hi = min(eff, key=eff.get), max(eff, key=eff.get)
        out[test] = {"min": eff[lo], "left_out_at_min": lo, "max": eff[hi], "left_out_at_max": hi}
    return out


def run(confirm: bool) -> dict:
    p = panel(confirm)
    primary = family(p, confirm)
    verdicts = {v["verdict"] for v in primary.values()}
    found = [t for t, v in primary.items() if v["verdict"] == "CHANNEL"]
    g2 = ("CHANNEL: P3 targets " + ", ".join(found) if found
          else "EXIT: no effect worth having in all three tests" if verdicts == {"NO EFFECT WORTH HAVING"}
          else "NO CHANNEL, NOT CONCLUSIVE: the owner decides at the G2 review")
    return {"mode": "confirm" if confirm else "design", "units": units(p, confirm),
            "primary": {"arm": "ex_price", "tests": primary, "g2_reading": g2, "decides_g2": confirm},
            "arms": {name: family(p, confirm, name) for name in ARMS[1:]},
            "dimson_beta": family(p, confirm, ar="_dimson"),
            "with_moves_over_10pct": family(p, confirm, clean=False),
            "leave_one_firm_out": leave_one_firm_out(p, confirm, primary)}


# ------------------------------------------------------------------ controls (before the tag)
def planted(d, test, target, rate, counts, null, jobs) -> dict:
    """bench.positive_control for a coefficient: random flags at the news rate, the target
    shifted by gamma x the tested column (r.flag, or the flag for vol). gamma, in units of
    the random-flag spread, is set by bisection so the median statistic over
    CALIBRATION_RUNS runs is 2.8 (the MDE); 100 fresh runs must find it in POWER_BAND."""
    sd, mu, dates = float(null.std(ddof=1)), float(null.mean()), d.session.to_numpy()

    def one(m, seed):
        z = (np.random.default_rng(seed).random(len(d)) < rate).astype(float)
        y, X = design_matrix(d, test, z, target)
        XtX, Xty = by_date(y + m * sd * X[:, 0], X, dates)
        return (solve(XtX, Xty)[0] - mu) / solve(XtX, Xty, counts)[:, 0].std(ddof=1)

    lo, hi = 0.5, 8.0
    for _ in range(8):
        mid = float(np.sqrt(lo * hi))
        med = np.median(bench.pmap(lambda s: one(mid, bench.SEED + 10_000 + s),
                                   range(bench.CALIBRATION_RUNS), jobs))
        lo, hi = (mid, hi) if med < 2.8 else (lo, mid)
    m = float(np.sqrt(lo * hi))
    stat = np.array(bench.pmap(lambda s: one(m, bench.SEED + 20_000 + s), range(bench.SEEDS), jobs))
    found = int((stat > bench.Z).sum())
    return {"gamma": m * sd, "gamma_in_noise_sd": m, "found": found, "of": bench.SEEDS,
            "band": list(bench.POWER_BAND), "median_stat": float(np.median(stat)),
            "pass": bench.POWER_BAND[0] <= found <= bench.POWER_BAND[1]}


def negative(p, d, test, target, flag, counts, null, news, jobs) -> dict:
    """bench.negative_controls for P2, 100 runs each. Targets block-permuted within each firm
    (20-trade blocks) get their own random-flag null per run. Stale news: every design-window
    headline moved 250+ sessions back on the design calendar (circularly), read against the
    real random-flag null."""
    cal = h3.calendar()
    cal = cal[(cal >= h3.START) & (cal < bench.FIRM_SEAL)]
    h = news[(news.day >= h3.START) & (news.day < bench.FIRM_SEAL)]
    pos = cal.searchsorted(h.day)
    tick, pos = h.ticker.to_numpy()[pos < len(cal)], pos[pos < len(cal)]
    step = max(1, (len(cal) - 500) // bench.SEEDS)
    trades = {t: (np.flatnonzero(p.ticker.to_numpy() == t), pd.DatetimeIndex(p.session[p.ticker == t]))
              for t in p.ticker.unique()}
    firms = [np.flatnonzero(d.ticker.to_numpy() == t) for t in d.ticker.unique()]

    def permuted(s):
        rng = np.random.default_rng(bench.SEED + 30_000 + s)
        y = d[target].to_numpy(float).copy()
        for ix in firms:
            y[ix] = bench.block_permute(y[ix], BLOCK, rng)
        e = d.assign(**{target: y})
        coef, se = estimate(e, test, flag, target, counts)
        return (coef[0] - noise(e, test, target, flag.mean(), 100_000 + 1000 * s).mean()) / se

    def stale(s):
        day = cal[(pos - 250 - step * s) % len(cal)]
        col = np.zeros(len(p))
        for t, (ix, td) in trades.items():
            col[ix] = flags(td, day[tick == t])
        e = rows(p.assign(news_ex_price=col), test, False, target)
        coef, se = estimate(e, test, e.news_ex_price.to_numpy(float), target, counts)
        return (coef[0] - null.mean()) / se

    out = {}
    for name, fn in (("permuted_labels", permuted), ("stale_news", stale)):
        stat = np.array(bench.pmap(fn, range(bench.SEEDS), jobs))
        found = int((np.abs(stat) > bench.Z).sum())
        out[name] = {"found": found, "of": bench.SEEDS, "max": bench.FALSE_MAX,
                     "found_positive": int((stat > bench.Z).sum()),
                     "found_negative": int((stat < -bench.Z).sum()), "pass": found <= bench.FALSE_MAX}
    return out


def controls(jobs: int) -> dict:
    """bench's G0 controls rebuilt for P2's coefficient tests: decisive arm, design window,
    G0's bands. None of them pairs the real news with the real targets."""
    news = issuer_news()
    p = panel(False, news)
    out = {}
    for test, k in TESTS.items():
        target = f"ar{k}"
        d = rows(p, test, False)
        flag = d.news_ex_price.to_numpy(float)
        null = noise(d, test, target, flag.mean())
        counts = block_counts(d.session.nunique())
        out[test] = {"n": len(d), "dates": int(d.session.nunique()), "news_days": int(flag.sum()),
                     "matched_noise": {"mean": float(null.mean()), "sd": float(null.std(ddof=1))},
                     "positive": planted(d, test, target, flag.mean(), counts, null, jobs),
                     **negative(p, d, test, target, flag, counts, null, arm(news, "ex_price"), jobs)}
    out["pass"] = all(v[c]["pass"] for v in out.values()
                      for c in ("positive", "permuted_labels", "stale_news"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controls", action="store_true", help="P2's own controls, design window")
    ap.add_argument("--confirm", action="store_true", help="the one sealed run; needs prereg-p2-v1")
    ap.add_argument("--jobs", type=int, default=1, help="workers for the controls")
    args = ap.parse_args()
    name = "p2_controls" if args.controls else "p2_confirm" if args.confirm else "p2_design"
    path = h3.CURATED / f"{name}.json"
    if args.confirm and path.exists():
        raise SystemExit(f"{path.name} exists: the confirmation run is made once")
    out = controls(args.jobs) if args.controls else run(args.confirm)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out["pass"] if args.controls else out["primary"]["g2_reading"], f"-> {path}")


if __name__ == "__main__":
    main()
