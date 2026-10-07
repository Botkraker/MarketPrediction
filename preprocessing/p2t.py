"""ADR-001 D1 revisit of P2 (§10): is P2's news-day volatility channel the market's first
reaction to news published after the close, or digestion of news it had already seen?

Plan: ADR-001 §8 (P2 card, track D1) and §10 ("revisit at G2 if D1 is approved"). Owner's
decisions (2026-10-07) in STATUS.md, "P2t decisions"; fixed in audit/PREREG_P2T.md before
the design run. Firm prices from 2021-01-01 stay blank until the annotated tag prereg-p2t-v1
exists, preprocessing/ matches it, and --confirm is given.

UNITS   P2's volatility units (AUDIT_REPORT §P2), phantom sessions dropped (data rule 2).
NEWS    an issuer headline (price reports excluded) belongs to the first trade on or after
        its date (P2 card). ilboursa headlines carry a publication time (D1,
        data/raw/ilboursa_d1): 'post' = published on that trade date at or after the 14:10
        close (ADR §2), so the next trade holds the first reaction; 'pre' = any earlier
        timed headline, which the market could already price that session; other outlets
        are 'untimed'. Three flags in one model (owner).
MODEL   |AR_1| = a_firm + g_pre pre + g_post post + g_untimed untimed + HAR(|r|) + h |r_m|.
TEST    g_pre and g_post, each minus its mean under 100 sets of random flags at its rate,
        over a yardstick each test passed its controls with (owner, after controls runs 1-2):
        g_pre sqrt(random-flag spread^2 + SE^2) (P0's rule), g_post the SE alone; SE from
        2,000 bootstraps of 20-date blocks; Holm across the two; SESOI 10% of
        the mean |AR_1| (P2's, owner). Reading: g_pre a CHANNEL = digestion; only g_post a
        CHANNEL = the first reaction (timing).

    OMP_NUM_THREADS=1 python3 preprocessing/p2t.py --controls --jobs 10
    OMP_NUM_THREADS=1 python3 preprocessing/p2t.py
    OMP_NUM_THREADS=1 python3 preprocessing/p2t.py --confirm
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import bench
import h3
import p2
from config import PRICE_REPORT_PATTERN
from features import phantom_sessions
from p2_posthoc import CORPORATE
from p3 import code_matches_tag

TAG = "prereg-p2t-v1"
D1 = h3.ROOT / "data" / "raw" / "ilboursa_d1" / "ilboursa_headlines.csv"
CLOSE, SESOI = 14 * 60 + 10, 0.10          # ADR §2; P2's SESOI (owner)
FLAGS = ("pre", "post", "untimed")
DECISIVE = ("pre", "post")
WIDE = {"pre": True, "post": False, "untimed": True}   # yardstick per test (owner, after controls runs 1-2)


# ------------------------------------------------------------------ data
def timed_news() -> pd.DataFrame:
    """Issuer headlines, one row per (headline, ticker): date, publication time where D1 has
    one (ilboursa, joined through the original raw file that the row ids index), and the
    price-report and corporate-action flags."""
    s = pd.read_parquet(h3.SCORED_V3, columns=["row_id", "source", "published_date", "headline_clean"])
    raw = pd.read_csv(h3.ROOT / "data" / "raw" / "ilboursa_headlines.csv", sep=";", encoding="utf-8-sig",
                      dtype=str, keep_default_na=False)
    d1 = pd.read_csv(D1, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d1 = d1.assign(key=d1.headline.str.strip() + "|" + d1.date).drop_duplicates("key").set_index("key")
    il = (s.source == "ilboursa").to_numpy()
    n = s.row_id[il].str.split("::").str[1].astype(int).to_numpy()
    key = pd.Series(index=s.index, dtype=object)
    key[il] = raw.headline.str.strip().to_numpy()[n] + "|" + raw.date.to_numpy()[n]
    at = pd.to_datetime(key.map(d1.published_at), format="%Y-%m-%d %H:%M")
    text = s.headline_clean.astype(str)
    low = text.str.lower()
    price = text.str.contains(PRICE_REPORT_PATTERN, case=False, regex=True, na=False)
    corporate = low.str.contains(CORPORATE, regex=True)
    day = pd.to_datetime(s.published_date).dt.normalize()
    parts = [pd.DataFrame({"ticker": t, "day": day[m], "published_at": at[m], "is_price_report": price[m],
                           "corporate": corporate[m]})
             for t, pat in h3.issuer_patterns().items() if (m := low.str.contains(pat)).any()]
    return pd.concat(parts, ignore_index=True)


def trade_dates() -> dict:
    """Each firm's trade dates (ALL_DATA.csv, phantom sessions dropped): dates only."""
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv", usecols=["Ticker", "Date"])
    s = pd.to_datetime(a.Date).dt.normalize()
    a = a.assign(session=s)[~s.isin(phantom_sessions())]
    return {t: pd.DatetimeIndex(np.sort(g.session.unique())) for t, g in a.groupby("Ticker")}


def flags3(news: pd.DataFrame, trades: dict, close: int = CLOSE) -> pd.DataFrame:
    """(ticker, session) -> pre / post / untimed. A headline belongs to the first trade on or
    after its date (P2 card); it is 'post' if timed and published on that trade date at or
    after the close."""
    parts = []
    for t, g in news.groupby("ticker"):
        td = trades.get(t)
        if td is None or not len(td):
            continue
        j = td.searchsorted(pd.DatetimeIndex(g.day), side="left")
        g = g[j < len(td)].assign(session=td[j[j < len(td)]])
        timed = g.published_at.notna()
        minutes = g.published_at.dt.hour * 60 + g.published_at.dt.minute
        post = timed & (g.day == g.session) & (minutes >= close)
        parts.append(g.assign(pre=timed & ~post, post=post, untimed=~timed))
    m = pd.concat(parts)
    return m.groupby(["ticker", "session"])[list(FLAGS)].any().astype(float).reset_index()


def data(confirm: bool):
    if confirm:
        bench.seal(pd.DataFrame({"session": pd.to_datetime([])}), "p2t", confirm=True)
        if not code_matches_tag(TAG):
            raise SystemExit(f"preprocessing/ differs from {TAG}: the confirmation run uses the tagged code")
    news = timed_news()
    return p2.panel(confirm, news, drop_phantoms=True), news, trade_dates()


def with_flags(rows: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    d = rows.merge(f, on=["ticker", "session"], how="left")
    d[list(FLAGS)] = d[list(FLAGS)].fillna(0.0)
    return d.sort_values("session", kind="stable").reset_index(drop=True)


def units(p, news, trades, confirm, keep=None, close=CLOSE) -> pd.DataFrame:
    """P2's volatility units with the three flags (0 on a day without such news)."""
    sel = news[~news.is_price_report] if keep is None else news[keep]
    return with_flags(p2.rows(p, "vol", confirm), flags3(sel, trades, close))


# ------------------------------------------------------------------ the test
def design(d: pd.DataFrame, F: np.ndarray):
    """y and X with firm intercepts removed (Frisch-Waugh); columns 0-2 are the flags."""
    firm = pd.factorize(d.ticker)[0]
    return (p2.within(np.abs(d.ar1.to_numpy(float)), firm),
            p2.within(np.c_[F, d[p2.HAR].to_numpy(float)], firm))


def fit(d, F, counts=None):
    """Coefficients, and their 20-date block-bootstrap draws when counts are given."""
    y, X = design(d, F)
    XtX, Xty = p2.by_date(y, X, d.session.to_numpy())
    return p2.solve(XtX, Xty), (None if counts is None else p2.solve(XtX, Xty, counts))


def noise(d, F, j, offset=0) -> np.ndarray:
    """Coefficient j with flag j replaced by random flags at its rate, 100 seeds."""
    rate, out = F[:, j].mean(), np.empty(bench.SEEDS)
    for s in range(bench.SEEDS):
        G = F.copy()
        G[:, j] = np.random.default_rng(bench.SEED + 60_000 + offset + s).random(len(d)) < rate
        y, X = design(d, G)
        out[s] = np.linalg.lstsq(X, y, rcond=None)[0][j]
    return out


def evaluate(d: pd.DataFrame) -> dict:
    F = d[list(FLAGS)].to_numpy(float)
    coef, boot = fit(d, F, p2.block_counts(d.session.nunique()))
    mean_abs = float(np.abs(d.ar1).mean())
    res = {"n": len(d), "dates": int(d.session.nunique()),
           "window": f"{d.session.min().date()}..{d.session.max().date()}",
           "news_days": {f: int(F[:, j].sum()) for j, f in enumerate(FLAGS)},
           "mean_abs_ar1": mean_abs, "sesoi": SESOI * mean_abs}
    for j, f in enumerate(FLAGS):
        null = noise(d, F, j)
        se = float(boot[:, j].std(ddof=1))
        scale = float(np.hypot(null.std(ddof=1), se)) if WIDE[f] else se
        e = float(coef[j] - null.mean())
        z = e / scale
        res[f] = {"estimate": float(coef[j]), "noise_mean": float(null.mean()), "noise_sd": float(null.std(ddof=1)),
                  "effect_vs_noise": e, "share_of_mean_abs_ar1": e / mean_abs, "se_boot": se, "scale": scale,
                  "z": z, "p": float(2 * stats.norm.sf(abs(z))), "ci95": [e - bench.Z * scale, e + bench.Z * scale],
                  "mde": 2.8 * scale}
    diff = boot[:, 1] - boot[:, 0]
    c, se = float(coef[1] - coef[0]), float(diff.std(ddof=1))
    res["post_minus_pre"] = {"estimate": c, "se_boot": se, "ci95": [c - bench.Z * se, c + bench.Z * se]}
    for f, p in bench.holm({f: res[f]["p"] for f in DECISIVE}).items():
        lo, hi = res[f]["ci95"]
        res[f] |= {"p_holm": p, "verdict": ("NO EFFECT WORTH HAVING" if -res["sesoi"] < lo and hi < res["sesoi"]
                                             else "CHANNEL" if p < 0.05 else "INCONCLUSIVE")}
    return res


def reading(res: dict) -> str:
    pre, post = res["pre"]["verdict"], res["post"]["verdict"]
    if pre == "CHANNEL":
        return "DIGESTION: pre-close news is followed by a bigger next move" + (
            "; after-close news too" if post == "CHANNEL" else f"; after-close: {post}")
    if post == "CHANNEL":
        return f"TIMING: after-close news only (its first reaction is the next trade); pre-close: {pre}"
    if pre == post == "NO EFFECT WORTH HAVING":
        return "NEITHER: no effect worth having before or after the close"
    return f"NOT CONCLUSIVE (pre-close {pre}, after-close {post}): the owner reads it at the review"


def run(confirm: bool) -> dict:
    p, news, trades = data(confirm)
    u = lambda **kw: units(p, news, trades, confirm, **kw)
    primary = evaluate(u())
    return {"mode": "confirm" if confirm else "design", "primary": primary, "reading": reading(primary),
            "decides": confirm,
            "secondary": {"all_headlines": evaluate(u(keep=np.ones(len(news), bool))),
                          "corporate_removed": evaluate(u(keep=(~news.is_price_report & ~news.corporate).to_numpy())),
                          "close_14_00": evaluate(u(close=14 * 60)), "close_14_30": evaluate(u(close=14 * 60 + 30))}}


# ------------------------------------------------------------------ controls (before the tag)
def controls(jobs: int) -> dict:
    """G0's controls for the two decisive coefficients, design window. None pairs the real
    news with the real targets: planted runs use random flags only; permuted targets; stale
    headlines (dates moved 250+ sessions on the design calendar, time of day kept)."""
    p, news, trades = data(False)
    rows = p2.rows(p, "vol", False)
    d = with_flags(rows, flags3(news[~news.is_price_report], trades))
    F = d[list(FLAGS)].to_numpy(float)
    rates, counts = F.mean(0), p2.block_counts(d.session.nunique())
    y0 = np.abs(d.ar1.to_numpy(float))
    firms = [np.flatnonzero(d.ticker.to_numpy() == t) for t in d.ticker.unique()]
    random_flags = lambda seed: (np.random.default_rng(seed).random(F.shape) < rates).astype(float)

    def stat(dd, FF, j, mu, sd):
        coef, boot = fit(dd, FF, counts)
        return float((coef[j] - mu) / np.hypot(sd, boot[:, j].std(ddof=1)))

    cal = h3.calendar()
    cal = cal[(cal >= h3.START) & (cal < bench.FIRM_SEAL)]
    hw = news[~news.is_price_report & (news.day >= h3.START) & (news.day < bench.FIRM_SEAL)]
    pos = cal.searchsorted(hw.day)
    hw, pos = hw[pos < len(cal)], pos[pos < len(cal)]
    tod = (hw.published_at - hw.published_at.dt.normalize()).to_numpy()
    step = max(1, (len(cal) - 500) // bench.SEEDS)

    def stale_rows(s):
        day = cal[(pos - 250 - step * s) % len(cal)]
        return with_flags(rows, flags3(hw.assign(day=day, published_at=day + tod), trades))

    out = {"n": len(d), "news_days": {f: int(F[:, j].sum()) for j, f in enumerate(FLAGS)}}
    for j, f in enumerate(DECISIVE):
        null = np.array([fit(d, random_flags(bench.SEED + 70_000 + s))[0][j] for s in range(bench.SEEDS)])
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        w = 1.0 if WIDE[f] else 0.0                  # the spread enters the yardstick only where chosen

        def planted(m, seed, j=j, mu=mu, sd=sd, w=w):
            G = random_flags(seed)
            return stat(d.assign(ar1=y0 + m * sd * G[:, j]), G, j, mu, w * sd)

        lo, hi = 0.5, 8.0
        for _ in range(8):
            mid = float(np.sqrt(lo * hi))
            med = np.median(bench.pmap(lambda s: planted(mid, bench.SEED + 10_000 + s), range(bench.CALIBRATION_RUNS), jobs))
            lo, hi = (mid, hi) if med < 2.8 else (lo, mid)
        g = float(np.sqrt(lo * hi))
        found = np.array(bench.pmap(lambda s: planted(g, bench.SEED + 20_000 + s), range(bench.SEEDS), jobs))

        def permuted(s, j=j):
            rng = np.random.default_rng(bench.SEED + 30_000 + s)
            y = y0.copy()
            for ix in firms:
                y[ix] = bench.block_permute(y[ix], 20, rng)
            e = d.assign(ar1=y)
            nul = noise(e, F, j, 100_000 + 1000 * s)
            return stat(e, F, j, float(nul.mean()), w * float(nul.std(ddof=1)))

        def stale(s, j=j, mu=mu, sd=sd, w=w):
            e = stale_rows(s)
            return stat(e, e[list(FLAGS)].to_numpy(float), j, mu, w * sd)

        n_found = int((found > bench.Z).sum())
        res = {"null_mean": mu, "null_sd": sd,
               "positive": {"gamma_in_null_sd": g, "found": n_found, "of": bench.SEEDS, "band": list(bench.POWER_BAND),
                            "median_stat": float(np.median(found)),
                            "pass": bench.POWER_BAND[0] <= n_found <= bench.POWER_BAND[1]}}
        for name, fn in (("permuted_labels", permuted), ("stale_news", stale)):
            z = np.array(bench.pmap(fn, range(bench.SEEDS), jobs))
            k = int((np.abs(z) > bench.Z).sum())
            res[name] = {"found": k, "of": bench.SEEDS, "max": bench.FALSE_MAX, "found_positive": int((z > bench.Z).sum()),
                         "found_negative": int((z < -bench.Z).sum()), "pass": k <= bench.FALSE_MAX}
        out[f] = res
    out["pass"] = all(out[f][c]["pass"] for f in DECISIVE for c in ("positive", "permuted_labels", "stale_news"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controls", action="store_true", help="the controls, design window")
    ap.add_argument("--confirm", action="store_true", help=f"the one sealed run; needs {TAG}")
    ap.add_argument("--jobs", type=int, default=1, help="workers for the controls")
    args = ap.parse_args()
    name = "p2t_controls" if args.controls else "p2t_confirm" if args.confirm else "p2t_design"
    path = h3.CURATED / f"{name}.json"
    if args.confirm and path.exists():
        raise SystemExit(f"{path.name} exists: the confirmation run is made once")
    out = controls(args.jobs) if args.controls else run(args.confirm)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out["pass"] if args.controls else out["reading"], f"-> {path}")


if __name__ == "__main__":
    main()
