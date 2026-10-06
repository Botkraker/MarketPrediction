"""Post-hoc checks of P2 (owner, 2026-10-06), after the review in AUDIT_REPORT §P2.

Design window 2016-2020 only. They cannot change G2: its sealed run is final. p2.py stays
exactly as pre-registered (tag prereg-p2-v1); each check feeds it different inputs, or
swaps one piece of its model for the duration of the check.

  phantom    the phantom sessions dropped from ALL_DATA and the Tunindex (data rule 2)
  corporate  issuer headlines about dividends, ex-dates, general meetings, bonus shares
             or capital increases removed from the news flags
  t8_range   the volatility model also gets the firm's same-session range ln(H/L) and
             |r|^2 (ADR T8); High/Low are sanity-checked first
  t8_dates   t8_range plus date effects next to the firm intercepts
  all        phantom + corporate + t8_dates

    OMP_NUM_THREADS=1 python3 preprocessing/p2_posthoc.py   # -> data/curated/p2_posthoc.json
"""
from __future__ import annotations

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd

import bench
import h3
import p2
from features import phantom_sessions

OUTPUT = h3.CURATED / "p2_posthoc.json"
PHANTOMS = phantom_sessions()             # fixed before any read is filtered
CORPORATE = (r"dividende|d[ée]tachement|coupon|mise en paiement|\bago\b|\bage\b|"
             r"assembl[ée]e g[ée]n[ée]rale|attribution gratuite|actions? gratuites?|"
             r"augmentation de capital")
T8 = ["range", "abs_r2"]
FROZEN_DESIGN = p2.design_matrix


@contextmanager
def patched(obj, name, value):
    """Swap one attribute while a check runs. ponytail: the frozen p2.py takes no such
    arguments; P3's code should take them as parameters instead."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def read_csv_without_phantoms(*args, read=pd.read_csv, **kwargs):
    """pd.read_csv minus rows dated on a phantom session (ALL_DATA's Date, the Tunindex's
    date). The calendar and the issuer list have neither column."""
    df = read(*args, **kwargs)
    for col in ("Date", "date"):
        if col in df:
            return df[~pd.to_datetime(df[col]).dt.normalize().isin(PHANTOMS)]
    return df


@contextmanager
def without_phantoms():
    with patched(h3, "phantom_sessions", lambda *a, **k: PHANTOMS), \
            patched(pd, "read_csv", read_csv_without_phantoms):
        yield


def read_parquet_without_corporate(*args, read=pd.read_parquet, **kwargs):
    df = read(*args, **kwargs)
    return df[~df.headline_clean.astype(str).str.contains(CORPORATE, case=False, regex=True, na=False)]


def firm_range() -> tuple[pd.DataFrame, dict]:
    """ln(High/Low) per firm trade day of the design window, after a sanity check; a day
    that fails it has no range and drops out of the T8 checks. Open is never read."""
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv", usecols=["Ticker", "Date", "High", "Low", "Close"])
    a["session"] = pd.to_datetime(a.Date).dt.normalize()
    a = a.drop_duplicates(["Ticker", "session"], keep="last")
    a = a[(a.session >= h3.START) & (a.session < bench.FIRM_SEAL)]
    outside = (a.Close > a.High) | (a.Close < a.Low)
    bad = (a.Low <= 0) | (a.High < a.Low) | outside
    check = {"firm_days": len(a), "low_not_positive": int((a.Low <= 0).sum()),
             "high_below_low": int((a.High < a.Low).sum()), "close_outside_range": int(outside.sum()),
             "zero_range": int((a.High == a.Low).sum())}
    a["range"] = np.where(bad, np.nan, np.log(a.High / a.Low))
    return a.rename(columns={"Ticker": "ticker"})[["ticker", "session", "range"]], check


def two_way(M: np.ndarray, d: pd.DataFrame, tol: float = 1e-12) -> np.ndarray:
    """Firm and date effects removed together, by alternating projections."""
    firm, date = pd.factorize(d.ticker)[0], pd.factorize(d.session)[0]
    for _ in range(1000):
        new = p2.within(p2.within(M, firm), date)
        if np.max(np.abs(new - M)) < tol:
            return new
        M = new
    raise RuntimeError("two-way demeaning did not converge")


def with_date_effects(d, test, news, target):
    """p2.design_matrix with date effects next to the volatility model's firm intercepts."""
    if test != "vol":
        return FROZEN_DESIGN(d, test, news, target)
    return two_way(np.abs(d[target].to_numpy(float)), d), two_way(np.c_[news, d[p2.HAR].to_numpy(float)], d)


def main() -> None:
    news = p2.issuer_news()
    with patched(pd, "read_parquet", read_parquet_without_corporate):
        news_nc = p2.issuer_news()
    base, base_nc = p2.panel(news=news), p2.panel(news=news_nc)
    with without_phantoms():
        clean, clean_nc = p2.panel(news=news), p2.panel(news=news_nc)
    if clean.session.isin(PHANTOMS).any():
        raise SystemExit("phantom rows survived the filter")
    rng, check = firm_range()
    t8 = lambda q: q.assign(abs_r2=q.r ** 2).merge(rng, on=["ticker", "session"], how="left")
    design = lambda h: h[(h.day >= h3.START) & (h.day < bench.FIRM_SEAL)]
    out = {"note": "post hoc (owner, 2026-10-06): design window 2016-2020 only; cannot change G2",
           "phantom_rows_dropped": int((base.session < bench.FIRM_SEAL).sum() - (clean.session < bench.FIRM_SEAL).sum()),
           "corporate_issuer_headlines_removed": len(design(news)) - len(design(news_nc)),
           "high_low_check": check, "checks": {}}
    c = out["checks"]
    c["phantom"] = p2.family(clean, False)
    c["corporate"] = p2.family(base_nc, False)
    with patched(p2, "HAR", p2.HAR + T8):
        c["t8_range"] = p2.family(t8(base), False)
        with patched(p2, "design_matrix", with_date_effects):
            c["t8_dates"] = p2.family(t8(base), False)
            c["all"] = p2.family(t8(clean_nc), False)
    with open(OUTPUT, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
