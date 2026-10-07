"""H3: news tone vs next-session direction (drop/flat/rise) and firm drawdowns.

Implements audit/PREREG_H3.md (tag prereg-h3-v1) and nothing else. Every number in
this file that could have been tuned is fixed there; the section is cited inline.

    python3 preprocessing/h3.py        # -> data/curated/h3_results.json  (~minutes, CPU)

Reuses features.map_to_next_session (rule 2), features.phantom_sessions (§3.1),
hypothesis_tests.holm. Direction needs class probabilities and in-fold tercile cut
points, which baseline.walk_forward (binary, regress/sign) does not provide, so
walk_forward_proba below is the 3-class counterpart with the same refit/embargo logic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from config import ISSUER_STOPWORDS, PRICE_REPORT_PATTERN
from features import DEFAULT_CALENDAR, DEFAULT_PRICES, map_to_next_session, phantom_sessions
from hypothesis_tests import holm

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
RAW = ROOT / "data" / "raw" / "bvmt"
SCORED_V3 = CURATED / "04_scored_v3_camembert.parquet"
SCORED_V2 = CURATED / "04_scored_v2_camembert.parquet"
OUTPUT = CURATED / "h3_results.json"

START = "2016-01-04"                          # §2
MIN_TRAIN, REFIT = 500, 20                    # §4
N_BOOT, SEED = 2000, 20261003                 # §8
F0 = ["ret_lag0", "ret_lag1", "ret_lag2", "dow_next_mon", "dow_next_tue",
      "dow_next_wed", "dow_next_thu", "rv20", "dd250"]          # §6
NEWS = ["tone", "has_news", "log_n"]
CRASH_K, CRASH_H = 0.12, 20                   # §5


# ---------------------------------------------------------------- index frame
def calendar() -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(pd.read_csv(DEFAULT_CALENDAR, parse_dates=["session_date"])
                           ["session_date"]).normalize()
    return cal.difference(phantom_sessions()).sort_values()


def index_frame() -> pd.DataFrame:
    px = pd.read_csv(DEFAULT_PRICES, encoding="utf-8-sig")
    px["session"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px[~px.session.isin(phantom_sessions())].sort_values("session").reset_index(drop=True)
    px["ret"] = px.close.pct_change()                      # recomputed across phantoms
    f = px[["session", "close", "ret"]].copy()
    for lag in (0, 1, 2):                                  # lags counted from the target
        f[f"ret_lag{lag}"] = f.ret.shift(lag)
    nxt = f.session.shift(-1).dt.dayofweek
    for d, name in enumerate(["mon", "tue", "wed", "thu"]):
        f[f"dow_next_{name}"] = (nxt == d).astype(float)
    f["rv20"] = f.ret.rolling(20).std()
    f["dd250"] = f.close / f.close.rolling(250, min_periods=60).max() - 1
    for h in (1, 5, 20):
        f[f"fwd{h}"] = f.close.shift(-h) / f.close - 1
    return f


def headlines(scored: Path) -> pd.DataFrame:
    s = pd.read_parquet(scored)
    s = s[s.sent_label.astype(str) != ""].copy()
    if "p_positive" in s:
        s["tone"] = s.p_positive - s.p_negative
    else:                                              # v2 file: labels only (§10.2)
        s["tone"] = s.sent_label.map({"negative": -1, "very_negative": -1, "neutral": 0,
                                      "positive": 1, "very_positive": 1})
    s["day"] = pd.to_datetime(s.published_date).dt.normalize()
    s["session"] = map_to_next_session(s.day, calendar())
    s["is_price_report"] = s.headline_clean.astype(str).str.contains(
        PRICE_REPORT_PATTERN, case=False, regex=True, na=False)
    return s.dropna(subset=["session"])


def add_news(frame: pd.DataFrame, news: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    agg = news.groupby("session").tone.agg(["mean", "size"]).reset_index()
    out = frame.merge(agg, on="session", how="left")
    out[f"tone{suffix}"] = out.pop("mean").fillna(0.0)
    n = out.pop("size").fillna(0)
    out[f"has_news{suffix}"] = (n > 0).astype(float)
    out[f"log_n{suffix}"] = np.log1p(n)
    return out


def direction_frame(scored: Path = SCORED_V3, start: str = START) -> pd.DataFrame:
    f, news = index_frame(), headlines(scored)
    f = add_news(f, news)
    f = add_news(f, news[~news.is_price_report], "_ex_price")
    f = add_news(f, news[news.is_price_report], "_placebo")
    return f[f.session >= pd.Timestamp(start)].reset_index(drop=True)


# ------------------------------------------------------------ walk-forward
def tercile_class(r: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.where(r < q[0], 0, np.where(r > q[1], 2, 1))        # drop / flat / rise


def residualise(X: np.ndarray, target: int, controls: list[int], fit_rows: slice):
    """§6 orthogonal arm: OLS of column `target` on `controls`, fitted on fit_rows only."""
    A = np.c_[np.ones(len(X)), X[:, controls]]
    beta, *_ = np.linalg.lstsq(A[fit_rows], X[fit_rows, target], rcond=None)
    X = X.copy()
    X[:, target] -= A @ beta
    return X


def walk_forward_proba(frame: pd.DataFrame, features: list[str], h: int,
                       resid: tuple[str, list[str]] | None = None,
                       min_train: int = MIN_TRAIN, refit: int = REFIT) -> pd.DataFrame:
    """3-class expanding walk-forward. Rows j < i - embargo train the model for row i;
    embargo = max(5, h) >= h, so every training target is fully observed by the close
    of session i (§4). Tercile cut points come from the training slice only (§5)."""
    embargo = max(5, h)
    data = frame.dropna(subset=features + [f"fwd{h}"]).reset_index(drop=True)
    X, r = data[features].to_numpy(float), data[f"fwd{h}"].to_numpy()
    if resid:
        t, c = features.index(resid[0]), [features.index(x) for x in resid[1]]
    rows, model = [], None
    for i in range(min_train, len(data)):
        if model is None or (i - min_train) % refit == 0:
            cut = i - embargo
            q = np.percentile(r[:cut], [100 / 3, 200 / 3])
            Xi = residualise(X, t, c, slice(0, cut)) if resid else X
            scaler = StandardScaler().fit(Xi[:cut])
            model = LogisticRegression(C=1.0, max_iter=2000).fit(
                scaler.transform(Xi[:cut]), tercile_class(r[:cut], q))
        p = np.zeros(3)                  # a class absent from training gets probability 0
        p[model.classes_] = model.predict_proba(scaler.transform(Xi[i:i + 1]))[0]
        rows.append({"session": data.session.iloc[i], "y": int(tercile_class(r[i:i + 1], q)[0]),
                     "p0": p[0], "p1": p[1], "p2": p[2]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation
def _ll(pred: pd.DataFrame) -> np.ndarray:
    P = pred[["p0", "p1", "p2"]].to_numpy()
    return -np.log(np.clip(P[np.arange(len(P)), pred.y.to_numpy()], 1e-12, 1))


def _auc(pred: pd.DataFrame, idx=None) -> float:
    p = pred if idx is None else pred.iloc[idx]
    return float(roc_auc_score(p.y, p[["p0", "p1", "p2"]], multi_class="ovr", average="macro"))


def summary(pred: pd.DataFrame) -> dict:
    P, y = pred[["p0", "p1", "p2"]].to_numpy(), pred.y.to_numpy()
    yhat = P.argmax(1)
    return {"n": len(pred), "log_loss": round(float(_ll(pred).mean()), 5),
            "auc_macro": round(_auc(pred), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, yhat)), 4),
            "brier": round(float(((P - np.eye(3)[y]) ** 2).sum(1).mean()), 5),
            "mcc": round(float(matthews_corrcoef(y, yhat)), 4),
            "predicted_share": (np.bincount(yhat, minlength=3) / len(y)).round(3).tolist(),
            "true_share": (np.bincount(y, minlength=3) / len(y)).round(3).tolist()}


def block_indices(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n, size=-(-n // block))
    return (starts[:, None] + np.arange(block)).ravel()[:n] % n     # circular blocks


def hac_dm(d: np.ndarray, lags: int) -> float:
    dbar, n = d.mean(), len(d)
    var = ((d - dbar) ** 2).mean()
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * ((d[k:] - dbar) * (d[:-k] - dbar)).mean()
    return float(2 * (1 - stats.norm.cdf(abs(dbar / np.sqrt(var / n)))))


def compare(treat: pd.DataFrame, base: pd.DataFrame, h: int) -> dict:
    """§8: log-loss difference (treatment - baseline), moving-block bootstrap."""
    if not treat.session.equals(base.session):
        raise ValueError("arms not aligned on the same sessions")
    d = _ll(treat) - _ll(base)
    block, rng = max(20, 2 * h), np.random.default_rng(SEED)
    boot_d, boot_auc = [], []
    for _ in range(N_BOOT):
        idx = block_indices(len(d), block, rng)
        boot_d.append(d[idx].mean())
        boot_auc.append(_auc(treat, idx) - _auc(base, idx))
    boot_d, boot_auc = np.array(boot_d), np.array(boot_auc)
    dbar = float(d.mean())
    p = float(np.mean(np.abs(boot_d - boot_d.mean()) >= abs(dbar)))
    lags = int(4 * (len(d) / 100) ** (2 / 9)) + h - 1
    return {"d_log_loss": round(dbar, 6), "ci95": np.percentile(boot_d, [2.5, 97.5]).round(6).tolist(),
            "p_bootstrap": round(p, 4), "mde_80": round(2.8 * float(boot_d.std()), 6),
            "d_auc": round(_auc(treat) - _auc(base), 4),
            "d_auc_ci95": np.percentile(boot_auc, [2.5, 97.5]).round(4).tolist(),
            "dm_p_hac": round(hac_dm(d, lags), 4), "treatment_better": dbar < 0,
            "block": block}


# ------------------------------------------------------------ H3a / H3b
ARMS = {"all": NEWS, "ex_price": [f"{c}_ex_price" for c in NEWS],
        "placebo": [f"{c}_placebo" for c in NEWS], "orthogonal": NEWS}


def run_direction(frame: pd.DataFrame) -> dict:
    out = {}
    base1 = walk_forward_proba(frame, F0, 1)
    arms, preds = {}, {}
    for arm, cols in ARMS.items():
        resid = ("tone", ["ret_lag0", "ret_lag1", "log_n"]) if arm == "orthogonal" else None
        preds[arm] = walk_forward_proba(frame, F0 + cols, 1, resid=resid)
        arms[arm] = {"summary": summary(preds[arm]), **compare(preds[arm], base1, 1)}
        print(f"  H3a {arm}: d_log_loss {arms[arm]['d_log_loss']:+.5f} p {arms[arm]['p_bootstrap']}", flush=True)
    adj = holm({a: v["p_bootstrap"] for a, v in arms.items()})
    for a in arms:
        arms[a]["p_holm"] = adj[a]
    passed = {a: arms[a]["treatment_better"] and arms[a]["p_holm"] < 0.05 for a in arms}
    harmful = {a: (not arms[a]["treatment_better"]) and arms[a]["p_holm"] < 0.05 for a in arms}
    out["H3a"] = {"baseline": summary(base1), "arms": arms, "passed": passed, "harmful": harmful,
                  "verdict": ("SIGNAL FOUND" if passed["all"] and passed["ex_price"] else
                              "MOMENTUM, NOT NEWS" if passed["all"] or passed["placebo"] else
                              "NEWS DEGRADES THE FORECAST" if harmful["all"] else "INCONCLUSIVE")}

    sec = {}
    for h in (5, 20):
        b, t = walk_forward_proba(frame, F0, h), walk_forward_proba(frame, F0 + NEWS, h)
        sec[f"h{h}"] = {"baseline": summary(b), "summary": summary(t), **compare(t, b, h)}
        print(f"  H3b h{h}: d_log_loss {sec[f'h{h}']['d_log_loss']:+.5f}", flush=True)
    adj = holm({k: v["p_bootstrap"] for k, v in sec.items()})
    for k in sec:
        sec[k]["p_holm"] = adj[k]
    out["H3b"] = sec

    # §10 robustness, uncorrected, never decisive
    late = (base1.session >= "2019-01-01").to_numpy()
    rob = {"all_2019plus": compare(preds["all"][late].reset_index(drop=True),
                                   base1[late].reset_index(drop=True), 1)}
    v2 = direction_frame(SCORED_V2, start="2019-01-01")
    b2, t2 = walk_forward_proba(v2, F0, 1), walk_forward_proba(v2, F0 + NEWS, 1)
    rob["v2_scores_2019_window"] = {"summary": summary(t2), "baseline": summary(b2), **compare(t2, b2, 1)}
    out["robustness"] = rob
    return out


# ------------------------------------------------------------ H3c firm panel
def issuer_patterns() -> dict[str, re.Pattern]:
    names = pd.read_csv(RAW / "sotcks_list.csv")
    out = {}
    for name, ticker in zip(names.name.str.strip(), names.ticker.str.strip()):
        if len(name) >= 4 and name.lower() not in ISSUER_STOPWORDS:
            out[ticker] = re.compile(r"\b" + re.escape(name.lower()) + r"\b")
    return out


def firm_panel(end: str = "2022-12-30") -> pd.DataFrame:
    """`end`: ADR-002 F runs H3c past ALL_DATA's last session (f.py)."""
    cal = calendar()
    cal = cal[(cal >= "2015-06-01") & (cal <= end)]
    a = pd.read_csv(RAW / "ALL_DATA.csv")
    a["Date"] = pd.to_datetime(a.Date).dt.normalize()
    a = a.drop_duplicates(["Ticker", "Date"], keep="last")
    index = index_frame()[["session", "ret_lag0", "rv20"]].rename(
        columns={"ret_lag0": "idx_ret_lag0", "rv20": "idx_rv20"})
    news = headlines(SCORED_V3)
    news = news[news.session <= cal.max()]
    low = news.headline_clean.astype(str).str.lower()
    tone_idx = add_news(pd.DataFrame({"session": cal}), news)[["session", "tone"]]
    parts = []
    for ticker, pat in issuer_patterns().items():
        g = a[a.Ticker == ticker]
        if g.empty:
            continue
        c = g.set_index("Date").Close.reindex(cal).ffill().dropna()
        if len(c) < 40 + CRASH_H:
            continue
        f = pd.DataFrame({"session": c.index, "close": c.to_numpy()})
        f["ticker"] = ticker
        f["ret"] = f.close.pct_change()
        for lag in (0, 1, 2):
            f[f"f_ret_lag{lag}"] = f.ret.shift(lag)
        f["f_rv20"] = f.ret.rolling(20).std()
        f["f_dd60"] = f.close / f.close.rolling(60, min_periods=20).max() - 1
        future_min = pd.concat([f.close.shift(-j) for j in range(1, CRASH_H + 1)], axis=1)
        f["event"] = (future_min.min(axis=1) / f.close - 1 < -CRASH_K).astype(float)
        f.loc[future_min.isna().any(axis=1), "event"] = np.nan     # last 20 sessions: no target
        f["age"] = np.arange(len(f))
        mine = news[low.str.contains(pat)].groupby("session").tone.agg(["sum", "size"])
        m = mine.reindex(f.session).fillna(0)
        s20, n20 = m["sum"].rolling(20, min_periods=1).sum(), m["size"].rolling(20, min_periods=1).sum()
        f["firm_tone20"] = np.where(n20 > 0, s20 / n20.replace(0, 1), 0.0)
        f["has_firm_news20"] = (n20 > 0).astype(float).to_numpy()
        parts.append(f)
    p = pd.concat(parts).merge(index, on="session", how="left").merge(tone_idx, on="session", how="left")
    p = p[(p.session >= START) & (p.age >= 40)].dropna(subset=["event"])
    return p.reset_index(drop=True)


FIRM_BASE = ["f_ret_lag0", "f_ret_lag1", "f_ret_lag2", "f_rv20", "f_dd60", "idx_ret_lag0", "idx_rv20"]
FIRM_NEWS = ["firm_tone20", "has_firm_news20", "tone"]


def walk_forward_panel(p: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Pooled logistic over dates: min_train 500 sessions, refit 20, embargo 20 (§4)."""
    p = p.dropna(subset=features).sort_values(["session", "ticker"]).reset_index(drop=True)
    sessions = np.sort(p.session.unique())
    pos = np.searchsorted(sessions, p.session.to_numpy())
    X, y = p[features].to_numpy(float), p.event.to_numpy().astype(int)
    out = []
    for k in range(MIN_TRAIN, len(sessions), REFIT):
        train = pos < k - CRASH_H
        scaler = StandardScaler().fit(X[train])
        model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
            scaler.transform(X[train]), y[train])
        thr = np.percentile(model.predict_proba(scaler.transform(X[train]))[:, 1], 95)
        test = (pos >= k) & (pos < k + REFIT)
        prob = model.predict_proba(scaler.transform(X[test]))[:, 1]
        out.append(p.loc[test, ["session", "ticker", "close", "event"]].assign(
            prob=prob, alarm=prob >= thr))
    return pd.concat(out).reset_index(drop=True)


def lead_times(pred: pd.DataFrame, panel: pd.DataFrame) -> list[int]:
    """Per event episode (a ticker's run of consecutive event rows) with an alarm: sessions
    from the first alarm to the first close below (1 - k) x that alarm row's close."""
    leads = []
    for ticker, g in pred.sort_values("session").groupby("ticker"):
        closes = panel[panel.ticker == ticker].set_index("session").close
        ev = g.event.to_numpy().astype(bool)
        run_id = np.cumsum(ev & ~np.r_[False, ev[:-1]])
        for rid in np.unique(run_id[ev]):
            ep = g[(run_id == rid) & ev]
            hits = ep[ep.alarm]
            if hits.empty:
                continue
            first = hits.iloc[0]
            after = closes[closes.index > first.session].iloc[:CRASH_H]
            breach = np.flatnonzero(after.to_numpy() < (1 - CRASH_K) * first.close)
            if len(breach):
                leads.append(int(breach[0] + 1))
    return leads


def run_crash() -> dict:
    panel = firm_panel()
    base, treat = walk_forward_panel(panel, FIRM_BASE), walk_forward_panel(panel, FIRM_BASE + FIRM_NEWS)
    assert base[["session", "ticker"]].equals(treat[["session", "ticker"]])
    y = treat.event.to_numpy().astype(int)

    def ap(pr, ix=slice(None)):
        return average_precision_score(y[ix], pr.prob.to_numpy()[ix])

    def ll(pr):
        q = np.clip(pr.prob.to_numpy(), 1e-12, 1 - 1e-12)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q))

    sessions = np.sort(treat.session.unique())
    by_s = treat.groupby("session").indices
    rng, boot = np.random.default_rng(SEED), []
    for _ in range(N_BOOT):          # time blocks: all firms on a date move together (§8)
        pick = sessions[block_indices(len(sessions), 20, rng)]
        ix = np.concatenate([by_s[s] for s in pick])
        boot.append(ap(treat, ix) - ap(base, ix))
    boot = np.array(boot)
    d = ap(treat) - ap(base)

    def desc(pr):
        alarms = pr.alarm.to_numpy()
        lt = lead_times(pr, panel)
        bins = pd.qcut(pr.prob, 10, labels=False, duplicates="drop")
        return {"pr_auc": round(float(ap(pr)), 4), "event_rate": round(float(y.mean()), 4),
                "recall_at_5pct_alarm": round(float(alarms[y == 1].mean()), 4),
                "alarm_rate": round(float(alarms.mean()), 4),
                "median_lead_sessions": float(np.median(lt)) if lt else None,
                "episodes_detected": len(lt), "log_loss": round(float(ll(pr).mean()), 5),
                "calibration_deciles": pr.assign(b=bins).groupby("b").agg(
                    pred=("prob", "mean"), obs=("event", "mean")).round(4).to_dict("list")}
    return {"panel_rows": len(panel), "tickers": int(panel.ticker.nunique()),
            "predictions": len(treat), "baseline": desc(base), "treatment": desc(treat),
            "d_pr_auc": round(float(d), 4), "ci95": np.percentile(boot, [2.5, 97.5]).round(4).tolist(),
            "p_bootstrap": round(float(np.mean(np.abs(boot - boot.mean()) >= abs(d))), 4),
            "mde_80": round(2.8 * float(boot.std()), 4),
            "d_log_loss": round(float(ll(treat).mean() - ll(base).mean()), 6)}


def main() -> None:
    frame = direction_frame()
    results = {"prereg": "audit/PREREG_H3.md @ prereg-h3-v1", "scored": SCORED_V3.name,
               "window_start": START, "phantom_sessions_removed": len(phantom_sessions()),
               **run_direction(frame)}
    print("  H3c firm panel...", flush=True)
    results["H3c"] = run_crash()
    OUTPUT.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({"H3a_verdict": results["H3a"]["verdict"]}, indent=2), f"-> {OUTPUT}")


if __name__ == "__main__":
    main()
