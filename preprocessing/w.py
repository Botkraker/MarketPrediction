"""ADR-001 W: H3c's price-only firm drawdown warning, calibrated and written up.

Plan: ADR-001 §8, track W ("in-fold calibration (isotonic or Platt; balanced class weights
inflate levels), reliability plot, lead times. A result on its own."). The owner chose W
after the G3 Exit; decisions (2026-10-06) in STATUS.md. The model is H3c's pre-registered
price-only model (h3.firm_panel, h3.FIRM_BASE, h3.walk_forward_panel), unchanged, on its
own window: an H3 replication, exempt from the seals.

SCREEN     headline numbers drop the rows whose 20-session window holds a one-session drop
           beyond -10% (presumed unadjusted ex-dates; P2/P3's rule); H3c as published is
           reported alongside (owner). Each sample gets its own calibrators.
CALIBRATE  Platt (logistic on the logit of H3c's probability), fitted for each 20-session
           block on every earlier prediction whose 20-session outcome was known before the
           block (owner); isotonic as a check. The first WARMUP prediction sessions only
           train the calibrators.
REPORT     PR-AUC with a 20-date block bootstrap; Brier and log loss before and after
           calibration; reliability by decile (audit/figures/w_reliability.png); alarms at
           H3c's threshold and at calibrated 10/15/20%; lead times (h3.lead_times); all by
           year. Plus one table of every ADR-001 primary result, read from the result files.

    OMP_NUM_THREADS=1 python3 preprocessing/w.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import h3  # noqa: E402

OUTPUT, FIGURE = h3.CURATED / "w_results.json", h3.ROOT / "audit" / "figures" / "w_reliability.png"
JUMP, WARMUP, LEVELS = 0.10, 260, (0.10, 0.15, 0.20)   # P2/P3's screen; ~1 year, a multiple of REFIT
SCORES = (("raw", "prob"), ("platt", "platt"), ("isotonic", "isotonic"))


def jump_windows(end: str = "2022-12-30") -> pd.DataFrame:
    """Per (ticker, session): does the next 20-session window hold a one-session drop beyond
    -JUMP? Closes built as in h3.firm_panel (carried over sessions without a trade)."""
    cal = h3.calendar()
    cal = cal[(cal >= "2015-06-01") & (cal <= end)]
    a = pd.read_csv(h3.RAW / "ALL_DATA.csv", usecols=["Ticker", "Date", "Close"])
    a["Date"] = pd.to_datetime(a.Date).dt.normalize()
    a = a.drop_duplicates(["Ticker", "Date"], keep="last")
    parts = []
    for t, g in a.groupby("Ticker"):
        c = g.set_index("Date").Close.reindex(cal).ffill().dropna()
        jump = (c / c.shift() - 1 < -JUMP).astype(float)
        ahead = jump[::-1].rolling(h3.CRASH_H, min_periods=1).max()[::-1].shift(-1).fillna(0) > 0
        parts.append(pd.DataFrame({"ticker": t, "session": c.index, "jump_window": ahead.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def calibrate(pred: pd.DataFrame, method: str) -> np.ndarray:
    """For each 20-session block after the warm-up, a calibrator fitted on every earlier
    prediction whose 20-session outcome was known before the block; NaN in the warm-up."""
    sessions = np.sort(pred.session.unique())
    pos = np.searchsorted(sessions, pred.session.to_numpy())
    x, y = pred.prob.to_numpy(float), pred.event.to_numpy(int)
    logit = np.log(x / (1 - x))
    out = np.full(len(pred), np.nan)
    for k in range(WARMUP, len(sessions), h3.REFIT):
        past, now = pos < k - h3.CRASH_H, (pos >= k) & (pos < k + h3.REFIT)
        if method == "platt":
            fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit[past, None], y[past])
            out[now] = fit.predict_proba(logit[now, None])[:, 1]
        else:
            fit = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x[past], y[past])
            out[now] = fit.predict(x[now])
    return out


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return {"pr_auc": float(average_precision_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "mean_prob": float(np.mean(p)), "event_rate": float(np.mean(y))}


def reliability(y: np.ndarray, p: np.ndarray) -> dict:
    g = pd.DataFrame({"p": p, "y": y, "b": pd.qcut(p, 10, labels=False, duplicates="drop")}).groupby("b")
    return {"pred": g.p.mean().round(4).tolist(), "obs": g.y.mean().round(4).tolist()}


def alarms(y: np.ndarray, flag: np.ndarray) -> dict:
    return {"alarm_rate": float(flag.mean()), "precision": float(y[flag].mean()) if flag.any() else None,
            "recall": float(flag[y == 1].mean())}


def leads(d: pd.DataFrame, panel: pd.DataFrame, flag: np.ndarray) -> dict:
    """h3.lead_times on this sample, plus how many drawdown episodes it holds."""
    lt = h3.lead_times(d.assign(alarm=flag), panel)
    ev = d.sort_values(["ticker", "session"]).groupby("ticker").event
    episodes = int(((d.event == 1) & ~ev.shift().eq(1).reindex(d.index)).sum())
    return {"episodes": episodes, "detected": len(lt), "median_lead": float(np.median(lt)) if lt else None,
            "lead_quartiles": np.percentile(lt, [25, 75]).tolist() if lt else None}


def bootstrap(d: pd.DataFrame) -> dict:
    """PR-AUC and Brier, 2,000 resamples of 20-session blocks of dates (H3c's scheme)."""
    sessions = np.sort(d.session.unique())
    by_s = d.groupby("session").indices
    rng = np.random.default_rng(h3.SEED)
    y, raw, platt = d.event.to_numpy(int), d.prob.to_numpy(float), d.platt.to_numpy(float)
    ap, b_raw, b_platt = [], [], []
    for _ in range(h3.N_BOOT):
        ix = np.concatenate([by_s[s] for s in sessions[h3.block_indices(len(sessions), 20, rng)]])
        ap.append(average_precision_score(y[ix], raw[ix]))
        b_raw.append(np.mean((raw[ix] - y[ix]) ** 2))
        b_platt.append(np.mean((platt[ix] - y[ix]) ** 2))
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    return {"pr_auc_ci95": ci(ap), "brier_raw_ci95": ci(b_raw), "brier_platt_ci95": ci(b_platt)}


def analyse(d: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """One sample (screened or as published), scored period only."""
    y = d.event.to_numpy(int)
    years = {}
    for yr, g in d.groupby(d.session.dt.year):
        gy = g.event.to_numpy(int)
        years[int(yr)] = {"n": len(g), "event_rate": float(gy.mean()),
                          "pr_auc": float(average_precision_score(gy, g.prob)),
                          "brier_raw": float(np.mean((g.prob - gy) ** 2)), "brier_platt": float(np.mean((g.platt - gy) ** 2)),
                          "h3c_alarm": alarms(gy, g.alarm.to_numpy())}
    levels = {f"platt_{round(100 * lv)}pct": (d.platt >= lv).to_numpy() for lv in LEVELS}
    return {"n": len(d), "window": f"{d.session.min().date()}..{d.session.max().date()}",
            "scores": {m: scores(y, d[c].to_numpy(float)) for m, c in SCORES},
            "reliability": {m: reliability(y, d[c].to_numpy(float)) for m, c in SCORES},
            "alarms": {"h3c_threshold": alarms(y, d.alarm.to_numpy())} | {k: alarms(y, v) for k, v in levels.items()},
            "lead_times": {"h3c_threshold": leads(d, panel, d.alarm.to_numpy())}
                          | {k: leads(d, panel, v) for k, v in levels.items()},
            "bootstrap": bootstrap(d), "by_year": years}


def summary(w: dict) -> str:
    """One table of every ADR-001 primary result, read from the result files."""
    load = lambda name: json.loads((h3.CURATED / f"{name}.json").read_text())
    ci = lambda c: f"[{c[0]:+.3g}, {c[1]:+.3g}]"
    rows = ["| phase | question | window | effect [95% CI] | MDE | SESOI | verdict |",
            "|---|---|---|---|---|---|---|"]
    h1 = load("p0_results_run3")["h1_v3"]
    rows.append(f"| P0 | H1 rerun: v3 tone → next index return (Clark-West, 4 arms) | {h1['window']} "
                f"| min Holm p {min(h1['holm']['cw'].values()):.2f} | — | none | INCONCLUSIVE |")
    for m in ("design", "confirm"):
        j = load(f"p1_{m}")
        d = j["primary"]
        rows.append(f"| P1 | news flow → next index variance (Δ QLIKE vs noise) | {j['window']} "
                    f"| {d['effect_vs_noise']:+.3g} {ci(d['ci95_vs_noise'])} | {d['mde_noise']:.3g} | {d['sesoi']:.3g} | {d['verdict']} |")
    for m in ("design", "confirm"):
        j = load(f"p2_{m}")
        for k, label in (("c1", "drift with news, next trade (c)"), ("c5", "drift with news, 5 trades (c)"),
                         ("vol", "news-day volatility (g)")):
            v = j["primary"]["tests"][k]
            rows.append(f"| P2 | {label} | {j['units']['window']} | {v['effect_vs_noise']:+.3g} {ci(v['ci95'])} "
                        f"| {v['mde']:.3g} | {v['sesoi']:.3g} | {v['verdict']} |")
    for m in ("design", "confirm"):
        j = load(f"p3_{m}")
        v = j["primary"]
        rows.append(f"| P3 | word list adds to v3 (Clark-West gain) | {j['window']} | {v['gain']:+.3g} {ci(v['ci95'])} "
                    f"| {v['mde']:.3g} | {v['sesoi']:.3g} | {v['verdict']} |")
    s = w["screened"]
    rows.append(f"| W | price-only 20-session drawdown warning: PR-AUC (event rate {s['scores']['raw']['event_rate']:.3g}) "
                f"| {s['window']} | {s['scores']['raw']['pr_auc']:.3g} {ci(s['bootstrap']['pr_auc_ci95'])} | — | — | descriptive |")
    return "\n".join(rows)


def figure(d: pd.DataFrame) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for col, label in (("prob", "H3c as published"), ("platt", "Platt"), ("isotonic", "isotonic")):
        r = reliability(d.event.to_numpy(int), d[col].to_numpy(float))
        ax.plot(r["pred"], r["obs"], marker="o", label=label)
    ax.plot([0, 0.8], [0, 0.8], color="grey", lw=0.8, ls="--", label="perfect calibration")
    ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="mean predicted probability (decile)",
           ylabel="observed share with a > 12% drawdown in 20 sessions",
           title=f"Firm drawdown warning, {d.session.min().year}-{d.session.max().year} (screened)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> None:
    panel = h3.firm_panel()
    pred = h3.walk_forward_panel(panel, h3.FIRM_BASE)
    reproduced = float(average_precision_score(pred.event.astype(int), pred.prob))   # H3c: 0.1789
    pred = pred.merge(jump_windows(), on=["ticker", "session"], how="left")
    out = {"h3c_pr_auc_reproduced": reproduced, "predictions": len(pred), "warmup_sessions": WARMUP,
           "jump_window_unknown": int(pred.jump_window.isna().sum())}
    pred["jump_window"] = pred.jump_window.fillna(False).astype(bool)
    out["jump_window_rows"] = int(pred.jump_window.sum())
    samples = {"screened": pred[~pred.jump_window].reset_index(drop=True), "as_published": pred}
    for name, d in samples.items():
        d = d.assign(platt=calibrate(d, "platt"), isotonic=calibrate(d, "isotonic"))
        samples[name] = d[d.platt.notna()].reset_index(drop=True)
        out[name] = analyse(samples[name], panel)
    out["summary_table"] = summary(out)
    figure(samples["screened"])
    with open(OUTPUT, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(out["summary_table"], f"\n-> {OUTPUT}, {FIGURE}")


if __name__ == "__main__":
    main()
