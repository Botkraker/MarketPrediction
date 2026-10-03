"""README figure: how much news changes each H3 forecast, with 95% intervals.

    python3 docs/make_figures.py      # -> docs/assets/h3_effects{,_dark}.svg

Reads data/curated/h3_results.json. Every effect is signed so that right = news helps:
the log-loss change is negated (lower loss is better), the PR-AUC change is used as is.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
THEMES = {  # chart chrome from the dataviz reference palette; one series, slot-1 blue
    "": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", mark="#2a78d6"),
    "_dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835", mark="#3987e5"),
}


def rows():
    r = json.loads((ROOT / "data" / "curated" / "h3_results.json").read_text())
    a, b, c = r["H3a"]["arms"]["all"], r["H3b"], r["H3c"]
    flip = lambda d, ci: (-d, -ci[1], -ci[0])
    direction = [("Next session", *flip(a["d_log_loss"], a["ci95"]), a["mde_80"]),
                 ("5 sessions", *flip(b["h5"]["d_log_loss"], b["h5"]["ci95"]), b["h5"]["mde_80"]),
                 ("20 sessions", *flip(b["h20"]["d_log_loss"], b["h20"]["ci95"]), b["h20"]["mde_80"])]
    crash = [("Firm drop > 12%", c["d_pr_auc"], *c["ci95"], c["mde_80"])]
    return direction, crash


def panel(ax, data, title, unit, t):
    for i, (label, est, lo, hi, mde) in enumerate(data):
        y = len(data) - 1 - i
        ax.plot([lo, hi], [y, y], color=t["mark"], lw=2, solid_capstyle="round", zorder=3)
        ax.plot(est, y, "o", ms=9, color=t["mark"], mec=t["surface"], mew=2, zorder=4)
        ax.plot(mde, y, "|", ms=14, mew=1.5, color=t["ink2"], zorder=2)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=t["ink"], fontsize=11)
    ax.axvline(0, color=t["ink2"], lw=1, zorder=1)
    span = max(max(abs(lo), abs(hi), mde) for _, _, lo, hi, mde in data) * 1.15
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_yticks([])
    ax.set_title(title, loc="left", color=t["ink"], fontsize=12, pad=10)
    ax.set_xlabel(unit, color=t["ink2"], fontsize=9.5)
    for x, ha, text in ((0, "left", "◀ news hurts"), (1, "right", "news helps ▶")):
        ax.annotate(text, xy=(x, 0), xycoords="axes fraction", xytext=(0, -24),
                    textcoords="offset points", ha=ha, va="top", color=t["ink2"], fontsize=9)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(t["surface"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    direction, crash = rows()
    for suffix, t in THEMES.items():
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 4.6), height_ratios=[3, 1.25],
                                   facecolor=t["surface"])
        panel(a, direction, "Direction forecast (drop / flat / rise)", "improvement in log-loss", t)
        panel(b, crash, "Crash early warning, 77 firms", "improvement in PR-AUC", t)
        fig.text(0.215, 0.015, "●  estimate     ━  95% interval     |  smallest effect the test could detect",
                 color=t["ink2"], fontsize=9)
        fig.subplots_adjust(left=0.2, right=0.97, top=0.93, bottom=0.17, hspace=1.0)
        fig.savefig(OUT / f"h3_effects{suffix}.svg", facecolor=t["surface"])
        plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
