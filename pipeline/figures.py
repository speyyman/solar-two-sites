"""Figures for the paper (PDF, light) and for the website (SVG, parchment).

Every figure reads results/*.json, results/daily_errors.csv or the prediction
tables written by `evaluate`; nothing is typed in by hand. Colours: the paper
uses a validated categorical trio (checked with the dataviz palette validator:
CVD dE >= 9.2, normal-vision dE >= 27.6 on #fcfcfb); the site variant uses ink
and one vermilion, with line style as the second cue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline.config import FIGURES, INTERIM, PRIMARY_NWP, RESULTS, SITES  # noqa: E402
from pipeline.evaluate import MAIN, REF  # noqa: E402


@dataclass(frozen=True)
class Theme:
    name: str
    ext: str
    surface: str
    ink: str
    muted: str
    grid: str
    c1: str  # primary series
    c2: str
    c3: str
    font: str


PAPER = Theme(
    "paper",
    "pdf",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "DejaVu Sans",
)
SITE = Theme(
    "site",
    "svg",
    "#ece4d0",
    "#0c1611",
    "#3d4a42",
    "#d3c9b0",
    "#c2410c",
    "#2f5e3c",
    "#6b5a2e",
    "DejaVu Sans Mono",
)
SITE_NAMES = {"arbuckle": "Arbuckle, CA", "kersey": "Kersey, CO"}
LABELS = {
    "persistence": "Persistence",
    "smart_persistence": "Smart persistence",
    f"physical/{PRIMARY_NWP}": "Physical (GFS)",
    f"physical_cal/{PRIMARY_NWP}": "Physical, calibrated (GFS)",
    f"gbm/{PRIMARY_NWP}": "Boosted trees (GFS)",
    "gbm/ecmwf": "Boosted trees (ECMWF)",
    "gbm/era5": "Boosted trees (ERA5, hindsight)",
    f"gbm_transfer/{PRIMARY_NWP}": "Boosted trees, other site (GFS)",
}


def style(t: Theme) -> None:
    plt.rcParams.update(
        {
            "font.family": t.font,
            "font.size": 8.5,
            "axes.edgecolor": t.muted,
            "axes.labelcolor": t.ink,
            "xtick.color": t.muted,
            "ytick.color": t.muted,
            "text.color": t.ink,
            "axes.facecolor": t.surface,
            "figure.facecolor": t.surface,
            "savefig.facecolor": t.surface,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": t.grid,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "solar-two-sites",  # stable ids, so reruns give identical SVG
            "pdf.fonttype": 42,
            "lines.linewidth": 1.6,
        }
    )


def save(fig: plt.Figure, name: str, t: Theme) -> None:
    out = FIGURES if t.name == "paper" else FIGURES / "site"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "solar-two-sites"} if t.ext == "pdf" else {"Date": None}
    if t.ext == "pdf":
        meta["CreationDate"] = None
    fig.savefig(out / f"{name}.{t.ext}", bbox_inches="tight", metadata=meta)
    if t.name == "paper":
        fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_skill(m: dict, t: Theme) -> None:
    rows = [k for k in LABELS if k != REF]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, key in zip(axes, SITES, strict=True):
        f = m["sites"][key]["full"]
        for i, c in enumerate(rows):
            s = f["point"][c]["skill"]
            lo, hi = f["skill_ci95"][c]
            col = t.c1 if c == MAIN else t.ink
            ax.plot([lo, hi], [i, i], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(s, i, "o", ms=5, color=col, mec=t.surface, mew=1.2)
            ax.text(hi + 0.012, i, f"{s:+.2f}", va="center", fontsize=7.5, color=t.muted)
        ax.axvline(0, color=t.muted, lw=0.8)
        ax.set_yticks(range(len(rows)), [LABELS[c] for c in rows])
        ax.invert_yaxis()
        ax.set_xlim(-0.2, 0.45)
        ax.set_title(SITE_NAMES[key], loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.supxlabel(
        "Skill against smart persistence, 95% block-bootstrap interval "
        "(1 = perfect, 0 = no better)",
        fontsize=8.5,
    )
    fig.tight_layout()
    save(fig, "skill", t)


def fig_sky(m: dict, t: Theme) -> None:
    series = [(REF, t.c2, "s"), (f"physical_cal/{PRIMARY_NWP}", t.c3, "^"), (MAIN, t.c1, "o")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    bins = ["clear", "mixed", "overcast"]
    for ax, key in zip(axes, SITES, strict=True):
        sky = m["sites"][key]["full"]["by_sky"]
        for j, (c, col, mk) in enumerate(series):
            x = np.arange(len(bins)) + (j - 1) * 0.18
            y = [sky[b]["point"][c]["nrmse"] for b in bins]
            ax.plot(x, y, mk, color=col, ms=6, mec=t.surface, mew=1.0, ls="none", label=LABELS[c])
        days = [sky[b]["days"] for b in bins]
        ax.set_xticks(range(len(bins)), [f"{b}\n{d} days" for b, d in zip(bins, days, strict=True)])
        ax.set_title(SITE_NAMES[key], loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE, fraction of AC rating")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "by_sky", t)


def fig_monthly(m: dict, t: Theme) -> None:
    series = [
        (REF, t.c2, (0, (4, 2))),
        (f"physical_cal/{PRIMARY_NWP}", t.c3, (0, (1, 1.5))),
        (MAIN, t.c1, "-"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"width_ratios": [18, 10]}
    )
    for ax, key in zip(axes, SITES, strict=True):
        bm = m["sites"][key]["full"]["by_month"]
        months = sorted(bm)
        for c, col, ls in series:
            ax.plot(
                range(len(months)),
                [bm[mo]["point"][c]["nrmse"] for mo in months],
                color=col,
                ls=ls,
                marker="o",
                ms=3,
                label=LABELS[c],
            )
        ax.set_xticks(
            range(0, len(months), 3), [months[i] for i in range(0, len(months), 3)], fontsize=7
        )
        ax.set_title(SITE_NAMES[key], loc="left", fontsize=9)
    axes[0].set_ylabel("Monthly RMSE, fraction of AC")
    axes[0].set_ylim(0, 0.34)
    # Legend under the panels, so it never sits on a line.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "monthly", t)


def fig_categories(fails: dict, t: Theme) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6))
    for ax, key in zip(axes, SITES, strict=True):
        cats = fails["sites"][key]["all_days_categories_abs_err_gt_1h"]
        items = sorted(cats.items(), key=lambda kv: kv[1])
        ax.barh(range(len(items)), [v for _, v in items], color=t.c1, height=0.62)
        ax.set_yticks(range(len(items)), [k for k, _ in items], fontsize=7.5)
        for i, (_, v) in enumerate(items):
            ax.text(v + 0.5, i, str(v), va="center", fontsize=7.5, color=t.muted)
        n = fails["sites"][key]["days_abs_err_gt_1h"]
        ax.set_title(
            f"{SITE_NAMES[key]}: {n} days off by over 1 rated hour", loc="left", fontsize=9
        )
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Days")
    fig.tight_layout()
    save(fig, "categories", t)


CASES = (
    ("arbuckle", "2024-10-17", "2024-10-19", "No output under a clear sky"),
    ("kersey", "2024-09-11", "2024-09-14", "One of two inverters off"),
    ("arbuckle", "2025-12-05", "2025-12-09", "Still December days"),
    ("kersey", "2024-08-08", "2024-08-10", "Heavy cloud, then cloud no source saw"),
)


def fig_cases(t: Theme) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, (key, a, b, title) in zip(axes.flat, CASES, strict=True):
        site = SITES[key]
        p = pd.read_parquet(INTERIM / f"{key}_predictions.parquet")
        p = p[(p["day"] >= a) & (p["day"] <= b)]
        x = p.index.tz_convert(site.tz).tz_localize(None)
        ax.fill_between(
            x,
            p[f"gbm_q10/{PRIMARY_NWP}"],
            p[f"gbm_q90/{PRIMARY_NWP}"],
            color=t.c1,
            alpha=0.18,
            lw=0,
            label="80% band (GFS)",
        )
        ax.plot(x, p[MAIN], color=t.c1, label="Forecast (GFS)")
        ax.plot(
            x,
            p["physical/era5"],
            color=t.c2,
            ls=(0, (4, 2)),
            lw=1.2,
            label="Physical model on ERA5 sky",
        )
        ax.plot(x, p["y_n"], color=t.ink, lw=1.4, label="Measured")
        ax.set_title(f"{SITE_NAMES[key]}, {a} to {b[5:]}\n{title}", loc="left", fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    for ax in axes[:, 0]:
        ax.set_ylabel("Power, fraction of AC")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "cases", t)


def run() -> None:
    m = json.loads((RESULTS / "metrics.json").read_text())
    fails = json.loads((RESULTS / "failures.json").read_text())
    for t in (PAPER, SITE):
        style(t)
        fig_skill(m, t)
        fig_sky(m, t)
        fig_monthly(m, t)
        fig_categories(fails, t)
        fig_cases(t)
    print("figures written to", FIGURES)
