"""The lab notes' figures, as SVG in the site's palette (drawn again on every rebuild)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from whiff_lab.data import ROOT  # noqa: E402
from whiff_lab.evaluate import TRAIN_LAST, run_expectancy  # noqa: E402
from whiff_lab.features import ERAS, STATE, counts, run_environment  # noqa: E402
from whiff_lab.models import Logistic, Shrunk, _two_rows, baseball_numbers  # noqa: E402

INK, MUTED, RULE, LEATHER, GRAPHITE = "#1c1e21", "#64615b", "#e3dfd7", "#d0612a", "#23262b"
FONT = "Archivo, 'Arial Narrow', sans-serif"
RESULTS = ROOT / "results"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 100,
        "svg.hashsalt": "whiff-lab",
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def save(fig, path: Path) -> Path:
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg = path.read_text()
    svg = re.sub(r"font-family:\s*'?DejaVu Sans'?", f"font-family: {FONT}", svg)
    svg = re.sub(r"font: (\d+(?:\.\d+)?)px 'DejaVu Sans'", rf"font: \1px {FONT}", svg)
    path.write_text(svg)
    return path


def coverage(out: Path) -> Path:
    f = json.loads((RESULTS / "findings.json").read_text())
    s = f["seasons"]
    years = [r["year"] for r in s]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(
        years, [r["games_all"] for r in s], width=0.85, color=RULE, label="All AL and NL games (game logs)"
    )
    ax.bar(
        years,
        [r["games_pbp"] or 0 for r in s],
        width=0.85,
        color=GRAPHITE,
        label="Games the lab uses (play-by-play)",
    )
    ax.set_ylabel("Games per season")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(1900, years[-1] + 1)
    return save(fig, out / "coverage.svg")


def sparsity(out: Path, c) -> Path:
    s = c[c.year <= TRAIN_LAST].groupby(STATE).n.sum()
    edges = [1, 10, 100, 1_000, 10_000, 100_000, 10**9]
    names = ["1-9", "10-99", "100-999", "1,000-9,999", "10,000-99,999", "100,000+"]
    states = [int(((s >= a) & (s < b)).sum()) for a, b in zip(edges, edges[1:], strict=False)]
    share = [float(s[(s >= a) & (s < b)].sum() / s.sum()) for a, b in zip(edges, edges[1:], strict=False)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a1.barh(names, states, color=GRAPHITE)
    a1.set_title("Game states, by how often they came up")
    a1.invert_yaxis()
    for i, v in enumerate(states):
        a1.text(v, i, f" {v:,}", va="center", fontsize=8.5, color=INK)
    a2.barh(names, share, color=LEATHER)
    a2.set_title("Share of plate appearances")
    a2.invert_yaxis()
    a2.set_yticklabels([])
    for i, v in enumerate(share):
        a2.text(v, i, f" {v:.1%}", va="center", fontsize=8.5, color=INK)
    a2.set_xlim(0, max(share) * 1.3)
    a1.set_xlim(0, max(states) * 1.3)
    fig.supylabel("Times seen, 1908-2016", fontsize=9, color=MUTED)
    return save(fig, out / "sparsity.svg")


def calibration(out: Path) -> Path:
    h = json.loads((RESULTS / "holdout.json").read_text())
    models = h["models"][1:]
    fig, axes = plt.subplots(1, len(models), figsize=(7.4, 2.3), sharey=True)
    for ax, m in zip(axes, models, strict=True):
        b = [x for x in m["calibration"] if x["n"] >= 500]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.scatter(
            [x["said"] for x in b],
            [x["happened"] for x in b],
            s=[8 + x["n"] / 6000 for x in b],
            color=LEATHER,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_title(
            m["name"].replace(" (empirical Bayes)", "").replace("Gradient", "Gradient\n"), fontsize=9
        )
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_aspect("equal")
    axes[0].set_ylabel("Home team won")
    fig.supxlabel("The model said (bins of 5 points; dot size: plate appearances)", fontsize=9, color=MUTED)
    return save(fig, out / "calibration.svg")


def shrinkage(out: Path, c) -> Path:
    train = c[c.year <= TRAIN_LAST]
    sh = Shrunk().fit(train)
    g = train.groupby(STATE)[["n", "w"]].sum().reset_index()
    g = g[g.n < 300]
    raw = (g.w / g.n).to_numpy()
    shr = sh.predict(g)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    sc = ax.scatter(
        raw,
        shr,
        c=np.log10(g.n),
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("lab", [LEATHER, "#e7a17c", GRAPHITE]),
        s=10,
        alpha=0.75,
        linewidth=0,
        rasterized=True,  # thousands of dots: one picture, not thousands of shapes
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, ticks=[0, 1, 2], pad=0.02)
    cb.ax.set_yticklabels(["1", "10", "100"])
    cb.set_label("Times seen", color=MUTED)
    ax.set_xlabel("The table: the home team's record from the state")
    ax.set_ylabel("Shrunk toward its parent state")
    ax.set_title(f"{len(g):,} states seen fewer than 300 times")
    return save(fig, out / "shrinkage.svg")


def logistic(out: Path, c) -> Path:
    train = c[c.year <= TRAIN_LAST]
    rpg = run_environment().set_index("year").rpg
    m = Logistic(run_expectancy(range(1908, TRAIN_LAST + 1)), rpg).fit(train)
    z = baseball_numbers(train, m.re, m.rpg, m.rpg_ref).z.to_numpy()
    bins = np.linspace(-4, 4, 41)
    idx = np.digitize(z, bins)
    xs, ys, ns = [], [], []
    for i in range(1, len(bins)):
        k = idx == i
        if train.n[k].sum() > 2000:
            xs.append(np.average(z[k], weights=train.n[k]))
            ys.append(train.w[k].sum() / train.n[k].sum())
            ns.append(train.n[k].sum())
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    zz = np.linspace(-4, 4, 200)
    X, yy, sw = _two_rows(train, pd.DataFrame({"z": z}, index=train.index))
    one = LogisticRegression().fit(X, yy, sample_weight=sw)  # two numbers: an intercept and a slope
    ax.plot(
        zz,
        one.predict_proba(pd.DataFrame({"z": zz}))[:, 1],
        color=MUTED,
        lw=1.2,
        ls=(0, (4, 3)),
        label=f"logistic curve, slope {one.coef_[0][0]:.2f} on z",
    )
    ax.scatter(
        xs,
        ys,
        s=[6 + n / 60000 for n in ns],
        color=LEATHER,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label="what happened, 1908-2016",
    )
    ax.set_xlabel("z: the home team's expected lead at the end, divided by how much can still change")
    ax.set_ylabel("Home team won")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper left", fontsize=8.5)
    return save(fig, out / "logistic.svg")


def rule(out: Path) -> Path:
    h = json.loads((RESULTS / "holdout.json").read_text())
    r = json.loads((RESULTS / "rule_change.json").read_text())
    names = [m["name"] for m in r["models"]][1:]
    before = [
        next(m for m in h["models"] if m["name"] == n)["log_loss"][
            "extra innings since 2020 (runner on second)"
        ]
        for n in names
    ]
    after = [m["log_loss"] for m in r["models"]][1:]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.barh(y - 0.2, before, height=0.38, color=RULE, label="fitted on 1908-2016 (never saw the rule)")
    ax.barh(y + 0.2, after, height=0.38, color=LEATHER, label="fitted on 1908-2022 (three seasons of it)")
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        ax.text(a, i - 0.2, f" {a:.3f}", va="center", fontsize=8, color=INK)
        ax.text(b, i + 0.2, f" {b:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [n.replace(" (empirical Bayes)", "") for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.62)
    ax.set_xlabel("Log loss in extra innings with the runner on second (lower is better)")
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1, 1.0), ncol=1)
    return save(fig, out / "rule.svg")


def eras(out: Path) -> Path:
    f = json.loads((RESULTS / "findings.json").read_text())
    spots = f["spots"][1:4]
    fig, axes = plt.subplots(1, len(spots), figsize=(7.6, 3.0), sharey=True)
    for ax, sp in zip(axes, spots, strict=True):
        for i, e in enumerate(ERAS):
            v = sp["eras"][e.key]
            ax.plot([v["lo"], v["hi"]], [i, i], color=GRAPHITE, alpha=0.35, lw=2.5, solid_capstyle="butt")
            ax.scatter(
                [v["p"]],
                [i],
                color=LEATHER if e.key in ("deadball", "liveball") else GRAPHITE,
                s=22,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(sp["name"].replace(", nobody on, nobody out", "\nnobody on, nobody out"), fontsize=8.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ERAS)), [f"{e.name} ({e.first}-)" for e in ERAS], fontsize=8)
    axes[0].invert_yaxis()
    fig.supxlabel(
        "The home team's chance to win (dot) and its 90% interval (line)", fontsize=9, color=MUTED, y=-0.04
    )
    return save(fig, out / "eras.svg")


def comebacks(out: Path) -> Path:
    f = json.loads((RESULTS / "findings.json").read_text())
    s = [r for r in f["seasons"] if r.get("under10") is not None]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.array([r["rpg"] for r in s])
    y = np.array([r["under10"] for r in s])
    ax.scatter(x, y, s=14, color=GRAPHITE, alpha=0.7, linewidth=0)
    for r in s:
        if r["year"] in (1908, 1930, 1968, 2000, 2025):
            ax.annotate(
                str(r["year"]),
                (r["rpg"], r["under10"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
                color=LEATHER,
            )
    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, a * xx + b, color=LEATHER, lw=1.5)
    ax.set_xlabel("Runs per team per game")
    ax.set_ylabel("Games won from below 10%")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title(f"One dot per season (correlation {f['comeback_rpg_r']:.2f})")
    return save(fig, out / "comebacks.svg")


def draw_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    c = counts()
    return [
        coverage(out),
        sparsity(out, c),
        calibration(out),
        shrinkage(out, c),
        logistic(out, c),
        rule(out),
        eras(out),
        comebacks(out),
    ]
