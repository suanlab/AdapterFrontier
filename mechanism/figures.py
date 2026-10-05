#!/usr/bin/env python3
"""Figures for the mechanism study, from mechanism/results/*.json.

Encoder = slot-1 blue, decoder = slot-2 orange (validated: protan dE 24.7,
normal dE 33.6); every series also differs in marker shape and is labelled
directly, so identity never rests on colour alone. Measures on different
scales go in separate panels, never on a second y-axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True)
C = {"encoder": "#2a78d6", "decoder": "#eb6834"}
MK = {"encoder": "o", "decoder": "s"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png"); plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                color=INK, fontsize=7, va="center")


def fig_selection():
    a = load("h5_selection")["mean_delta_by_M_by_family"]
    b = load("h5b_selection_healthy")["mean_delta_by_M_by_family"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), sharey=True)
    for ax, data, title in ((axes[0], a, "All baseline candidates"),
                            (axes[1], b, "Candidates that trained")):
        for f in ("encoder", "decoder"):
            M = np.array(sorted(int(m) for m in data[f])); y = np.array([100 * data[f][str(m)] for m in M])
            ax.plot(M, y, color=C[f], marker=MK[f], markersize=5, markeredgecolor="white", markeredgewidth=0.8)
            label_end(ax, M[-1], y[-1], f, C[f])
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_xlim(0.8, 32); ax.set_title(title, color=INK, loc="left"); ax.set_xlabel("baseline candidates M")
    axes[0].set_ylabel("ensemble − selected single (pp)")
    save(fig, "fig_selection_pressure")


def fig_g_by_size():
    rows = load("g_by_size_exploratory")["rows"]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    rng = np.random.default_rng(0)
    for f in ("encoder", "decoder"):
        r = [x for x in rows if x["fam"] == f]
        xs = np.array([x["size"] for x in r]) * np.exp(rng.uniform(-0.06, 0.06, len(r)))
        ax.scatter(xs, [100 * x["G_tt"] for x in r], s=22, color=C[f], marker=MK[f],
                   edgecolor="white", linewidth=0.8, label=f, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xscale("log"); ax.set_xlabel("base model parameters (M)")
    ax.set_ylabel("G (pp)")
    ax.set_title("Extra rank and epochs: trained baseline − pool member", color=INK, loc="left", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "fig_quality_gap_by_size")


def fig_teq():
    d = load("teq_formula")
    fam = lambda p: "encoder" if (p.startswith("corpus:") or "bert" in p) else "decoder"
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    lo, hi = 0.98, 1.8
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    for f in ("encoder", "decoder"):
        r = [x for x in d["pools"] if fam(x["pool"]) == f]
        ax.scatter([x["pred_B"] for x in r], [x["T_eq"] for x in r], s=24, color=C[f], marker=MK[f],
                   edgecolor="white", linewidth=0.8, label=f, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"predicted $\sqrt{1+\pi s^2_{\mathrm{margin}}/8}$")
    ax.set_ylabel(r"observed $T_{\mathrm{eq}}$")
    ax.text(0.04, 0.95, f"R² = {d['B_margin_probit']['r2']:.2f}, {d['n_pools']} pools",
            transform=ax.transAxes, color=INK, fontsize=7, va="top")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig_implicit_temperature")


def fig_epochs():
    r = load("h3_new_runs")["all_runs"]
    ep = [1, 2, 4]; keys = ["ep1", "ep2", "ep4"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.9), gridspec_kw={"wspace": 0.45})
    panels = (("mean_acc", "member accuracy", 1.0), ("D_soft_vote", "averaging gain D (pp)", 100.0))
    for ax, (k, lab, sc) in zip(axes, panels):
        ax.plot(ep, [sc * r[x][k] for x in keys], color=C["decoder"], marker=MK["decoder"],
                markersize=5, markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xticks(ep); ax.set_xlabel("epochs"); ax.set_title(lab, color=INK, loc="left")
    fig.suptitle("Qwen-2.5-0.5B, MNLI, 5 members per point", x=0.02, ha="left", fontsize=7, color=MUTED, y=1.04)
    save(fig, "fig_diversity_by_epochs")


if __name__ == "__main__":
    fig_selection(); fig_g_by_size(); fig_teq(); fig_epochs()
