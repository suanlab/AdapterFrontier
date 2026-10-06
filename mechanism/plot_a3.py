#!/usr/bin/env python3
"""Figure: A3 equal-budget comparison on sealed test sets (PREREG_AB.md §9.1).

Rows: accuracy difference and calibrated-NLL difference (ensemble minus
single); columns: SNLI and Yahoo Answers. x = budget in epoch-units (equal
tokens for both pipelines); error bars are the joint 95% bootstrap CI over
test examples for the mean of two replicates. Reads results/a3_confirmatory.json.

Usage: python3 mechanism/plot_a3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
# family hue (blue encoder, orange decoder); the size-matched A3c pair is the
# lighter shade with a dashed line and square markers (secondary encoding)
SERIES = {"bert": ("BERT-base", "#1c5cab", "-", "o"), "q05": ("Qwen2.5-0.5B", "#c24a1c", "-", "o"),
          "bertl": ("BERT-large", "#6da7ec", "--", "s"), "smol": ("SmolLM2-360M", "#f29a6e", "--", "s")}
OFFSETS = {"bert": 0.94, "q05": 0.98, "bertl": 1.02, "smol": 1.06}
TASKS = {"snli": "SNLI", "yahoo": "Yahoo Answers"}
BUDGETS = (4, 8, 16)
ROWS = (("delta_acc", 100.0, "Δ accuracy (pp)"),
        ("dnll_calibrated", 1.0, "Δ NLL, both scaled"))
# One cell is dominated by a single failed run (the only single candidate of
# replicate 2 at B=4); it is drawn as an off-scale marker, not to scale.
OFFSCALE = {("snli", "bertl", 4)}


def main() -> int:
    cells = json.loads((HERE / "results/a3_confirmatory.json").read_text())["cells"]
    cells.update(json.loads((HERE / "results/a3c_confirmatory.json").read_text())["cells"])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.9), sharex=True)
    for i, (key, scale, ylabel) in enumerate(ROWS):
        for j, (task, tname) in enumerate(TASKS.items()):
            ax = axes[i, j]
            ax.axhline(0, color="#555555", lw=0.8)
            for bb, (label, color, ls, mk) in SERIES.items():
                xs, ys, lo, hi = [], [], [], []
                for B in BUDGETS:
                    if (task, bb, B) in OFFSCALE:
                        continue
                    c = cells[f"{task}/{bb}/B{B}"][key]
                    xs.append(B * OFFSETS[bb])   # small dodge, same scale
                    ys.append(scale * c["mean"])
                    lo.append(scale * (c["mean"] - c["ci95"][0]))
                    hi.append(scale * (c["ci95"][1] - c["mean"]))
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=2, ls=ls, marker=mk, ms=5,
                            capsize=3, elinewidth=1.2, markeredgecolor="white", markeredgewidth=1.5)
            for (t, b, B) in OFFSCALE:
                if t == task:
                    v = scale * cells[f"{t}/{b}/B{B}"][key]["mean"]
                    lo_, hi_ = ax.get_ylim()
                    y = hi_ if v > hi_ else lo_ if v < lo_ else v
                    ax.plot([B * OFFSETS[b]], [y], marker="^" if v > 0 else "v", color=SERIES[b][1], ms=7, clip_on=False)
                    ax.annotate(f"off scale: {v:+.2f}" + ("pp" if key == "delta_acc" else "") + " (failed single)",
                                (B * OFFSETS[b], y), xytext=(9, 0), textcoords="offset points",
                                fontsize=7, color="#555555", va="center", annotation_clip=False)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
            ax.set_xlim(3.2, 24)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            ax.tick_params(labelsize=8.5)
            if i == 0:
                ax.set_title(tname, fontsize=10, loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if i == 1:
                ax.set_xlabel("budget (epochs over 20k rows; equal tokens)", fontsize=8.5)
    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker=mk, label=l) for l, c, ls, mk in SERIES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_a3_equal_budget.{ext}", dpi=200, bbox_inches="tight")
        fig.savefig(HERE.parent / "paper/figures" / f"fig_a3_equal_budget.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/fig_a3_equal_budget.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
