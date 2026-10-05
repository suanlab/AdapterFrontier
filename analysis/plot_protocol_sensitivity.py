#!/usr/bin/env python3
"""Figure 1: the apparent PEFT-ensembling benefit is a product of
evaluation protocol.

Panel A (same denominator): on the 48 pools that carry both baselines, the
share of cells adjudicated SUPPORTED and REVERSED against the weak baseline
(best_of_n) and the better-resourced n_rank single, per family. Every pair of
bars compares the same pools, so the drop is the baseline and nothing else.
An earlier version chained stages with different denominators.

Panel B (distribution): per-cell accuracy deltas for the strict encoder
comparison (encoder x n_rank x accuracy, n=92), colored by post-FDR
verdict. The mass sits left of zero; no cell is SUPPORTED.

Reads the same clean subset as analysis/protocol_sensitivity.py
(HellaSwag excluded: stale selection split; GSM8K excluded: contaminated).

Usage: python3 analysis/plot_protocol_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from protocol_sensitivity import load_accuracy_cells, _verdict_stats  # noqa: E402

C_SUP = "#2E7D32"     # green
C_REV = "#C62828"     # red
C_UNS = "#BDBDBD"     # gray
C_WEAK = "#86b6ef"    # light blue: best_of_n
C_STRONG = "#1c5cab"  # dark blue: n_rank (ordinal pair, validated)


def main() -> int:
    cells = load_accuracy_cells()
    enc = [c for c in cells if c["family"] == "encoder"]

    import json
    matched = json.loads((ROOT / "analysis/protocol_sensitivity.json").read_text()
                         )["S2_no_compute_match"]["matched_pools"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---------------- Panel A: same pools, two baselines ----------------
    groups = []
    for fam, name in (("encoder", "Encoder"), ("decoder", "Decoder")):
        m = matched[fam]
        for key, verdict in (("pct_supported", "SUP"), ("pct_reversed", "REV")):
            groups.append((f"{name}\n{verdict}", m["best_of_n"][key], m["n_rank"][key],
                           m["n_pools"]))
    x = np.arange(len(groups)) + np.array([0, 0, 0.5, 0.5])
    w = 0.36
    weak = [g[1] for g in groups]
    strong = [g[2] for g in groups]
    axA.bar(x - w / 2, weak, w, color=C_WEAK, label="vs best-of-N (weak)")
    axA.bar(x + w / 2, strong, w, color=C_STRONG, label="vs n_rank (stronger single)")
    for xi, a_, b_ in zip(x, weak, strong):
        axA.text(xi - w / 2, a_ + 0.8, f"{a_:.1f}", ha="center", fontsize=8.5, color="#333333")
        axA.text(xi + w / 2, b_ + 0.8, f"{b_:.1f}", ha="center", fontsize=8.5, color="#333333")
    axA.set_xticks(x)
    axA.set_xticklabels([g[0] for g in groups], fontsize=9)
    for xc, fam in ((x[0:2].mean(), "encoder"), (x[2:4].mean(), "decoder")):
        axA.text(xc, -8.0, f"{matched[fam]['n_pools']} pools, "
                 f"{matched[fam]['n_rank']['n_cells']} cells per bar",
                 ha="center", fontsize=8, color="#555555")
    axA.set_ylabel("% of cells (post-FDR verdict)", fontsize=10)
    axA.set_title("A  Same pools, stronger baseline", fontsize=10.5, loc="left",
                  fontweight="bold")
    axA.set_ylim(0, max(weak + strong) * 1.25)
    axA.legend(frameon=False, fontsize=9, loc="upper left")
    axA.spines[["top", "right"]].set_visible(False)

    # ---------------- Panel B: strict encoder distribution ----------------
    strict = [c for c in enc if c["baseline_kind"] == "n_rank"]
    d = np.array([c["diff_pp"] for c in strict])
    verd = [c["verdict"] for c in strict]
    colors = [C_REV if v == "reversed" else (C_SUP if v == "supported" else C_UNS)
              for v in verd]
    order = np.argsort(d)
    axB.scatter(d[order], np.arange(len(d)), c=[colors[i] for i in order],
                s=16, edgecolors="none")
    axB.axvline(0, color="black", lw=1.0)
    axB.axvline(d.mean(), color="#1565C0", lw=1.2, ls="--")
    axB.text(d.mean() - 0.25, len(d) * 0.55, f"mean {d.mean():+.2f}pp",
             color="#1565C0", fontsize=9, ha="right")
    axB.set_xlabel("Δ accuracy vs the n_rank single (pp)", fontsize=10)
    axB.set_ylabel("cells (sorted)", fontsize=10)
    n_rev = sum(1 for v in verd if v == "reversed")
    axB.set_title(f"B  Strict encoder comparison: 0/{len(d)} SUPPORTED, "
                  f"{n_rev} REVERSED", fontsize=10.5, loc="left", fontweight="bold")
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_ylim(-2, len(d) * 1.12)
    # data runs bottom-left -> top-right, so the upper-left corner is free
    axB.legend(handles=[mpatches.Patch(color=C_REV, label="REVERSED"),
                        mpatches.Patch(color=C_UNS, label="unsupported")],
               frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    figdir = ROOT / "paper/figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"fig1_protocol_sensitivity.{ext}",
                    dpi=200, bbox_inches="tight")

    print("Panel A (matched pools):")
    for label, a_, b_, n in groups:
        print(f"  {label.replace(chr(10), ' '):12s} pools={n}  best_of_n {a_:5.1f}%  n_rank {b_:5.1f}%")
    print(f"Panel B: n={len(d)} strict encoder cells, mean {d.mean():+.3f}pp, "
          f"reversed {n_rev} ({100*n_rev/len(d):.1f}%), supported "
          f"{sum(1 for v in verd if v=='supported')}")
    print(f"wrote {figdir}/fig1_protocol_sensitivity.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
