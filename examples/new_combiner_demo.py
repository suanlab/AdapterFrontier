#!/usr/bin/env python3
"""Evaluate a NEW combination rule on the released pools, without retraining.

This is the end-to-end path the paper's resource claim rests on (2026-10-04
panel, B-M4 / C-M5 / D-M2 / E-M4). It needs only files in the release:
per-member validation and test logits of the A3 equal-budget runs
(mechanism/runs/a3_*), the split helper, and the paired adjudication below.

The example rule is "calibrate each member, then average": every member gets
its own temperature, fitted on val_combine, before its probabilities are
averaged. It is compared against the equal-budget single S(16) of
PREREG_AB.md §9.1 on the test split, with a paired bootstrap CI, exactly as
the paper's cells are adjudicated. To try another rule, replace
`combine()`; it receives each member's validation and test logits plus the
validation labels and returns test probabilities.

Usage: python3 examples/new_combiner_demo.py [--task snli|yahoo] [--backbone bert|q05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mechanism"))
from ab_common import RUNS, logits, softmax, validation_slices  # noqa: E402

E_SEEDS = tuple(range(201, 209))     # E(16), replicate 1
S_SEEDS = tuple(range(251, 255))     # S(16), replicate 1
T_GRID = np.logspace(np.log10(0.05), np.log10(20), 200)


def fit_temperature(z, y):
    nll = [-np.log(softmax(z, T)[np.arange(len(y)), y] + 1e-12).mean() for T in T_GRID]
    return float(T_GRID[int(np.argmin(nll))])


def combine(val_logits, test_logits, y_val, comb):
    """The new rule: per-member temperature, then probability averaging."""
    temps = [fit_temperature(zv[comb], y_val[comb]) for zv in val_logits]
    return np.mean([softmax(zt, T) for zt, T in zip(test_logits, temps)], axis=0)


def adjudicate(p_new, p_base, y, B=5000, seed=0):
    d = (p_new.argmax(-1) == y).astype(float) - (p_base.argmax(-1) == y).astype(float)
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    verdict = "supported" if lo > 0 else "reversed" if hi < 0 else "unsupported"
    return d.mean(), (lo, hi), verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["snli", "yahoo"], default="snli")
    ap.add_argument("--backbone", choices=["bert", "q05"], default="q05")
    args = ap.parse_args()

    def load(arm, seed, epoch):
        d = RUNS / f"a3_{args.task}_{args.backbone}_{arm}_s{seed}"
        zv, yv = logits(d, epoch, "validation")
        zt, yt = logits(d, epoch, "test")
        return zv, yv, zt, yt

    E = [load("E", s, 2) for s in E_SEEDS]
    S = [load("S", s, 4) for s in S_SEEDS]
    y_val, y_test = E[0][1], E[0][3]
    sl = validation_slices(len(y_val))
    sel, comb = sl["val_selection"], sl["val_combine"]

    k = int(np.argmax([(zv[sel].argmax(-1) == y_val[sel]).mean() for zv, *_ in S]))
    p_single = softmax(S[k][2])
    p_new = combine([e[0] for e in E], [e[2] for e in E], y_val, comb)
    p_plain = np.mean([softmax(e[2]) for e in E], axis=0)

    for name, p in (("plain soft vote", p_plain), ("new rule", p_new)):
        delta, (lo, hi), v = adjudicate(p, p_single, y_test)
        print(f"{args.task}/{args.backbone}  {name:16s} vs S(16): "
              f"{100*delta:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
