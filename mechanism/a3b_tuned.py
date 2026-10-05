#!/usr/bin/env python3
"""Experiment A3b (mechanism/PREREG_AB.md §9.2): E(16) against a single that
tunes its recipe inside the same budget.

S_tuned(16) = the val_selection-best of four 4-epoch configurations, one seed
each: c1 (r=8, lr 1e-4), c2 (r=8, lr 3e-4), c3 (r=32, lr 1e-4) and c4 (r=32,
lr 3e-4; the §9.1 S run). E(16) is the §9.1 ensemble of eight r=8 2-epoch
members, unchanged. Same outcomes, CIs and adjudication as a3_analysis.py.

This script was committed before any A3b run started (§9.2).

Usage: python3 mechanism/a3b_tuned.py [--dev]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3_analysis import BACKBONES, OUT, PREDICTED_SIGN, TASKS, adjudicate, joint_ci  # noqa: E402
from ab_common import RUNS, logits, metrics, softmax, validation_slices  # noqa: E402
from b1_calibrate import fit_T, nll_vec, power_T  # noqa: E402

E_SEEDS = (tuple(range(201, 209)), tuple(range(211, 219)))
CONFIGS = ("c1", "c2", "c3", "c4")
SEEDS = ({"c1": 301, "c2": 302, "c3": 303, "c4": 251},
         {"c1": 311, "c2": 312, "c3": 313, "c4": 261})


def config_dir(task, bb, cfg, seed) -> Path:
    if cfg == "c4":
        return RUNS / f"a3_{task}_{bb}_S_s{seed}"
    return RUNS / f"a3b_{task}_{bb}_{cfg}_s{seed}"


def load(d: Path, epoch: int, split: str):
    if not (d / "metrics.json").exists():
        raise FileNotFoundError(d)
    zv, yv = logits(d, epoch, "validation")
    if split == "test":
        zt, yt = logits(d, epoch, "test")
    else:
        zt, yt = None, None
    return zv, yv, zt, yt


def one_replicate(task, bb, rep, split):
    E = [load(RUNS / f"a3_{task}_{bb}_E_s{s}", 2, split) for s in E_SEEDS[rep]]
    C = {c: load(config_dir(task, bb, c, SEEDS[rep][c]), 4, split) for c in CONFIGS}
    y_val = E[0][1]
    for zv, yv, _, _ in E + list(C.values()):
        if not np.array_equal(yv, y_val):
            raise SystemExit(f"{task}/{bb}: validation labels do not align")
    sl = validation_slices(len(y_val))
    sel, comb, dev = sl["val_selection"], sl["val_combine"], sl["dev_test"]
    if split == "test":
        y_ev = E[0][3]
        ev = lambda t: t[2]
    else:
        y_ev = y_val[dev]
        ev = lambda t: t[0][dev]

    sel_acc = {c: float((C[c][0][sel].argmax(-1) == y_val[sel]).mean()) for c in CONFIGS}
    best = max(CONFIGS, key=lambda c: sel_acc[c])
    zS_val, zS_ev = C[best][0], ev(C[best])
    pE_val = np.mean([softmax(t[0]) for t in E], 0)
    pE_ev = np.mean([softmax(ev(t)) for t in E], 0)
    T_S, _ = fit_T(lambda T: softmax(zS_val[comb], T), y_val[comb])
    T_E, _ = fit_T(lambda T: power_T(pE_val[comb], T), y_val[comb])
    pS_ev, pS_ts, pE_ts = softmax(zS_ev), softmax(zS_ev, T_S), power_T(pE_ev, T_E)
    corr = lambda p: (p.argmax(-1) == y_ev).astype(float)
    cfg_meta = {c: {k: metrics(config_dir(task, bb, c, SEEDS[rep][c]))[k]
                    for k in ("lora_r", "learning_rate") if k in metrics(config_dir(task, bb, c, SEEDS[rep][c]))}
                for c in CONFIGS}
    return {
        "selected_config": best, "val_selection_acc": sel_acc, "configs": cfg_meta,
        "eval_acc_by_config": {c: float((ev(C[c]).argmax(-1) == y_ev).mean()) for c in CONFIGS},
        "T_S": T_S, "T_E": T_E,
        "acc_E": float(corr(pE_ev).mean()), "acc_S_tuned": float(corr(pS_ev).mean()),
        "_vec": {"dacc": corr(pE_ev) - corr(pS_ev),
                 "dnll_ts": nll_vec(pE_ts, y_ev) - nll_vec(pS_ts, y_ev)},
    }


def verdict(cells):
    if any(c == "REVERSED" for c in cells):
        return "FAILS"
    return "REPLICATES" if sum(c == "SUPPORTED" for c in cells) >= 3 else "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    split = "dev_test" if args.dev else "test"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"_note": ("EXPLORATORY dev_test run" if args.dev else "CONFIRMATORY: PREREG_AB.md §9.2"),
           "git_sha": sha, "eval_split": split, "cells": {}}
    a3, bp = [], []
    for task in TASKS:
        for bb in BACKBONES:
            key = f"{task}/{bb}/B16"
            try:
                reps = [one_replicate(task, bb, r, split) for r in range(2)]
            except FileNotFoundError as e:
                res["cells"][key] = {"status": f"missing {e}"}; print(key, "missing"); continue
            seed = zlib.crc32(("a3b|" + key).encode()) % (2**31)
            d_acc, d_nll = joint_ci(reps, "dacc", seed), joint_ci(reps, "dnll_ts", seed + 1)
            sign = PREDICTED_SIGN[bb]
            cell = {"delta_acc": d_acc, "dnll_calibrated": d_nll,
                    "A3_adjudication": adjudicate(d_acc, sign),
                    "B_adjudication": adjudicate({"ci95": [-d_nll["ci95"][1], -d_nll["ci95"][0]]}, sign),
                    "replicates": [{k: v for k, v in r.items() if k != "_vec"} for r in reps]}
            res["cells"][key] = cell
            a3.append(cell["A3_adjudication"]); bp.append(cell["B_adjudication"])
            picks = [r["selected_config"] for r in reps]
            print(f"{key:16s} picked {picks} dAcc {100*d_acc['mean']:+.2f}pp "
                  f"[{100*d_acc['ci95'][0]:+.2f},{100*d_acc['ci95'][1]:+.2f}] {cell['A3_adjudication']:11s} | "
                  f"dNLL_TS {d_nll['mean']:+.4f} [{d_nll['ci95'][0]:+.4f},{d_nll['ci95'][1]:+.4f}] "
                  f"{cell['B_adjudication']}")
    res["A3_P2"] = {"cells": a3, "verdict": verdict(a3) if len(a3) == 4 else "incomplete"}
    res["B_P2"] = {"cells": bp, "verdict": verdict(bp) if len(bp) == 4 else "incomplete"}
    print(f"\nA3-P2: {res['A3_P2']['verdict']}   B-P2: {res['B_P2']['verdict']}")
    out = OUT / ("a3b_dev_exploratory.json" if args.dev else "a3b_confirmatory.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
