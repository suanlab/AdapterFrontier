#!/usr/bin/env python3
"""Experiment A3c (mechanism/PREREG_AB.md §9.4): family or size?

The §9.1 pipeline, unchanged, on a size-matched pair from opposite families:
bert-large-uncased (encoder, 335M; tag `bertl`) and SmolLM2-360M (decoder,
362M; tag `smol`). Predictions A3-P3 and B-P3: the ensemble is worse than the
single on `bertl` and better on `smol`, at B in {8, 16} on both tasks.

Committed before any A3c run (§9.4). `--dev` runs the same pipeline on the
validation dev_test slice, for code checks only.

Usage: python3 mechanism/a3c_analysis.py [--dev]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_analysis as base  # noqa: E402

BACKBONES = {"bertl": "bert-large-uncased", "smol": "SmolLM2-360M"}
PREDICTED_SIGN = {"bertl": -1, "smol": +1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    split = "dev_test" if args.dev else "test"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"_note": ("EXPLORATORY dev_test run" if args.dev else "CONFIRMATORY: PREREG_AB.md §9.4"),
           "git_sha": sha, "eval_split": split, "cells": {}}
    a3p, bp = [], []
    for task in base.TASKS:
        for bb in BACKBONES:
            for B in base.BUDGETS:
                key = f"{task}/{bb}/B{B}"
                try:
                    reps = [base.one_replicate(task, bb, B, r, split) for r in range(len(base.REPLICATES))]
                except FileNotFoundError as e:
                    res["cells"][key] = {"status": f"missing {e}"}; print(key, "missing"); continue
                seed = zlib.crc32(("a3c|" + key).encode()) % (2**31)
                d_acc = base.joint_ci(reps, "dacc", seed)
                d_nll = base.joint_ci(reps, "dnll_ts", seed + 1)
                sign = PREDICTED_SIGN[bb]
                cell = {
                    "delta_acc": d_acc, "delta_acc_greedy": base.joint_ci(reps, "dacc_greedy", seed + 2),
                    "dnll_calibrated": d_nll, "dnll_uncalibrated": base.joint_ci(reps, "dnll_raw", seed + 3),
                    "risk90_EminusS": base.risk90_ci(reps, seed + 4),
                    "replicates": [{k: v for k, v in r.items() if k != "_vec"} for r in reps],
                    "predicted": B in base.PREDICTED_BUDGETS,
                    "A3_adjudication": base.adjudicate(d_acc, sign),
                    "B_adjudication": base.adjudicate({"ci95": [-d_nll["ci95"][1], -d_nll["ci95"][0]]}, sign),
                }
                res["cells"][key] = cell
                if cell["predicted"]:
                    a3p.append(cell["A3_adjudication"]); bp.append(cell["B_adjudication"])
                print(f"{key:18s} dAcc {100*d_acc['mean']:+.2f}pp [{100*d_acc['ci95'][0]:+.2f},"
                      f"{100*d_acc['ci95'][1]:+.2f}] {cell['A3_adjudication']:11s} | dNLL_TS {d_nll['mean']:+.4f} "
                      f"[{d_nll['ci95'][0]:+.4f},{d_nll['ci95'][1]:+.4f}] {cell['B_adjudication']}")
    res["A3_P3"] = {"cells": a3p, "verdict": base.verdict(a3p) if len(a3p) == 8 else "incomplete"}
    res["B_P3"] = {"cells": bp, "verdict": base.verdict(bp) if len(bp) == 8 else "incomplete"}
    print(f"\nA3-P3: {res['A3_P3']['verdict']}   B-P3: {res['B_P3']['verdict']}")
    out = base.OUT / ("a3c_dev_exploratory.json" if args.dev else "a3c_confirmatory.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
