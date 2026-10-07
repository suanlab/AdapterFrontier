#!/usr/bin/env python3
"""Experiment A3d (mechanism/PREREG_AB.md §9.6): a tuned ensemble against the
tuned single.

E_tuned(16): eight 2-epoch members, two seeds for each of four member recipes
(r in {8, 32} x lr in {3e-4, 1e-4}), combined by greedy forward selection on
val_selection (primary) or a soft vote over all eight (secondary).
S_tuned(16): the val_selection-best of four 4-epoch configurations (§9.2).
Both cost 16 epoch-units. Same outcomes and adjudication as a3b_tuned.py.

Committed before any A3d run (§9.6). `--dev` runs on the validation dev_test
slice, for code checks only.

Usage: python3 mechanism/a3d_analysis.py [--dev]
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
import a3b_tuned as tb  # noqa: E402
from a3_analysis import BACKBONES, OUT, PREDICTED_SIGN, TASKS, adjudicate, greedy_subset, joint_ci  # noqa: E402
from ab_common import softmax, validation_slices  # noqa: E402
from b1_calibrate import fit_T, nll_vec, power_T  # noqa: E402

MEMBER_SEEDS = (
    {"m1": (201, 202), "m2": (401, 402), "m3": (403, 404), "m4": (405, 406)},
    {"m1": (211, 212), "m2": (411, 412), "m3": (413, 414), "m4": (415, 416)},
)


def member_dir(task, bb, cfg, seed) -> Path:
    if cfg == "m1":
        return tb.RUNS / f"a3_{task}_{bb}_E_s{seed}"
    return tb.RUNS / f"a3d_{task}_{bb}_{cfg}_s{seed}"


def one_replicate(task, bb, rep, split):
    members = [(cfg, s, tb.load(member_dir(task, bb, cfg, s), 2, split))
               for cfg, seeds in MEMBER_SEEDS[rep].items() for s in seeds]
    C = {c: tb.load(tb.config_dir(task, bb, c, tb.SEEDS[rep][c]), 4, split) for c in tb.CONFIGS}
    y_val = members[0][2][1]
    for _, _, (zv, yv, _, _) in members:
        if not np.array_equal(yv, y_val):
            raise SystemExit(f"{task}/{bb}: validation labels do not align")
    sl = validation_slices(len(y_val))
    sel, comb, dev = sl["val_selection"], sl["val_combine"], sl["dev_test"]
    if split == "test":
        y_ev = members[0][2][3]
        ev = lambda t: t[2]
    else:
        y_ev = y_val[dev]
        ev = lambda t: t[0][dev]

    # tuned single, exactly as in §9.2
    best = max(tb.CONFIGS, key=lambda c: float((C[c][0][sel].argmax(-1) == y_val[sel]).mean()))
    zS_val, zS_ev = C[best][0], ev(C[best])
    # tuned ensemble: greedy subset of the eight members on val_selection
    g = greedy_subset([t[0][sel] for _, _, t in members], y_val[sel])
    pG_val = np.mean([softmax(members[i][2][0]) for i in g], 0)
    pG_ev = np.mean([softmax(ev(members[i][2])) for i in g], 0)
    pA_ev = np.mean([softmax(ev(t)) for _, _, t in members], 0)

    T_S, _ = fit_T(lambda T: softmax(zS_val[comb], T), y_val[comb])
    T_G, _ = fit_T(lambda T: power_T(pG_val[comb], T), y_val[comb])
    pS_ev, pS_ts, pG_ts = softmax(zS_ev), softmax(zS_ev, T_S), power_T(pG_ev, T_G)
    corr = lambda p: (p.argmax(-1) == y_ev).astype(float)
    return {
        "selected_single_config": best,
        "greedy_members": [f"{members[i][0]}:s{members[i][1]}" for i in g],
        "T_S": T_S, "T_E": T_G,
        "acc_E_tuned": float(corr(pG_ev).mean()), "acc_E_all8": float(corr(pA_ev).mean()),
        "acc_S_tuned": float(corr(pS_ev).mean()),
        "_vec": {"dacc": corr(pG_ev) - corr(pS_ev), "dacc_all8": corr(pA_ev) - corr(pS_ev),
                 "dnll_ts": nll_vec(pG_ts, y_ev) - nll_vec(pS_ts, y_ev)},
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
    res = {"_note": ("EXPLORATORY dev_test run" if args.dev else "CONFIRMATORY: PREREG_AB.md §9.6"),
           "git_sha": sha, "eval_split": split, "cells": {}}
    a3, bp = [], []
    for task in TASKS:
        for bb in BACKBONES:
            key = f"{task}/{bb}/B16"
            try:
                reps = [one_replicate(task, bb, r, split) for r in range(2)]
            except FileNotFoundError as e:
                res["cells"][key] = {"status": f"missing {e}"}; print(key, "missing"); continue
            seed = zlib.crc32(("a3d|" + key).encode()) % (2**31)
            d_acc, d_nll = joint_ci(reps, "dacc", seed), joint_ci(reps, "dnll_ts", seed + 1)
            sign = PREDICTED_SIGN[bb]
            cell = {"delta_acc": d_acc, "delta_acc_all8": joint_ci(reps, "dacc_all8", seed + 2),
                    "dnll_calibrated": d_nll,
                    "A3_adjudication": adjudicate(d_acc, sign),
                    "B_adjudication": adjudicate({"ci95": [-d_nll["ci95"][1], -d_nll["ci95"][0]]}, sign),
                    "replicates": [{k: v for k, v in r.items() if k != "_vec"} for r in reps]}
            res["cells"][key] = cell
            a3.append(cell["A3_adjudication"]); bp.append(cell["B_adjudication"])
            print(f"{key:16s} dAcc {100*d_acc['mean']:+.2f}pp [{100*d_acc['ci95'][0]:+.2f},"
                  f"{100*d_acc['ci95'][1]:+.2f}] {cell['A3_adjudication']:11s} | all8 "
                  f"{100*cell['delta_acc_all8']['mean']:+.2f}pp | dNLL_TS {d_nll['mean']:+.4f} "
                  f"[{d_nll['ci95'][0]:+.4f},{d_nll['ci95'][1]:+.4f}] {cell['B_adjudication']}")
    res["A3_P4"] = {"cells": a3, "verdict": verdict(a3) if len(a3) == 4 else "incomplete"}
    res["B_P4"] = {"cells": bp, "verdict": verdict(bp) if len(bp) == 4 else "incomplete"}
    print(f"\nA3-P4: {res['A3_P4']['verdict']}   B-P4: {res['B_P4']['verdict']}")
    out = OUT / ("a3d_dev_exploratory.json" if args.dev else "a3d_confirmatory.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
