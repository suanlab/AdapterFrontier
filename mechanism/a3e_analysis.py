#!/usr/bin/env python3
"""Experiment A3e (mechanism/PREREG_AB.md §9.7): averaging and allocation,
separated, on six new replicates.

Per backbone x task x replicate: a bank of eight 2-epoch runs (b1..b8:
r in {8,32} x lr in {3e-4,1e-4} x 2 seeds) and four 4-epoch configuration
runs (c1..c4 as §9.2, with epoch-2 checkpoints). Arms:
  E_tuned  greedy forward selection over the bank, soft vote
  S_bank   best single run of the same bank (val_selection)
  S_es     best of {c1..c4} x {epoch 2, epoch 4} (val_selection)
Estimands: averaging = E_tuned - S_bank; allocation = E_tuned - S_es; and the
calibrated NLL difference of each (temperatures on val_combine).

Primary: paired t-interval over the six replicate-level differences (95%,
5 df). Secondary: example-level paired bootstrap of the replicate mean.

Committed before any A3e run (§9.7). `--dev` runs on validation dev_test for
code checks only.

Usage: python3 mechanism/a3e_analysis.py [--dev]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a3_analysis import adjudicate, greedy_subset, joint_ci  # noqa: E402
from ab_common import RUNS, logits, softmax, validation_slices  # noqa: E402
from b1_calibrate import fit_T, nll_vec, power_T  # noqa: E402

TASKS = ("snli", "yahoo")
BACKBONES = ("bert", "q05")
N_REP = 6
BANK = [(f"b{j+1}", r, lr) for j, (r, lr) in enumerate((r, lr) for r in (8, 32) for lr in ("3e-4", "1e-4") for _ in range(2))]
CONFIGS = [("c1", 8, "1e-4"), ("c2", 8, "3e-4"), ("c3", 32, "1e-4"), ("c4", 32, "3e-4")]
SIGN = {"averaging": {"bert": +1, "q05": +1}, "allocation": {"bert": -1, "q05": +1}}
RESULT_STEM = "a3e"   # output file stem; a3f_analysis.py overrides it


def bank_seed(i, j):
    return 600 + 20 * i + j + 1


def cfg_seed(i, k):
    return 600 + 20 * i + 11 + k


def run_dir(task, bb, name, seed) -> Path:
    return RUNS / f"a3e_{task}_{bb}_{name}_s{seed}"


def load(d, epoch, split):
    if not (d / "metrics.json").exists():
        raise FileNotFoundError(d)
    zv, yv = logits(d, epoch, "validation")
    zt, yt = (logits(d, epoch, "test") if split == "test" else (None, None))
    return zv, yv, zt, yt


def one(task, bb, i, split):
    bank = [load(run_dir(task, bb, name, bank_seed(i, j)), 2, split) for j, (name, _, _) in enumerate(BANK)]
    es = [load(run_dir(task, bb, c, cfg_seed(i, k)), e, split) for k, (c, _, _) in enumerate(CONFIGS) for e in (2, 4)]
    y_val = bank[0][1]
    for t in bank + es:
        if not np.array_equal(t[1], y_val):
            raise SystemExit(f"{task}/{bb}/rep{i}: validation labels do not align")
    sl = validation_slices(len(y_val)); sel, comb, dev = sl["val_selection"], sl["val_combine"], sl["dev_test"]
    if split == "test":
        y_ev = bank[0][3]; ev = lambda t: t[2]
    else:
        y_ev = y_val[dev]; ev = lambda t: t[0][dev]
    vacc = lambda z: float((z[sel].argmax(-1) == y_val[sel]).mean())
    corr = lambda p: (p.argmax(-1) == y_ev).astype(float)

    g = greedy_subset([t[0][sel] for t in bank], y_val[sel])
    pG_val = np.mean([softmax(bank[k][0]) for k in g], 0)
    pG = np.mean([softmax(ev(bank[k])) for k in g], 0)
    kB = int(np.argmax([vacc(t[0]) for t in bank]))
    kS = int(np.argmax([vacc(t[0]) for t in es]))
    T_G, _ = fit_T(lambda T: power_T(pG_val[comb], T), y_val[comb])
    T_B, _ = fit_T(lambda T: softmax(bank[kB][0][comb], T), y_val[comb])
    T_S, _ = fit_T(lambda T: softmax(es[kS][0][comb], T), y_val[comb])
    nG = nll_vec(power_T(pG, T_G), y_ev)
    nB = nll_vec(softmax(ev(bank[kB]), T_B), y_ev)
    nS = nll_vec(softmax(ev(es[kS]), T_S), y_ev)
    cG, cB, cS = corr(pG), corr(softmax(ev(bank[kB]))), corr(softmax(ev(es[kS])))
    return {"vec": {"averaging": cG - cB, "allocation": cG - cS, "nll_averaging": nG - nB, "nll_allocation": nG - nS},
            "acc": {"E_tuned": float(cG.mean()), "S_bank": float(cB.mean()), "S_es": float(cS.mean())},
            "S_bank_pick": BANK[kB][0], "S_es_pick": f"{CONFIGS[kS // 2][0]}@epoch{2 if kS % 2 == 0 else 4}",
            "greedy": [BANK[k][0] for k in g]}


def t_interval(diffs):
    d = np.asarray(diffs, dtype=float); n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.975, n - 1) * se)
    return {"mean": m, "ci95": [m - h, m + h], "per_replicate": d.tolist()}


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
    res = {"_note": ("EXPLORATORY dev_test run" if args.dev else f"CONFIRMATORY: PREREG_AB.md ({RESULT_STEM})"),
           "git_sha": sha, "eval_split": split, "cells": {}}
    verdicts = {"A3_P5a": [], "A3_P5b": [], "B_P5a": [], "B_P5b": []}
    for task in TASKS:
        for bb in BACKBONES:
            key = f"{task}/{bb}/B16"
            try:
                reps = [one(task, bb, i, split) for i in range(N_REP)]
            except FileNotFoundError as e:
                res["cells"][key] = {"status": f"missing {e}"}; print(key, "missing"); continue
            cell = {"replicates": [{k: v for k, v in r.items() if k != "vec"} for r in reps]}
            seed = zlib.crc32(("a3e|" + key).encode()) % (2**31)
            for j, est in enumerate(("averaging", "allocation")):
                acc_t = t_interval([r["vec"][est].mean() for r in reps])
                nll_t = t_interval([r["vec"][f"nll_{est}"].mean() for r in reps])
                sign = SIGN[est][bb]
                cell[est] = {"accuracy_t": acc_t, "nll_t": nll_t,
                             "accuracy_bootstrap": joint_ci([{"_vec": {"d": r["vec"][est]}} for r in reps], "d", seed + 2 * j),
                             "A_adjudication": adjudicate(acc_t, sign),
                             # NLL: ensemble lower is the predicted sign whenever accuracy predicts the ensemble wins
                             "B_adjudication": adjudicate({"ci95": [-nll_t["ci95"][1], -nll_t["ci95"][0]]}, sign)}
            verdicts["A3_P5a"].append(cell["averaging"]["A_adjudication"]); verdicts["B_P5a"].append(cell["averaging"]["B_adjudication"])
            verdicts["A3_P5b"].append(cell["allocation"]["A_adjudication"]); verdicts["B_P5b"].append(cell["allocation"]["B_adjudication"])
            res["cells"][key] = cell
            for est in ("averaging", "allocation"):
                a = cell[est]["accuracy_t"]; n = cell[est]["nll_t"]
                print(f"{key:15s} {est:10s} acc {100*a['mean']:+.2f}pp [{100*a['ci95'][0]:+.2f},{100*a['ci95'][1]:+.2f}] "
                      f"{cell[est]['A_adjudication']:11s} | NLL {n['mean']:+.4f} [{n['ci95'][0]:+.4f},{n['ci95'][1]:+.4f}] "
                      f"{cell[est]['B_adjudication']}")
    for k, v in verdicts.items():
        res[k] = {"cells": v, "verdict": verdict(v) if len(v) == 4 else "incomplete"}
    print("\n" + "   ".join(f"{k}: {res[k]['verdict']}" for k in verdicts))
    out = Path(__file__).resolve().parent / "results" / (f"{RESULT_STEM}_dev_exploratory.json" if args.dev else f"{RESULT_STEM}_confirmatory.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
