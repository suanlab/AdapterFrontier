#!/usr/bin/env python3
"""Experiment A3 and prediction B-P1, confirmatory (mechanism/PREREG_AB.md §9.1).

Per (task, backbone, budget B, replicate):
  E(B): soft vote over the first B/2 ensemble-recipe adapters (r=8, 2 epochs)
  S(B): the val_selection-best of the first B/4 single-recipe adapters
        (r=32, 4 epochs)
Both cost B epoch-units, which is the same token count within a task.

Primary (A3-P1): Delta = acc(E) - acc(S) on the sealed test. A cell (task,
backbone, B) is adjudicated on the mean Delta of its two replicates, with a
CI from resampling test examples jointly for both replicates.
B-P1: NLL(E+TS) - NLL(S+TS), temperatures fitted on val_combine (§5).
Secondary: greedy-subset E, uncalibrated NLL, risk at 90% coverage.

`--dev` runs the identical pipeline on the dev_test slice of validation
instead of the sealed test. That is for checking the code before
mechanism/TEST_FREEZE exists; its numbers are exploratory and are written to a
separate file.

Usage: python3 mechanism/a3_analysis.py [--dev]
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
from ab_common import RUNS, logits, metrics, softmax, validation_slices  # noqa: E402
from b1_calibrate import (aurc, confidences, fit_T, nll_vec, power_T,  # noqa: E402
                          threshold_for)

TASKS = ("snli", "yahoo")
BACKBONES = {"bert": "bert-base-uncased", "q05": "Qwen2.5-0.5B"}
BUDGETS = (4, 8, 16)
REPLICATES = ({"E": tuple(range(201, 209)), "S": tuple(range(251, 255))},
              {"E": tuple(range(211, 219)), "S": tuple(range(261, 265))})
PREDICTED_SIGN = {"bert": -1, "q05": +1}       # A3-P1 on Delta; B-P1 on -dNLL
PREDICTED_BUDGETS = (8, 16)
B_BOOT = 5000
OUT = Path(__file__).resolve().parent / "results"


def run_dir(task, bb, arm, seed) -> Path:
    return RUNS / f"a3_{task}_{bb}_{arm}_s{seed}"


def load_arm(task, bb, arm, seeds, eval_split):
    """Validation logits (for selection and temperatures) and evaluation logits."""
    val, ev, y_val, y_ev, failed = [], [], None, None, []
    for s in seeds:
        d = run_dir(task, bb, arm, s)
        if not (d / "metrics.json").exists():
            raise FileNotFoundError(d)
        z, y = logits(d, 2 if arm == "E" else 4, "validation")
        if y_val is not None and not np.array_equal(y, y_val):
            raise SystemExit(f"{d}: validation labels do not align")
        y_val = y
        val.append(z)
        if eval_split == "test":
            zt, yt = logits(d, 2 if arm == "E" else 4, "test")
            if y_ev is not None and not np.array_equal(yt, y_ev):
                raise SystemExit(f"{d}: test labels do not align")
            y_ev = yt; ev.append(zt)
        ck = metrics(d)["checkpoints"][-1]
        failed.append(bool(ck["failed"]))
    return val, ev, y_val, y_ev, failed


def greedy_subset(members_sel, y_sel) -> list[int]:
    """Forward selection on val_selection accuracy of the soft vote, no repeats."""
    probs = [softmax(z) for z in members_sel]
    chosen, best = [], -1.0
    while True:
        cand = [(float((np.mean([probs[j] for j in chosen + [i]], 0).argmax(-1) == y_sel).mean()), i)
                for i in range(len(probs)) if i not in chosen]
        if not cand:
            return chosen
        a, i = max(cand)
        if a <= best:
            return chosen
        chosen.append(i); best = a


def one_replicate(task, bb, B, rep, eval_split):
    seeds_E = REPLICATES[rep]["E"][: B // 2]
    seeds_S = REPLICATES[rep]["S"][: B // 4]
    vE, eE, y_val, y_ev, fE = load_arm(task, bb, "E", seeds_E, eval_split)
    vS, eS, y_val2, _, fS = load_arm(task, bb, "S", seeds_S, eval_split)
    if not np.array_equal(y_val, y_val2):
        raise SystemExit(f"{task}/{bb}: E and S validation labels differ")
    sl = validation_slices(len(y_val))
    sel, comb = sl["val_selection"], sl["val_combine"]
    if eval_split == "dev_test":
        dev = sl["dev_test"]
        eE = [z[dev] for z in vE]; eS = [z[dev] for z in vS]; y_ev = y_val[dev]

    k = int(np.argmax([float((z[sel].argmax(-1) == y_val[sel]).mean()) for z in vS]))
    zS_val, zS_ev = vS[k], eS[k]
    pE_val = np.mean([softmax(z) for z in vE], 0)
    pE_ev = np.mean([softmax(z) for z in eE], 0)
    g = greedy_subset([z[sel] for z in vE], y_val[sel])
    pG_ev = np.mean([softmax(eE[i]) for i in g], 0)

    T_S, _ = fit_T(lambda T: softmax(zS_val[comb], T), y_val[comb])
    T_E, _ = fit_T(lambda T: power_T(pE_val[comb], T), y_val[comb])
    pS_ev, pS_ts = softmax(zS_ev), softmax(zS_ev, T_S)
    pE_ts = power_T(pE_ev, T_E)

    # selective, secondary: each side picks its confidence on val_combine by AURC
    def selective(p_raw_val, p_cal_val, z_val, p_raw_ev, p_cal_ev, z_ev):
        cv = confidences(p_raw_val, p_cal_val, z_val)
        cor_val = (p_raw_val.argmax(-1) == y_val).astype(float)
        best = min(cv, key=lambda n: aurc(cv[n][comb], cor_val[comb]))
        thr = threshold_for(cv[best][comb], 0.90)
        keep = confidences(p_raw_ev, p_cal_ev, z_ev)[best] >= thr
        return best, keep
    zE_val = np.mean(vE, 0); zE_ev = np.mean(eE, 0)
    cS, keepS = selective(softmax(zS_val), softmax(zS_val, T_S), zS_val, pS_ev, pS_ts, zS_ev)
    cE, keepE = selective(pE_val, power_T(pE_val, T_E), zE_val, pE_ev, pE_ts, zE_ev)

    corr = lambda p: (p.argmax(-1) == y_ev).astype(float)
    return {
        "seeds_E": list(seeds_E), "seeds_S": list(seeds_S), "selected_S_seed": seeds_S[k],
        "greedy_members": [seeds_E[i] for i in g], "T_S": T_S, "T_E": T_E,
        "failed_E": fE, "failed_S": fS, "conf_S": cS, "conf_E": cE,
        "_vec": {"dacc": corr(pE_ev) - corr(pS_ev), "dacc_greedy": corr(pG_ev) - corr(pS_ev),
                 "dnll_ts": nll_vec(pE_ts, y_ev) - nll_vec(pS_ts, y_ev),
                 "dnll_raw": nll_vec(pE_ev, y_ev) - nll_vec(pS_ev, y_ev),
                 "errE": 1 - corr(pE_ev), "errS": 1 - corr(pS_ev), "keepE": keepE, "keepS": keepS},
        "acc_E": float(corr(pE_ev).mean()), "acc_S": float(corr(pS_ev).mean()),
        "acc_E_greedy": float(corr(pG_ev).mean()),
        "nll_E_ts": float(nll_vec(pE_ts, y_ev).mean()), "nll_S_ts": float(nll_vec(pS_ts, y_ev).mean()),
    }


def joint_ci(reps, key, seed):
    """Mean over replicates of a per-example difference; CI resamples examples jointly."""
    vecs = [r["_vec"][key] for r in reps]
    n = len(vecs[0])
    rng = np.random.RandomState(seed)
    obs = float(np.mean([v.mean() for v in vecs]))
    boots = np.empty(B_BOOT)
    for b in range(B_BOOT):
        i = rng.randint(0, n, n)
        boots[b] = np.mean([v[i].mean() for v in vecs])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": obs, "ci95": [float(lo), float(hi)]}


def risk90_ci(reps, seed):
    n = len(reps[0]["_vec"]["errE"])
    rng = np.random.RandomState(seed)
    def stat(i):
        return np.mean([r["_vec"]["errE"][i][r["_vec"]["keepE"][i]].mean()
                        - r["_vec"]["errS"][i][r["_vec"]["keepS"][i]].mean() for r in reps])
    obs = float(stat(np.arange(n)))
    boots = [stat(rng.randint(0, n, n)) for _ in range(B_BOOT)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": obs, "ci95": [float(lo), float(hi)]}


def adjudicate(ci, sign):
    """SUPPORTED if the CI excludes 0 in the predicted direction, REVERSED if
    it excludes 0 in the other direction, unsupported otherwise."""
    lo, hi = ci["ci95"]
    if lo > 0:
        return "SUPPORTED" if sign > 0 else "REVERSED"
    if hi < 0:
        return "SUPPORTED" if sign < 0 else "REVERSED"
    return "unsupported"


def verdict(cells):
    rev = sum(c == "REVERSED" for c in cells)
    sup = sum(c == "SUPPORTED" for c in cells)
    if rev:
        return "FAILS"
    return "REPLICATES" if sup >= 4 else "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()
    split = "dev_test" if args.dev else "test"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"_note": ("EXPLORATORY dev_test run of the confirmatory pipeline" if args.dev else
                     "CONFIRMATORY: PREREG_AB.md §9.1 on the sealed test"),
           "git_sha": sha, "eval_split": split, "cells": {}}
    a3_pred, b_pred = [], []
    for task in TASKS:
        for bb in BACKBONES:
            for B in BUDGETS:
                key = f"{task}/{bb}/B{B}"
                try:
                    reps = [one_replicate(task, bb, B, r, split) for r in range(len(REPLICATES))]
                except FileNotFoundError as e:
                    res["cells"][key] = {"status": f"missing {e}"}; print(key, "missing"); continue
                seed = zlib.crc32(key.encode()) % (2**31)
                d_acc = joint_ci(reps, "dacc", seed)
                d_nll = joint_ci(reps, "dnll_ts", seed + 1)
                sign = PREDICTED_SIGN[bb]
                cell = {
                    "delta_acc": d_acc, "delta_acc_greedy": joint_ci(reps, "dacc_greedy", seed + 2),
                    "dnll_calibrated": d_nll, "dnll_uncalibrated": joint_ci(reps, "dnll_raw", seed + 3),
                    "risk90_EminusS": risk90_ci(reps, seed + 4),
                    "replicates": [{k: v for k, v in r.items() if k != "_vec"} for r in reps],
                    "predicted": B in PREDICTED_BUDGETS,
                    "A3_adjudication": adjudicate(d_acc, sign),
                    # B-P1 predicts dNLL < 0 for Qwen (ensemble better) and >= 0 for BERT
                    "B_adjudication": adjudicate({"ci95": [-d_nll["ci95"][1], -d_nll["ci95"][0]]}, sign),
                }
                res["cells"][key] = cell
                if cell["predicted"]:
                    a3_pred.append(cell["A3_adjudication"]); b_pred.append(cell["B_adjudication"])
                print(f"{key:18s} dAcc {100*d_acc['mean']:+.2f}pp [{100*d_acc['ci95'][0]:+.2f},{100*d_acc['ci95'][1]:+.2f}] "
                      f"{cell['A3_adjudication']:11s} | dNLL_TS {d_nll['mean']:+.4f} "
                      f"[{d_nll['ci95'][0]:+.4f},{d_nll['ci95'][1]:+.4f}] {cell['B_adjudication']}")
    res["A3_P1"] = {"cells": a3_pred, "verdict": verdict(a3_pred) if len(a3_pred) == 8 else "incomplete"}
    res["B_P1"] = {"cells": b_pred, "verdict": verdict(b_pred) if len(b_pred) == 8 else "incomplete"}
    print(f"\nA3-P1: {res['A3_P1']['verdict']}   B-P1: {res['B_P1']['verdict']}")
    out = OUT / ("a3_dev_exploratory.json" if args.dev else "a3_confirmatory.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
