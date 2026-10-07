#!/usr/bin/env python3
"""Post-hoc decomposition of the equal-budget gap (panel 2, A-M1 / B-M1 /
D-M1 / E-M2). Not pre-registered: it reads the sealed test logits after the
freeze, so every number here is EXPLORATORY.

E(B) and S(B) differ in three ways at once: averaging, member recipe (rank 8,
2 epochs vs rank 32, 4 epochs) and selection (best of B/4). Using only the
released logits, split

    Delta = acc(E) - acc(S)
          = [acc(E) - acc(M)]  +  [acc(M) - acc(S)]
            averaging term        recipe-and-selection term

where M is the val_selection-best single member of E (rank 8, 2 epochs). The
averaging term is the gain from averaging members of one recipe; the recipe
term compares the best member that recipe produced with the single pipeline.
A third arm, M_all = the val_selection-best checkpoint among all of E's members
and S's candidates, shows what a single gets when it may choose its recipe
from runs already trained.

Also reported, from the A2 pilot (SNLI dev_test only, exploratory): the
ensemble of three rank-8 members against the best of three rank-32 members at
the *same* epoch (2 or 4), which holds training length fixed.

Usage: python3 mechanism/a3_decompose.py
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_analysis as base  # noqa: E402
from ab_common import RUNS, logits, softmax, validation_slices  # noqa: E402

BACKBONES = ("bert", "q05", "bertl", "smol")
B = 16


def one(task, bb, rep):
    seeds_E = base.REPLICATES[rep]["E"][: B // 2]
    seeds_S = base.REPLICATES[rep]["S"][: B // 4]
    E = [(logits(base.run_dir(task, bb, "E", s), 2, "validation"), logits(base.run_dir(task, bb, "E", s), 2, "test"))
         for s in seeds_E]
    S = [(logits(base.run_dir(task, bb, "S", s), 4, "validation"), logits(base.run_dir(task, bb, "S", s), 4, "test"))
         for s in seeds_S]
    y_val, y_te = E[0][0][1], E[0][1][1]
    sel = validation_slices(len(y_val))["val_selection"]
    vacc = lambda z: float((z[sel].argmax(-1) == y_val[sel]).mean())
    corr = lambda z: (z.argmax(-1) == y_te).astype(float)
    kE = int(np.argmax([vacc(v[0]) for v, _ in E]))
    kS = int(np.argmax([vacc(v[0]) for v, _ in S]))
    all_runs = [v for v in E] + [v for v in S]
    kA = int(np.argmax([vacc(v[0]) for v, _ in all_runs]))
    cE = corr(np.mean([softmax(t[0]) for _, t in E], 0))
    cM = corr(E[kE][1][0]); cS = corr(S[kS][1][0]); cA = corr(all_runs[kA][1][0])
    return {"vec": {"delta": cE - cS, "averaging": cE - cM, "recipe": cM - cS, "e_vs_choose_any": cE - cA},
            "acc": {"E": cE.mean(), "M_best_member": cM.mean(), "S": cS.mean(), "M_all": cA.mean()},
            "M_all_is": "E member" if kA < len(E) else "S candidate"}


def ci(vecs, seed):
    rng = np.random.RandomState(seed); n = len(vecs[0])
    obs = float(np.mean([v.mean() for v in vecs]))
    boots = [np.mean([v[i].mean() for v in vecs]) for i in (rng.randint(0, n, n) for _ in range(5000))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return [round(100 * obs, 2), round(100 * lo, 2), round(100 * hi, 2)]


def a2_equal_epoch():
    out = {}
    for tag in ("bert", "q05"):
        z = {}
        for r in (8, 32):
            for e in (2, 4):
                for s in (101, 102, 103):
                    z[(r, e, s)] = logits(RUNS / f"a2_{tag}_r{r}_s{s}", e, "validation")
        y = z[(8, 2, 101)][1]
        sl = validation_slices(len(y)); sel, dev = sl["val_selection"], sl["dev_test"]
        for e in (2, 4):
            pE = np.mean([softmax(z[(8, e, s)][0][dev]) for s in (101, 102, 103)], 0)
            k = max((101, 102, 103), key=lambda s: (z[(32, e, s)][0][sel].argmax(-1) == y[sel]).mean())
            out[f"{tag}_epoch{e}"] = round(100 * float((pE.argmax(-1) == y[dev]).mean()
                                                   - (z[(32, e, k)][0][dev].argmax(-1) == y[dev]).mean()), 2)
    return out


def main() -> int:
    res = {"_note": __doc__.split("\n\n")[0] + " " + __doc__.split("\n\n")[1], "B": B, "cells": {}}
    for task in base.TASKS:
        for bb in BACKBONES:
            reps = [one(task, bb, r) for r in range(2)]
            seed = zlib.crc32(f"dec|{task}|{bb}".encode()) % (2**31)
            cell = {k: ci([r["vec"][k] for r in reps], seed + i) for i, k in enumerate(("delta", "averaging", "recipe", "e_vs_choose_any"))}
            cell["acc"] = {k: round(100 * float(np.mean([r["acc"][k] for r in reps])), 2) for k in reps[0]["acc"]}
            cell["M_all_is"] = [r["M_all_is"] for r in reps]
            res["cells"][f"{task}/{bb}"] = cell
            print(f"{task:5s} {bb:5s} Delta {cell['delta']}  averaging {cell['averaging']}  "
                  f"recipe {cell['recipe']}  E vs choose-any {cell['e_vs_choose_any']}  M_all from {cell['M_all_is']}")
    res["a2_equal_epoch_dev_test_pp"] = a2_equal_epoch()
    print("A2 pilot, equal epochs (dev_test):", res["a2_equal_epoch_dev_test_pp"])
    out = Path(__file__).resolve().parent / "results" / "a3_decompose_exploratory.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
