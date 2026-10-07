#!/usr/bin/env python3
"""Post-hoc same-bank comparisons (panel 2, A-M1 / B-M1 / C-M1 / D-M1 / E-M2).
EXPLORATORY: the test logits were unsealed before this analysis was written.

Each A3d replicate trained eight 2-epoch runs (r in {8, 32} x lr in {3e-4,
1e-4} x 2 seeds) for 16 epoch-units. On *those same runs* two pipelines
differ only in how they combine:

  E_tuned  greedy forward selection over the eight, soft vote (as §9.6)
  S_bank   the val_selection-best single run of the eight

so E_tuned - S_bank isolates averaging at identical runs and identical cost.
Two further comparisons use an early-stopping tuned single, S_es: the
val_selection-best checkpoint among {epoch 2, epoch 4} of the four 4-epoch
configuration runs of §9.2 (c1-c4), also 16 epoch-units.

  E_tuned - S_es     tuned ensemble vs early-stopping tuned single
  E_plain - S_es     the §9.1 ensemble (eight r8 lr3e-4 members) vs S_es

CIs: mean over the two replicates, paired bootstrap over test examples
(B = 5,000), as in the registered analyses.

Usage: python3 mechanism/a3_samebank.py
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3b_tuned as tb  # noqa: E402
from a3_analysis import greedy_subset, joint_ci  # noqa: E402
from a3d_analysis import MEMBER_SEEDS, member_dir  # noqa: E402
from ab_common import logits, softmax, validation_slices  # noqa: E402
from b1_calibrate import fit_T, nll_vec, power_T  # noqa: E402

TASKS = ("snli", "yahoo")
BACKBONES = ("bert", "q05")


def lv(d, epoch):
    return logits(d, epoch, "validation"), logits(d, epoch, "test")


def one(task, bb, rep):
    bank = [lv(member_dir(task, bb, cfg, s), 2) for cfg, seeds in MEMBER_SEEDS[rep].items() for s in seeds]
    plain = [lv(tb.RUNS / f"a3_{task}_{bb}_E_s{s}", 2) for s in tb.E_SEEDS[rep]]
    es = [lv(tb.config_dir(task, bb, c, tb.SEEDS[rep][c]), e) for c in tb.CONFIGS for e in (2, 4)]
    (zv0, y_val), (_, y_te) = bank[0]
    sl = validation_slices(len(y_val)); sel, comb = sl["val_selection"], sl["val_combine"]
    vacc = lambda z: float((z[sel].argmax(-1) == y_val[sel]).mean())
    corr = lambda p: (p.argmax(-1) == y_te).astype(float)

    g = greedy_subset([v[0][sel] for v, _ in bank], y_val[sel])
    pG_val = np.mean([softmax(bank[i][0][0]) for i in g], 0)
    pG = np.mean([softmax(bank[i][1][0]) for i in g], 0)
    kB = int(np.argmax([vacc(v[0]) for v, _ in bank]))
    kE = int(np.argmax([vacc(v[0]) for v, _ in es]))
    pP = np.mean([softmax(t[0]) for _, t in plain], 0)
    zB_val, zB = bank[kB][0][0], bank[kB][1][0]
    zS_val, zS = es[kE][0][0], es[kE][1][0]

    T_G, _ = fit_T(lambda T: power_T(pG_val[comb], T), y_val[comb])
    T_B, _ = fit_T(lambda T: softmax(zB_val[comb], T), y_val[comb])
    T_S, _ = fit_T(lambda T: softmax(zS_val[comb], T), y_val[comb])
    nG, nB, nS = (nll_vec(power_T(pG, T_G), y_te), nll_vec(softmax(zB, T_B), y_te), nll_vec(softmax(zS, T_S), y_te))
    return {
        "vec": {"Etuned_minus_Sbank": corr(pG) - corr(softmax(zB)),
                "Etuned_minus_Ses": corr(pG) - corr(softmax(zS)),
                "Eplain_minus_Ses": corr(pP) - corr(softmax(zS)),
                "nll_Etuned_minus_Sbank": nG - nB, "nll_Etuned_minus_Ses": nG - nS},
        "acc": {"E_tuned": float(corr(pG).mean()), "S_bank": float(corr(softmax(zB)).mean()),
                "S_es": float(corr(softmax(zS)).mean()), "E_plain": float(corr(pP).mean())},
        "S_es_pick": ["c1", "c2", "c3", "c4"][kE // 2] + f"@epoch{2 if kE % 2 == 0 else 4}",
        "S_bank_pick": kB, "greedy_size": len(g),
    }


def main() -> int:
    res = {"_note": "EXPLORATORY post-hoc same-bank comparisons; see module docstring.", "cells": {}}
    keys = ("Etuned_minus_Sbank", "Etuned_minus_Ses", "Eplain_minus_Ses", "nll_Etuned_minus_Sbank", "nll_Etuned_minus_Ses")
    for task in TASKS:
        for bb in BACKBONES:
            reps = [one(task, bb, r) for r in range(2)]
            seed = zlib.crc32(f"samebank|{task}|{bb}".encode()) % (2**31)
            cell = {k: joint_ci([{"_vec": {k: r["vec"][k]}} for r in reps], k, seed + i) for i, k in enumerate(keys)}
            cell["acc"] = {k: round(100 * float(np.mean([r["acc"][k] for r in reps])), 2) for k in reps[0]["acc"]}
            cell["S_es_pick"] = [r["S_es_pick"] for r in reps]
            cell["greedy_size"] = [r["greedy_size"] for r in reps]
            res["cells"][f"{task}/{bb}"] = cell
            f = lambda k, s=100: f"{s*cell[k]['mean']:+.2f} [{s*cell[k]['ci95'][0]:+.2f},{s*cell[k]['ci95'][1]:+.2f}]"
            print(f"{task:5s} {bb:4s} Etuned-Sbank {f('Etuned_minus_Sbank')}pp | Etuned-Ses {f('Etuned_minus_Ses')}pp | "
                  f"Eplain-Ses {f('Eplain_minus_Ses')}pp | NLL Etuned-Sbank {f('nll_Etuned_minus_Sbank', 1)} | "
                  f"S_es {cell['S_es_pick']}")
    out = Path(__file__).resolve().parent / "results" / "a3_samebank_exploratory.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
