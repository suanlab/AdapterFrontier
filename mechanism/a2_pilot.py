#!/usr/bin/env python3
"""Experiment A2 pilot analysis (mechanism/PREREG_AB.md §3). Descriptive only.

Twelve SNLI trajectories: {BERT-base, Qwen-0.5B} x r in {8, 32} (alpha = 2r)
x seeds {101, 102, 103}, each a 4-epoch schedule with checkpoints at epochs 2
(mid-trajectory) and 4. A "cell" is one (backbone, r, epoch); it has three
seeds.

For every pairing of an ensemble cell (soft vote over its three seeds) with a
single cell (the val_selection-best of its three seeds) we report, on
dev_test, Delta = D - G - S, where
    D = acc(ensemble) - mean member acc         (diversity gain)
    G = mean single-candidate acc - mean member acc   (quality gap)
    S = acc(selected single) - mean candidate acc     (selection gain)
and each arm's cost in processed train tokens (every seed it trained counts).
Both arms train three trajectories, so at the same epoch they cost the same;
r changes training FLOPs by well under 1% and is free in this unit.

dev_test was seen for accuracy in H8, so all of this is exploratory, and the
pilot decides only whether the pipeline works (the §3 expansion rule).

Usage: python3 mechanism/a2_pilot.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import zlib
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_common import RUNS, logits, metrics, softmax, validation_slices  # noqa: E402

BACKBONES = {"bert": "bert-base-uncased", "q05": "Qwen2.5-0.5B"}
RANKS = (8, 32)
EPOCHS = (2, 4)
SEEDS = (101, 102, 103)
B = 5000
OUT = Path(__file__).resolve().parent / "results" / "a2_pilot.json"


def acc(z: np.ndarray, y: np.ndarray) -> float:
    return float((z.argmax(-1) == y).mean())


def pipeline_checks(tag: str) -> tuple[dict, list[str]]:
    """The §3 expansion rule: nothing lost, labels align, token counts consistent."""
    problems, runs, y0 = [], {}, None
    for r, s in product(RANKS, SEEDS):
        run = RUNS / f"a2_{tag}_r{r}_s{s}"
        if not (run / "metrics.json").exists():
            problems.append(f"{run.name}: missing"); continue
        m = metrics(run)
        cks = {c["epoch"]: c for c in m["checkpoints"]}
        per_epoch = cks[2]["train_tokens"] / 2
        if cks[4]["train_tokens"] != 4 * per_epoch:
            problems.append(f"{run.name}: token count {cks[4]['train_tokens']} != 4 x {per_epoch}")
        for e in EPOCHS:
            _, y = logits(run, e, "validation")
            if y0 is None:
                y0 = y
            elif not np.array_equal(y, y0):
                problems.append(f"{run.name} epoch{e}: labels do not align")
        runs[(r, s)] = m
    return runs, problems


def backbone(tag: str) -> dict:
    runs, problems = pipeline_checks(tag)
    if problems:
        return {"status": "incomplete", "problems": problems}
    any_run = RUNS / f"a2_{tag}_r{RANKS[0]}_s{SEEDS[0]}"
    _, y = logits(any_run, 2, "validation")
    sl = validation_slices(len(y))
    sel, dev = sl["val_selection"], sl["dev_test"]

    z = {(r, e, s): logits(RUNS / f"a2_{tag}_r{r}_s{s}", e, "validation")[0]
         for r, e, s in product(RANKS, EPOCHS, SEEDS)}
    tokens = {(r, e, s): {c["epoch"]: c for c in runs[(r, s)]["checkpoints"]}[e]["train_tokens"]
              for r, e, s in product(RANKS, EPOCHS, SEEDS)}
    secs = {(r, e, s): {c["epoch"]: c for c in runs[(r, s)]["checkpoints"]}[e]["train_seconds"]
            for r, e, s in product(RANKS, EPOCHS, SEEDS)}
    failed = {f"r{r}_s{s}_e{e}": {c["epoch"]: c for c in runs[(r, s)]["checkpoints"]}[e]["failed"]
              for r, e, s in product(RANKS, EPOCHS, SEEDS)}

    cells = {}
    for r, e in product(RANKS, EPOCHS):
        accs = [acc(z[(r, e, s)][dev], y[dev]) for s in SEEDS]
        cells[f"r{r}_e{e}"] = {"dev_test_acc_mean": float(np.mean(accs)),
                               "dev_test_acc_sd": float(np.std(accs, ddof=1)),
                               "dev_test_acc_by_seed": accs,
                               "tokens_per_seed": tokens[(r, e, SEEDS[0])],
                               "gpu_seconds_by_seed": [secs[(r, e, s)] for s in SEEDS]}

    pairings = []
    for (re_, ee), (rs, es) in product(product(RANKS, EPOCHS), repeat=2):
        members = [z[(re_, ee, s)] for s in SEEDS]
        p_ens = np.mean([softmax(m) for m in members], axis=0)
        cands = [z[(rs, es, s)] for s in SEEDS]
        k = int(np.argmax([acc(c[sel], y[sel]) for c in cands]))
        a_ens = acc(p_ens[dev], y[dev])
        m_mem = float(np.mean([acc(m[dev], y[dev]) for m in members]))
        m_cand = float(np.mean([acc(c[dev], y[dev]) for c in cands]))
        a_sing = acc(cands[k][dev], y[dev])
        D, G, S = a_ens - m_mem, m_cand - m_mem, a_sing - m_cand
        # paired bootstrap of Delta over dev_test examples
        ce = (p_ens[dev].argmax(-1) == y[dev]).astype(float)
        cs = (cands[k][dev].argmax(-1) == y[dev]).astype(float)
        rng = np.random.RandomState(zlib.crc32(f"{tag}|{re_}|{ee}|{rs}|{es}".encode()) % (2**31))
        n = len(dev)
        boots = [(ce[i] - cs[i]).mean() for i in (rng.randint(0, n, n) for _ in range(B))]
        cost_e = sum(tokens[(re_, ee, s)] for s in SEEDS)
        cost_s = sum(tokens[(rs, es, s)] for s in SEEDS)
        pairings.append({
            "ensemble": f"r{re_}_e{ee}", "single": f"r{rs}_e{es}",
            "selected_seed": SEEDS[k],
            "delta": a_ens - a_sing, "delta_ci95": [float(v) for v in np.quantile(boots, [0.025, 0.975])],
            "D": D, "G": G, "S": S, "identity_residual": (D - G - S) - (a_ens - a_sing),
            "cost_tokens_ensemble": cost_e, "cost_tokens_single": cost_s,
            "cost_ratio_single_over_ensemble": cost_s / cost_e,
            "equal_cost": cost_e == cost_s,
        })
    return {"status": "ok", "cells": cells, "pairings": pairings, "failed_flags": failed}


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"_note": "EXPLORATORY A2 pilot on SNLI dev_test; PREREG_AB.md §3.", "git_sha": sha,
           "backbones": {}}
    ok_all = True
    for tag, name in BACKBONES.items():
        r = backbone(tag)
        res["backbones"][name] = r
        if r["status"] != "ok":
            ok_all = False
            print(f"{name}: {r['status']} ({len(r['problems'])} problems, first: {r['problems'][0]})")
            continue
        print(f"\n{name}")
        for c, v in r["cells"].items():
            print(f"  {c:7s} acc {v['dev_test_acc_mean']:.4f} ± {v['dev_test_acc_sd']:.4f}  "
                  f"tokens/seed {v['tokens_per_seed']:,}")
        for p in r["pairings"]:
            if p["equal_cost"] and p["ensemble"].startswith("r8") and p["single"].startswith("r32"):
                print(f"  ens {p['ensemble']} vs single {p['single']}: Delta {100*p['delta']:+.2f}pp "
                      f"[{100*p['delta_ci95'][0]:+.2f},{100*p['delta_ci95'][1]:+.2f}]  "
                      f"D {100*p['D']:+.2f} G {100*p['G']:+.2f} S {100*p['S']:+.2f}")
    res["expansion_rule_passed"] = ok_all
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nexpansion rule (pipeline works): {ok_all}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
