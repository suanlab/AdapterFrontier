#!/usr/bin/env python3
"""Experiment A3f (mechanism/PREREG_AB.md §9.8): §9.7's averaging and
allocation estimands on Qwen2.5-1.5B, Yahoo Answers, B=16, four replicates.

Reuses a3e_analysis with a different backbone, task, replicate count, seed
base, run prefix and output stem. Its result keys A3_P5a/A3_P5b/B_P5a/B_P5b
are the §9.8 predictions A3-P6a/A3-P6b/B-P6a/B-P6b. Committed before any A3f run.

Usage: python3 mechanism/a3f_analysis.py [--dev]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3e_analysis as e  # noqa: E402

e.TASKS = ("yahoo",)
e.BACKBONES = ("q15",)
e.N_REP = 4
e.SIGN = {"averaging": {"q15": +1}, "allocation": {"q15": +1}}
e.bank_seed = lambda i, j: 700 + 20 * i + j + 1
e.cfg_seed = lambda i, k: 700 + 20 * i + 11 + k
e.run_dir = lambda task, bb, name, seed: e.RUNS / f"a3f_{task}_{bb}_{name}_s{seed}"


def verdict(cells):
    # one cell per prediction (§9.8)
    if not cells:
        return "incomplete"
    return {"SUPPORTED": "REPLICATES", "REVERSED": "FAILS"}.get(cells[0], "inconclusive")


e.verdict = verdict
e.RESULT_STEM = "a3f"

if __name__ == "__main__":
    sys.exit(e.main())
