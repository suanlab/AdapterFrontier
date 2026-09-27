#!/usr/bin/env python3
"""Does the headline accuracy null track the baseline's budget advantage?

Post-hoc and descriptive; not part of the pre-registration.

The headline slice compares 23 encoder pools against the `n_rank` single
adapter, which measured from the manifests received a median 1.45x the pool's
training GPU-hours (baseline_budget_audit.py). A reviewer can fairly object
that a null against a better-resourced competitor is biased towards the null,
so it cannot by itself support a claim made "at fixed budget".

No pool in the corpus has a baseline at exactly 1.0x, so that question cannot
be answered directly. What the corpus can answer is whether the null weakens
as the baseline's advantage shrinks. If the extra budget were what produced
the null, the pools closest to parity should show the ensemble doing better.
This script reports, per pool, the budget ratio against the post-FDR accuracy
verdicts on the same 92 cells the paper headlines, plus a rank correlation.

Output: analysis/budget_dose_response.json
"""
from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "budget_dose_response.json"

ENCODERS = ("bert", "roberta", "deberta")
CELL = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_n_rank$")
# The slice "closest to parity": every pool whose baseline received at most
# 1.3x the pool's GPU-hours. Chosen as the smallest round threshold that keeps
# more than a quarter of the 23 pools; the per-pool table is released so any
# other threshold can be read off it.
NEAR_PARITY = 1.3


def headline_cells() -> dict[str, list[dict]]:
    """The 92-cell slice: clean encoder classification, accuracy, vs n_rank."""
    by_pool: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(ANALYSIS.glob("*_cm_*_vs_n_rank.json")):
        m = CELL.match(f.stem)
        if not m:
            continue            # the _ECE files do not match: accuracy only
        pool = m["pool"]
        if ("hellaswag" in pool or "gsm8k" in pool
                or pool.startswith(("baseline", "smoke"))
                or not any(e in pool for e in ENCODERS)):
            continue
        d = json.loads(f.read_text())
        by_pool[pool].append({
            "method": m["method"],
            "delta": d["accuracy_diff"],
            "verdict": d["adjudication_post_fdr"],
        })
    return by_pool


def main() -> None:
    audit = {p["pool_id"]: p for p in
             json.loads((ANALYSIS / "baseline_budget_audit.json").read_text())["pairs"]}
    cells = headline_cells()
    missing = sorted(set(cells) - set(audit))
    if missing:
        raise SystemExit(f"no budget ratio for headline pools: {missing}")

    pools = []
    for pool, cs in cells.items():
        v = collections.Counter(c["verdict"] for c in cs)
        pools.append({
            "pool_id": pool,
            "budget_ratio": audit[pool]["ratio"],
            "mean_delta_pp": 100 * statistics.mean(c["delta"] for c in cs),
            "n_cells": len(cs),
            "supported": v.get("supported", 0),
            "reversed": v.get("reversed", 0),
            "unsupported": v.get("unsupported", 0),
        })
    pools.sort(key=lambda p: p["budget_ratio"])

    rho, p_rho = spearmanr([p["budget_ratio"] for p in pools],
                           [p["mean_delta_pp"] for p in pools])
    near = [p for p in pools if p["budget_ratio"] <= NEAR_PARITY]

    out = {
        "note": ("Post-hoc, descriptive. Headline 92-cell slice split by the "
                 "n_rank baseline's realised GPU-hour ratio to its pool."),
        "n_pools": len(pools),
        "n_cells": sum(p["n_cells"] for p in pools),
        "spearman_rho_ratio_vs_mean_delta": round(float(rho), 3),
        "spearman_p": round(float(p_rho), 3),
        "near_parity_threshold": NEAR_PARITY,
        "near_parity": {
            "n_pools": len(near),
            "n_cells": sum(p["n_cells"] for p in near),
            "supported": sum(p["supported"] for p in near),
            "reversed": sum(p["reversed"] for p in near),
            "min_ratio": min(p["budget_ratio"] for p in near),
            "max_ratio": max(p["budget_ratio"] for p in near),
        },
        "pools": pools,
    }
    OUT.write_text(json.dumps(out, indent=1))

    np_ = out["near_parity"]
    print(f"{out['n_pools']} pools / {out['n_cells']} cells")
    print(f"Spearman(ratio, mean delta) = {out['spearman_rho_ratio_vs_mean_delta']:+.3f}"
          f"  (p = {out['spearman_p']:.3f})")
    print(f"ratio <= {NEAR_PARITY}: {np_['n_pools']} pools "
          f"({np_['min_ratio']}x-{np_['max_ratio']}x), {np_['n_cells']} cells, "
          f"{np_['supported']} SUPPORTED, {np_['reversed']} REVERSED")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
