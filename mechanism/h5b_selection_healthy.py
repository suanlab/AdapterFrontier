#!/usr/bin/env python3
"""H5b (exploratory, not registered): selection pressure among candidates
that trained.

H5's M=1 arm draws a random baseline candidate, and some baselines are mostly
failed runs (training_health.py). A random draw then often lands on a failure,
so part of the large encoder Delta at M=1 could be "the ensemble beats a run
that never trained" rather than "the ensemble beats a typical trained single".
This repeats H5 with failed candidates removed, judged test-free from each
candidate's final training loss (>= 0.85 ln C), exactly as in
analysis/training_health.py.

Candidate order in ensemble_results matches the pool manifest; for every
baseline that has a failed candidate, test accuracy and the manifest's
validation accuracy correlate at r > 0.95, which confirms the alignment there.
"""
from __future__ import annotations

import json
import math

import numpy as np

from common import ROOT, headline_pairs, result, save

REPEATS = 200
MS = (1, 2, 4, 8, 16)


def trained_mask(baseline_path: str, k: int) -> np.ndarray:
    name = baseline_path.split("/")[-1][:-5]
    m = json.loads((ROOT / "pools" / f"{name}.json").read_text())
    r = [a["metrics"]["train_loss"] / math.log(a["metrics"]["num_labels"])
         for a in m["adapters"]]
    assert len(r) == k, name
    return np.array([x < 0.85 for x in r])


def main() -> None:
    rng = np.random.default_rng(0)
    out_pools = []
    for pr in headline_pairs():
        E, B = result(pr["ensemble"]), result(pr["baseline"])
        y = E["y"]; e_c = np.asarray(E["methods"]["soft_vote"]["predictions_test"]) == y
        keep = trained_mask(pr["baseline"], B["P"].shape[0])
        Bc = (B["P"][keep] == y[None, :]); K, N = Bc.shape
        Ms = [m for m in MS if m <= K]
        delta = {m: [] for m in Ms}
        for _ in range(REPEATS):
            perm = rng.permutation(N); A, Bh = perm[: N // 2], perm[N // 2:]
            order = rng.permutation(K); accA, accB = Bc[:, A].mean(1), Bc[:, Bh].mean(1)
            for m in Ms:
                sub = order[:m]; delta[m].append(e_c[Bh].mean() - accB[sub[np.argmax(accA[sub])]])
        out_pools.append({"pool": pr["pool"], "family": pr["family"],
                          "candidates_trained": int(K), "candidates_total": int(len(keep)),
                          "delta_by_M": {m: float(np.mean(delta[m])) for m in Ms}})
    fam = {f: {str(m): float(np.mean([p["delta_by_M"][m] for p in out_pools
                                     if p["family"] == f and m in p["delta_by_M"]]))
               for m in MS} for f in ("encoder", "decoder")}
    save("h5b_selection_healthy", {"exploratory": True, "repeats": REPEATS,
                                   "mean_delta_by_M_by_family": fam, "pools": out_pools})
    print("H5b (exploratory): baseline candidates that trained only")
    for f, v in fam.items():
        print(f"  {f:8s} " + "  ".join(f"M={m}:{100*x:+.2f}pp" for m, x in v.items()))


if __name__ == "__main__":
    main()
