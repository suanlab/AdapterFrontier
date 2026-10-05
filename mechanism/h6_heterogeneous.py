#!/usr/bin/env python3
"""H6: heterogeneous populations need quality weighting (mechanism/PREREG.md §3).

MNLI, QNLI, AG News. One member per backbone (the member with median
accuracy on half A), drawn from each backbone's seed-only Pool-A. Weighted
vote with weights estimated on half A, scored on half B, against the best
single member of any backbone selected on half A. 200 cross-fit repeats.

Weights (§9, deviation 2): the criterion uses the K-class Nitzan-Paroush
weight ln((K-1) a / (1-a)); the registered binary weight ln(a / (1-a)) and
the unweighted vote (seen in §0, not confirmatory) are reported too.
Only files sharing the task's majority test-index set are used.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json

import numpy as np

from common import ROOT, result, save, verdict

TASKS = ("mnli", "qnli", "agnews")
REPEATS = 200


def weighted_vote(P, w, K):
    score = np.zeros((K, P.shape[1]))
    for p, wi in zip(P, w):
        score[p, np.arange(P.shape[1])] += wi
    return score.argmax(0)


def main() -> None:
    rng = np.random.default_rng(0)
    out = {"repeats": REPEATS, "tasks": {}}
    wins = 0
    for t in TASKS:
        files = sorted(glob.glob(str(ROOT / f"ensemble_results/pool_a_{t}_*.json")))
        recs = [result(str(f).replace(str(ROOT) + "/", "")) for f in files]
        h = [hashlib.md5(r["test_indices"].tobytes()).hexdigest() for r in recs]
        common_h = collections.Counter(h).most_common(1)[0][0]
        recs = [r for r, x in zip(recs, h) if x == common_h]
        y = recs[0]["y"]; K = int(y.max()) + 1; N = len(y)
        res = collections.defaultdict(list)
        for _ in range(REPEATS):
            perm = rng.permutation(N); A, Bh = perm[: N // 2], perm[N // 2:]
            reps, accA = [], []
            best_a, best_pred = -1, None
            for r in recs:
                cA = (r["P"][:, A] == y[A]).mean(1)
                i = int(np.argsort(cA)[len(cA) // 2])
                reps.append(r["P"][i]); accA.append(cA[i])
                j = int(np.argmax(cA))
                if cA[j] > best_a:
                    best_a, best_pred = cA[j], r["P"][j]
            P = np.stack(reps); a = np.clip(np.array(accA), 1e-6, 1 - 1e-6)
            w_k = np.log((K - 1) * a / (1 - a)); w_b = np.log(a / (1 - a))
            accB = lambda pred: float((pred[Bh] == y[Bh]).mean())
            res["weighted_kclass"].append(accB(weighted_vote(P, w_k, K)))
            res["weighted_binary_registered"].append(accB(weighted_vote(P, w_b, K)))
            res["unweighted"].append(accB(weighted_vote(P, np.ones(len(P)), K)))
            res["best_single"].append(accB(best_pred))
        m = {k: float(np.mean(v)) for k, v in res.items()}
        m["n_backbones"] = len(recs)
        m["frac_repeats_weighted_ge_best"] = float(np.mean(
            np.array(res["weighted_kclass"]) >= np.array(res["best_single"])))
        out["tasks"][t] = m
        wins += m["weighted_kclass"] >= m["best_single"]
    passed = wins >= 2
    out["tasks_weighted_ge_best"] = f"{wins}/3"
    out["verdict"] = verdict(passed, wins == 0)
    save("h6_heterogeneous", out)
    print(f"{'task':8s} n   weighted(K) weighted(bin) unweighted best_single  P(w>=best)")
    for t, m in out["tasks"].items():
        print(f"{t:8s} {m['n_backbones']:2d}  {m['weighted_kclass']:.4f}      {m['weighted_binary_registered']:.4f}       "
              f"{m['unweighted']:.4f}     {m['best_single']:.4f}    {m['frac_repeats_weighted_ge_best']:.2f}")
    print(f"weighted heterogeneous >= best single in {out['tasks_weighted_ge_best']} tasks -> {out['verdict']}")


if __name__ == "__main__":
    main()
