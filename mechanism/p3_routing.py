#!/usr/bin/env python3
"""P3b (exploratory, not registered): how much of the oracle headroom is
reachable without training a router?

P3 found that "some member is right" exceeds the selected single by 9-10pp in
both families. Two training-free rules, from stored predictions and max
probabilities only, cross-fitted on halves of the test split (200 repeats):

  R1  max-confidence member: per example, follow the member most confident in
      its own prediction.
  R2  confidence gate: use the n_rank single adapter when its confidence is at
      least tau, and the soft_vote ensemble otherwise; tau chosen on half A.
      Inference cost = 1 + N x (fraction routed to the ensemble).
"""
from __future__ import annotations

import numpy as np

from common import headline_pairs, result, save

REPEATS = 200
TAUS = np.linspace(0.34, 0.999, 60)


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for pr in headline_pairs():
        E, B = result(pr["ensemble"]), result(pr["baseline"]); y = E["y"]; N = len(y)
        sv = np.asarray(E["methods"]["soft_vote"]["predictions_test"])
        bs = B["methods"]["best_single"]
        sp, sc = np.asarray(bs["predictions_test"]), np.asarray(bs["confidence_test"])
        r1 = E["P"][E["C"].argmax(0), np.arange(N)]
        oracle = (E["P"] == y[None, :]).any(0)
        acc = {k: [] for k in ("single", "soft_vote", "R1", "R2", "oracle")}; frac = []
        for _ in range(REPEATS):
            perm = rng.permutation(N); A, Bh = perm[: N // 2], perm[N // 2:]
            gate = lambda idx, t: np.where(sc[idx] >= t, sp[idx], sv[idx])
            tau = TAUS[int(np.argmax([(gate(A, t) == y[A]).mean() for t in TAUS]))]
            acc["R2"].append((gate(Bh, tau) == y[Bh]).mean()); frac.append((sc[Bh] < tau).mean())
            for k, pred in (("single", sp), ("soft_vote", sv), ("R1", r1)):
                acc[k].append((pred[Bh] == y[Bh]).mean())
            acc["oracle"].append(oracle[Bh].mean())
        M = E["P"].shape[0]
        rows.append({"pool": pr["pool"], "family": pr["family"], "M": M,
                     **{k: float(np.mean(v)) for k, v in acc.items()},
                     "R2_frac_to_ensemble": float(np.mean(frac)),
                     "R2_cost": float(1 + M * np.mean(frac))})
    out = {"exploratory": True, "repeats": REPEATS, "by_family": {}, "pools": rows}
    print(f"{'family':8s} {'single':>7s} {'soft':>7s} {'R1':>7s} {'R2':>7s} {'oracle':>7s}  R2 to-ens  R2 cost   share of headroom (R2)")
    for f in ("encoder", "decoder"):
        rs = [r for r in rows if r["family"] == f]
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("single", "soft_vote", "R1", "R2", "oracle",
                                                             "R2_frac_to_ensemble", "R2_cost")}
        m["R2_share_of_oracle_headroom"] = (m["R2"] - m["single"]) / (m["oracle"] - m["single"])
        m["pools_R2_beats_both"] = f"{sum(r['R2'] > max(r['single'], r['soft_vote']) for r in rs)}/{len(rs)}"
        out["by_family"][f] = m
        print(f"{f:8s} {m['single']:.4f}  {m['soft_vote']:.4f}  {m['R1']:.4f}  {m['R2']:.4f}  {m['oracle']:.4f}  "
              f"{m['R2_frac_to_ensemble']:.0%}       {m['R2_cost']:.1f}x     {m['R2_share_of_oracle_headroom']:.1%}"
              f"   (R2 beats single and ensemble in {m['pools_R2_beats_both']} pools)")
    save("p3_routing", out)


if __name__ == "__main__":
    main()
