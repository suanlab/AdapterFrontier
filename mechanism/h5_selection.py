#!/usr/bin/env python3
"""H5: the baseline's edge is selection pressure (mechanism/PREREG.md §3).

Cross-fitting on the test split, because the stored files carry test
predictions only: each repeat splits the test examples in half, selects the
best of M baseline candidates on half A and scores everything on half B.
Candidate subsets are nested (the first M of one random permutation) and the
same splits are reused for every M, so differences across M are not
Monte-Carlo noise from independent draws.

S_pred is the registered formula, sigma * sqrt(2 ln M) * lambda. Because
sqrt(2 ln M) is a poor approximation to the expected maximum of M normals at
small M (1.18 vs 0.56 at M=2), the exact expected maximum is also reported,
labelled secondary.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from common import headline_pairs, result, save, verdict

REPEATS = 200
MS = (1, 2, 4, 8, 16)


def expected_max_normal(m: int) -> float:
    """E[max of m iid N(0,1)] by numerical integration (secondary analysis)."""
    x = np.linspace(-8, 8, 20001)
    integrate = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 has trapz only
    return float(integrate(x * m * norm.pdf(x) * norm.cdf(x) ** (m - 1), x))


def main() -> None:
    rng = np.random.default_rng(0)
    pools, obs, pred, pred_exact, cells = [], [], [], [], []
    for pr in headline_pairs():
        E, B = result(pr["ensemble"]), result(pr["baseline"])
        y, ens = E["y"], np.asarray(E["methods"]["soft_vote"]["predictions_test"])
        Bc = (B["P"] == y[None, :])                 # candidates x examples, correct?
        e_c = (ens == y)
        K, N = Bc.shape
        Ms = [m for m in MS if m <= K]
        delta = {m: [] for m in Ms}; s_obs = {m: [] for m in Ms}
        for _ in range(REPEATS):
            perm = rng.permutation(N); A, Bh = perm[: N // 2], perm[N // 2:]
            order = rng.permutation(K)
            accA, accB = Bc[:, A].mean(1), Bc[:, Bh].mean(1)
            for m in Ms:
                sub = order[:m]
                pick = sub[np.argmax(accA[sub])]
                delta[m].append(e_c[Bh].mean() - accB[pick])
                s_obs[m].append(accB[pick] - accB.mean())
        d_mean = {m: float(np.mean(delta[m])) for m in Ms}
        monotone = all(d_mean[Ms[i + 1]] <= d_mean[Ms[i]] for i in range(len(Ms) - 1))
        sigma = float(B["acc"].std(ddof=1)); a = float(B["acc"].mean())
        lam = sigma ** 2 / (sigma ** 2 + a * (1 - a) / (N // 2))
        for m in Ms:
            obs.append(float(np.mean(s_obs[m])))
            pred.append(sigma * np.sqrt(2 * np.log(m)) * lam)
            pred_exact.append(sigma * expected_max_normal(m) * lam)
            cells.append((pr["pool"], m))
        pools.append({"pool": pr["pool"], "family": pr["family"], "K": K,
                      "sigma_candidates": sigma, "lambda": lam,
                      "delta_by_M": d_mean, "monotone": monotone})

    obs, pred, pred_exact = map(np.asarray, (obs, pred, pred_exact))
    def r2(p): return float(1 - ((obs - p) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())
    frac_mono = float(np.mean([p["monotone"] for p in pools]))
    r2_reg, r2_exact = r2(pred), r2(pred_exact)
    passed = frac_mono >= 0.8 and r2_reg >= 0.5
    out = {
        "repeats": REPEATS, "M": list(MS), "n_pools": len(pools),
        "fraction_pools_monotone": frac_mono,
        "r2_S_registered_formula": r2_reg,
        "r2_S_exact_expected_max_secondary": r2_exact,
        "verdict": verdict(passed, not passed),
        "mean_delta_by_M_by_family": {
            f: {str(m): float(np.mean([p["delta_by_M"][m] for p in pools
                                       if p["family"] == f and m in p["delta_by_M"]]))
                for m in MS}
            for f in ("encoder", "decoder")},
        "sigma_candidates_by_family": {
            f: float(np.median([p["sigma_candidates"] for p in pools if p["family"] == f]))
            for f in ("encoder", "decoder")},
        "pools": pools,
    }
    save("h5_selection", out)
    print(f"H5 over {len(pools)} pools, {REPEATS} cross-fit repeats")
    print(f"  Delta(M) non-increasing in {frac_mono:.0%} of pools (need >= 80%)")
    print(f"  S_pred registered R2 = {r2_reg:+.3f} (need >= 0.5); exact-max secondary R2 = {r2_exact:+.3f}")
    for f, v in out["mean_delta_by_M_by_family"].items():
        print(f"  {f:8s} mean Delta by M: " + "  ".join(f"M={m}:{100*x:+.2f}pp" for m, x in v.items()))
    print(f"  median candidate-accuracy sd: {out['sigma_candidates_by_family']}")
    print(f"  -> {out['verdict']}")


if __name__ == "__main__":
    main()
