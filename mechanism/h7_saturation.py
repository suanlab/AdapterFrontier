#!/usr/bin/env python3
"""H7: saturation follows 1 - 1/N, magnitude follows 1 - rho
(mechanism/PREREG.md §3, deviation 1 in §9).

Confirmatory: Spearman(rho, D(N_max)) <= -0.3 over the 48 headline pools,
with D from the stored full-pool soft_vote.

The first criterion (soft_vote D(N)/D(N_max) vs the 1 - 1/N law) is not
testable as registered. Two exploratory substitutes are reported and labelled:
majority_vote subsets on the headline pools, and soft_vote subsets on the six
cached-logit pools.
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import ROOT, headline_pairs, plurality, result, save, verdict

NS = (2, 4, 8)
DRAWS = 100


def err_corr(P, y):
    E = (P != y[None, :]).astype(float)
    E = E[E.std(1) > 0]
    if len(E) < 2:
        return float("nan")
    c = np.corrcoef(E)
    return float(c[np.triu_indices(len(E), 1)].mean())


def law(n, nmax):
    return (1 - 1 / n) / (1 - 1 / nmax)


def curve(score_subset, M, rng):
    """Mean D(N) over random subsets, normalised by D(M)."""
    full = score_subset(np.arange(M))
    out = {}
    for n in NS:
        if n >= M:
            continue
        out[n] = float(np.mean([score_subset(rng.choice(M, n, replace=False))
                                for _ in range(DRAWS)]))
    return full, out


def main() -> None:
    rng = np.random.default_rng(0)
    rhos, Ds, fam = [], [], []
    mv_dev = []
    for pr in headline_pairs():
        E = result(pr["ensemble"]); y = E["y"]; K = int(max(y.max(), E["P"].max())) + 1
        rho = err_corr(E["P"], y)
        D = float(E["methods"]["soft_vote"]["accuracy_test"] - E["acc"].mean())
        rhos.append(rho); Ds.append(D); fam.append(pr["family"])

        def mv_D(idx, P=E["P"]):
            sub = P[idx]
            return float((plurality(sub, K) == y).mean() - (sub == y).mean())
        M = E["P"].shape[0]
        full, c = curve(mv_D, M, rng)
        if full > 0.002:
            mv_dev += [abs(c[n] / full - law(n, M)) for n in c]

    ok = ~np.isnan(rhos)
    r = spearmanr(np.array(rhos)[ok], np.array(Ds)[ok])
    confirm_pass = r.correlation <= -0.3
    fam_rho = {f: float(np.mean([x for x, g in zip(rhos, fam) if g == f])) for f in ("encoder", "decoder")}

    # Exploratory: soft_vote subsets on the cached-logit pools
    sv_dev = []
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text())
        npys = sorted((ROOT / "ensemble_cache").glob(f"{d['pool_id']}__*.npy"))
        if len(npys) < 2:
            continue
        test = np.asarray(d["test_indices"]); y = np.asarray(d["labels_test"])
        L = np.stack([np.load(f) for f in npys])[:, test]
        prob = np.exp(L - L.max(-1, keepdims=True)); prob /= prob.sum(-1, keepdims=True)
        acc_i = (L.argmax(-1) == y).mean(1)

        def sv_D(idx):
            return float((prob[idx].mean(0).argmax(-1) == y).mean() - acc_i[idx].mean())
        full, c = curve(sv_D, L.shape[0], rng)
        if full > 0.002:
            sv_dev += [abs(c[n] / full - law(n, L.shape[0])) for n in c]

    out = {
        "confirmatory": {"spearman_rho_vs_D": float(r.correlation), "p": float(r.pvalue),
                         "n_pools": int(ok.sum()), "criterion": "<= -0.3",
                         "verdict": "PASS" if confirm_pass else ("FALSIFIED" if r.correlation >= 0 else "PARTIAL")},
        "first_criterion": "NOT TESTABLE as registered (§9 deviation 1)",
        "mean_rho_by_family": fam_rho,
        "exploratory_majority_vote_median_abs_dev": float(np.median(mv_dev)) if mv_dev else None,
        "exploratory_majority_vote_n_points": len(mv_dev),
        "exploratory_soft_vote_cached_median_abs_dev": float(np.median(sv_dev)) if sv_dev else None,
        "exploratory_soft_vote_cached_n_points": len(sv_dev),
    }
    save("h7_saturation", out)
    print(f"H7 confirmatory: Spearman(rho, D_softvote) = {r.correlation:+.3f} (p={r.pvalue:.3f}, n={ok.sum()}) "
          f"-> {out['confirmatory']['verdict']}")
    print(f"  mean error correlation: {fam_rho}")
    print(f"  first criterion: {out['first_criterion']}")
    print(f"  exploratory 1-1/N law, median |dev|: majority_vote headline "
          f"{out['exploratory_majority_vote_median_abs_dev']} (n={len(mv_dev)}); "
          f"soft_vote cached {out['exploratory_soft_vote_cached_median_abs_dev']} (n={len(sv_dev)})")


if __name__ == "__main__":
    main()
