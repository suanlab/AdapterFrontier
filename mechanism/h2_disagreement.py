#!/usr/bin/env python3
"""H2: one parameter predicts the diversity gain (mechanism/PREREG.md §3).

D = a_E - abar is modelled as kappa * delta, delta the label-free fraction of
members disagreeing with the plurality vote. kappa is fitted leave-one-pool-
out, separately per combination rule. For majority_vote, D <= delta holds
exactly (the ensemble can only gain on examples where members disagree), so
kappa <= 1 there by construction; for soft_vote it is an empirical question.
"""
from __future__ import annotations

import numpy as np

from common import headline_pairs, plurality, result, save, verdict

RULES = ("soft_vote", "majority_vote")


def main() -> None:
    rows = []
    for pr in headline_pairs():
        E, B = result(pr["ensemble"]), result(pr["baseline"])
        K = int(max(E["y"].max(), E["P"].max())) + 1
        abar = float(E["acc"].mean())
        delta = float((E["P"] != plurality(E["P"], K)[None, :]).mean())
        bbar = float(B["acc"].mean())
        bs = B["methods"]["best_single"]
        b_star = float(bs["accuracy_test"])
        assert abs(b_star - pr["b_star_cell"]) < 1e-9, pr["pool"]
        for rule in RULES:
            a_E = float(E["methods"][rule]["accuracy_test"])
            rows.append({**{k: pr[k] for k in ("pool", "family", "task")},
                         "rule": rule, "abar": abar, "delta": delta,
                         "D": a_E - abar, "G": bbar - abar, "S": b_star - bbar,
                         "Delta": a_E - b_star})

    out = {"n_pools": len({r["pool"] for r in rows}), "by_rule": {}}
    for rule in RULES:
        rs = [r for r in rows if r["rule"] == rule]
        D = np.array([r["D"] for r in rs]); d = np.array([r["delta"] for r in rs])
        Dhat, kappas = np.empty_like(D), []
        for i in range(len(rs)):
            m = np.arange(len(rs)) != i
            k = float((D[m] * d[m]).sum() / (d[m] ** 2).sum())
            kappas.append(k); Dhat[i] = k * d[i]
        r2 = 1 - ((D - Dhat) ** 2).sum() / ((D - D.mean()) ** 2).sum()
        Delta = np.array([r["Delta"] for r in rs])
        Delta_hat = Dhat - np.array([r["G"] + r["S"] for r in rs])
        nz = Delta != 0
        sign_acc = float((np.sign(Delta_hat[nz]) == np.sign(Delta[nz])).mean())
        fam = {}
        for f in ("encoder", "decoder"):
            idx = np.array([r["family"] == f for r in rs])
            fam[f] = {"mean_delta": float(d[idx].mean()), "mean_D": float(D[idx].mean()),
                      "mean_G": float(np.mean([r["G"] for r, i in zip(rs, idx) if i])),
                      "mean_S": float(np.mean([r["S"] for r, i in zip(rs, idx) if i]))}
        passed = (0 < min(kappas) and max(kappas) <= 1 and r2 >= 0.5 and sign_acc >= 0.8)
        falsified = r2 < 0.3 or sign_acc < 0.7
        out["by_rule"][rule] = {
            "kappa_full": float((D * d).sum() / (d ** 2).sum()),
            "kappa_lopo_range": [min(kappas), max(kappas)],
            "lopo_r2_D": float(r2), "sign_accuracy_Delta": sign_acc,
            "n_cells": int(len(rs)), "by_family": fam,
            "verdict": verdict(passed, falsified)}
    out["rows"] = rows
    save("h2_disagreement", out)

    print(f"H2 over {out['n_pools']} pools")
    for rule, o in out["by_rule"].items():
        print(f"  {rule:14s} kappa={o['kappa_full']:.3f} (LOPO {o['kappa_lopo_range'][0]:.3f}-"
              f"{o['kappa_lopo_range'][1]:.3f})  LOPO R2(D)={o['lopo_r2_D']:+.3f}  "
              f"sign acc={o['sign_accuracy_Delta']:.1%}  -> {o['verdict']}")
        for f, v in o["by_family"].items():
            print(f"      {f:8s} delta={v['mean_delta']:.4f}  D={v['mean_D']:+.4f}  "
                  f"G={v['mean_G']:+.4f}  S={v['mean_S']:+.4f}")


if __name__ == "__main__":
    main()
