#!/usr/bin/env python3
"""Why one temperature can replace the population (App. tempmech).

Post-hoc analysis of the six pools whose logits cache survives (the same pools
as temperature_control.py). Probability averaging over members' logits z_i is
closely approximated by the mean-logit model at a single temperature,

    mean_i softmax(z_i)  ~=  softmax(zbar / T_eq),

the multiclass analogue of MacKay's probit approximation, E[sigmoid(z)] ~=
sigmoid(mu / sqrt(1 + pi s^2 / 8)) for z ~ N(mu, s^2). T_eq is fitted without
labels. The ensemble therefore sits at one point on the mean model's
temperature curve -- a point set by how much the members disagree, not by how
much temperature the model needs (T*, fitted on val_combine). That predicts
when averaging improves calibration over the mean model: when T_eq moves
toward T*, i.e. |ln T* - ln T_eq| < |ln T*|.

Output: analysis/temperature_mechanism.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from temperature_control import (ece_equal_mass, fit_temperature,  # noqa: E402
                                 labels_for_task)

CACHE = ROOT / "ensemble_cache"
OUT = ROOT / "analysis" / "temperature_mechanism.json"


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ece(p, y):
    return float(ece_equal_mass(p.max(-1), p.argmax(-1) == y))


def main() -> int:
    rows = []
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text()); pid = d["pool_id"]
        npys = sorted(CACHE.glob(f"{pid}__*.npy"))
        if len(npys) < 2:
            continue
        L = np.stack([np.load(f) for f in npys]); L = L - L.mean(-1, keepdims=True)
        test, comb = np.asarray(d["test_indices"]), np.asarray(d["val_combine_indices"])
        y_full = labels_for_task(d["task"], L.shape[1])
        if y_full is None or not np.array_equal(y_full[test], np.asarray(d["labels_test"])):
            print(f"  !! {pid}: labels unavailable or inconsistent, skipping")
            continue
        y = y_full[test]; Lt = L[:, test]; zbar = Lt.mean(0); p_ens = softmax(Lt).mean(0)

        def kl(T):
            q = softmax(zbar / T)
            return float((p_ens * (np.log(p_ens + 1e-12) - np.log(q + 1e-12))).sum(-1).mean())
        T_eq = float(minimize_scalar(kl, bounds=(0.25, 10), method="bounded").x)
        T_star = float(fit_temperature(L[:, comb].mean(0), y_full[comb]))
        e_ens, e_1 = ece(p_ens, y), ece(softmax(zbar), y)
        predicted = bool(abs(np.log(T_star) - np.log(T_eq)) < abs(np.log(T_star)))
        rows.append({
            "pool_id": pid, "n_adapters": int(L.shape[0]), "T_eq": T_eq, "T_star": T_star,
            "kl_at_T_eq": kl(T_eq), "kl_at_T1": kl(1.0),
            "ece_ensemble": e_ens, "ece_mean_T1": e_1,
            "ece_mean_T_eq": ece(softmax(zbar / T_eq), y),
            "ece_mean_T_star": ece(softmax(zbar / T_star), y),
            "predicted_ensemble_improves": predicted,
            "observed_ensemble_improves": bool(e_ens < e_1)})
    if not rows:
        print("no pool with a logits cache and consistent labels")
        return 1
    n = len(rows)
    out = {
        "note": ("Post hoc. T_eq fitted label-free on test; T* is the NLL-optimal temperature "
                 "of the mean-logit model on val_combine."),
        "n_pools": n,
        "teq_reproduces_ensemble_ece_within_0p01": sum(abs(r["ece_mean_T_eq"] - r["ece_ensemble"]) <= 0.01 for r in rows),
        "rule_predicts_improvement": sum(r["predicted_ensemble_improves"] == r["observed_ensemble_improves"] for r in rows),
        "pools": rows,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{'pool':44s} T_eq  T*    ECE ens  ECE(T_eq)  ECE(1)   ECE(T*)  improves pred/obs")
    for r in rows:
        print(f"{r['pool_id'][:44]:44s} {r['T_eq']:.2f}  {r['T_star']:.2f}  {r['ece_ensemble']:.4f}   "
              f"{r['ece_mean_T_eq']:.4f}    {r['ece_mean_T1']:.4f}   {r['ece_mean_T_star']:.4f}   "
              f"{r['predicted_ensemble_improves']}/{r['observed_ensemble_improves']}")
    print(f"T_eq reproduces the ensemble's ECE within 0.01 in {out['teq_reproduces_ensemble_ece_within_0p01']}/{n}; "
          f"the rule predicts improvement in {out['rule_predicts_improvement']}/{n}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
