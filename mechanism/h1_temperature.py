#!/usr/bin/env python3
"""H1: a population's calibration effect is an implicit temperature
(mechanism/PREREG.md §3).

For the six pools with a logits cache. The ensemble is probability averaging
of class-centred logits. T_eq is fitted label-free; T* is the NLL-optimal
temperature for the mean-logit model on val_combine, with labels rebuilt and
cross-checked exactly as in analysis/temperature_control.py. (a) and (d) were
seen in the exploratory pass (§0) and are reported, not counted.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from common import ROOT, save, verdict

sys.path.insert(0, str(ROOT / "analysis"))
from temperature_control import (ece_equal_mass, fit_temperature,  # noqa: E402
                                 labels_for_task)

CACHE = ROOT / "ensemble_cache"


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ece(p, y):
    return ece_equal_mass(p.max(-1), p.argmax(-1) == y)


def main() -> None:
    rows = []
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text()); pid = d["pool_id"]
        npys = sorted(CACHE.glob(f"{pid}__*.npy"))
        if len(npys) < 2:
            continue
        L = np.stack([np.load(f) for f in npys]); L = L - L.mean(-1, keepdims=True)
        test, comb = np.asarray(d["test_indices"]), np.asarray(d["val_combine_indices"])
        y_full = labels_for_task(d["task"], L.shape[1])
        assert y_full is not None and np.array_equal(y_full[test], np.asarray(d["labels_test"]))
        y_t, y_c = y_full[test], y_full[comb]
        Lt = L[:, test]; zbar = Lt.mean(0); p_ens = softmax(Lt).mean(0)

        def kl(T):
            q = softmax(zbar / T)
            return float((p_ens * (np.log(p_ens + 1e-12) - np.log(q + 1e-12))).sum(-1).mean())
        T_eq = float(minimize_scalar(kl, bounds=(0.25, 10), method="bounded").x)
        T_star = float(fit_temperature(L[:, comb].mean(0), y_c))

        ece_ens, ece_mean1 = ece(p_ens, y_t), ece(softmax(zbar), y_t)
        ece_teq = ece(softmax(zbar / T_eq), y_t)
        rule_improves = abs(np.log(T_star) - np.log(T_eq)) < abs(np.log(T_star))
        s2_i = Lt.var(0).mean(-1)
        shrink_i = softmax(zbar).max(-1) - p_ens.max(-1)
        rows.append({
            "pool": pid, "M": int(L.shape[0]), "T_eq": T_eq, "T_star_mean_logit": T_star,
            "kl_T_eq": kl(T_eq), "kl_T1": kl(1.0),
            "ece_ensemble": ece_ens, "ece_mean_logit_T1": ece_mean1,
            "ece_mean_logit_T_eq": ece_teq, "ece_mean_logit_T_star": ece(softmax(zbar / T_star), y_t),
            "b_abs_err": abs(ece_teq - ece_ens),
            "c_predicted_improves": bool(rule_improves),
            "c_observed_improves": bool(ece_ens < ece_mean1),
            "d_spearman": float(spearmanr(s2_i, shrink_i).correlation),
        })

    n = len(rows)
    b_ok = sum(r["b_abs_err"] <= 0.01 for r in rows)
    c_ok = sum(r["c_predicted_improves"] == r["c_observed_improves"] for r in rows)
    a_ok = sum(r["kl_T_eq"] <= 0.5 * r["kl_T1"] for r in rows)
    d_ok = sum(r["d_spearman"] > 0.3 for r in rows)
    passed = b_ok >= 5 and c_ok >= 5
    out = {"n_pools": n, "a_not_confirmatory": f"{a_ok}/{n}", "b": f"{b_ok}/{n}",
           "c": f"{c_ok}/{n}", "d_not_confirmatory": f"{d_ok}/{n}",
           "temp_scaled_mean_beats_ensemble": f"{sum(r['ece_mean_logit_T_star'] < r['ece_ensemble'] for r in rows)}/{n}",
           "verdict": verdict(passed, not passed), "pools": rows}
    save("h1_temperature", out)
    print(f"{'pool':40s} T_eq  T*    ECE ens  ECE(T_eq) ECE(1)  ECE(T*)  pred/obs improve")
    for r in rows:
        print(f"{r['pool'][:40]:40s} {r['T_eq']:.2f}  {r['T_star_mean_logit']:.2f}  {r['ece_ensemble']:.4f}   "
              f"{r['ece_mean_logit_T_eq']:.4f}   {r['ece_mean_logit_T1']:.4f}  {r['ece_mean_logit_T_star']:.4f}  "
              f"{r['c_predicted_improves']!s:5s}/{r['c_observed_improves']}")
    print(f"(b) |ECE(T_eq)-ECE(ens)|<=0.01: {out['b']}   (c) rule matches: {out['c']}   "
          f"[(a) {out['a_not_confirmatory']}, (d) {out['d_not_confirmatory']} not confirmatory]")
    print(f"mean-logit model at T* beats the ensemble: {out['temp_scaled_mean_beats_ensemble']}")
    print(f"-> {out['verdict']}")


if __name__ == "__main__":
    main()
