#!/usr/bin/env python3
"""Step 3 (exploratory): a better predictor of the implicit temperature T_eq.

H1 found the probit form T_pred = sqrt(1 + pi s^2 / 8), with s^2 the mean
across-member variance of class-centred logits, right in ordering but too low
at high spread (1.11 predicted vs 1.43 observed). MacKay's approximation is
derived for a *binary* logit, i.e. a margin. In a multiclass model the
confidence is governed by the margin between the top two classes, so this
script compares three predictors of T_eq across every pool with logits:

  A  isotropic: s^2 = mean over classes of the across-member logit variance
  B  margin:    s^2 = across-member variance of (z_top - z_second), with top and
                second fixed by the mean-logit model on each example
  C  margin, with the constant fitted (T = sqrt(1 + c s^2)) by least squares

Pools: the six corpus pools with a logits cache, the seven H3 arms
(Qwen-2.5-0.5B MNLI) and whatever H8 SNLI pools have finished.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from common import ROOT, save

sys.path.insert(0, str(ROOT))
from splits import split_indices  # noqa: E402

RUNS = ROOT / "mechanism" / "runs"


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def pool_stats(L):
    """L: members x examples x classes (test slice)."""
    L = L - L.mean(-1, keepdims=True)
    zbar = L.mean(0); p_ens = softmax(L).mean(0)

    def kl(T):
        q = softmax(zbar / T)
        return float((p_ens * (np.log(p_ens + 1e-12) - np.log(q + 1e-12))).sum(-1).mean())
    T_eq = float(minimize_scalar(kl, bounds=(0.25, 10), method="bounded").x)
    s2_iso = float(L.var(0).mean())
    order = np.argsort(-zbar, -1); n = np.arange(L.shape[1])
    margin = L[:, n, order[:, 0]] - L[:, n, order[:, 1]]          # members x examples
    s2_margin = float(margin.var(0).mean())
    return T_eq, s2_iso, s2_margin


def pools():
    for rf in sorted(glob.glob(str(ROOT / "ensemble_results/pilot_c1/*.json"))):
        d = json.loads(Path(rf).read_text())
        npys = sorted((ROOT / "ensemble_cache").glob(f"{d['pool_id']}__*.npy"))
        if len(npys) >= 2:
            yield f"corpus:{d['pool_id']}", np.stack([np.load(f) for f in npys])[:, np.asarray(d["test_indices"])]
    arms = {"ep1": "h3a_ep1_s", "ep2": "h3b_all_s", "ep4": "h3a_ep4_s", "head": "h3b_head_s",
            "lora": "h3b_lora_s", "data": "h3b_data_s"}
    for arm, pre in arms.items():
        L = [np.load(d / "logits_validation_matched.npy") for d in sorted(RUNS.glob(pre + "*"))
             if (d / "metrics.json").exists()]
        if len(L) >= 2:
            _, _, test = split_indices(L[0].shape[0], seed=0)
            yield f"h3:{arm}", np.stack(L)[:, test]
    for tag in ("bert", "q05", "q15", "q3"):
        for rec in ("pool", "base"):
            L = []
            for d in sorted(RUNS.glob(f"h8_{tag}_{rec}_s*")):
                if (d / "metrics.json").exists() and json.loads((d / "metrics.json").read_text())["train_loss_over_ln3"] < 0.85:
                    L.append(np.load(d / "logits_validation.npy"))
            if len(L) >= 2:
                _, _, test = split_indices(L[0].shape[0], seed=0)
                yield f"h8:{tag}/{rec}", np.stack(L)[:, test]


def main() -> None:
    rows = []
    for name, L in pools():
        T_eq, s_iso, s_m = pool_stats(L)
        rows.append({"pool": name, "M": int(L.shape[0]), "T_eq": T_eq, "s2_iso": s_iso, "s2_margin": s_m})
    T = np.array([r["T_eq"] for r in rows])
    pred_A = np.sqrt(1 + np.pi * np.array([r["s2_iso"] for r in rows]) / 8)
    sm = np.array([r["s2_margin"] for r in rows])
    pred_B = np.sqrt(1 + np.pi * sm / 8)
    c = float(minimize_scalar(lambda c: ((np.sqrt(1 + c * sm) - T) ** 2).sum(), bounds=(0, 5), method="bounded").x)
    pred_C = np.sqrt(1 + c * sm)
    r2 = lambda p: float(1 - ((T - p) ** 2).sum() / ((T - T.mean()) ** 2).sum())
    out = {"exploratory": True, "n_pools": len(rows),
           "A_isotropic_probit": {"r2": r2(pred_A), "mean_abs_err": float(np.abs(T - pred_A).mean())},
           "B_margin_probit": {"r2": r2(pred_B), "mean_abs_err": float(np.abs(T - pred_B).mean())},
           "C_margin_fitted": {"c": c, "pi_over_8": float(np.pi / 8), "r2": r2(pred_C),
                               "mean_abs_err": float(np.abs(T - pred_C).mean())},
           "spearman_T_eq_vs_s2_margin": float(spearmanr(T, sm).correlation),
           "spearman_T_eq_vs_s2_iso": float(spearmanr(T, [r["s2_iso"] for r in rows]).correlation),
           "pools": rows}
    for r, a, b, cc in zip(rows, pred_A, pred_B, pred_C):
        r.update({"pred_A": float(a), "pred_B": float(b), "pred_C": float(cc)})
    save("teq_formula", out)
    print(f"{'pool':42s} M   T_eq   A(iso)  B(margin) C(fit)")
    for r in rows:
        print(f"{r['pool'][:42]:42s} {r['M']:2d}  {r['T_eq']:.2f}   {r['pred_A']:.2f}    {r['pred_B']:.2f}     {r['pred_C']:.2f}")
    print(f"\n{len(rows)} pools.  R2 / mean |err|:  A isotropic {out['A_isotropic_probit']['r2']:+.2f} / "
          f"{out['A_isotropic_probit']['mean_abs_err']:.3f}   B margin {out['B_margin_probit']['r2']:+.2f} / "
          f"{out['B_margin_probit']['mean_abs_err']:.3f}   C fitted c={c:.3f} (pi/8={np.pi/8:.3f}) "
          f"{out['C_margin_fitted']['r2']:+.2f} / {out['C_margin_fitted']['mean_abs_err']:.3f}")
    print(f"Spearman(T_eq, s2): margin {out['spearman_T_eq_vs_s2_margin']:+.2f}, isotropic {out['spearman_T_eq_vs_s2_iso']:+.2f}")


if __name__ == "__main__":
    main()
