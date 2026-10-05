#!/usr/bin/env python3
"""H10: the implicit-temperature result (H1) on the new SNLI pools
(mechanism/PREREG.md §10).

Eight pools: {BERT-base, Qwen-2.5 0.5B/1.5B/3B} x {pool recipe, baseline
recipe}, trained adapters only. Same quantities as H1 and
analysis/temperature_mechanism.py; T* is fitted on the val_combine slice.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from common import ROOT, save, verdict

sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
from splits import split_indices  # noqa: E402
from temperature_control import ece_equal_mass, fit_temperature  # noqa: E402

RUNS = ROOT / "mechanism" / "runs"
MODELS = [("bert", "BERT-base", "encoder"), ("q05", "Qwen-2.5-0.5B", "decoder"),
          ("q15", "Qwen-2.5-1.5B", "decoder"), ("q3", "Qwen-2.5-3B", "decoder")]


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ece(p, y):
    return float(ece_equal_mass(p.max(-1), p.argmax(-1) == y))


def check_complete_and_aligned(runs_dir, per_arm=6):
    """Refuse a verdict unless every model x recipe has its registered number of
    finished runs and every run shares one label array and one logit shape."""
    import hashlib, collections
    cnt, labels, shapes = collections.Counter(), set(), set()
    for d in sorted(runs_dir.glob("h8_*")):
        if not (d / "metrics.json").exists():
            continue
        _, tag, rec = d.name.split("_")[:3]
        cnt[(tag, rec)] += 1
        labels.add(hashlib.md5(np.load(d / "labels_validation.npy").tobytes()).hexdigest())
        shapes.add(np.load(d / "logits_validation.npy", mmap_mode="r").shape)
    short = {f"{t}/{r}": n for (t, r), n in cnt.items() if n < per_arm}
    missing = [f"{t}/{r}" for t in ("bert", "q05", "q15", "q3") for r in ("pool", "base") if (t, r) not in cnt]
    problems = []
    if short or missing:
        problems.append(f"incomplete arms: {short} missing: {missing}")
    if len(labels) > 1:
        problems.append(f"{len(labels)} different label arrays across runs")
    if len(shapes) > 1:
        problems.append(f"logit shapes differ: {sorted(shapes)}")
    return problems


def main() -> None:
    rows, missing = [], []
    for tag, name, fam in MODELS:
        for recipe in ("pool", "base"):
            runs, y = [], None
            for d in sorted(RUNS.glob(f"h8_{tag}_{recipe}_s*")):
                if (d / "metrics.json").exists():          # finished runs only
                    m = json.loads((d / "metrics.json").read_text())
                    y = np.load(d / "labels_validation.npy")
                    if m["train_loss_over_ln3"] < 0.85:
                        runs.append(np.load(d / "logits_validation.npy"))
            if len(runs) < 2:
                missing.append(f"{name}/{recipe} ({len(runs)} trained)"); continue
            _, comb, test = split_indices(len(y), seed=0)
            L = np.stack(runs); L = L - L.mean(-1, keepdims=True)
            Lt, yt = L[:, test], y[test]; zbar = Lt.mean(0); p_ens = softmax(Lt).mean(0)

            def kl(T):
                q = softmax(zbar / T)
                return float((p_ens * (np.log(p_ens + 1e-12) - np.log(q + 1e-12))).sum(-1).mean())
            T_eq = float(minimize_scalar(kl, bounds=(0.25, 10), method="bounded").x)
            T_star = float(fit_temperature(L[:, comb].mean(0), y[comb]))
            e_ens, e_1, e_teq = ece(p_ens, yt), ece(softmax(zbar), yt), ece(softmax(zbar / T_eq), yt)
            rows.append({"pool": f"{name}/{recipe}", "family": fam, "M": len(runs),
                         "T_eq": T_eq, "T_star": T_star, "ece_ensemble": e_ens,
                         "ece_mean_T1": e_1, "ece_mean_T_eq": e_teq,
                         "ece_mean_T_star": ece(softmax(zbar / T_star), yt),
                         "b_ok": abs(e_teq - e_ens) <= 0.01,
                         "c_pred": bool(abs(np.log(T_star) - np.log(T_eq)) < abs(np.log(T_star))),
                         "c_obs": bool(e_ens < e_1)})
    problems = check_complete_and_aligned(RUNS)
    if missing or problems:
        out = {"verdict": "NOT TESTABLE (runs incomplete or misaligned)", "missing": missing,
               "completeness_problems": problems, "pools": rows}
    else:
        b = sum(r["b_ok"] for r in rows); c = sum(r["c_pred"] == r["c_obs"] for r in rows)
        out = {"b": f"{b}/8", "c": f"{c}/8", "verdict": verdict(b >= 6 and c >= 6, b <= 4 or c <= 4),
               "pools": rows}
    save("h10_temperature_snli", out)
    print(f"{'pool':22s} fam      M  T_eq  T*    ECE ens  ECE(T_eq) ECE(1)   ECE(T*)  improves pred/obs")
    for r in rows:
        print(f"{r['pool']:22s} {r['family']:8s} {r['M']}  {r['T_eq']:.2f}  {r['T_star']:.2f}  {r['ece_ensemble']:.4f}   "
              f"{r['ece_mean_T_eq']:.4f}    {r['ece_mean_T1']:.4f}   {r['ece_mean_T_star']:.4f}   {r['c_pred']}/{r['c_obs']}")
    print(("missing: " + ", ".join(missing)) if missing else f"(b) {out['b']}  (c) {out['c']}")
    print(f"H10 -> {out['verdict']}")


if __name__ == "__main__":
    main()
