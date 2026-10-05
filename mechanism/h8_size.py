#!/usr/bin/env python3
"""H8: what extra budget buys a single adapter shrinks with model size
(mechanism/PREREG.md §10, §9 deviation 3).

SNLI, outside the corpus. Per model: G_tt = mean test accuracy of the trained
baseline-recipe adapters (r=32, 4 ep) minus that of the trained pool-recipe
adapters (r=8, 2 ep), failed runs removed by the training-loss rule.

Confirmatory: G_tt decreases monotonically 0.5B > 1.5B > 3B within Qwen-2.5,
and G_tt(BERT-base) > G_tt(Qwen-2.5-0.5B); falsified if G_tt(3B) >= G_tt(0.5B).
Exploratory: the paper's comparison replayed on new data -- the pool-recipe
soft_vote ensemble against the baseline-recipe adapter with the best accuracy
on the val_selection slice.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import ROOT, save, verdict

sys.path.insert(0, str(ROOT))
from splits import split_indices  # noqa: E402

RUNS = ROOT / "mechanism" / "runs"
MODELS = [("bert", "BERT-base", 110), ("q05", "Qwen-2.5-0.5B", 494),
          ("q15", "Qwen-2.5-1.5B", 1544), ("q3", "Qwen-2.5-3B", 3086)]


def load(tag, recipe):
    runs = []
    for d in sorted(RUNS.glob(f"h8_{tag}_{recipe}_s*")):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        runs.append((m, np.load(d / "logits_validation.npy")))
    return runs


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


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
    out = {"models": {}}
    y = None
    for tag, name, size in MODELS:
        pool, base = load(tag, "pool"), load(tag, "base")
        if y is None and (pool or base):
            y = np.load(next(d for d in RUNS.glob(f"h8_{tag}_*_s*")
                             if (d / "labels_validation.npy").exists()) / "labels_validation.npy")
        if not pool or not base:
            out["models"][name] = {"status": f"incomplete ({len(pool)} pool, {len(base)} base)"}
            continue
        sel, _, test = split_indices(len(y), seed=0)
        ok = lambda rs: [r for r in rs if r[0]["train_loss_over_ln3"] < 0.85]
        pt, bt = ok(pool), ok(base)
        acc = lambda rs, idx: np.array([(r[1][idx].argmax(-1) == y[idx]).mean() for r in rs])
        G = float(acc(bt, test).mean() - acc(pt, test).mean()) if pt and bt else float("nan")
        ens = softmax(np.stack([r[1][test] for r in pt])).mean(0).argmax(-1) == y[test]
        b_pick = bt[int(np.argmax(acc(bt, sel)))]
        delta = float(ens.mean() - (b_pick[1][test].argmax(-1) == y[test]).mean())
        out["models"][name] = {
            "params_m": size, "n_pool": len(pool), "n_base": len(base),
            "n_pool_trained": len(pt), "n_base_trained": len(bt),
            "pool_mean_acc": float(acc(pt, test).mean()), "base_mean_acc": float(acc(bt, test).mean()),
            "G_tt": G, "D_pool_soft_vote": float(ens.mean() - acc(pt, test).mean()),
            "Delta_ensemble_vs_selected_baseline_exploratory": delta}
    ms = out["models"]
    problems = check_complete_and_aligned(RUNS)
    out["completeness_problems"] = problems
    done = all("G_tt" in ms.get(n, {}) for _, n, _ in MODELS) and not problems
    if done:
        g = [ms[n]["G_tt"] for _, n, _ in MODELS]
        mono = g[1] > g[2] > g[3]
        passed = mono and g[0] > g[1]
        out["verdict"] = verdict(passed, g[3] >= g[1])
    else:
        out["verdict"] = "NOT TESTABLE (runs incomplete)"
    save("h8_size", out)
    print(f"{'model':16s} trained(pool/base)  pool acc  base acc   G_tt     D_pool   Delta(expl.)")
    for _, n, _ in MODELS:
        v = ms[n]
        if "G_tt" not in v:
            print(f"{n:16s} {v['status']}"); continue
        print(f"{n:16s} {v['n_pool_trained']}/{v['n_pool']} {v['n_base_trained']}/{v['n_base']}          "
              f"{v['pool_mean_acc']:.4f}   {v['base_mean_acc']:.4f}   {100*v['G_tt']:+.2f}pp  "
              f"{100*v['D_pool_soft_vote']:+.2f}pp  {100*v['Delta_ensemble_vs_selected_baseline_exploratory']:+.2f}pp")
    print(f"H8 -> {out['verdict']}")


if __name__ == "__main__":
    main()
