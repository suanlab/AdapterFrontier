#!/usr/bin/env python3
"""H3a / H3b on the new Qwen-2.5-0.5B MNLI runs (mechanism/PREREG.md §3).

H3a: does diversity shrink with training? Arms ep1, ep2 (= h3b_all), ep4.
H3b: where does diversity come from? Arms head / lora / data / all / none.
Every quantity is computed on the paper's test split of validation_matched.
Runs that failed to train (final loss >= 0.85 ln 3) are flagged and the
results are reported with and without them (§5).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from common import ROOT, save, verdict

sys.path.insert(0, str(ROOT))
from splits import split_indices  # noqa: E402

RUNS = ROOT / "mechanism" / "runs"
ARMS = {
    "ep1": "h3a_ep1_s", "ep2": "h3b_all_s", "ep4": "h3a_ep4_s",
    "head": "h3b_head_s", "lora": "h3b_lora_s", "data": "h3b_data_s",
    "all": "h3b_all_s", "none": "h3b_none_r",
}


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_arm(prefix: str, drop_failed: bool):
    runs = []
    for d in sorted(RUNS.glob(prefix + "*")):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        if drop_failed and m["train_loss_over_ln3"] >= 0.85:
            continue
        runs.append((d.name, m, np.load(d / "logits_validation_matched.npy")))
    return runs


def stats(runs, y, test):
    L = np.stack([r[2][test] for r in runs])            # M x N x 3
    P = L.argmax(-1); correct = P == y[None, :]
    acc = correct.mean(1)
    plural = np.stack([(P == k).sum(0) for k in range(3)]).argmax(0)
    delta = float((P != plural[None, :]).mean())
    pairs = list(itertools.combinations(range(len(P)), 2))
    disagree = float(np.mean([(P[a] != P[b]).mean() for a, b in pairs])) if pairs else 0.0
    sv = softmax(L).mean(0).argmax(-1) == y
    Lc = L - L.mean(-1, keepdims=True)
    return {"M": len(runs), "mean_acc": float(acc.mean()), "sd_acc": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "delta": delta, "pairwise_disagreement": disagree,
            "D_soft_vote": float(sv.mean() - acc.mean()),
            "logit_spread": float(Lc.var(0).mean())}


def main() -> None:
    y_full = None
    out = {}
    for drop in (False, True):
        res = {}
        for arm, prefix in ARMS.items():
            runs = load_arm(prefix, drop)
            if not runs:
                continue
            if y_full is None:
                y_full = np.load(RUNS / runs[0][0] / "labels_validation_matched.npy")
            _, _, test = split_indices(len(y_full), seed=0)
            res[arm] = stats(runs, y_full[test], test) if len(runs) >= 2 else {"M": len(runs)}
            res[arm]["failed_runs"] = [r[0] for r in load_arm(prefix, False)
                                       if r[1]["train_loss_over_ln3"] >= 0.85]
        out["without_failed" if drop else "all_runs"] = res

    r = out["all_runs"]
    h3a = h3b = "NOT TESTABLE (runs incomplete)"
    if all(k in r and r[k].get("M", 0) >= 2 for k in ("ep1", "ep4")):
        d1, d4 = r["ep1"]["D_soft_vote"], r["ep4"]["D_soft_vote"]
        dl1, dl4 = r["ep1"]["delta"], r["ep4"]["delta"]
        h3a = verdict(d4 < d1 and dl4 < dl1, d4 >= d1)
    if all(k in r and r[k].get("M", 0) >= 2 for k in ("head", "all")):
        ratio = r["head"]["delta"] / r["all"]["delta"] if r["all"]["delta"] > 0 else float("nan")
        out["head_over_all_delta"] = ratio
        h3b = verdict(ratio >= 0.5, ratio < 0.25)
    out["H3a"], out["H3b"] = h3a, h3b
    save("h3_new_runs", out)
    print(f"{'arm':6s} M  mean acc  sd     delta   pair-dis  D_sv     spread  failed")
    for arm, s in r.items():
        if "delta" in s:
            print(f"{arm:6s} {s['M']}  {s['mean_acc']:.4f}  {s['sd_acc']:.4f}  {s['delta']:.4f}  "
                  f"{s['pairwise_disagreement']:.4f}  {100*s['D_soft_vote']:+.2f}pp  {s['logit_spread']:.3f}  {s['failed_runs']}")
        else:
            print(f"{arm:6s} {s['M']}  (fewer than 2 finished runs)")
    print(f"H3a: {h3a}   H3b: {h3b}" + (f"   (head/all delta = {out['head_over_all_delta']:.2f})"
                                       if "head_over_all_delta" in out else ""))


if __name__ == "__main__":
    main()
