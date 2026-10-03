#!/usr/bin/env python3
"""Replace the mixed corpus BH batch with one BH batch per arm, each over valid
p-values (fixes the error disclosed as L14).

What was wrong
--------------
scripts/apply_corpus_bh_fdr.py ran one BH over all 1,028 cells. The 514 ECE
cells contributed `sign_flip_p` values from compute_match's ECE permutation,
which swaps the (confidence, correctness) *pair* between arms and so tests
whether the two systems share one joint distribution -- which averaging breaks
by construction whether or not ECE moves (median ECE p 4e-4 against 0.13 for
accuracy). Those p-values were invalid, and pooling them with the accuracy
p-values loosened the accuracy cutoff (0.0145 -> 0.0248).

What this script does
---------------------
* Accuracy cells keep their pre-registered paired sign-flip p (10,000
  permutations, as stored) and get their own BH batch.
* ECE cells get a valid two-sided p by inverting a paired bootstrap of the ECE
  difference (B = 5,000, equal-mass 15-bin ECE), exactly the procedure of
  analysis/ece_test_correction.py, which computed it for the 92 headline cells
  only. Seeds match that script on n_rank cells so those p-values agree.
* Each arm's BH uses q = 0.05. The verdict rule is unchanged:
  post-FDR = the CI verdict if the cell passes BH, else "unsupported".
* The old mixed-batch fields are preserved under `legacy_mixed_batch`.

Usage:  python3 scripts/apply_bh_by_arm.py [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from compute_match import _ece_equal_mass  # noqa: E402

B = 5000
Q = 0.05
SUMMARY = ROOT / "analysis" / "bh_by_arm_summary.json"


def arrays(path: str, method: str):
    d = json.loads((ROOT / path.lstrip("./")).read_text())
    y = np.asarray(d["labels_test"])
    blk = d["methods"][method]
    return (np.asarray(blk["confidence_test"], dtype=np.float64),
            (np.asarray(blk["predictions_test"]) == y).astype(np.float64))


def ece_p(cell_path: str) -> tuple[str, float]:
    cell = json.loads(Path(cell_path).read_text())
    e, b = cell["ensemble"], cell["baseline"]
    conf_b, corr_b = arrays(b["result_path"], b["method"])
    conf_e, corr_e = arrays(e["result_path"], e["method"])
    pid, m = e["pool_id"], e["method"]
    key = f"{pid}|{m}" if b["kind"] == "n_rank" else f"{pid}|{m}|{b['kind']}"
    rng = np.random.RandomState(zlib.crc32(key.encode()) % (2**31))
    n = len(conf_b)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.randint(0, n, size=n)
        boots[i] = (_ece_equal_mass(conf_b[idx], corr_b[idx])
                    - _ece_equal_mass(conf_e[idx], corr_e[idx]))
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return cell_path, float(max(p, 1.0 / (B + 1)))


def bh(pvals: dict[str, float], q: float):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    qvals, prev = {}, 1.0
    for rank in range(m, 0, -1):                       # step-up, monotone
        path, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        qvals[path] = prev
    passed = [p for (path, p) in items if qvals[path] <= q]
    cutoff = max(passed) if passed else 0.0
    return qvals, cutoff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    files = sorted(str(f) for f in (ROOT / "analysis").glob("**/*_cm_*.json"))
    cells = {f: json.loads(Path(f).read_text()) for f in files}
    acc = {f: c["sign_flip_p"] for f, c in cells.items() if not f.endswith("_ECE.json")}
    ece_files = [f for f in files if f.endswith("_ECE.json")]
    with Pool(args.workers) as pool:
        ece = dict(pool.map(ece_p, ece_files, chunksize=4))

    out = {"q": Q, "n_cells": len(files), "arms": {}}
    for arm, pv in (("accuracy", acc), ("ece", ece)):
        qv, cutoff = bh(pv, Q)
        counts = {"pre": {}, "post": {}}
        for f in pv:
            c = cells[f]
            if "legacy_mixed_batch" not in c:
                c["legacy_mixed_batch"] = {k: c.get(k) for k in (
                    "bh_q_value", "bh_pass", "bh_q_threshold", "bh_batch_size", "bh_batch_hash",
                    "holm_adjusted_p_batch", "adjudication_post_fdr", "adjudication")}
            pre = c["adjudication_pre_fdr"]
            c["bh_arm"] = arm
            c["p_for_bh"] = pv[f]
            if arm == "ece":
                c["ece_p_paired_bootstrap"] = pv[f]
            c["bh_q_value"] = qv[f]
            c["bh_pass"] = qv[f] <= Q
            c["bh_q_threshold"] = Q
            c["bh_batch_size"] = len(pv)
            post = pre if c["bh_pass"] else "unsupported"
            c["adjudication_post_fdr"] = post
            c["adjudication"] = post
            counts["pre"][pre] = counts["pre"].get(pre, 0) + 1
            counts["post"][post] = counts["post"].get(post, 0) + 1
        h = hashlib.sha256("\n".join(sorted(Path(f).name for f in pv)).encode()).hexdigest()[:16]
        for f in pv:
            cells[f]["bh_batch_hash"] = h
        out["arms"][arm] = {"n_cells": len(pv), "bh_cutoff_p": cutoff, "bh_batch_hash": h,
                            "counts_pre_fdr": counts["pre"], "counts_post_fdr": counts["post"],
                            "median_p": float(np.median(list(pv.values())))}
    if not args.dry_run:
        for f, c in cells.items():
            Path(f).write_text(json.dumps(c, indent=2))
        SUMMARY.write_text(json.dumps(out, indent=2))
    for arm, a in out["arms"].items():
        print(f"{arm:8s} n={a['n_cells']}  BH cutoff p={a['bh_cutoff_p']:.4g}  median p={a['median_p']:.3g}  "
              f"pre {a['counts_pre_fdr']}  post {a['counts_post_fdr']}")
    print("dry run, nothing written" if args.dry_run else f"wrote {len(cells)} cells and {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
