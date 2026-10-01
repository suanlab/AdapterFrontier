#!/usr/bin/env python3
"""Did both arms of every headline comparison actually train?

Post-hoc robustness check, and deliberately test-free.

A pool-vs-baseline comparison is only informative if both arms trained. The
`n_rank` baseline is a best-of-N over its members, so if most of its members
never learned, its selection pressure collapses and best-of-20 becomes, in
effect, best-of-1. The clearest case is Qwen-2.5-3B MNLI: 19 of its 20
baseline adapters end training at the loss of a uniform guess, and that pool
supplies 4 of the 19 decoder cells the paper reports as SUPPORTED.

Deciding which comparisons to drop by looking at their test results would
itself be the kind of post-hoc selection the paper measures. So "failed to
train" is defined from each adapter's own final *training* loss, which never
sees the evaluation split:

    failed  <=>  train_loss >= THRESHOLD * ln(num_labels)

ln(C) is the cross-entropy of a uniform prediction. That is the right
reference only for label-balanced tasks, and the five tasks in the headline
slice (MNLI, QNLI, AG News, ANLI, SST-2) are near balance. BoolQ (about 62/38)
is not, and would need a class-prior reference instead; it has no `n_rank`
baseline and is outside this check.

A comparison is excluded when a majority of either arm's adapters failed. The
script reports the headline verdicts with and without those comparisons, and
repeats the flagging at neighbouring thresholds to show it does not hinge on
the exact value.

Output: analysis/training_health.json
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS, POOLS = ROOT / "analysis", ROOT / "pools"
OUT = ANALYSIS / "training_health.json"

THRESHOLD = 0.85
SENSITIVITY = (0.80, 0.85, 0.90)
ENCODERS = ("bert", "roberta", "deberta")
CELL = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_n_rank$")


def trained_fraction(manifest: str, threshold: float) -> tuple[int, int]:
    """-> (n adapters with a recorded training loss, n that trained)."""
    d = json.loads((POOLS / f"{manifest}.json").read_text())
    ratios = []
    for a in d["adapters"]:
        m = a.get("metrics") or {}
        if m.get("train_loss") is not None and m.get("num_labels"):
            ratios.append(m["train_loss"] / math.log(m["num_labels"]))
    return len(ratios), sum(r < threshold for r in ratios)


def headline_cells() -> list[dict]:
    cells = []
    for f in sorted(ANALYSIS.glob("*_cm_*_vs_n_rank.json")):
        m = CELL.match(f.stem)
        if not m:
            continue
        pool = m["pool"]
        if ("hellaswag" in pool or "gsm8k" in pool
                or pool.startswith(("baseline", "smoke"))):
            continue
        d = json.loads(f.read_text())
        cells.append({
            "pool": pool,
            "baseline": Path(d["baseline"]["result_path"]).stem,
            "family": "encoder" if any(e in pool for e in ENCODERS) else "decoder",
            "verdict": d["adjudication_post_fdr"],
        })
    return cells


def failed(manifest: str, threshold: float) -> bool:
    n, ok = trained_fraction(manifest, threshold)
    return n > 0 and ok / n < 0.5


def summarise(cs: list[dict]) -> dict:
    v = collections.Counter(c["verdict"] for c in cs)
    n = len(cs)
    return {"n_cells": n, "n_pools": len({c["pool"] for c in cs}),
            "supported": v.get("supported", 0), "reversed": v.get("reversed", 0),
            "pct_reversed": round(100 * v.get("reversed", 0) / n, 1) if n else None}


def main() -> None:
    cells = headline_cells()
    manifests = sorted({c["pool"] for c in cells} | {c["baseline"] for c in cells})

    health = {}
    for m in manifests:
        n, ok = trained_fraction(m, THRESHOLD)
        health[m] = {"n": n, "trained": ok,
                     "fraction_trained": round(ok / n, 3) if n else None}
    flagged = {m for m in manifests if failed(m, THRESHOLD)}

    def keep(c: dict) -> bool:
        return c["pool"] not in flagged and c["baseline"] not in flagged

    out = {
        "note": ("Post hoc, test-free. An adapter failed if its final train loss "
                 f">= {THRESHOLD} x ln(num_labels); a comparison is excluded when a "
                 "majority of either arm failed."),
        "threshold": THRESHOLD,
        "flagged_manifests": sorted(flagged),
        "flagged_by_threshold": {
            str(t): sorted(m for m in manifests if failed(m, t)) for t in SENSITIVITY},
        "health": health,
    }
    for fam in ("encoder", "decoder"):
        fc = [c for c in cells if c["family"] == fam]
        out[fam] = {"all": summarise(fc),
                    "both_arms_trained": summarise([c for c in fc if keep(c)]),
                    "excluded_pools": sorted({c["pool"] for c in fc if not keep(c)})}
    OUT.write_text(json.dumps(out, indent=1))

    print(f"flagged at {THRESHOLD}: {out['flagged_manifests']}")
    same = len({tuple(v) for v in out["flagged_by_threshold"].values()}) == 1
    print(f"same flags at {SENSITIVITY}: {same}")
    for fam in ("encoder", "decoder"):
        a, k = out[fam]["all"], out[fam]["both_arms_trained"]
        print(f"{fam}: all {a['supported']}/{a['n_cells']} SUP {a['reversed']} REV"
              f"  ->  both arms trained {k['supported']}/{k['n_cells']} SUP "
              f"{k['reversed']} REV ({k['pct_reversed']}%), {k['n_pools']} pools;"
              f" excluded {out[fam]['excluded_pools']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
