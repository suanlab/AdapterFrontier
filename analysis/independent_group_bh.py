#!/usr/bin/env python3
"""Independence robustness for the corpus BH: one cell per independent unit.

BH assumes roughly independent p-values, and the four combination rules of a
(pool, baseline_kind, metric) unit share members, so their p-values are
correlated. Here each unit contributes one representative cell (`soft_vote`)
and BH runs per arm over the representatives, on the same valid p-values the
per-cell batch uses (`p_for_bh`: accuracy sign-flip p; ECE paired-bootstrap p,
see scripts/apply_bh_by_arm.py). An earlier, unscripted version ran one mixed
batch over the invalid ECE permutation p-values (L14).

Usage: python3 analysis/independent_group_bh.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = re.compile(r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_"
                  r"(?P<kind>best_of_n|n_rank|n_steps|n_data)(?P<ece>_ECE)?\.json$")
Q = 0.05


def bh_pass(pvals: list[float], q: float) -> tuple[list[bool], float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            k = rank
    cutoff = pvals[order[k - 1]] if k else 0.0
    return [pvals[i] <= cutoff and k > 0 for i in range(m)], cutoff


def main() -> int:
    reps = {"accuracy": [], "ece": []}
    for f in sorted(glob.glob(str(ROOT / "analysis/*_cm_soft_vote_vs_*.json"))):
        m = CELL.match(Path(f).name)
        if not m:
            continue
        d = json.loads(Path(f).read_text())
        reps["ece" if m.group("ece") else "accuracy"].append(
            (Path(f).name, d["p_for_bh"], d["adjudication_pre_fdr"]))
    out = {"q": Q, "representative": "soft_vote", "arms": {}}
    tot = {"n": 0, "supported": 0, "reversed": 0}
    for arm, items in reps.items():
        passed, cutoff = bh_pass([p for _, p, _ in items], Q)
        post = [v if ok else "unsupported" for (_, _, v), ok in zip(items, passed)]
        c = {"n_groups": len(items), "bh_cutoff_p": cutoff,
             "supported": post.count("supported"), "reversed": post.count("reversed")}
        out["arms"][arm] = c
        tot["n"] += len(items); tot["supported"] += c["supported"]; tot["reversed"] += c["reversed"]
        print(f"{arm:8s} groups={len(items)} cutoff={cutoff:.4g} SUP {c['supported']} REV {c['reversed']}")
    tot["pct_supported"] = round(100 * tot["supported"] / tot["n"], 1)
    tot["pct_reversed"] = round(100 * tot["reversed"] / tot["n"], 1)
    out["total"] = tot
    print(f"total    groups={tot['n']} SUP {tot['supported']} ({tot['pct_supported']}%) "
          f"REV {tot['reversed']} ({tot['pct_reversed']}%)")
    (ROOT / "analysis/independent_group_bh.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
