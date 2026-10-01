#!/usr/bin/env python3
"""Encoder vs decoder, against the same strong single adapter.

The paper's headline accuracy null is an encoder statement: 0 of 92 clean
encoder classification cells beat the `n_rank` single. Until v1.3.n the body
did not report the other half of the same comparison. The 25 decoder
classification pools were adjudicated against the same kind of baseline,
which received the same budget advantage (median 1.45x the pool's GPU-hours,
all 25 pairs baseline-favoured; baseline_budget_audit.json), and they come out
differently: some cells are SUPPORTED and none are REVERSED.

Reporting only the encoder half would be the selective reporting this paper
is about, so this script produces both halves side by side, with enough
structure (per task, per method, per pool) to show where the decoder gain
sits. It also records the best_of_n comparison per family, so the size of
shortcut S2 can be read off for decoders as well as encoders.

The slice definition matches the headline: classification pools only
(HellaSwag and GSM8K excluded), accuracy and ECE cells, post-FDR verdicts.

Output: analysis/family_split.json
"""
from __future__ import annotations

import collections
import json
import re
import statistics
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "family_split.json"

ENCODERS = ("bert", "roberta", "deberta")
TASKS = ("mnli", "qnli", "agnews", "boolq", "anli", "sst2")
CELL = re.compile(
    r"^(?P<pool>.+?)_cm_(?P<method>[a-z_]+?)_vs_(?P<base>n_rank|best_of_n)(?P<ece>_ECE)?$")


def family(pool: str) -> str:
    return "encoder" if any(e in pool for e in ENCODERS) else "decoder"


def task(pool: str) -> str:
    return next(t for t in TASKS if f"_{t}_" in f"{pool}_")


def load() -> list[dict]:
    cells = []
    for f in sorted(ANALYSIS.glob("*_cm_*.json")):
        m = CELL.match(f.stem)
        if not m:
            continue
        pool = m["pool"]
        if ("hellaswag" in pool or "gsm8k" in pool
                or pool.startswith(("baseline", "smoke"))):
            continue
        d = json.loads(f.read_text())
        cells.append({
            "pool": pool, "family": family(pool), "task": task(pool),
            "method": m["method"], "baseline": m["base"],
            "metric": "ece" if m["ece"] else "acc",
            "delta": d.get("accuracy_diff"),
            "ci_low": d.get("ci_low"), "ci_high": d.get("ci_high"),
            "pre": d["adjudication_pre_fdr"], "post": d["adjudication_post_fdr"],
        })
    return cells


def verdicts(cs: list[dict], key: str = "post") -> dict:
    v = collections.Counter(c[key] for c in cs)
    n = len(cs)
    return {"n": n, "supported": v.get("supported", 0),
            "reversed": v.get("reversed", 0),
            "pct_supported": round(100 * v.get("supported", 0) / n, 1) if n else None,
            "pct_reversed": round(100 * v.get("reversed", 0) / n, 1) if n else None}


def main() -> None:
    cells = load()
    out: dict = {"note": ("Clean classification cells, post-FDR unless marked pre. "
                          "Encoder = BERT/RoBERTa/DeBERTa; decoder = the rest.")}
    for fam in ("encoder", "decoder"):
        acc = [c for c in cells if c["family"] == fam and c["baseline"] == "n_rank"
               and c["metric"] == "acc"]
        ece = [c for c in cells if c["family"] == fam and c["baseline"] == "n_rank"
               and c["metric"] == "ece"]
        bon = [c for c in cells if c["family"] == fam and c["baseline"] == "best_of_n"
               and c["metric"] == "acc"]
        pools = sorted({c["pool"] for c in acc})
        sup_pools = sorted({c["pool"] for c in acc if c["post"] == "supported"})
        rev_pools = sorted({c["pool"] for c in acc if c["post"] == "reversed"})
        # Pool level, the inferential unit of L1d: one mean delta per pool
        # (averaging its four method cells), then a two-sided sign test.
        pool_mean = {p: statistics.mean(c["delta"] for c in acc if c["pool"] == p)
                     for p in pools}
        n_pos = sum(v > 0 for v in pool_mean.values())
        n_neg = sum(v < 0 for v in pool_mean.values())
        sign_p = binomtest(n_pos, n_pos + n_neg, 0.5).pvalue
        out[fam] = {
            "n_pools": len(pools),
            "acc_vs_n_rank": verdicts(acc),
            "acc_vs_n_rank_pre_fdr": verdicts(acc, "pre"),
            "acc_mean_delta_pp": round(100 * statistics.mean(c["delta"] for c in acc), 2),
            "acc_median_delta_pp": round(100 * statistics.median(c["delta"] for c in acc), 2),
            "acc_median_ci_halfwidth_pp": round(100 * statistics.median(
                (c["ci_high"] - c["ci_low"]) / 2 for c in acc), 2),
            "acc_cells_excluding_plus_1pp": sum(c["ci_high"] < 0.01 for c in acc),
            "pools_with_a_supported_cell": len(sup_pools),
            "pools_with_a_reversed_cell": len(rev_pools),
            "pools_mean_delta_positive": n_pos,
            "pools_mean_delta_negative": n_neg,
            "pool_sign_test_p": round(float(sign_p), 5),
            "supported_cells_by_task": dict(collections.Counter(
                c["task"] for c in acc if c["post"] == "supported")),
            "acc_by_method": {m: verdicts([c for c in acc if c["method"] == m])
                              for m in sorted({c["method"] for c in acc})},
            "ece_vs_n_rank": verdicts(ece),
            "acc_vs_best_of_n": verdicts(bon),
        }
    OUT.write_text(json.dumps(out, indent=1))

    for fam in ("encoder", "decoder"):
        o = out[fam]
        a, e, b = o["acc_vs_n_rank"], o["ece_vs_n_rank"], o["acc_vs_best_of_n"]
        print(f"{fam}: {o['n_pools']} pools")
        print(f"  acc vs n_rank    : {a['supported']}/{a['n']} SUP, {a['reversed']} REV, "
              f"mean {o['acc_mean_delta_pp']:+.2f}pp, median CI half-width "
              f"{o['acc_median_ci_halfwidth_pp']}pp")
        print(f"  SUPPORTED by task: {o['supported_cells_by_task']}; "
              f"{o['pools_with_a_supported_cell']} pools with >=1")
        print(f"  pool level       : mean delta >0 in {o['pools_mean_delta_positive']}, <0 in "
              f"{o['pools_mean_delta_negative']} (sign test p = {o['pool_sign_test_p']}); "
              f"{o['pools_with_a_reversed_cell']} pools with a REVERSED cell")
        print(f"  ece vs n_rank    : {e['supported']}/{e['n']} SUP, {e['reversed']} REV")
        print(f"  acc vs best_of_n : {b['supported']}/{b['n']} SUP ({b['pct_supported']}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
