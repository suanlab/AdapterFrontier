#!/usr/bin/env python3
"""P1: what a population buys that a scalar cannot -- ranking
(mechanism/PREREG.md §3). Also P3, the per-example oracle (descriptive).

Temperature scaling is monotone in the logits' scale, so it cannot reorder a
single model's predictions by confidence; selective prediction (abstain on the
least confident) is therefore untouched by "one adapter plus one temperature".
E-AURC = AURC - AURC of a perfect ranking at the same error rate, which
separates ranking quality from accuracy.
"""
from __future__ import annotations

import numpy as np

from common import headline_pairs, result, save, verdict


def aurc(conf, correct):
    order = np.argsort(-conf, kind="stable")
    err = (~correct[order]).astype(float)
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def e_aurc(conf, correct):
    ideal = np.sort(correct.astype(float))[::-1]           # all correct first
    err = 1 - ideal
    opt = float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())
    return aurc(conf, correct) - opt


def main() -> None:
    rows = []
    for pr in headline_pairs():
        E, B = result(pr["ensemble"]), result(pr["baseline"]); y = E["y"]
        sv = E["methods"]["soft_vote"]; bs = B["methods"]["best_single"]
        ce = np.asarray(sv["predictions_test"]) == y; cb = np.asarray(bs["predictions_test"]) == y
        fe, fb = np.asarray(sv["confidence_test"]), np.asarray(bs["confidence_test"])
        oracle = float((E["P"] == y[None, :]).any(0).mean())
        rows.append({"pool": pr["pool"], "family": pr["family"],
                     "acc_ens": float(ce.mean()), "acc_single": float(cb.mean()),
                     "aurc_ens": aurc(fe, ce), "aurc_single": aurc(fb, cb),
                     "eaurc_ens": e_aurc(fe, ce), "eaurc_single": e_aurc(fb, cb),
                     "oracle_any_member": oracle})
    n = len(rows)
    wins = sum(r["eaurc_ens"] < r["eaurc_single"] for r in rows)
    frac = wins / n
    fam = {}
    for f in ("encoder", "decoder"):
        rs = [r for r in rows if r["family"] == f]
        fam[f] = {"pools": len(rs),
                  "eaurc_ens_better": f"{sum(r['eaurc_ens'] < r['eaurc_single'] for r in rs)}/{len(rs)}",
                  "aurc_ens_better": f"{sum(r['aurc_ens'] < r['aurc_single'] for r in rs)}/{len(rs)}",
                  "median_eaurc_ens": float(np.median([r["eaurc_ens"] for r in rs])),
                  "median_eaurc_single": float(np.median([r["eaurc_single"] for r in rs])),
                  "mean_oracle_minus_single_pp": float(100 * np.mean([r["oracle_any_member"] - r["acc_single"] for r in rs])),
                  "mean_ens_minus_single_pp": float(100 * np.mean([r["acc_ens"] - r["acc_single"] for r in rs]))}
    out = {"n_pools": n, "eaurc_ens_better": f"{wins}/{n}", "fraction": frac,
           "verdict": verdict(frac >= 0.7, frac <= 0.5), "by_family": fam, "pools": rows}
    save("p1_selective", out)
    print(f"P1: ensemble E-AURC below single in {wins}/{n} pools ({frac:.0%}; need >= 70%) -> {out['verdict']}")
    for f, v in fam.items():
        print(f"  {f:8s} E-AURC ens better {v['eaurc_ens_better']} (raw AURC {v['aurc_ens_better']}); "
              f"median E-AURC ens {v['median_eaurc_ens']:.4f} vs single {v['median_eaurc_single']:.4f}")
        print(f"           P3 oracle - single: {v['mean_oracle_minus_single_pp']:+.2f}pp   "
              f"ensemble - single: {v['mean_ens_minus_single_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
