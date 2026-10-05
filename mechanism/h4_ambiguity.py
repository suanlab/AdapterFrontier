#!/usr/bin/env python3
"""H4: ensemble gains concentrate on ambiguous items (mechanism/PREREG.md §3).

ChaosNLI-MNLI (Nie et al., 2020) gives 100 human labels per item. The copy
used is tasksource/chaos-mnli-ambiguity (1,599 items, label counts summing to
100, original pairIDs), the official Dropbox link returning an HTML page.
Items are aligned to GLUE validation_matched by exact (premise, hypothesis)
text; all 1,599 align and 647 fall in the paper's test split.

D per item = 1[soft_vote correct] - mean member correctness. Pooled over the
decoder MNLI pools that have a SUPPORTED accuracy cell against n_rank, split
into terciles of human-label entropy. Encoder MNLI pools are reported as an
exploratory contrast.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from datasets import load_dataset

from common import ROOT, headline_pairs, result, save, verdict

sys.path.insert(0, str(ROOT))
from splits import split_indices  # noqa: E402


def supported_pools() -> set[str]:
    s = set()
    for f in (ROOT / "analysis").glob("*_mnli_*_cm_*_vs_n_rank.json"):
        if f.stem.endswith("_ECE"):
            continue
        d = json.loads(f.read_text())
        if d["adjudication_post_fdr"] == "supported":
            s.add(d["ensemble"]["pool_id"])
    return s


def main() -> None:
    rows = json.loads((ROOT / "mechanism/data/chaosnli_mnli_m.json").read_text())
    ev = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    key = {(p.strip(), h.strip()): i for i, (p, h) in enumerate(zip(ev["premise"], ev["hypothesis"]))}
    ent = {key[(r["premise"].strip(), r["hypothesis"].strip())]: float(r["entropy"]) for r in rows}
    _, _, test = split_indices(len(ev), seed=0)
    pos = {int(t): j for j, t in enumerate(test)}            # dataset index -> test position
    items = sorted(i for i in ent if i in pos)
    e = np.array([ent[i] for i in items]); cols = np.array([pos[i] for i in items])
    cut = np.quantile(e, [1 / 3, 2 / 3])
    terc = np.digitize(e, cut)                               # 0 low, 1 mid, 2 high

    sup = supported_pools()
    groups = {"decoder_supported": [], "encoder_mnli_exploratory": []}
    for pr in headline_pairs():
        if pr["task"] != "mnli":
            continue
        E = result(pr["ensemble"])
        assert np.array_equal(E["test_indices"], test), pr["pool"]
        y = E["y"][cols]
        ens = np.asarray(E["methods"]["soft_vote"]["predictions_test"])[cols] == y
        mem = (E["P"][:, cols] == y[None, :]).mean(0)
        D = ens.astype(float) - mem
        if pr["family"] == "decoder" and pr["pool"] in sup:
            groups["decoder_supported"].append(D)
        elif pr["family"] == "encoder":
            groups["encoder_mnli_exploratory"].append(D)

    out = {"n_items_test": int(len(items)), "entropy_cuts": cut.tolist(), "groups": {}}
    for g, Ds in groups.items():
        Dm = np.mean(np.stack(Ds), 0)                        # average over pools
        by = [float(Dm[terc == k].mean()) for k in range(3)]
        out["groups"][g] = {"n_pools": len(Ds), "D_by_tercile_low_mid_high": by}
    lo, _, hi = out["groups"]["decoder_supported"]["D_by_tercile_low_mid_high"]
    out["verdict"] = verdict(hi > 2 * lo and hi > 0, hi <= lo)
    save("h4_ambiguity", out)
    print(f"H4: {len(items)} ChaosNLI items in the test split; entropy tercile cuts {np.round(cut, 3).tolist()}")
    for g, v in out["groups"].items():
        print(f"  {g:26s} ({v['n_pools']} pools) D by tercile low/mid/high: "
              + "  ".join(f"{100*x:+.2f}pp" for x in v["D_by_tercile_low_mid_high"]))
    print(f"  -> {out['verdict']}")


if __name__ == "__main__":
    main()
