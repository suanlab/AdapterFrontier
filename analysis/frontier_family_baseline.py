#!/usr/bin/env python3
"""How much of the frontier R^2 is just "encoder or decoder"?

Post hoc. The headline frontier (accuracy vs n_rank, 48 pools) pools the two
model families, and the paper now reports that they sit on opposite sides of
zero: encoder deltas are mostly negative, decoder deltas mostly positive. A
regression that only learned which family a pool belongs to could therefore
score a respectable out-of-pool R^2 without learning anything about
diversity. That is a fair reviewer objection to the claim that diversity
predicts the delta.

This script answers it with the paper's own machinery: the same slice, the
same design-matrix builder, the same leave-one-pool-out folds and the same
ridge (`frontier.lopo_cv_r2`) as the headline number, fitted three ways:

  family only          one column: is the base model a decoder?
  frontier features    the paper's headline design matrix
  features + family    both, to see whether family adds anything on top

Output: analysis/frontier_family_baseline.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from frontier import (  # noqa: E402
    build_design_matrix_diagnostics,
    load_cm_cells,
    load_diversity,
    lopo_cv_r2,
)

OUT = ROOT / "analysis" / "frontier_family_baseline.json"
ENCODERS = ("bert", "roberta", "deberta")


def main() -> None:
    adir = ROOT / "analysis"
    diversity = load_diversity(adir)
    cells = [c for c in load_cm_cells(adir)
             if not c["pool_id"].startswith("baseline_")
             and c["metric"] == "accuracy" and c["baseline_kind"] == "n_rank"]
    diag = build_design_matrix_diagnostics(cells, diversity, include_method_onehots=True)
    X, y, used = diag["X"], diag["y"], diag["used_rows"]
    pools = [r["pool_id"] for r in used]
    fam = np.array([[0.0 if any(e in p for e in ENCODERS) else 1.0] for p in pools])

    fits = {
        "family_only": lopo_cv_r2(fam, y, pools, ridge=0.1),
        "frontier_features": lopo_cv_r2(X, y, pools, ridge=0.1),
        "features_plus_family": lopo_cv_r2(np.column_stack([X, fam]), y, pools, ridge=0.1),
    }
    r2 = {k: v.get("r2_lopo") for k, v in fits.items()}
    out = {
        "note": ("Post hoc. Same slice (accuracy vs n_rank), rows, LOPO folds and "
                 "ridge as the headline frontier R^2."),
        "n_cells": int(len(used)), "n_pools": len(set(pools)),
        "n_decoder_pools": len({p for p, f in zip(pools, fam[:, 0]) if f == 1.0}),
        "r2_lopo": {k: (round(v, 3) if v is not None else None) for k, v in r2.items()},
        "features_over_family": (round(r2["frontier_features"] - r2["family_only"], 3)
                                 if None not in (r2["frontier_features"], r2["family_only"])
                                 else None),
        "skipped_folds": {k: len(v.get("skipped_folds", [])) for k, v in fits.items()},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{out['n_cells']} cells, {out['n_pools']} pools ({out['n_decoder_pools']} decoder)")
    for k, v in out["r2_lopo"].items():
        print(f"  {k:22s} LOPO R^2 = {v:+.3f}")
    print(f"  diversity features add {out['features_over_family']:+.3f} over family alone")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
