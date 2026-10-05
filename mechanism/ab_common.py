"""Loading helpers for experiments A and B (mechanism/PREREG_AB.md).

The one rule this module exists to enforce: SNLI test logits stay sealed until
mechanism/TEST_FREEZE is committed (PREREG_AB.md §8). Every analysis reads
logits through `logits()`, which refuses the test split before the freeze.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splits import split_indices  # noqa: E402

RUNS = ROOT / "mechanism" / "runs"
FREEZE = ROOT / "mechanism" / "TEST_FREEZE"


class SealedSplitError(RuntimeError):
    pass


def logits(run: Path, epoch: int | None, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(logits, labels) of one checkpoint. `epoch=None` reads an H8 run."""
    if split == "test" and not FREEZE.exists():
        raise SealedSplitError(f"{run}: test logits are sealed until {FREEZE} exists")
    base = run if epoch is None else run / f"epoch{epoch}"
    return (np.load(base / f"logits_{split}.npy").astype(np.float64),
            np.load(base / f"labels_{split}.npy"))


def validation_slices(n: int) -> dict[str, np.ndarray]:
    """val_selection / val_combine / dev_test over SNLI validation (split seed 0)."""
    sel, comb, dev = split_indices(n, seed=0)
    return {"val_selection": sel, "val_combine": comb, "dev_test": dev}


def metrics(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text())


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
