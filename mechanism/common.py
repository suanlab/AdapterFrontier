"""Shared loaders for the mechanism study (mechanism/PREREG.md).

The headline slice is defined exactly as in analysis/family_split.py: pools
with an n_rank accuracy cell, classification only, HellaSwag/GSM8K/baselines
and the smoke test excluded.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
RESULTS = ROOT / "mechanism" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ENCODERS = ("bert", "roberta", "deberta")
TASKS = ("mnli", "qnli", "agnews", "boolq", "anli", "sst2")
CELL = re.compile(r"^(?P<pool>.+?)_cm_soft_vote_vs_n_rank$")


def family(pool: str) -> str:
    return "encoder" if any(e in pool for e in ENCODERS) else "decoder"


def task(pool: str) -> str:
    return next(t for t in TASKS if f"_{t}_" in f"{pool}_")


@lru_cache(maxsize=None)
def result(path: str) -> dict:
    """Arrays from an ensemble_results file: P, C (members x examples), y."""
    d = json.loads((ROOT / path).read_text())
    return {
        "pool_id": d["pool_id"],
        "P": np.asarray(d["individual_predictions_test"], dtype=np.int64),
        "C": np.asarray(d["individual_confidence_test"], dtype=np.float64),
        "acc": np.asarray(d["individual_accuracy_test"], dtype=np.float64),
        "y": np.asarray(d["labels_test"], dtype=np.int64),
        "test_indices": np.asarray(d["test_indices"], dtype=np.int64),
        "methods": d["methods"],
        "task": d.get("task"),
    }


def headline_pairs() -> list[dict]:
    pairs = []
    for f in sorted(ANALYSIS.glob("*_cm_soft_vote_vs_n_rank.json")):
        m = CELL.match(f.stem)
        if not m:
            continue
        pool = m["pool"]
        if ("hellaswag" in pool or "gsm8k" in pool
                or pool.startswith(("baseline", "smoke"))):
            continue
        d = json.loads(f.read_text())
        pairs.append({
            "pool": pool, "family": family(pool), "task": task(pool),
            "ensemble": d["ensemble"]["result_path"].lstrip("./"),
            "baseline": d["baseline"]["result_path"].lstrip("./"),
            "b_star_cell": d["baseline"]["accuracy_test"],
        })
    return pairs


def plurality(P: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.stack([(P == k).sum(0) for k in range(n_classes)])
    return counts.argmax(0)


def save(name: str, obj: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, default=float))
    return path


def verdict(passed: bool | None, falsified: bool | None) -> str:
    if passed:
        return "PASS"
    if falsified:
        return "FALSIFIED"
    return "PARTIAL" if passed is False else "NOT TESTABLE"
