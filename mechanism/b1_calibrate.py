#!/usr/bin/env python3
"""Experiments B1 and B2 (mechanism/PREREG_AB.md §5, §6) on H8 pools.

B1 asks whether the probability ensemble keeps any calibration advantage once
the selected single gets the same post-hoc freedom, one temperature fitted on
val_combine. B2 asks the same of error ranking (selective prediction), with
each side free to pick its confidence function on val_combine.

Pilot run, so every number is EXPLORATORY: H8 already reported accuracy on
this dev_test slice of SNLI validation. The confirmatory run uses the sealed
SNLI test of the A2/A3 trajectories after mechanism/TEST_FREEZE.

Per backbone the ensemble is the six H8 pool-recipe seeds (r=8, 2 epochs), and
the single is the val_selection-best of the six baseline-recipe seeds (r=32,
4 epochs), the n_rank analogue.

Usage: python3 mechanism/b1_calibrate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from ab_common import RUNS, logits, softmax, validation_slices  # noqa: E402
from compute_match import _ece_equal_mass  # noqa: E402

BACKBONES = {"bert": "bert-base-uncased", "q05": "Qwen2.5-0.5B",
             "q15": "Qwen2.5-1.5B", "q3": "Qwen2.5-3B"}
SEEDS = (11, 21, 31, 41, 51, 61)
T_GRID = np.logspace(np.log10(0.05), np.log10(20), 200)
B = 5000
COVERAGES = (0.80, 0.90, 0.95)
DELTA_NLL = 0.01
DELTA_RISK = 0.005
OUT = Path(__file__).resolve().parent / "results" / "b1_h8_pilot.json"
EPS = 1e-12


# ---- probability transforms --------------------------------------------------

def power_T(p: np.ndarray, T: float) -> np.ndarray:
    """q_T(c) ∝ p(c)^(1/T): the temperature of a probability vector."""
    return softmax(np.log(np.clip(p, EPS, 1.0)), T)


def nll_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def fit_T(score_fn, y: np.ndarray) -> tuple[float, bool]:
    """Grid temperature minimising mean NLL; also reports a grid-edge hit."""
    losses = [nll_vec(score_fn(T), y).mean() for T in T_GRID]
    i = int(np.argmin(losses))
    return float(T_GRID[i]), i in (0, len(T_GRID) - 1)


# ---- metrics -----------------------------------------------------------------

def brier_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.eye(p.shape[1])[y]
    return ((p - onehot) ** 2).sum(-1)


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    conf, correct = p.max(-1), (p.argmax(-1) == y).astype(float)
    return {"accuracy": float(correct.mean()), "nll": float(nll_vec(p, y).mean()),
            "brier": float(brier_vec(p, y).mean()),
            **{f"ece{k}": float(_ece_equal_mass(conf, correct, n_bins=k)) for k in (15, 10, 20)}}


def paired_ci(diff: np.ndarray, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    n = len(diff)
    boots = np.array([diff[rng.randint(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)]}


# ---- selective prediction ----------------------------------------------------

def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf, kind="stable")
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(conf) + 1)
    return float(risk.mean())


def threshold_for(conf: np.ndarray, coverage: float) -> float:
    """Smallest confidence kept when the top `coverage` share is accepted."""
    return float(np.quantile(conf, 1 - coverage))


def confidences(p_raw: np.ndarray, p_cal: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    def margin(a):
        s = np.sort(a, -1)
        return s[:, -1] - s[:, -2]
    return {"msp": p_raw.max(-1), "msp_ts": p_cal.max(-1),
            "prob_margin_ts": margin(p_cal), "logit_margin": margin(z)}


# ---- one backbone ------------------------------------------------------------

def load(kind: str, tag: str) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    zs, names, y = [], [], None
    for s in SEEDS:
        run = RUNS / f"h8_{tag}_{kind}_s{s}"
        if not (run / "metrics.json").exists():
            continue
        z, lab = logits(run, None, "validation")
        if y is not None and not np.array_equal(y, lab):
            raise SystemExit(f"{run}: labels do not align")
        y = lab; zs.append(z); names.append(run.name)
    return zs, y, names


def backbone(tag: str) -> dict | None:
    pool, y, pool_names = load("pool", tag)
    base, y2, base_names = load("base", tag)
    if len(pool) < len(SEEDS) or len(base) < len(SEEDS):
        return {"status": "incomplete", "n_pool": len(pool), "n_base": len(base)}
    if not np.array_equal(y, y2):
        raise SystemExit(f"{tag}: pool and baseline labels differ")
    sl = validation_slices(len(y))
    sel, comb, dev = sl["val_selection"], sl["val_combine"], sl["dev_test"]

    # single: val_selection-best baseline seed (accuracy)
    sel_acc = [float((z[sel].argmax(-1) == y[sel]).mean()) for z in base]
    k = int(np.argmax(sel_acc))
    z_s = base[k]
    T_s, edge_s = fit_T(lambda T: softmax(z_s[comb], T), y[comb])

    # ensemble: mean of member softmaxes; power temperature on top
    p_e = np.mean([softmax(z) for z in pool], axis=0)
    T_e, edge_e = fit_T(lambda T: power_T(p_e[comb], T), y[comb])
    # secondary arms
    T_m = [fit_T(lambda T, z=z: softmax(z[comb], T), y[comb])[0] for z in pool]
    p_mts = np.mean([softmax(z, t) for z, t in zip(pool, T_m)], axis=0)
    z_bar = np.mean(pool, axis=0)
    T_l, _ = fit_T(lambda T: softmax(z_bar[comb], T), y[comb])

    arms = {"S": softmax(z_s), "S+TS": softmax(z_s, T_s),
            "E": p_e, "E+TS": power_T(p_e, T_e),
            "E_memberTS": p_mts, "E_meanlogit+TS": softmax(z_bar, T_l)}
    yd = y[dev]
    out = {"status": "ok", "single_run": base_names[k], "single_val_selection_acc": sel_acc[k],
           "pool_runs": pool_names,
           "temperatures": {"S": T_s, "E_power": T_e, "E_meanlogit": T_l, "members": T_m},
           "grid_edge_hit": {"S": edge_s, "E": edge_e},
           "arms_dev_test": {a: summary(p[dev], yd) for a, p in arms.items()}}

    # B1 primary: NLL(E+TS) - NLL(S+TS), negative = ensemble better
    d_nll = nll_vec(arms["E+TS"][dev], yd) - nll_vec(arms["S+TS"][dev], yd)
    ci = paired_ci(d_nll, seed=zlib.crc32(tag.encode()) % (2**31))
    out["B1_primary_nll_EminusS_calibrated"] = ci
    out["B1_single_noninferior"] = bool(-ci["ci95"][0] < DELTA_NLL)  # NLL(S+TS)-NLL(E+TS) upper bound
    out["B1_uncalibrated_nll_EminusS"] = paired_ci(
        nll_vec(arms["E"][dev], yd) - nll_vec(arms["S"][dev], yd), seed=1 + zlib.crc32(tag.encode()) % (2**31))
    d_brier = brier_vec(arms["E+TS"][dev], yd) - brier_vec(arms["S+TS"][dev], yd)
    out["B1_brier_EminusS_calibrated"] = paired_ci(d_brier, seed=2 + zlib.crc32(tag.encode()) % (2**31))

    # B2: each side picks its confidence on val_combine by AURC
    sides = {"S": confidences(arms["S"], arms["S+TS"], z_s),
             "E": confidences(arms["E"], arms["E+TS"], z_bar)}
    correct = {"S": (arms["S"].argmax(-1) == y).astype(float),
               "E": (arms["E"].argmax(-1) == y).astype(float)}
    b2 = {}
    for side, confs in sides.items():
        picks = {name: aurc(c[comb], correct[side][comb]) for name, c in confs.items()}
        best = min(picks, key=picks.get)
        c = confs[best]
        cov = {}
        for target in COVERAGES:
            thr = threshold_for(c[comb], target)
            keep = c[dev] >= thr
            cov[str(target)] = {"threshold": thr, "realised_coverage": float(keep.mean()),
                                "risk": float(1 - correct[side][dev][keep].mean())}
        b2[side] = {"confidence": best, "aurc_val_combine": picks,
                    "aurc_dev_test": aurc(c[dev], correct[side][dev]), "coverage": cov,
                    "_keep90": c[dev] >= threshold_for(c[comb], 0.90)}
    # primary: paired risk difference at 90% coverage (E - S); positive = ensemble worse
    rng = np.random.RandomState(3 + zlib.crc32(tag.encode()) % (2**31))
    kS, kE = b2["S"].pop("_keep90"), b2["E"].pop("_keep90")
    eS, eE = 1 - correct["S"][dev], 1 - correct["E"][dev]
    n = len(dev)
    boots = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        boots.append(eE[i][kE[i]].mean() - eS[i][kS[i]].mean())
    obs = eE[kE].mean() - eS[kS].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    b2["primary_risk90_EminusS"] = {"mean": float(obs), "ci95": [float(lo), float(hi)]}
    # risk(S) - risk(E) has upper bound -lo; the single is non-inferior if that is below delta
    b2["single_noninferior"] = bool(-lo < DELTA_RISK)
    out["B2"] = b2
    return out


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"_note": ("EXPLORATORY pilot of PREREG_AB.md B1/B2 on H8 SNLI pools; dev_test was "
                     "already seen for accuracy in H8. Negative differences favour the ensemble."),
           "git_sha": sha, "delta_nll": DELTA_NLL, "delta_risk": DELTA_RISK, "backbones": {}}
    for tag, name in BACKBONES.items():
        r = backbone(tag)
        res["backbones"][name] = r
        if r["status"] != "ok":
            print(f"{name:14s} {r}")
            continue
        a = r["arms_dev_test"]
        p = r["B1_primary_nll_EminusS_calibrated"]
        q = r["B2"]["primary_risk90_EminusS"]
        print(f"{name:14s} NLL S {a['S']['nll']:.4f} S+TS {a['S+TS']['nll']:.4f} "
              f"E {a['E']['nll']:.4f} E+TS {a['E+TS']['nll']:.4f} | "
              f"dNLL(E+TS - S+TS) {p['mean']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}] | "
              f"acc S {a['S']['accuracy']:.4f} E {a['E']['accuracy']:.4f} | "
              f"risk@90 E-S {q['mean']:+.4f} [{q['ci95'][0]:+.4f},{q['ci95'][1]:+.4f}]")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
