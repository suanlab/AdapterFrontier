#!/usr/bin/env python3
"""E3: External reproducibility audit. Re-derive all paper headline numbers
from released artifacts only.

External party should be able to:
    pip install -r requirements.txt
    python3 scripts/reproduce_paper_numbers.py

…and see every claim recomputed from JSON with PASS/FAIL markers.
Runtime: < 30 seconds.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def header(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def check(name, actual, expected, tol=0.005):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        ok = abs(actual - expected) < tol
    else:
        ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:55s} expected={expected}  actual={actual}")
    return ok


def main():
    header("E3: External reproducibility audit")
    print(f"Root: {ROOT}")
    passes, fails = 0, 0
    def tally(ok):
        nonlocal passes, fails
        passes += int(ok); fails += int(not ok)

    # 1. Corpus + adjudication counts
    header("1. Corpus + adjudication (Abstract, §3)")
    cm_files = list(ROOT.glob("**/*_cm_*.json"))
    print(f"  cm files: {len(cm_files)}")
    tally(check("Total cm cells", len(cm_files), 1028, tol=2))

    # Each cell carries p_for_bh, bh_q_value and adjudication_pre/post_fdr,
    # written by scripts/apply_bh_valid_exact.py: BH per arm over the 927 valid
    # cells only (HellaSwag and the smoke test excluded), exact binomial p for
    # accuracy, ECE p capped at 1. We re-derive each arm's BH here.
    cells = []
    for fn in cm_files:
        d = json.loads(Path(fn).read_text())
        cells.append({
            "name": Path(fn).name, "arm": d.get("bh_arm"), "p": d.get("p_for_bh"),
            "pre": d.get("adjudication_pre_fdr"), "post": d.get("adjudication_post_fdr"),
            "excluded": d.get("excluded_from_multiplicity"), "batch_hash": d.get("bh_batch_hash"),
        })
    valid = [c for c in cells if not c["excluded"]]
    tally(check("cells excluded from multiplicity (HellaSwag + smoke)", len(cells) - len(valid), 101))
    tally(check("excluded cells carry verdict 'excluded'",
                all(c["post"] == "excluded" for c in cells if c["excluded"]), True))
    n = len(valid)
    tally(check("valid cells", n, 927))
    sup_pre = sum(1 for c in valid if c["pre"] == "supported")
    rev_pre = sum(1 for c in valid if c["pre"] == "reversed")
    tally(check("% SUPPORTED (pre-FDR, valid cells)", round(100*sup_pre/n, 1), 34.5, tol=0.06))
    tally(check("% REVERSED  (pre-FDR, valid cells)", round(100*rev_pre/n, 1), 14.9, tol=0.06))
    sup_post = sum(1 for c in valid if c["post"] == "supported")
    rev_post = sum(1 for c in valid if c["post"] == "reversed")
    tally(check("% SUPPORTED (post-FDR, per-arm BH)", round(100*sup_post/n, 1), 27.8, tol=0.06))
    tally(check("% REVERSED  (post-FDR, per-arm BH)", round(100*rev_post/n, 1), 13.2, tol=0.06))
    tally(check("ECE p never above 1", max(c["p"] for c in cells if c["arm"] == "ece"), 1.0, tol=1e-9)
          if max(c["p"] for c in cells if c["arm"] == "ece") <= 1 else check("ECE p never above 1", "above 1", 1.0))

    tally(check("every cell has a BH arm", sorted({c["arm"] for c in cells}), ["accuracy", "ece"]))
    for arm, want_n, want_cut in (("accuracy", 515, 0.0127), ("ece", 412, 0.0280)):
        arm_cells = [c for c in valid if c["arm"] == arm]
        ps = sorted((c["p"], c["pre"], c["post"]) for c in arm_cells)
        m, q, cutoff = len(ps), 0.05, 0.0
        for k, (pv, _, _) in enumerate(ps, 1):
            if pv <= k*q/m: cutoff = pv
        rec = sum(1 for pv, pre, _ in ps if pre in ("supported", "reversed") and pv <= cutoff)
        stored = sum(1 for _, _, post in ps if post in ("supported", "reversed"))
        tally(check(f"{arm}: valid cells in the BH batch", m, want_n))
        tally(check(f"{arm}: BH cutoff (recomputed)", round(cutoff, 4), want_cut, tol=0.0006))
        tally(check(f"{arm}: stored post-FDR significant == recomputed", stored, rec))

    # exact accuracy p for one cell, re-derived from its per-example predictions
    from scipy.stats import binomtest
    probe = "pool_a_mnli_qwen25_05b_cm_soft_vote_vs_n_rank.json"
    cd = json.loads((ROOT / "analysis" / probe).read_text())
    def correct(side):
        r = json.loads((ROOT / cd[side]["result_path"].lstrip("./")).read_text())
        return np.asarray(r["methods"][cd[side]["method"]]["predictions_test"]) == np.asarray(r["labels_test"])
    ce, cb = correct("ensemble"), correct("baseline")
    n10, n01 = int((ce & ~cb).sum()), int((~ce & cb).sum())
    tally(check("exact sign-flip p re-derived (probe cell)",
                round(binomtest(n10, n10 + n01, 0.5).pvalue, 8), round(cd["p_for_bh"], 8), tol=1e-8))

    hashes = {c["batch_hash"] for c in valid if c["batch_hash"]}
    tally(check("one bh_batch_hash per arm", len(hashes), 2))

    # 2. Frontier R²
    header("2. Frontier R² (§3, Table 5)")
    f = ROOT / "analysis/frontier_r2_cis.json"
    if f.exists():
        d = json.loads(f.read_text())
        v = d.get("accuracy__n_rank", {})
        tally(check("Corpus-wide R²_LOPO (48 pools, enc+dec)", round(v.get("r2_lopo", 0), 3), 0.597, tol=0.005))
        ci = v.get("r2_lopo_ci95", [0, 0])
        tally(check("R² CI lower", round(ci[0], 3), 0.303, tol=0.01))
        tally(check("R² CI upper", round(ci[1], 3), 0.710, tol=0.01))
        tally(check("n_cells", v.get("n_cells"), 192))
        tally(check("n_pools", v.get("n_pools"), 48))
    else:
        print(f"  SKIP: {f} not found")

    # 2b. Audit fix Critical 3: headline R² survives method-onehot ablation
    #     and emits explicit drop diagnostics (no silent row drops).
    f2 = ROOT / "analysis/frontier_with_drop_audit.json"
    if f2.exists():
        d = json.loads(f2.read_text())
        sl = d.get("by_slice", {}).get("accuracy__vs__n_rank", {})
        tally(check("Audit: headline n_rank R²", round(sl.get("r2_lopo", 0), 3), 0.597, tol=0.005))
        dd = sl.get("drop_diagnostics", {})
        tally(check("Audit: zero rows dropped on n_rank", dd.get("n_rows_dropped", -1), 0))
        ab = sl.get("ablation_no_method_onehots", {})
        delta = ab.get("r2_lopo_delta_vs_headline")
        if isinstance(delta, (int, float)):
            tally(check("Audit: method-onehot ablation |Δ R²| < 0.01",
                         abs(delta) < 0.01, True))
    else:
        print(f"  SKIP: {f2} not found")

    # 3. H(c|conf) gap
    header("3. H(c|conf) gap (§4.2, Table 6)")
    f = ROOT / "analysis/info_mechanism.json"
    if f.exists():
        d = json.loads(f.read_text())
        fs = d.get("family_summary", {})
        enc = fs.get("encoder_bertfam", {})
        dec = fs.get("decoder", {})
        tally(check("Encoder H mean", round(enc.get("mean_H_corr_given_conf_bits", 0), 3), 0.556, tol=0.01))
        tally(check("Decoder H mean", round(dec.get("mean_H_corr_given_conf_bits", 0), 3), 0.437, tol=0.01))
        gap = (enc.get("mean_H_corr_given_conf_bits", 0) - dec.get("mean_H_corr_given_conf_bits", 0)) / max(enc.get("mean_H_corr_given_conf_bits", 1), 1e-9)
        tally(check("Enc→Dec H gap %", round(100*gap, 0), 21, tol=2))

    # 4. Distillation Qwen-0.5B
    header("4. Distillation Qwen-0.5B MNLI (§5)")
    f1 = ROOT / "distilled_adapters/sweep/qwen05b_a0.9_t4/distill_metrics.json"
    f2 = ROOT / "ensemble_results/pool_a_mnli_qwen25_05b.json"
    if f1.exists() and f2.exists():
        d = json.loads(f1.read_text())
        e = json.loads(f2.read_text())
        bs = e["methods"]["best_single"]["accuracy_test"]
        sv = e["methods"]["soft_vote"]["accuracy_test"]
        ds = d["accuracy_test"]
        rec = (ds - bs) / (sv - bs) * 100
        tally(check("Distill recovery %", round(rec, 0), 87, tol=3))

    # 5. RETRACTED 2026-07-28 — the GSM8K digit-shift contamination probe was
    #    invalid (gold answer shifted with the question), so its numbers are
    #    no longer asserted here and the claim is withdrawn in the paper (L12).

    # 6. vLLM serving
    header("6. vLLM multi-LoRA (§5)")
    for model, fname, exp_ratio in [
        ("Qwen-0.5B", "analysis/vllm_bench_qwen05b.json", 1.075),
        ("Llama-3.1-8B-Inst", "analysis/vllm_bench_llama31_8b.json", 1.067),
    ]:
        f = ROOT / fname
        if f.exists():
            d = json.loads(f.read_text())
            tally(check(f"{model} ratio", round(d["multi_over_single_ratio"], 3), exp_ratio, tol=0.005))

    # 7. RETRACTED 2026-07-28 — the E1 decontaminated ensemble probe scored both
    #    arms against the invalid shifted gold labels, so its delta is an
    #    artifact and is no longer asserted (paper L12(ii)).

    # 8. Metric-general accuracy-specificity + H3 refutation (§4.2 new paragraph)
    header("8. Accuracy-specificity is metric-general; refinement not the mechanism")
    fw = ROOT / "analysis/w2_calibration_asymmetry.json"
    if fw.exists():
        d = json.loads(fw.read_text())
        def r2_of(slice_key):
            v = d.get(slice_key)
            return round(v["r2_lopo"], 2) if v and v.get("r2_lopo") is not None else None
        tally(check("accuracy n_rank R2", r2_of("accuracy__n_rank"), 0.60, tol=0.02))
        tally(check("ECE n_rank R2 (unpredictable)", r2_of("ece__n_rank"), -0.06, tol=0.03))
        tally(check("Brier n_rank R2 (unpredictable)", r2_of("brier__n_rank"), -0.30, tol=0.03))
        tally(check("NLL n_rank R2 (predictable)", r2_of("nll__n_rank"), 0.42, tol=0.03))
        tally(check("MCE n_rank R2 (predictable)", r2_of("mce__n_rank"), 0.22, tol=0.03))
    else:
        print(f"  SKIP: {fw} not found")

    fh = ROOT / "analysis/h3_mechanism.json"
    if fh.exists():
        d = json.loads(fh.read_text())
        tally(check("H3 resolution sigma / accuracy sigma",
                     round(d.get("res_std_over_acc_std", 0), 1), 0.2, tol=0.05))
        tally(check("H3 corr(resolution_gain, accuracy_gain)",
                     round(d.get("corr_resolution_accuracy", 0), 2), -0.10, tol=0.05))
        tally(check("H3 mechanism refuted (verdict_supported False)",
                     d.get("verdict_supported"), False))
    else:
        print(f"  SKIP: {fh} not found")

    # 9. Protocol sensitivity (§Protocol sensitivity, Table 1, Fig. 1)
    header("9. Protocol sensitivity: the four evaluation shortcuts")
    fp = ROOT / "analysis/protocol_sensitivity.json"
    if fp.exists():
        d = json.loads(fp.read_text())
        s1 = d["S1_selection_on_test"]
        tally(check("S1 select-on-test inflation (all, pp)",
                     round(s1["all"]["mean_inflation_pp"], 2), 1.12, tol=0.02))
        tally(check("S1 encoder inflation (pp)",
                     round(s1["encoder"]["mean_inflation_pp"], 2), 1.30, tol=0.02))
        s2 = d["S2_no_compute_match"]
        tally(check("S2 weak-baseline SUPPORTED %",
                     s2["all"]["best_of_n"]["pct_supported"], 21.9, tol=0.1))
        tally(check("S2 n_rank SUPPORTED %",
                     s2["all"]["n_rank"]["pct_supported"], 6.8, tol=0.1))
        tally(check("S2 encoder strict SUPPORTED % (headline: zero)",
                     s2["encoder"]["n_rank"]["pct_supported"], 0.0, tol=0.001))
        tally(check("S2 encoder strict REVERSED %",
                     s2["encoder"]["n_rank"]["pct_reversed"], 29.3, tol=0.1))
        tally(check("S2 encoder strict n_cells",
                     s2["encoder"]["n_rank"]["n_cells"], 92))
        tally(check("S2 encoder strict mean Δ (pp)",
                     round(s2["encoder"]["n_rank"]["mean_diff_pp"], 2), -1.27, tol=0.02))
        # every combination rule at zero SUPPORTED
        by_m = d["S2b_strict_encoder_by_method"]
        tally(check("S2b all methods at 0% SUPPORTED",
                     max(v["pct_supported"] for v in by_m.values()), 0.0, tol=0.001))
        mp = s2["matched_pools"]
        tally(check("S2 matched: pools carrying both baselines", mp["all"]["n_pools"], 48))
        tally(check("S2 matched: SUPPORTED % vs best_of_n", mp["all"]["best_of_n"]["pct_supported"], 21.9, tol=0.06))
        tally(check("S2 matched: SUPPORTED % vs n_rank", mp["all"]["n_rank"]["pct_supported"], 6.8, tol=0.06))
        tally(check("S2 matched encoder: SUP % best_of_n -> n_rank",
                     (mp["encoder"]["best_of_n"]["pct_supported"], mp["encoder"]["n_rank"]["pct_supported"]), (9.8, 0.0)))
        tally(check("S2 matched decoder: SUP % best_of_n -> n_rank",
                     (mp["decoder"]["best_of_n"]["pct_supported"], mp["decoder"]["n_rank"]["pct_supported"]), (33.0, 13.0)))
        s3 = d["S3_aggregate_reporting"]["all"]
        tally(check("S3 cells hidden in no-effect tasks", s3["significant_cells_hidden_in_neutral_tasks"], 36))
        tally(check("S3 % hidden", s3["pct_hidden_significant"], 22.0, tol=0.06))
        mnli = next(t for t in s3["tasks"] if t["task"] == "mnli")
        tally(check("S3 MNLI task mean reads as no-effect",
                     mnli["task_level_reads_as"], "no effect"))
        tally(check("S3 significant cells hidden inside MNLI",
                     mnli["reversed_cells_inside"] + mnli["supported_cells_inside"], 30))
    else:
        print(f"  SKIP: {fp} not found")


    # 10. Correction-procedure sensitivity (App. correction)
    header("10. Multiple-comparison procedure sensitivity")
    fc = ROOT / "analysis/correction_sensitivity.json"
    if fc.exists():
        d = json.loads(fc.read_text())
        acc, ece = d["accuracy"], d["ECE"]
        tally(check("accuracy SUPPORTED = 0 under ALL procedures",
                     d["summary"]["accuracy_supported_zero_under_all_procedures"], True))
        tally(check("accuracy REVERSED, raw CI", acc["raw_ci"]["reversed"], 38))
        tally(check("accuracy REVERSED, within-table Holm", acc["holm_within_table"]["reversed"], 15))
        tally(check("accuracy REVERSED, whole-arm Holm (exact p)", acc["holm_corpus_by_arm"]["reversed"], 12))
        tally(check("accuracy REVERSED, corpus BH", acc["bh_corpus_q05"]["reversed"], 27))
        tally(check("ECE SUPPORTED, corpus BH", ece["bh_corpus_q05"]["supported"], 51))
        tally(check("ECE SUPPORTED, within-table Holm", ece["holm_within_table"]["supported"], 43))
        tally(check("ECE SUPPORTED, corpus Holm (nothing survives)",
                     ece["holm_corpus_by_arm"]["supported"], 0))
    else:
        print(f"  SKIP: {fc} not found")

    # 11. Baseline budget audit (the n_rank baseline is NOT compute-matched)
    header("11. Realised baseline training budget")
    fb = ROOT / "analysis/baseline_budget_audit.json"
    if fb.exists():
        d = json.loads(fb.read_text())
        tally(check("median baseline/pool GPU-h ratio", round(d["ratio_median"], 2), 1.45, tol=0.02))
        tally(check("max ratio", round(d["ratio_max"], 2), 2.96, tol=0.02))
        tally(check("pairs where baseline got more compute", d["n_pairs_baseline_favoured"], 47))
        tally(check("total pairs", d["n_pairs"], 48))
    else:
        print(f"  SKIP: {fb} not found")

    # 11b. Does the null track that budget advantage? (§1, §4; post-hoc)
    header("11b. Headline null vs the baseline's budget advantage")
    fd = ROOT / "analysis/budget_dose_response.json"
    if fd.exists():
        d = json.loads(fd.read_text())
        np_ = d["near_parity"]
        tally(check("headline slice pools", d["n_pools"], 23))
        tally(check("headline slice cells", d["n_cells"], 92))
        tally(check("Spearman(ratio, mean delta)", d["spearman_rho_ratio_vs_mean_delta"], -0.28, tol=0.006))
        tally(check("Spearman p", d["spearman_p"], 0.20, tol=0.006))
        tally(check("near-parity (<=1.3x) pools", np_["n_pools"], 6))
        tally(check("near-parity cells", np_["n_cells"], 24))
        tally(check("near-parity SUPPORTED", np_["supported"], 0))
        tally(check("near-parity REVERSED", np_["reversed"], 8))
        tally(check("near-parity min ratio", round(np_["min_ratio"], 2), 0.90, tol=0.006))
    else:
        print(f"  SKIP: {fd} not found")


    # 11c. The decoder half of the headline comparison (abstract, §1, §4)
    header("11c. Encoder vs decoder against the same n_rank single")
    ff = ROOT / "analysis/family_split.json"
    if ff.exists():
        d = json.loads(ff.read_text())
        e, dec = d["encoder"], d["decoder"]
        tally(check("encoder acc cells", e["acc_vs_n_rank"]["n"], 92))
        tally(check("encoder acc SUPPORTED", e["acc_vs_n_rank"]["supported"], 0))
        tally(check("decoder pools", dec["n_pools"], 25))
        tally(check("decoder acc cells", dec["acc_vs_n_rank"]["n"], 100))
        tally(check("decoder acc SUPPORTED (post-FDR)", dec["acc_vs_n_rank"]["supported"], 13))
        tally(check("decoder acc REVERSED (post-FDR)", dec["acc_vs_n_rank"]["reversed"], 0))
        tally(check("decoder acc SUPPORTED (pre-FDR)", dec["acc_vs_n_rank_pre_fdr"]["supported"], 31))
        tally(check("decoder acc mean delta pp", dec["acc_mean_delta_pp"], 0.56, tol=0.006))
        tally(check("decoder SUPPORTED cells all on MNLI", dec["supported_cells_by_task"], {"mnli": 13}))
        tally(check("decoder pools with a SUPPORTED cell", dec["pools_with_a_supported_cell"], 5))
        tally(check("decoder ECE SUPPORTED", dec["ece_vs_n_rank"]["supported"], 57))
        tally(check("decoder ECE REVERSED", dec["ece_vs_n_rank"]["reversed"], 10))
        tally(check("decoder acc SUPPORTED vs best_of_n (S2)", dec["acc_vs_best_of_n"]["supported"], 33))
        tally(check("encoder pools with mean delta < 0", e["pools_mean_delta_negative"], 20))
        tally(check("decoder pools with mean delta > 0", dec["pools_mean_delta_positive"], 22))
        tally(check("encoder pool sign test p < 0.001", e["pool_sign_test_p"] < 0.001, True))
        tally(check("decoder pool sign test p < 0.001", dec["pool_sign_test_p"] < 0.001, True))
    else:
        print(f"  SKIP: {ff} not found")

    # 11d. Training health: drop comparisons whose arms mostly failed to train
    header("11d. Robustness to arms that failed to train (test-free)")
    fh = ROOT / "analysis/training_health.json"
    if fh.exists():
        d = json.loads(fh.read_text())
        tally(check("flags identical at thresholds 0.80/0.85/0.90",
                     len({tuple(v) for v in d["flagged_by_threshold"].values()}), 1))
        tally(check("Qwen-3B MNLI baseline flagged",
                     "baseline_compute_matched_mnli_qwen25_3b" in d["flagged_manifests"], True))
        tally(check("Qwen-3B MNLI baseline: 1 of 20 trained",
                     d["health"]["baseline_compute_matched_mnli_qwen25_3b"]["trained"], 1))
        k = d["decoder"]["both_arms_trained"]
        tally(check("decoder, both arms trained: cells", k["n_cells"], 96))
        tally(check("decoder, both arms trained: SUPPORTED", k["supported"], 9))
        tally(check("decoder, both arms trained: REVERSED", k["reversed"], 0))
        k = d["encoder"]["both_arms_trained"]
        tally(check("encoder, both arms trained: cells", k["n_cells"], 72))
        tally(check("encoder, both arms trained: SUPPORTED", k["supported"], 0))
        tally(check("encoder, both arms trained: % REVERSED", k["pct_reversed"], 29.2, tol=0.06))
    else:
        print(f"  SKIP: {fh} not found")

    # 11e. Is the frontier just the encoder/decoder split?
    header("11e. Frontier vs a family-only baseline (same folds)")
    fb2 = ROOT / "analysis/frontier_family_baseline.json"
    if fb2.exists():
        d = json.loads(fb2.read_text())["r2_lopo"]
        tally(check("frontier features R2 (headline)", d["frontier_features"], 0.597, tol=0.002))
        tally(check("family-only R2", d["family_only"], 0.203, tol=0.002))
        tally(check("features + family R2", d["features_plus_family"], 0.569, tol=0.002))
    else:
        print(f"  SKIP: {fb2} not found")

    # 11f. Why one temperature replaces the population (App. tempmech)
    header("11f. The ensemble as an implicit temperature (App. tempmech)")
    fm = ROOT / "analysis/temperature_mechanism.json"
    if fm.exists():
        d = json.loads(fm.read_text())
        by = {p["pool_id"]: p for p in d["pools"]}
        tally(check("tempmech pools", d["n_pools"], 6))
        tally(check("T_eq reproduces ensemble ECE within 0.01", d["teq_reproduces_ensemble_ece_within_0p01"], 5))
        tally(check("rule predicts ECE improvement", d["rule_predicts_improvement"], 5))
        a = by["pool_a_anli_roberta_base_local10"]
        tally(check("ANLI T_eq", round(a["T_eq"], 2), 1.04, tol=0.006))
        tally(check("ANLI T*", round(a["T_star"], 1), 3.8, tol=0.06))
        tally(check("ANLI ensemble ECE", round(a["ece_ensemble"], 2), 0.25, tol=0.006))
        tally(check("ANLI mean model ECE at T*", round(a["ece_mean_T_star"], 2), 0.06, tol=0.006))
        tally(check("BoolQ/MNLI T_eq in [1.3, 1.45]",
                     all(1.25 <= by[k]["T_eq"] <= 1.45 for k in ("pool_a_boolq_roberta_base_local10",
                                                                  "pool_c_mnli_roberta_base_method_mixed_local8")), True))
    else:
        print(f"  SKIP: {fm} not found")

    ft = ROOT / "analysis/temperature_control.json"
    if ft.exists():
        pools = [p["pool_id"] for p in json.loads(ft.read_text())["pools"]]
        tally(check("temperature-control pools are all encoders",
                     all(any(e in p for e in ("bert", "roberta", "deberta")) for p in pools), True))

    # 12. Diversity-feature leakage bound (L1c)
    header("12. Diversity-feature leakage check")
    fl = ROOT / "analysis/leakage_check_diversity.json"
    if fl.exists():
        d = json.loads(fl.read_text())
        sm = d["summary"]
        tally(check("disagreement relative shift < 5%",
                     sm["disagreement_rate"]["mean_relative_shift"] < 0.05, True))
        tally(check("logit-correlation relative shift < 1%",
                     sm["logit_correlation"]["mean_relative_shift"] < 0.01, True))
        tally(check("pool ordering preserved (all 3 features)",
                     all(d["pool_ordering_preserved"].values()), True))
    else:
        print(f"  SKIP: {fl} not found")


    # 13. Power / equivalence and cell independence (L1d, §4 precision)
    header("13. Precision, equivalence, and cell independence")
    fp2 = ROOT / "analysis/power_and_independence.json"
    if fp2.exists():
        d = json.loads(fp2.read_text())
        tally(check("cells excluding a +1.0pp gain", d["equivalence"]["1.0"]["excluded"], 67))
        tally(check("cells excluding a +0.5pp gain", d["equivalence"]["0.5"]["excluded"], 57))
        tally(check("median CI half-width (pp)",
                     round(d["ci_half_width_pp"]["median"], 2), 0.94, tol=0.02))
        big = d["by_test_size"]["n_test >= 2000"]
        tally(check("well-powered cells", big["n"], 68))
        tally(check("well-powered excluding +1.0pp", big["excl_1.0pp"], 62))
        mr = d["method_redundancy"]
        tally(check("soft_vote/logit_avg agree <0.1pp", mr["agree_within_0.1pp"], 20))
        tally(check("soft_vote/logit_avg bit-identical", mr["bit_identical"], 10))
    else:
        print(f"  SKIP: {fp2} not found")

    # ---- Table `tab:frontier_cross`: cross-model-class LOPO R^2.
    # Previously unasserted, and the table's numbers could not be regenerated
    # from the release at all because only ridge was ever implemented.
    header("Cross-model-class frontier (Table tab:frontier_cross)")
    fp3 = ROOT / "analysis/frontier_model_classes.json"
    if fp3.exists():
        d = json.loads(fp3.read_text())["by_slice"]
        expected = {
            "accuracy__vs__n_rank":    (192, 48, +0.597, +0.276, +0.285),
            "ece__vs__n_rank":         (192, 48, -0.061, +0.295, +0.180),
            "accuracy__vs__best_of_n": (265, 64, +0.312, +0.519, +0.365),
            "ece__vs__best_of_n":      (265, 64, +0.288, +0.416, +0.340),
        }
        for key, (nc, npool, rg, rf, gb) in expected.items():
            s = d[key]
            tally(check(f"{key} cells", s["n_cells"], nc))
            tally(check(f"{key} pools", s["n_pools"], npool))
            tally(check(f"{key} ridge", s["ridge"]["r2_lopo"], rg, tol=0.01))
            tally(check(f"{key} RF", s["random_forest"]["r2_lopo"], rf, tol=0.01))
            tally(check(f"{key} GBM", s["gradient_boosting"]["r2_lopo"], gb, tol=0.01))
        # The fairness claim in S3: ridge is the cross-class max on the headline
        # slice and on no other, so the fixed choice never buys R^2.
        tally(check("slices where ridge is cross-class max",
                     sum(s.get("ridge_is_cross_class_max", False) for s in d.values()), 1))
        tally(check("ridge is max on the headline slice",
                     d["accuracy__vs__n_rank"]["ridge_is_cross_class_max"], True))
    else:
        print(f"  SKIP: {fp3} not found")

    # ---- What the calibration gain is, and is not (S4).
    header("Calibration arm: proper scores and per-rule split")
    fp4 = ROOT / "analysis/ece_test_correction.json"
    if fp4.exists():
        d = json.loads(fp4.read_text())
        tally(check("clean encoder calibration cells", d["n_cells"], 92))
        tally(check("pools", d["n_pools"], 23))
        adj = d["ece_adjudication"]
        tally(check("ECE SUPPORTED (raw CI)", adj["raw_ci"]["supported"], 52))
        tally(check("ECE REVERSED (raw CI)", adj["raw_ci"]["reversed"], 17))
        tally(check("ECE SUPPORTED (pre-registered within-table Holm)",
                     adj["holm_within_table_prereg"]["supported"], 43))
        ps = d["proper_scores"]
        tally(check("multiclass Brier improved", ps["multiclass_brier_improved"], 34))
        tally(check("multiclass Brier mean diff",
                     round(ps["multiclass_brier_mean_diff"], 4), -0.0062, tol=0.0005))
        tally(check("NLL improved", ps["nll_improved"], 47))
        tally(check("top-label Brier SUPPORTED", ps["toplabel_brier_supported"], 22))
        tally(check("top-label Brier REVERSED", ps["toplabel_brier_reversed"], 36))
        pm = d["per_method"]
        for m, exp in (("soft_vote", 17), ("logit_avg", 16),
                       ("greedy_soup", 16), ("majority_vote", 3)):
            tally(check(f"ECE SUPPORTED {m}", pm[m]["ece_supported"], exp))
        tally(check("ECE reversals that are majority_vote",
                     d["ece_reversals_from_majority_vote"], 16))
        ex = d["excluding_majority_vote"]
        tally(check("ECE SUPPORTED excluding majority_vote", ex["ece_supported"], 49))
        tally(check("cells excluding majority_vote", ex["n_cells"], 69))
        tally(check("pools SUPPORTED on every averaging rule",
                     ex["pools_supported_on_every_rule"], 14))
    else:
        print(f"  SKIP: {fp4} not found")

    # ---- The recalibration control (title claim, S4).
    header("Temperature control on the calibration arm")
    fp5 = ROOT / "analysis/temperature_control.json"
    if fp5.exists():
        d = json.loads(fp5.read_text())
        tally(check("pools with a retained logits cache", d["n_pools"], 6))
        b = d["ensemble_better_than"]
        tally(check("ensemble beats uncalibrated single (pools)",
                     b["uncalibrated_single"]["pools"], 4))
        tally(check("ensemble beats temperature-scaled single (pools)",
                     b["temperature_scaled_single"]["pools"], 2))
        tally(check("mean ECE gain vs temperature-scaled single",
                     round(b["temperature_scaled_single"]["mean_ece_gain"], 4),
                     -0.0316, tol=0.001))
        tally(check("ensemble beats single, both arms scaled (pools)",
                     b["temperature_scaled_both_arms"]["pools"], 3))
        tally(check("median fitted temperature",
                     round(d["median_fitted_temperature"], 2), 1.32, tol=0.02))
    else:
        print(f"  SKIP: {fp5} not found")

    # ---- Table `tab:cost`: previously ungenerated and unasserted.
    header("Cost of blind ensembling (Table tab:cost)")
    fp6 = ROOT / "analysis/cost_savings.json"
    if fp6.exists():
        d = json.loads(fp6.read_text())
        expected = {   # n, blind dpp, blind %REV, blind cost, routed dpp, routed %REV, routed cost
            "Enc acc": (54, -1.0, 20, 20, +0.1, 7, 5),
            "Dec acc": (50, +0.6, 0, 18, +0.8, 0, 15),
            "Enc ECE": (53, +2.4, 8, 20, +3.6, 0, 20),
            "Dec ECE": (50, +3.5, 0, 18, +3.6, 0, 18),
            "All":     (207, +1.4, 7, 19, +2.0, 2, 17),
        }
        for row, (n, bd, br, bc, rd, rr, rc) in expected.items():
            s = d[row]
            tally(check(f"{row} n", s["n"], n))
            tally(check(f"{row} blind mean delta pp",
                         round(s["blind_mean_delta_pp"], 1), bd, tol=0.06))
            tally(check(f"{row} blind % reversed", s["blind_pct_reversed"], br))
            tally(check(f"{row} blind median cost", s["blind_median_cost"], bc))
            tally(check(f"{row} routed mean delta pp",
                         round(s["routed_mean_delta_pp"], 1), rd, tol=0.06))
            tally(check(f"{row} routed % reversed", s["routed_pct_reversed"], rr))
            tally(check(f"{row} routed median cost", s["routed_median_cost"], rc))
    else:
        print(f"  SKIP: {fp6} not found")

    # ---- Independence robustness (App. app:stats): one soft_vote cell per unit, BH per arm.
    header("Independent-group BH (App. app:stats)")
    fp7 = ROOT / "analysis/independent_group_bh.json"
    if fp7.exists():
        d = json.loads(fp7.read_text())
        tally(check("groups, accuracy arm", d["arms"]["accuracy"]["n_groups"], 129))
        tally(check("groups, ECE arm", d["arms"]["ece"]["n_groups"], 103))
        tally(check("BH cutoff, accuracy groups", round(d["arms"]["accuracy"]["bh_cutoff_p"], 4), 0.0143, tol=6e-5))
        tally(check("BH cutoff, ECE groups", round(d["arms"]["ece"]["bh_cutoff_p"], 4), 0.0356, tol=6e-5))
        tally(check("groups SUPPORTED post-FDR", d["total"]["supported"], 98))
        tally(check("groups REVERSED post-FDR", d["total"]["reversed"], 18))
        tally(check("groups SUPPORTED %", d["total"]["pct_supported"], 42.2, tol=0.06))
        tally(check("groups REVERSED %", d["total"]["pct_reversed"], 7.8, tol=0.06))
        ece_rev = [f for f in ROOT.glob("analysis/*_cm_*_ECE.json")
                   if json.loads(f.read_text())["adjudication_post_fdr"] == "reversed"]
        tally(check("ECE cells REVERSED post-FDR", len(ece_rev), 73))
        tally(check("... of which majority_vote", sum("_cm_majority_vote_" in f.name for f in ece_rev), 67))
    else:
        print(f"  SKIP: {fp7} not found")

    # ---- App. app:ecequal (per-rule ECE, post-FDR) and the corpus split behind Fig. landscape.
    header("ECE by rule (App. app:ecequal) and corpus split")
    sys.path.insert(0, str(ROOT / "analysis"))
    from protocol_sensitivity import load_accuracy_cells
    enc_pools = sorted({c["pool_id"] for c in load_accuracy_cells()
                        if c["family"] == "encoder" and c["baseline_kind"] == "n_rank"})
    def ece_verdict(pool, m):
        f = ROOT / f"analysis/{pool}_cm_{m}_vs_n_rank_ECE.json"
        return json.loads(f.read_text())["adjudication_post_fdr"]
    prob_rules = ("soft_vote", "logit_avg", "greedy_soup")
    by_rule = {m: sum(ece_verdict(p, m) == "supported" for p in enc_pools)
               for m in prob_rules + ("majority_vote",)}
    for m, n in (("soft_vote", 17), ("logit_avg", 15), ("greedy_soup", 16), ("majority_vote", 3)):
        tally(check(f"encoder ECE SUPPORTED, {m}", by_rule[m], n))
    tally(check("probability-averaging ECE SUPPORTED of 69", sum(by_rule[m] for m in prob_rules), 48))
    tally(check("pools SUPPORTED on all three averaging rules",
                sum(all(ece_verdict(p, m) == "supported" for m in prob_rules) for p in enc_pools), 14))
    names = [f.name for f in ROOT.glob("analysis/*_cm_*.json")]
    tally(check("baseline self-comparison cells", sum(n.startswith("baseline_") for n in names), 100))
    tally(check("smoke-test cells", sum(n.startswith("smoke") for n in names), 1))
    tally(check("HellaSwag main-pool cells",
                sum("hellaswag" in n and not n.startswith("baseline_") for n in names), 100))

    # ---- v1.3.r: dependence, the pre-registered equal-budget test, measured cost
    header("Dependence robustness (Sec. findings, L1d)")
    dr = json.loads((ROOT / "analysis/dependence_robustness.json").read_text())["families"]
    e, dcd = dr["encoder"], dr["decoder"]
    tally(check("encoder baseline clusters", e["n_baseline_clusters"], 10))
    tally(check("decoder baseline clusters", dcd["n_baseline_clusters"], 15))
    tally(check("encoder cluster sign (+/-)", (e["cluster_sign_test"]["positive"], e["cluster_sign_test"]["negative"]), (2, 8)))
    tally(check("encoder cluster sign p", round(e["cluster_sign_test"]["p"], 2), 0.11, tol=0.006))
    tally(check("decoder cluster sign (+/-)", (dcd["cluster_sign_test"]["positive"], dcd["cluster_sign_test"]["negative"]), (14, 1)))
    tally(check("encoder cluster mean / CI", [round(e["cluster_bootstrap_mean_delta_pp"]["mean"], 2)] +
                [round(v, 2) for v in e["cluster_bootstrap_mean_delta_pp"]["ci95"]], [-1.20, -2.30, -0.37], tol=0.006))
    tally(check("decoder cluster mean / CI", [round(dcd["cluster_bootstrap_mean_delta_pp"]["mean"], 2)] +
                [round(v, 2) for v in dcd["cluster_bootstrap_mean_delta_pp"]["ci95"]], [0.66, -0.09, 1.37], tol=0.006))
    tally(check("BY encoder accuracy (SUP, REV)", (e["BY_accuracy"]["supported"], e["BY_accuracy"]["reversed"]), (0, 17)))
    tally(check("BY decoder accuracy SUP", dcd["BY_accuracy"]["supported"], 8))
    tally(check("BY encoder ECE (SUP, REV)", (e["BY_ece"]["supported"], e["BY_ece"]["reversed"]), (44, 17)))

    header("Pre-registered equal-budget test (Sec. a3)")
    a3 = json.loads((ROOT / "mechanism/results/a3_confirmatory.json").read_text())
    tally(check("A3-P1 verdict", a3["A3_P1"]["verdict"], "REPLICATES"))
    tally(check("B-P1 verdict", a3["B_P1"]["verdict"], "REPLICATES"))
    tally(check("A3: all 12 accuracy cells SUPPORTED",
                sum(c["A3_adjudication"] == "SUPPORTED" for c in a3["cells"].values()), 12))
    tally(check("A3: all 12 NLL cells SUPPORTED",
                sum(c["B_adjudication"] == "SUPPORTED" for c in a3["cells"].values()), 12))
    accs = {k: round(100 * c["delta_acc"]["mean"], 1) for k, c in a3["cells"].items()}
    tally(check("BERT range (pp)", (min(v for k, v in accs.items() if "/bert/" in k), max(v for k, v in accs.items() if "/bert/" in k)), (-2.8, -1.0), tol=0.06))
    tally(check("Qwen range (pp)", (min(v for k, v in accs.items() if "/q05/" in k), max(v for k, v in accs.items() if "/q05/" in k)), (1.0, 3.2), tol=0.06))
    b16 = {k: round(a3["cells"][k]["dnll_calibrated"]["mean"], 3) for k in ("snli/bert/B16", "yahoo/bert/B16", "snli/q05/B16", "yahoo/q05/B16")}
    tally(check("B=16 calibrated NLL deltas", list(b16.values()), [0.063, 0.025, -0.059, -0.125], tol=0.0006))
    tally(check("replicates agree in sign everywhere",
                all(len({(r["acc_E"] - r["acc_S"]) > 0 for r in c["replicates"]}) == 1 for c in a3["cells"].values()), True))
    a3b = json.loads((ROOT / "mechanism/results/a3b_confirmatory.json").read_text())
    tally(check("A3-P2 verdict", a3b["A3_P2"]["verdict"], "REPLICATES"))
    tally(check("B-P2 verdict", a3b["B_P2"]["verdict"], "REPLICATES"))
    tb = {k: round(100 * c["delta_acc"]["mean"], 1) for k, c in a3b["cells"].items()}
    tally(check("tuned single: deltas (snli/bert, snli/q05, yahoo/bert, yahoo/q05)",
                [tb["snli/bert/B16"], tb["snli/q05/B16"], tb["yahoo/bert/B16"], tb["yahoo/q05/B16"]], [-2.6, 1.1, -1.1, 1.7], tol=0.06))
    oracle_ok = all(max(r["eval_acc_by_config"].values()) < r["acc_E"]
                    for k, c in a3b["cells"].items() if "/q05/" in k for r in c["replicates"])
    tally(check("no Qwen config reaches the ensemble even picked on test", oracle_ok, True))
    sec5 = lambda n: n.startswith("a3b_") or any(n.startswith(f"a3_{t}_{b}_") for t in ("snli", "yahoo") for b in ("bert", "q05"))
    n_runs = sum(1 for d in (ROOT / "mechanism/runs").iterdir()
                 if (d.name.startswith("a2_") or sec5(d.name)) and (d / "metrics.json").exists())
    tally(check("released runs with test logits and adapters", n_runs, 132))

    picked = {}
    for k, c in a3b["cells"].items():
        if "/q05/" in k:
            picked[k] = (round(100 * np.mean([max(r["eval_acc_by_config"].values()) for r in c["replicates"]]), 1),
                         round(100 * np.mean([r["acc_E"] for r in c["replicates"]]), 1))
    tally(check("test-picked Qwen single vs ensemble (SNLI)", picked["snli/q05/B16"], (87.3, 88.4), tol=0.06))
    tally(check("test-picked Qwen single vs ensemble (Yahoo)", picked["yahoo/q05/B16"], (73.8, 75.6), tol=0.06))
    runs_sec5 = sum(1 for d in (ROOT / "mechanism/runs").iterdir()
                    if sec5(d.name) and (d / "metrics.json").exists())
    tally(check("runs of Sec. 5 (A3 + A3b)", runs_sec5, 120))

    header("Family or size? A3c (Sec. a3)")
    a3c = json.loads((ROOT / "mechanism/results/a3c_confirmatory.json").read_text())
    tally(check("A3-P3 verdict", a3c["A3_P3"]["verdict"], "REPLICATES"))
    tally(check("B-P3 verdict", a3c["B_P3"]["verdict"], "REPLICATES"))
    tally(check("A3-P3 predicted cells (SUP, REV)", (a3c["A3_P3"]["cells"].count("SUPPORTED"), a3c["A3_P3"]["cells"].count("REVERSED")), (6, 0)))
    tally(check("B-P3 predicted cells SUPPORTED", a3c["B_P3"]["cells"].count("SUPPORTED"), 8))
    cc = {k: round(100 * v["delta_acc"]["mean"], 1) for k, v in a3c["cells"].items()}
    bl = [v for k, v in cc.items() if "/bertl/" in k and not k.endswith("B4") or k == "yahoo/bertl/B4"]
    tally(check("BERT-large range excluding the failed-run cell (pp)", (min(bl), max(bl)), (-2.0, -0.9), tol=0.06))
    tally(check("SmolLM2 Yahoo B=16 / SNLI B=8 (pp)", (cc["yahoo/smol/B16"], cc["snli/smol/B8"]), (1.1, 0.4), tol=0.06))
    failed = [d.name for d in (ROOT / "mechanism/runs").iterdir()
              if d.name.startswith("a3_") and ("_bertl_" in d.name or "_smol_" in d.name)
              and json.loads((d / "metrics.json").read_text())["checkpoints"][-1]["failed"]]
    tally(check("A3c runs failing the training-loss rule", sorted(failed), ["a3_snli_bertl_S_s261", "a3_yahoo_bertl_E_s208"]))
    tally(check("A3c lr check: all four healthy", all(not json.loads((d / "metrics.json").read_text())["checkpoints"][-1]["failed"]
                for d in (ROOT / "mechanism/runs").iterdir() if d.name.startswith("a3c_check_")), True))

    header("Tuned ensemble A3d and post-hoc decomposition (Sec. a3)")
    a3d = json.loads((ROOT / "mechanism/results/a3d_confirmatory.json").read_text())
    tally(check("A3-P4 / B-P4 verdicts", (a3d["A3_P4"]["verdict"], a3d["B_P4"]["verdict"]), ("REPLICATES", "REPLICATES")))
    dd = {k: round(100 * c["delta_acc"]["mean"], 1) for k, c in a3d["cells"].items()}
    tally(check("A3d deltas (snli/bert, yahoo/bert, snli/q05, yahoo/q05)",
                [dd["snli/bert/B16"], dd["yahoo/bert/B16"], dd["snli/q05/B16"], dd["yahoo/q05/B16"]], [-0.9, -0.5, 0.8, 1.5], tol=0.06))
    dec = json.loads((ROOT / "mechanism/results/a3_decompose_exploratory.json").read_text())["cells"]
    av = [c["averaging"][0] for c in dec.values()]
    tally(check("averaging term range, 8 cells (pp)", (round(min(av), 1), round(max(av), 1)), (0.4, 0.8), tol=0.06))
    tally(check("averaging term CI excludes 0 in all 8", all(c["averaging"][1] > 0 for c in dec.values()), True))
    enc_r = [c["recipe"][0] for k, c in dec.items() if k.split("/")[1] in ("bert", "bertl")]
    dec_r = [c["recipe"][0] for k, c in dec.items() if k.split("/")[1] in ("q05", "smol")]
    tally(check("recipe term encoders (min, max)", (round(min(enc_r), 1), round(max(enc_r), 1)), (-3.3, -1.4), tol=0.06))
    tally(check("recipe term decoders (min, max)", (round(min(dec_r), 1), round(max(dec_r), 1)), (-0.1, 2.4), tol=0.06))
    sb = json.loads((ROOT / "mechanism/results/a3_samebank_exploratory.json").read_text())["cells"]
    tally(check("same bank: E_tuned - S_bank > 0 (CI) in all 4", all(c["Etuned_minus_Sbank"]["ci95"][0] > 0 for c in sb.values()), True))
    tally(check("same bank range (pp)", (round(100 * min(c["Etuned_minus_Sbank"]["mean"] for c in sb.values()), 1),
                                         round(100 * max(c["Etuned_minus_Sbank"]["mean"] for c in sb.values()), 1)), (0.1, 0.5), tol=0.06))
    tally(check("E_tuned - S_es (snli/bert, yahoo/bert, snli/q05, yahoo/q05)",
                [round(100 * sb[k]["Etuned_minus_Ses"]["mean"], 1) for k in ("snli/bert", "yahoo/bert", "snli/q05", "yahoo/q05")],
                [-0.9, -0.2, 0.8, 1.1], tol=0.06))
    # pinned to the run families the paper reports, so later runs (A3e) do not move it
    n_runs_all = sum(1 for d in (ROOT / "mechanism/runs").iterdir()
                     if d.name.startswith(("a2_", "a3_", "a3b_", "a3c_", "a3d_", "h3a_", "h3b_", "h8_"))
                     and (d / "metrics.json").exists())
    sec5_all = sum(1 for d in (ROOT / "mechanism/runs").iterdir()
                   if d.name.startswith(("a3_", "a3b_", "a3d_")) and (d / "metrics.json").exists())
    tally(check("mechanism runs in total / runs of Sec. 5", (n_runs_all, sec5_all), (361, 264)))
    with_test = sum(1 for d in (ROOT / "mechanism/runs").iterdir()
                    if d.name.startswith(("a2_", "a3_", "a3b_", "a3c_", "a3d_"))
                    and any((d / f"epoch{e}" / "logits_test.npy").exists() for e in (2, 4)))
    tally(check("runs with released test logits (Sec. 5 + pilot + checks)", with_test, 280))

    header("Measured serving cost (App. latency)")
    xs = []
    for bb in ("bert", "q05"):
        r = json.loads((ROOT / f"mechanism/results/serve_cost_{bb}.json").read_text())["rows"]
        xs += [r["ensemble_N8"]["latency_x_vs_merged_single"], r["ensemble_N8_batched"]["latency_x_vs_merged_single"]]
    tally(check("N=8 latency range vs merged single", (round(min(xs)), round(max(xs))), (11, 18)))

    header("Six-pool temperature control with CIs (App. tempmech)")
    tc = json.loads((ROOT / "analysis/temperature_control.json").read_text())
    tally(check("both scaled: no ECE CI excludes 0",
                all(r["both_scaled_ece_gain_ci95"][0] < 0 < r["both_scaled_ece_gain_ci95"][1] for r in tc["pools"]), True))
    tally(check("both scaled: NLL favours ensemble in 6/6", sum(r["both_scaled_nll_gain"] > 0 for r in tc["pools"]), 6))
    tally(check("both scaled: NLL significant pools", sum(r["both_scaled_nll_gain_ci95"][0] > 0 for r in tc["pools"]), 3))
    tally(check("ECE gain vs scaled single without ANLI", round(tc["without_anli"]["mean_gain_vs_temp_scaled"], 4), 0.0004, tol=0.00006))

    header("Decoder temperature control, exploratory (App. tempmech)")
    dex = json.loads((ROOT / "analysis/temperature_control.json").read_text())["decoder_exploratory"]
    tally(check("decoder pools rebuilt", dex["n_pools"], 6))
    tally(check("decoder both scaled: NLL favours ensemble / significant",
                (dex["both_scaled_nll_favours_ensemble"], dex["both_scaled_nll_significant"]), (6, 6)))
    tally(check("decoder both scaled: ECE significant either way", dex["both_scaled_ece_significant_either_way"], 1))

    header("LMC barriers, standard definition (App. lmc)")
    lmc = json.loads((ROOT / "analysis/lmc_barrier_standard.json").read_text())
    tally(check("LMC barriers (BERT A, BERT C, Qwen A, DeBERTa A)",
                [round(lmc[k]["barrier_standard_mean"], 3) for k in
                 ("pool_a_mnli_bert", "pool_c_mnli_bert_method_mixed", "pool_a_mnli_qwen25_05b", "pool_a_mnli_deberta_v3_base")],
                [0.911, 0.902, 0.463, 0.016], tol=0.0006))
    tally(check("LMC pairs per pool", sorted({v["n_pairs"] for v in lmc.values()}), [5]))

    header("Frontier CI, regenerated (App. stats)")
    fb = json.loads((ROOT / "analysis/frontier_cluster_bootstrap.json").read_text())
    tally(check("frontier LOPO R^2", round(fb["r2_lopo"], 3), 0.597, tol=0.0006))
    tally(check("frontier cluster-bootstrap CI", [round(v, 2) for v in fb["r2_lopo_ci95_cluster_bootstrap"]], [-0.42, 0.75], tol=0.006))
    tally(check("leave-one-(model,task)-out R^2", round(fb["r2_leave_one_model_task_out"], 3), 0.593, tol=0.0006))
    tally(check("(model, task) groups", fb["n_model_task_groups"], 25))

    header("P_a: encoder best_of_n on the matched pools (Sec. findings)")
    mp_enc = json.loads((ROOT / "analysis/protocol_sensitivity.json").read_text())["S2_no_compute_match"]["matched_pools"]["encoder"]
    tally(check("encoder vs best_of_n SUP / REV %", (mp_enc["best_of_n"]["pct_supported"], mp_enc["best_of_n"]["pct_reversed"]), (9.8, 7.6)))

    header("Summary")
    total = passes + fails
    print(f"\n  PASS: {passes}/{total}")
    print(f"  FAIL: {fails}/{total}")
    if fails == 0:
        print("\n  All paper headline numbers reproduce from released JSONs.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
