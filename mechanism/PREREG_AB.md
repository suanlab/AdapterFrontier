# Pre-registration: budget crossover (A) and both-sides calibration (B)

Written 2026-10-03, before any run or analysis described here. Committed to
git before the first A2 trajectory starts; the commit SHA is the freeze point.
Changes after that point go to §9 with a date and a reason, and say whether
they were made before or after the affected results were seen.

Source of the design: `IDEA.md` §6 (experiments A and B), adapted to what the
H8 infrastructure (`mechanism/train_h8.py`) already does.

## 1. Questions

**A. What does a fixed training budget buy, a pool or a stronger single?**
Two estimands, kept apart (IDEA.md §6A):

- *Reuse*: the pool is already trained (sunk cost). Compare its
  validation-selected member against subset/full ensembles of it.
- *Train under budget*: nothing is trained yet. Compare a single pipeline and
  an ensemble pipeline under the same total training budget B, counting every
  run each pipeline needed, including its search and failed runs.

The paper's `n_rank` baseline answers neither cleanly. It is kept as a
separate historical arm and is not re-labelled "compute-matched".

**B. Once both sides are calibrated, does the ensemble keep any advantage?**
In probability quality (NLL, Brier, ECE) and in error ranking (selective
prediction), against a single adapter given the same post-hoc freedom.

## 2. Data and splits

- Task for the pilot: SNLI. Train on the first 20,000 rows of `train` with
  label ≥ 0, as in H8.
- `validation` (label ≥ 0, 9,842 rows) is split 40/20/40 by
  `splits.split_indices` (split seed 0) into `val_selection`, `val_combine`
  and `dev_test`. H8 already reported accuracy on this split, so **every
  result on `dev_test` is exploratory**.
- `test` (label ≥ 0) is **sealed**. A2 and later runs write its logits to
  `logits_test.npy`, but no analysis script may read them until §8's freeze
  file exists. Before writing this document no project run had produced SNLI
  test predictions (H8 dumps validation only; the corpus has no SNLI pools).
- Second task, for expansion only: non-NLI, with an official test split of at
  least 5,000 examples that the corpus never used. It is chosen and recorded
  in §9 before any of its runs start. Candidates: `yahoo_answers_topics`,
  `dbpedia_14`.

Selection (members, best single, greedy subsets) uses only `val_selection`.
Temperatures, confidence choices and selective thresholds use only
`val_combine`.

## 3. Experiment A2: rank × epoch pilot

| Factor | Levels |
|---|---|
| backbone | `bert-base-uncased` (bs 32), `Qwen/Qwen2.5-0.5B` (bs 16) |
| LoRA rank r | 8, 32, with `lora_alpha = 2r` in both, so α/r is matched (H8's baseline used α/r = 1) |
| schedule | one 4-epoch run per trajectory; linear decay, 6% warmup, lr 3e-4 |
| checkpoints | end of epoch 2 and epoch 4 of the same trajectory |
| seeds | 101, 102, 103 (not used anywhere in the project before) |

That is 2 × 2 × 3 = **12 trajectories, 24 checkpoints**. Everything else is
the H8 recipe: target modules, dropout 0, bf16, max length 128, weight decay
0.01, deterministic flags.

An epoch-2 checkpoint of a 4-epoch schedule has not finished its learning-rate
decay, so it is **not** the 2-epoch recipe. It is reported as "mid-trajectory"
and never compared with H8's 2-epoch pool recipe as though it were.

Recorded per checkpoint: logits on all of `validation` and `test`, the LoRA
adapter weights, wall-clock GPU seconds since the trajectory started, train
tokens processed, mean train loss of the last logged window, and the health
flag (train loss ≥ 0.85·ln 3 = failed, as in the paper).

**Cost unit.** The GPUs are shared with unrelated jobs, so wall-clock seconds
are noisy and are reported only as a secondary measure. The primary cost is
*processed train tokens*. Within a backbone every arm uses the same base
model, the same data and the same sequence lengths, and LoRA rank changes the
FLOPs by well under 1%, so tokens are proportional to training FLOPs. One
r = 32 epoch and one r = 8 epoch therefore cost the same. That is the point:
the budget advantage of the paper's `n_rank` single came from its search over
seeds and epochs, not from rank.

### A2 analysis (descriptive; decides only whether to expand)

Per backbone, on `dev_test`:

1. Accuracy of each (r, checkpoint) cell, mean ± SD over seeds.
2. The paper's decomposition Δ = D − G − S for each pairing of an ensemble
   arm (3 members of one (r, checkpoint) cell, soft vote) with a single arm
   (the `val_selection`-best of 3 seeds of another cell).
3. Cost of every arm in processed train tokens (GPU seconds alongside),
   including every seed the arm selected over.
4. Equal-cost pairings in the pilot: the ensemble of three r = 8 epoch-2
   checkpoints against the best of three r = 32 epoch-2 checkpoints, and the
   same at epoch 4. Their measured cost ratio is reported next to them; the
   pilot does not claim they are matched.

**Expansion rule.** Expand to A3 if the pipeline works: every logit file
aligns with its labels, no trajectory is lost, and the token count of every
checkpoint equals its epochs times the per-epoch count. The rule does not
depend on which arm wins.

## 4. Experiment A3: end-to-end equal-budget comparison (after the pilot)

Three budget levels, in processed train tokens per backbone, are set in §9
from the pilot before any A3 run. Each pipeline gets budget B, and its cost
includes every run it trains, including search, failed runs and retries. GPU
seconds are reported alongside. A pipeline may not look at candidates outside its budget.
Performance is evaluated once on the sealed `test`. The tolerance on matching
(±5% was proposed) is fixed in §9 after the pilot, before A3 results exist.

## 5. Experiment B1: calibrate both sides

Four arms on the same pool:

| arm | definition |
|---|---|
| S | the `val_selection`-best single (by accuracy) |
| S+TS | S with one temperature on its logits |
| E | probability ensemble: mean of member softmaxes |
| E+TS | E with a power temperature, q_T(c) ∝ p_E(c)^(1/T) |

T is chosen to minimise NLL on `val_combine` over a fixed grid of 200
log-spaced values in [0.05, 20]. If the minimum falls at a grid edge, that is
reported. Two further arms are reported but are not primary: per-member TS
before averaging, and TS of the mean-logit ensemble.

- **Primary metric**: paired NLL difference, E+TS minus S+TS, on the reporting
  split, with a 95% paired percentile bootstrap CI (B = 5,000).
- **Secondary**: multiclass Brier, ECE (15 equal-mass bins; also 10 and 20
  bins as a sensitivity check), accuracy.
- **Unit**: one pool, i.e. one backbone and recipe. No pooling across
  backbones beyond counting pools by sign.
- **Non-inferiority**: we say "a calibrated single is enough" only if the
  upper CI bound of NLL(S+TS) − NLL(E+TS) is below δ_NLL = 0.01 nats. A
  non-significant difference alone is not reported as equivalence.

**Pilot data**: the four H8 backbones. The ensemble is the six pool-recipe
seeds; the single is the best of the six baseline-recipe seeds. This is
exploratory because `dev_test` has been seen. The six cached corpus pools
(App. tempmech) are reported alongside, also exploratory.
**Confirmatory data**: A2/A3 runs on the sealed SNLI `test` and on the second
task's test, after §8.

## 6. Experiment B2: selective prediction against a strong single

Confidence functions, offered to both sides:

- maximum softmax probability (MSP);
- MSP after the B1 temperature;
- top-2 margin of the (calibrated) probabilities;
- top-2 logit margin (single only; for the ensemble, the mean logits).

Each side's confidence function is chosen on `val_combine` by AURC.

Metrics:
- AURC and the full risk–coverage curve;
- risk at 80/90/95% coverage, with the threshold set on `val_combine` and the
  realised test coverage reported next to the risk.

- **Primary**: paired risk difference at 90% coverage, bootstrap CI.
- **Non-inferiority margin**: δ_risk = 0.5 pp.

E-AURC alone is not read as information independent of accuracy (Traub et
al., 2024).

## 7. Common rules

- No missing probability vector is reconstructed or imputed. Pools without
  full vectors are out of B.
- Negative and null results are reported with effect sizes and CIs.
- Every result JSON records the git SHA, the run directories used and the
  split hash.
- Results that disagree with a hypothesis do not cause it to be re-specified;
  re-specification goes to §9 as an amendment and is marked post hoc.

## 8. Freeze procedure for sealed test data

Before any script reads `logits_test.npy`:

1. The A3 budget levels and the second task are recorded in §9.
2. The analysis scripts that will read the test logits are committed.
3. A file `mechanism/TEST_FREEZE` is committed containing that commit's SHA.

The loading helper refuses to open `logits_test.npy` unless
`mechanism/TEST_FREEZE` exists.

## 9. Amendments

### 9.1 A3 design, second task and confirmatory predictions (2026-10-03)

Written after the 12 A2 trajectories finished but **before `a2_pilot.py` was
run on them**, so no A2 effect size had been seen. No A3 run has started.
Sealed test logits have not been read.

**Second task.** `yahoo_answers_topics` (10 topics). The project has never
used it. Train: the first 20,000 rows of `train.shuffle(seed=0)`.
Validation: the next 10,000 rows of that shuffle, split 40/20/40 by
`split_indices` like SNLI's. Test: the official 60,000-row test split,
sealed. Input: question title and content as the first segment, best answer
as the second, max length 128.

**Budget unit.** One epoch over the task's 20,000 training rows. Within a
task every epoch processes the same tokens, so budgets match exactly in
tokens (tolerance 0%). Budget levels: **B ∈ {4, 8, 16}** epoch-units per
pipeline.

**Pipelines** (backbones and batch sizes as in §3; lr 3e-4, alpha = 2r):

| pipeline | trains | combines |
|---|---|---|
| E(B), ensemble | N = B/2 adapters, r = 8, full 2-epoch schedule | soft vote over all N, no selection (primary); greedy subset on `val_selection` (secondary) |
| S(B), single | K = B/4 adapters, r = 32, full 4-epoch schedule | the `val_selection`-best of the K (at B = 4, K = 1, so there is no selection) |

E mirrors the corpus pool recipe and S the `n_rank` recipe, now at equal
budget. Rank × epoch is A2's question and is not crossed again here.

**Replicates.** Two, with disjoint seeds: replicate 1 uses seeds 201–208
(E members) and 251–254 (S candidates); replicate 2 uses 211–218 and 261–264.
Within a replicate, budgets are nested (E(4) uses the first 2 members, E(8)
the first 4; S(8) the first 2 candidates).

Run count: 16 E-member runs and 8 S runs per backbone and task (both
replicates), so 24 × 2 backbones × 2 tasks = 96 runs.

**Primary outcome.** Δ(B) = acc(E(B)) − acc(S(B)) on the sealed test, per
(backbone, task, B, replicate), with a paired bootstrap CI over test examples
(B = 5,000). A cell is adjudicated on the mean of its two replicates' Δ, with
a CI from resampling test examples jointly for both replicates:

- SUPPORTED if the CI excludes 0 in the predicted direction;
- REVERSED if it excludes 0 in the other direction;
- unsupported otherwise.

**Prediction A3-P1** (from the corpus split and H9): Δ(B) < 0 for BERT-base
and Δ(B) > 0 for Qwen-0.5B, on both tasks, at B = 8 and B = 16. B = 4 is
reported without a prediction, because S has no selection there.
- *Replicates* if none of the 8 predicted cells is REVERSED and at least 4
  are SUPPORTED.
- *Fails* if any predicted cell is REVERSED.
- Otherwise inconclusive.

**Prediction B-P1** (from the B1 pilot): at B = 8 and B = 16,
NLL(E+TS) − NLL(S+TS) < 0 for Qwen-0.5B and ≥ 0 for BERT-base on both tasks,
with §5's definitions and the same adjudication and pass rule. Risk at 90%
coverage (§6) is reported as secondary.

**Freeze.** `a3_analysis.py` is committed before `mechanism/TEST_FREEZE`;
only then is any test logit read (§8).

### 9.2 A3b: a single that tunes its recipe inside the budget (2026-10-05)

Written after the A3 confirmatory results (§9.1, `ab100b5`) were seen. It
answers the obvious objection to them: S used one fixed recipe (r = 32, lr
3e-4), which on Qwen may simply be mistuned (A2 already showed r = 32 below
r = 8 there). This amendment and its analysis script are committed **before
any A3b run starts**. Because `mechanism/TEST_FREEZE` already exists, that
ordering is the only protection, and it is stated here so it can be checked
against the git log.

**Pipeline.** S_tuned(16) trains four configurations, one seed each, all with
a full 4-epoch schedule and alpha = 2r, and keeps the `val_selection`-best:

| config | r | lr | replicate-1 seed | replicate-2 seed |
|---|---|---|---|---|
| c1 | 8 | 1e-4 | 301 | 311 |
| c2 | 8 | 3e-4 | 302 | 312 |
| c3 | 32 | 1e-4 | 303 | 313 |
| c4 | 32 | 3e-4 | existing S run s251 | existing S run s261 |

Its cost is 4 × 4 = 16 epoch-units, the same as E(16) in §9.1, which it is
compared against: the same eight members per replicate, unchanged. c4 reuses
the §9.1 S runs because they are exactly that configuration. New runs: 3
configs × 2 replicates × 2 backbones × 2 tasks = 24.

**Outcomes.** As in §9.1:
- Delta = acc(E(16)) − acc(S_tuned(16)) on the test set;
- NLL(E+TS) − NLL(S_tuned+TS), with temperatures fitted on val_combine;
- the replicate mean, with a joint CI over test examples, adjudicated
  SUPPORTED / REVERSED / unsupported.

The selected configuration per replicate is reported.

**Predictions.**
- A3-P2: Delta < 0 for BERT-base and Delta > 0 for Qwen-0.5B on both tasks
  (4 cells).
- B-P2: the calibrated NLL difference has the same signs as in B-P1.

For each prediction:
- *Replicates* if no cell is REVERSED and at least 3 of 4 are SUPPORTED.
- *Fails* if any cell is REVERSED.
- Otherwise inconclusive.

If A3-P2 fails for Qwen, the §9.1 Qwen result is reported as an artefact of
the fixed recipe, in the body and not only in an appendix.

**Not done here.** A tuned *ensemble* (the same search applied to members) is
not run. The question is whether the single's recipe explains §9.1, not which
pipeline is optimal.

### 9.3 Errata (2026-10-05, after all results)

Two statements above are imprecise. Neither changes a result.

- §9.1 says it was "written after the 12 A2 trajectories finished". The
  commit (`fcb98c4`, 23:43:01) came three minutes before the last A2
  trajectory finished (23:46:06). The substantive claim holds: no A2 effect
  size had been computed (`a2_pilot.json` was written at 23:46:20).
- §8 says `TEST_FREEZE` holds the SHA of the commit with the analysis
  scripts. It holds the SHA of the commit that added the runs (`34caa62`).
  The analysis scripts were committed earlier (`ad1fe51`) and are unchanged
  since; the file says so.

One A3b run (`a3b_yahoo_q05_c1_s301`) was killed by an unrelated job on the
shared GPU at 19% of training. It was rerun with the same seed and
configuration, and no partial output was used (`6cdf327`).
