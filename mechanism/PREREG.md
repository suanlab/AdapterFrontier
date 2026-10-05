# Mechanism study: pre-registration

**Status:** written and committed before any analysis below was run, except
where §0 says otherwise. The commit that adds this file is the registration;
`git log -- mechanism/PREREG.md` dates it. Changes after that commit go in §9,
with a date and a reason, and never overwrite the original text.

**Relation to the paper.** The paper (`paper/emnlp.tex`) audits *whether* PEFT
ensembling beats a stronger single adapter. This study asks *why* the answer
splits by model family and what a population buys that a scalar cannot. It
uses the paper's released cells and per-example predictions, plus a small set
of new Qwen-2.5-0.5B MNLI runs. Nothing here changes a paper number.

---

## 0. What was already seen (honest disclosure)

On 2026-10-02, before this file existed, an exploratory pass computed:

- For the six pools with a logits cache: the label-free equivalent temperature
  `T_eq`, the mean across-member logit variance `s2`, the residual KL at `T_eq`
  and at `T=1`, and the within-pool Spearman correlation between per-example
  `s2_i` and per-example confidence shrinkage. Result: KL(T_eq) < KL(1) in 6/6,
  and the correlation was positive in 6/6 (+0.19 to +0.94). **H1(a) and H1(d)
  are therefore not confirmatory.** H1(b) and H1(c) were not computed.
- On MNLI, pool_a only: within-pool and cross-backbone error correlations, and
  an *unweighted* heterogeneous majority vote (one median member per backbone,
  and top-k random members). The unweighted vote did not beat the best single
  member. **H6's unweighted arm is therefore not confirmatory.** The weighted
  arm, and QNLI and AG News, were not computed.
- No other hypothesis below has been examined in any form.

## 1. Data

- **Headline slice (H2, H5, H7, P1):** the 48 pools with an `n_rank` baseline,
  classification only (HellaSwag, GSM8K excluded), as in `analysis/family_split.py`:
  23 encoder pools, 25 decoder pools. Per-example test predictions from
  `ensemble_results/<pool>.json` and `ensemble_results/<baseline>.json`.
  Test split: `split_indices(n, seed=0)`, as in the paper.
- **Cached logits (H1):** the six `ensemble_results/pilot_c1/*.json` pools with
  `ensemble_cache/*.npy`, as in `analysis/temperature_control.py`.
- **New runs (H3a, H3b):** Qwen-2.5-0.5B, MNLI, the exact recipe of
  `sweep_configs/pool_a_mnli_qwen25_05b.json` (40k train, r=8, alpha=16,
  lr 3e-4, bs 16, seq 128, LoRA on q,v, bf16), with the epoch count and the
  three random sources varied as specified below. Logits are dumped for the full
  `validation_matched` (n=9815); all evaluation uses the paper's test split.

## 2. Definitions

For a pool with members `i = 1..M` on test examples `j`:

- `a_i` member accuracy; `abar` mean member accuracy.
- `a_E` ensemble accuracy for a given combination rule.
- **D** (diversity gain) = `a_E - abar`.
- **delta** (label-free disagreement) = mean over `j` of the fraction of members
  whose prediction differs from the plurality vote.
- For the `n_rank` baseline with candidates `k`: `bbar` mean candidate accuracy,
  `b*` the accuracy of the adapter the paper selected.
- **G** (quality gap) = `bbar - abar`. **S** (selection gain) = `b* - bbar`.
- **Delta** = `a_E - b*` = D - G - S (an identity).
- **rho** = mean pairwise Pearson correlation of members' 0/1 error vectors.
- **T_eq** = argmin_T KL( mean_i softmax(z_i) || softmax(zbar / T) ), class-
  centred logits, no labels used. **T*** = NLL-optimal temperature for the
  mean-logit model, fitted on `val_combine` (as in temperature_control.py).
- **Failed adapter**: final training loss >= 0.85 ln C (as in training_health.py).

## 3. Hypotheses and falsification criteria

Each hypothesis states the prediction, what would falsify it, and the data.
"Pass" requires every listed criterion.

### H1 — A population's calibration effect is an implicit temperature
- (a) *[not confirmatory, §0]* KL(T_eq) <= 0.5 KL(1) in >= 5/6 pools.
- (b) ECE of softmax(zbar / T_eq) is within 0.01 of the ensemble's ECE in
  >= 5/6 pools (15-bin equal-mass, test split).
- (c) The ensemble improves on the mean-logit model at T=1 exactly when T_eq
  lies between 1 and 2T*-1 on the log scale, i.e. when |ln T* - ln T_eq| <
  |ln T*|. Prediction: the sign of the ECE change matches this rule in >= 5/6.
- (d) *[not confirmatory, §0]* within-pool Spearman(s2_i, shrink_i) > 0.3 in
  >= 5/6.
- **Falsified** if (b) or (c) fails.

### H2 — One parameter predicts the diversity gain
- Model: D ~ kappa * delta, kappa fitted by leave-one-pool-out, separately for
  `soft_vote` and `majority_vote`, over the 48 headline pools.
- Predictions: kappa in (0, 1]; LOPO R^2 for D >= 0.5; and the sign of
  Deltahat = kappa*delta - G - S matches the sign of the observed Delta in
  >= 80% of pool x rule cells.
- **Falsified** if LOPO R^2 < 0.3 or sign accuracy < 70%.
- Note: G and S are computed from member accuracies, so sign accuracy measures
  how well kappa*delta stands in for D, not an independent forecast.

### H3a — Decoder diversity comes from under-convergence (new runs)
- Qwen-0.5B MNLI, 5 members per arm, all three random sources varied, at
  1, 2 and 4 epochs.
- Prediction: delta and D both decrease from 1 to 4 epochs.
- **Falsified** if D(4 ep) >= D(1 ep).

### H3b — Where the diversity comes from (new runs)
- Qwen-0.5B MNLI, 2 epochs, 5 members per arm. Arms: vary only the
  classification-head seed; only the LoRA-initialisation seed; only the
  data-order seed; all three; none (3 repeats, the GPU-nondeterminism floor).
- Prediction (assumption 5 of the plan): the head-only arm produces at least 50%
  of the all-vary arm's delta.
- **Falsified** if head-only delta < 25% of all-vary delta. Results between 25%
  and 50% are reported as partial.

### H4 — Gains concentrate on ambiguous items (if ChaosNLI can be aligned)
- MNLI test items that appear in ChaosNLI, split into terciles of human-label
  entropy. D for `soft_vote`, pooled over the decoder MNLI pools with a
  SUPPORTED cell.
- Prediction: D(top tercile) > 2 x D(bottom tercile).
- **Falsified** if D(top) <= D(bottom). If fewer than 300 test items align, H4 is
  reported as not testable rather than run.

### H5 — The baseline's edge is selection pressure
- Cross-fit: split the test set in half at random (200 repeats). Select the best
  of M random baseline candidates on half A; measure Delta on half B, for
  M in {1, 2, 4, 8, 16} (capped at the number of candidates).
- Predictions: Delta(M) is non-increasing in M for >= 80% of pools (soft_vote);
  and S_pred(M) = sigma_B sqrt(2 ln M) * lambda, lambda = sigma_B^2 /
  (sigma_B^2 + a(1-a)/n_A), explains >= 50% of the variance of the observed
  S(M) over pools x M.
- **Falsified** if either fails.

### H6 — Heterogeneous populations need quality weighting
- MNLI, QNLI, AG News (>= 3 backbones each). Cross-fit as in H5. One median
  member per backbone; weights w = ln(a/(1-a)) estimated on half A (the
  Nitzan–Paroush optimal weighting for independent voters); weighted vote
  evaluated on half B, against the best single member of any backbone selected
  on half A.
- Prediction: the weighted heterogeneous vote >= best single in >= 2 of 3 tasks.
- **Falsified** if it loses on all 3.

### H7 — Saturation follows 1 - 1/N, magnitude follows 1 - rho
- Correction to the plan, recorded before testing: the plan said the plateau
  location is predicted by rho. Under the equicorrelated model the fraction of
  the maximum variance reduction reached at N is 1 - 1/N regardless of rho;
  rho sets the size of the gain, not where it saturates.
- Predictions: for soft_vote, D(N)/D(N_max) is within 0.15 of
  (1 - 1/N)/(1 - 1/N_max) at N in {2, 4, 8}, median over pools with
  D(N_max) > 0.002; and Spearman(rho, D(N_max)) <= -0.3.
- **Falsified** if the median deviation exceeds 0.25 or the correlation is >= 0.

### P1 — What a population buys that a scalar cannot: ranking
- Temperature scaling is monotone, so it cannot change the order in which a
  single model's predictions are ranked by confidence. Selective prediction is
  therefore a place where a population could beat "one adapter plus one
  temperature" in principle.
- For each headline pool: excess area under the risk–coverage curve (E-AURC,
  i.e. AURC minus the oracle AURC at the same accuracy) for the soft_vote
  ensemble ranked by its confidence, and for the selected baseline ranked by its
  max probability.
- Prediction: ensemble E-AURC < single E-AURC in >= 70% of pools.
- **Falsified** if <= 50%.

## 4. What is descriptive only

- P3: the per-example oracle (correct if any member is correct), as an upper
  bound on routing. No hypothesis.
- Per-family breakdowns of every quantity above.

## 5. Analysis rules

- All hypotheses are reported, passed or failed, in `mechanism/RESULTS.md`.
- No correction across hypotheses; each has its own pre-stated criterion.
- If a computation cannot be run as specified (missing data, an alignment that
  fails), it is reported as not testable with the reason, not substituted.
- New runs that fail to train (§2) are kept, flagged, and reported both with and
  without them.

## 9. Deviations

Recorded 2026-10-02, after inspecting the stored data format and **before**
running any analysis in §3.

1. **H7, first criterion: not testable as specified.** The headline results
   store each member's predicted label and its maximum probability, not the
   full probability vector, so the soft_vote of a member *subset* cannot be
   recomputed. Per §5 this is reported as not testable, not substituted. Two
   analyses are added and labelled exploratory: the same curve for
   majority_vote on the headline pools, and for soft_vote on the six pools with
   a logits cache. H7's second criterion (Spearman(rho, D(N_max)) for the full
   soft_vote) needs only stored quantities and stays confirmatory.
2. **H6 weighting.** ln(a/(1-a)) is the Nitzan–Paroush weight for *binary*
   votes. For K classes with symmetric errors the optimal weight is
   ln((K-1) a / (1-a)), and the binary formula gives a negative weight to any
   voter between chance (1/K) and 1/2. The K-class weight is used for the
   criterion; the registered formula is also reported.

3. **H8 design scaled down for compute** (recorded 2026-10-03, before any H8
   run). The test runs on two A6000s shared with unrelated jobs, and the
   registered design (8 + 8 adapters x 4 models at 40k training rows) is about
   80 GPU-hours. It is run with **6 + 6 adapters per model and 20,000 training
   rows**. The task is SNLI (outside the corpus; examples with label -1
   removed), evaluated on the paper's 40% test slice of its validation split.
   Each model keeps its corpus batch size and learning rate (BERT-base bs 32,
   Qwen-2.5 0.5B/1.5B/3B bs 16/8/4, lr 3e-4, LoRA on query/value or
   q_proj/v_proj); the pool recipe is r=8, alpha=16, 2 epochs, and the
   baseline recipe is the n_rank configuration r=32, alpha=32, 4 epochs. The
   smaller N lowers power; it does not change the prediction or the
   falsification rule.

4. **P1's premise is false for multiclass confidence** (recorded 2026-10-03,
   after an external read). P1 says temperature scaling "cannot change the
   order in which a single model's predictions are ranked by confidence". That
   holds for binary models and for a raw top-2 margin, but not for the maximum
   softmax probability with three or more classes: logits [4, 0, 0] and
   [3, 0, -20] give MSP 0.9647 > 0.9526 at T = 1 and 0.7870 < 0.8176 at T = 2.
   P1's E-AURC comparison is therefore not evidence of a benefit "a temperature
   cannot buy"; it compared the ensemble against an *uncalibrated* single's MSP.
   The comparison stands as recorded; its interpretation is withdrawn, and a
   test against temperature-scaled, margin and other post-hoc single
   confidences is needed before any such claim.
5. **H10's pool count** (recorded 2026-10-03). H10 says the eight SNLI pools
   are "half of them decoders"; three of the four models are decoders, so six
   of the eight pools are. The prediction and criterion are unchanged. H8, H9
   and H10 all use the same SNLI runs and are not independent tests.

## 10. Hypotheses registered after the first analysis pass

Registered 2026-10-02, **after** seeing the results of §3 on the existing data.
They are therefore confirmatory only on data not yet collected; the existing
corpus generated them and cannot test them.

### H8 — What extra budget buys a single adapter shrinks with model size
- Source: with failed runs removed on both sides, the quality gap G (trained
  baseline mean minus trained pool-member mean) falls with base-model size
  (Spearman rho = -0.69 over 48 pools; -0.46 within decoders), and it is this
  term, not diversity, that separates the families (G = +3.7pp encoder,
  +0.1pp decoder; error correlation 0.71 vs 0.70).
- Mechanistic reading: larger pretrained models have lower intrinsic dimension
  for fine-tuning (Aghajanyan et al., 2021), so rank 8 already saturates them
  and extra rank or epochs buy little; diversity is then the only lever left.
- Test on new data: one task outside the corpus, three sizes of one model family
  (Qwen-2.5 0.5B, 1.5B, 3B) plus BERT-base, each with 8 trained adapters at
  the pool recipe (r=8, 2 epochs) and 8 at the baseline recipe (r=32, 4 epochs),
  failed runs removed by §2.
- Prediction: G_tt decreases monotonically with size within Qwen-2.5, and
  G_tt(BERT-base) > G_tt(Qwen-2.5-0.5B).
- **Falsified** if G_tt(3B) >= G_tt(0.5B).

### H9 — The family split replicates out of sample (registered 2026-10-03)
- Registered while the H8 SNLI runs are training. At registration all 12 BERT
  runs had finished and a partial BERT result had been printed (3 pool, 1
  baseline adapter: Delta = -2.7pp), so **the BERT part is not confirmatory**.
  Of the 36 Qwen runs, one (`h8_q05_pool_s11`) had finished and had not been
  inspected.
- Quantity: on the SNLI test slice, Delta = accuracy of the soft_vote ensemble
  of the trained pool-recipe adapters minus that of the trained baseline-recipe
  adapter selected on the val_selection slice -- the paper's comparison, on a
  task outside the corpus. Computed by `mechanism/h8_size.py`.
- Prediction, from the decomposition (Delta = D - G - S, with G shrinking as
  models grow): Delta > 0 for each of Qwen-2.5 0.5B, 1.5B and 3B; and
  D_pool > G_tt for each of them. (BERT, non-confirmatory: Delta < 0.)
- **Falsified** if Delta <= 0 for two or more of the three Qwen models.

### H10 — H1 replicates on new pools, decoders included (registered 2026-10-03)
- H1 passed on six *encoder* pools, the only ones with cached logits in the
  corpus. The H8 SNLI runs save logits, so they form eight new pools (four
  models x two recipes, trained adapters only), half of them decoders.
- At registration the BERT runs had finished and only their accuracies had been
  printed; T_eq, T* and ECE had not been computed for any SNLI pool.
- Quantities as in H1: T_eq fitted label-free on the test slice; T* the
  NLL-optimal temperature of the mean-logit model on the val_combine slice of
  `split_indices`; 15-bin equal-mass ECE.
- Predictions: (b) |ECE(softmax(zbar/T_eq)) - ECE(ensemble)| <= 0.01 in >= 6
  of 8 pools; (c) the rule |ln T* - ln T_eq| < |ln T*| predicts whether the
  ensemble improves ECE over the mean-logit model at T=1 in >= 6 of 8 pools.
- **Falsified** if (b) or (c) holds in 4 or fewer of the 8 pools.
