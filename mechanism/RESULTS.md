# Mechanism study: results (updated 2026-10-03)

Pre-registration: `mechanism/PREREG.md` (registered at `da3a291`, deviations
in §9 recorded at `e599c16` before any analysis, H8 added at `cdbb13e`). Every
number below is produced by the script named beside it and saved under
`mechanism/results/`. The 33 H3 runs finished on 2026-10-02 (none failed to
train); the 48 H8 runs on SNLI finished on 2026-10-03.

## Verdicts

| | Hypothesis | Verdict | Key number | Script |
|---|---|---|---|---|
| H1 | A population's calibration effect is an implicit temperature | **PASS** | (b) 5/6, (c) 5/6 | `h1_temperature.py` |
| H2 | One parameter (kappa x disagreement) predicts the diversity gain | **FALSIFIED** | LOPO R^2 = 0.27 (soft_vote), 0.00 (majority) | `h2_disagreement.py` |
| H3a | Decoder diversity comes from under-convergence | **FALSIFIED** (opposite direction) | D +1.00 / +1.32 / +2.30pp at 1 / 2 / 4 epochs | `h3_new_runs.py` |
| H3b | Head seed alone gives >= 50% of the diversity | **PASS** (but see below) | head 87%, LoRA 74%, data order 86% of all-vary delta | `h3_new_runs.py` |
| H4 | Gains concentrate on ambiguous items | **FALSIFIED** | decoder D by entropy tercile +1.71 / +1.42 / +1.56pp | `h4_ambiguity.py` |
| H5 | The baseline's edge is selection pressure | **FALSIFIED** | monotone in 60% of pools (need 80%); S formula R^2 = 0.57 | `h5_selection.py` |
| H6 | Heterogeneous populations win with quality weighting | **FALSIFIED** | 0/3 tasks | `h6_heterogeneous.py` |
| H7 | Diversity gain scales with 1 - rho | **PASS** (2nd criterion) | Spearman(rho, D) = -0.63; 1st criterion not testable (§9) | `h7_saturation.py` |
| P1 | A population buys ranking a temperature cannot | **PARTIAL; premise withdrawn** (§9 dev. 4) | 25/48 pools vs an *uncalibrated* single; decoders 22/25, encoders 3/23 | `p1_selective.py` |
| H8 | Extra budget buys less as models grow | **PASS as registered; the size trend is not shown** (see below) | G_tt +3.09 (BERT) / -0.83 / -0.89 / -12.72pp (Qwen 0.5B/1.5B/3B) | `h8_size.py` |
| H9 | The family split replicates out of sample (SNLI) | **PASS** | Delta +1.17 / +1.42 / +1.96pp (Qwen); BERT -2.46pp (non-confirmatory) | `h8_size.py` |
| H10 | H1 replicates on new pools, decoders included (SNLI) | **PASS** | (b) 8/8, (c) 7/8 | `h10_temperature_snli.py` |

Of the confirmatory tests, H1, H3b, H7, H8, H9 and H10 pass; H2, H3a, H4, H5 and H6
are falsified; P1 is partial. H8, H9 and H10 share the same 48 SNLI runs and are
not independent tests.
The falsifications are informative: each removes an explanation the plan
proposed for the encoder/decoder split.

## What the data say

**1. The family split is not about diversity.** Mean pairwise error
correlation is the same in both families (encoder 0.71, decoder 0.70; H7), and
encoder pools actually disagree more (delta 0.092 vs 0.056) and gain more from
averaging (D = +2.7pp vs +1.1pp; H2). The plan's "clonal population" account
(assumption 3) does not explain the split.

**2. Nor is it selection pressure, once failed runs are removed.** H5's
headline curve looked decisive (encoder Delta +5.4pp against one random
baseline candidate, -1.1pp against the best of 16), but a random candidate is
often a run that never trained. With failed candidates removed (exploratory,
`h5b_selection_healthy.py`), encoder ensembles lose to a *random trained*
baseline adapter at every M (-0.95pp at M=1, -1.03pp at M=16) and decoder
ensembles beat it at every M (+1.02pp to +0.48pp).

**3. It is what the extra budget buys the single adapter.** Decomposing
Delta = D - G - S with trained adapters only (exploratory,
`decomposition_trained_exploratory.json`):

| soft_vote | D (diversity) | G (quality gap) | S (selection) | Delta |
|---|---|---|---|---|
| encoder | +2.72pp | **+3.68pp** | +0.24pp | -1.20pp |
| decoder | +1.11pp | **+0.08pp** | +0.47pp | +0.56pp |

On encoders, spending the pool's budget on rank and epochs makes one adapter
3.7pp better than a typical pool member, which averaging cannot recover. On
decoders the same spend buys almost nothing, so even a small averaging gain
wins. G falls with base-model size (trained vs trained: +1.3pp at 110M, +1.9pp
at 340M, about 0 to -0.7pp from 0.5B to 1.7B; Spearman -0.69 overall, -0.46
within decoders; `g_by_size_exploratory.json`). That is consistent with larger
models having lower intrinsic dimension (Aghajanyan et al., 2021), but size and
family are confounded in this corpus, which is why it is registered as H8 for
new data rather than claimed.

**4. A population's calibration effect is a temperature it did not choose
(H1).** Probability averaging is closely approximated by the mean-logit model
at one temperature T_eq, which the members' logit spread sets. Whether the
ensemble improves calibration is then predicted by where T_eq sits relative to
the temperature the model needs (T*): 5/6 pools correctly. On ANLI, T* = 3.8
but the ensemble behaves like T = 1.04, so it stays badly overconfident
(ECE 0.25; one fitted temperature gives 0.06). On BoolQ and MNLI the mean model
is already calibrated (T* about 1) and the ensemble's T_eq of 1.3 to 1.4 makes it
*underconfident*, raising ECE. This is the mechanism behind the paper's
temperature control.

**5. Decoder ensembles rank their errors better than an uncalibrated single
(P1) -- but this is not yet evidence of something a scalar cannot buy.**
Decoder ensembles rank their own errors better than the selected single
adapter in 22/25 pools (median E-AURC 0.021 vs 0.029); encoder ensembles do so
in 3/23. P1 was framed on the premise that temperature scaling cannot reorder
predictions by confidence. That is false for the maximum softmax probability
with three or more classes (PREREG §9, deviation 4: logits [4,0,0] and
[3,0,-20] swap order between T=1 and T=2). The comparison was against the
single's raw MSP, so the result says only that decoder ensembles beat an
*uncalibrated* single at selective prediction. Whether anything survives
against a temperature-scaled single, a raw top-2 margin or other post-hoc
confidence scores (e.g. Cattelan & Silva, UAI 2024) is untested.

**6. Two proposed remedies do not work.** Gains do not concentrate on items
humans find ambiguous (H4: flat across ChaosNLI entropy terciles for decoders,
*decreasing* for encoders). And a quality-weighted vote of one median member per
backbone loses to the best single member in all three tasks tried (H6).

**7. Headroom.** The per-example oracle (correct if any member is) is 9 to 10pp
above the selected single in both families (P3), far above what any
combination rule here achieves. If that headroom is reachable, it is by
per-example routing, not averaging.

**8. Decoder diversity grows with training, from any perturbation (H3).** On
Qwen-2.5-0.5B MNLI with all seeds varied, the members' disagreement *rises*
with training (delta 0.044 / 0.051 / 0.069 at 1 / 2 / 4 epochs) and so does the
averaging gain (D +1.00 / +1.32 / +2.30pp), while mean member accuracy peaks at
2 epochs (0.838) and falls at 4 (0.830). Members that overfit overfit in
different directions, and averaging recovers it: the 4-epoch ensemble (0.853)
beats the 2-epoch one (0.851) although its members are worse. Their logit
spread grows eightfold (0.18 to 1.40), which is H1's mechanism at work: the
average cools overconfident members. H3a predicted the opposite and is
falsified. H3b passes as registered -- varying only the classification-head
seed gives 87% of the all-seeds disagreement -- but varying only the LoRA
initialisation (74%) or only the data order (86%) does nearly as much, so the
right reading is sensitivity to *any* perturbation, not the head specifically.
Three runs with every seed fixed were bit-identical (delta = 0), so all of the
observed diversity comes from the seeds and none from GPU nondeterminism.

**9. The oracle headroom is not reachable by training-free routing
(exploratory, `p3_routing.py`).** Following the most confident member (R1) does
no better than averaging. A confidence gate (R2: the n_rank single when it is
confident, the ensemble otherwise, threshold chosen on half the test split)
captures 0% of the oracle headroom on encoders and 6.7% on decoders. Its
practical use is cost, not accuracy: on decoders it matches the ensemble's
accuracy (0.8550 vs 0.8546) while calling the ensemble on only 19% of examples,
at about 4x the single adapter's inference instead of about 19x.

**10. The implicit temperature is predicted by margin spread, not logit spread
(exploratory, `teq_formula.py`).** H1's probit form, using the across-member
variance of all class-centred logits, ordered the pools correctly but ran low
at high spread. The probit approximation is derived for a binary logit, and in
a multiclass model confidence is governed by the margin between the top two
classes. Using the across-member variance of that margin, with the theoretical
constant pi/8 and nothing fitted, raises R^2 for T_eq from 0.33 to 0.75 over 16
pools (six corpus encoder pools, six Qwen-0.5B MNLI arms and the four SNLI pools
finished at the time; Spearman 0.95). These are not 16 independent settings:
the six MNLI arms share one backbone and task, and the SNLI pools share one
task. The pool set grew while runs finished, so the R^2 is tied to the
`teq_formula.json` written on 2026-10-03 and will not be re-selected as more
pools arrive. Fitting the constant gives c = 0.31 against pi/8 = 0.39
and R^2 = 0.83. The two largest misses are the method-mixed MNLI pool and BoolQ,
where mixing PEFT methods plausibly makes the logit spread non-Gaussian.

## Coverage of the execution plan

| Plan step | What was done | Not done, and why |
|---|---|---|
| 1. Mechanism variables | D, G, S, delta, rho, T_eq for every headline pool (`h2`, `h5`, `h7`, `h1`) | — |
| 2. Hypothesis tests on existing data | H1, H2, H4, H5, H6, H7, P1 against pre-registered criteria | — |
| 3. Refine the theory | Probit form tested; rule for when averaging helps calibration (H1c); margin-variance form, R^2 0.33 -> 0.75 with no fitted constant (finding 10) | A form for method-mixed pools, whose logit spread is not Gaussian |
| 4. Source of diversity | H3a/H3b, 33 runs | MEU-10 (per-architecture tuned vs shared HP): not run; the training-health analysis covers the failure-rate side of it |
| 5. Matched inference compute | — | MEU-9 dropped: the corpus recipe uses LoRA dropout 0, so a single classification adapter has no stochasticity to sample; self-consistency is a generation-task comparison |
| 6. Alternative paradigms | P1 selective prediction, P2 weighted heterogeneous vote (H6), P3 oracle, training-free routing (P3b) | A learned per-example router (needs input features not in the stored results); P4 diversity-trained populations |
| 7. Out-of-sample prediction | H8, H9, H10 on SNLI, registered before the Qwen results; all pass | A second out-of-corpus task (planned in `PREREG_AB.md`) |
| 8. Paper | H1 added to the paper as an appendix (`analysis/temperature_mechanism.py`) | Decomposition and P1 kept out of the October submission as exploratory |

## Deviations and caveats

- H7's first criterion is not testable as registered (no stored per-member
  probability vectors); the exploratory substitutes agree with the 1 - 1/N law
  for soft_vote on the six cached pools (median |dev| 0.095) and are marginal
  for majority_vote (0.17).
- H6 used the K-class Nitzan–Paroush weight (§9); the registered binary weight
  gives the same verdict.
- H4 used the tasksource copy of ChaosNLI-MNLI because the official Dropbox link
  returned an HTML page; it has the original 1,599 items, pairIDs and 100-label
  counts.
- Findings 2 and 3 are exploratory: they were not registered, and they use the
  training-loss failure rule from `analysis/training_health.py`.
- H1 rests on six encoder pools, the only ones whose logits survive.

## Out of sample: H8, H9, H10 on SNLI

**What passes cleanly.** The family split replicates on a task outside the
corpus (H9). Against the val_selection-best baseline-recipe adapter, the
pool-recipe ensemble wins on all three Qwen sizes (+1.17, +1.42, +1.96pp) and
loses on BERT-base (-2.46pp, non-confirmatory because BERT had been seen). The
decomposition behaves as predicted: the extra budget buys BERT +3.09pp (G_tt)
and buys every Qwen size nothing or less than nothing, so a ~0.8pp diversity
gain decides the comparison. H1's temperature account also replicates, now on
decoder pools (H10: (b) 8/8, (c) 7/8).

**What H8's pass does not show.** H8 predicted G falling *monotonically with
size* within Qwen-2.5. The registered criterion holds (G(3B) < G(0.5B)), but:

- G(0.5B) = -0.83pp and G(1.5B) = -0.89pp differ by 0.06pp, far inside seed
  noise, so there is no measured trend from 0.5B to 1.5B;
- the 3B value reflects an unstable baseline recipe (r=32, alpha=32, lr 3e-4,
  bs 4). Two of its six runs are flagged as failed (train loss >= 0.85 ln 3),
  and two more sit just under the flag (0.84, 0.76) at 0.65 and 0.73
  validation accuracy.

The ordering survives stricter thresholds (0.80, 0.75 and 0.70 leave 3, 2 and
2 baseline runs and G(3B) = -8.07, -2.64 and -2.64pp;
`results/h8_threshold_sensitivity.json`, post hoc), but with two runs that is
weak evidence. Read H8 as: the budget buys nothing on Qwen at any size tested
and something on BERT. It is not evidence for a size law. The H8 baseline
recipe also uses alpha/r = 1 against the pool's 2; `PREREG_AB.md` (A2) removes
that confound.

## Exploratory: calibrating both sides (B1/B2 pilot)

`b1_calibrate.py` (`PREREG_AB.md` §5–6, run on the H8 pools; exploratory,
because the dev_test slice had been seen for accuracy). With one temperature
fitted on val_combine for the single and a power temperature for the ensemble:

| backbone | NLL S+TS | NLL E+TS | E+TS - S+TS [95% CI] | risk@90% E - S |
|---|---|---|---|---|
| BERT-base | 0.451 | 0.500 | +0.049 [+0.037, +0.061] | +2.5pp |
| Qwen-2.5-0.5B | 0.371 | 0.318 | -0.054 [-0.067, -0.041] | -1.5pp |
| Qwen-2.5-1.5B | 0.295 | 0.252 | -0.043 [-0.057, -0.029] | -1.0pp |
| Qwen-2.5-3B | 0.292 | 0.235 | -0.058 [-0.072, -0.044] | -1.1pp |

The temperature closes most of the single's calibration gap (Qwen-0.5B NLL
0.576 -> 0.371) but not all of it, and on decoders the ensemble keeps an
advantage in both probability quality and error ranking. Part of that is
accuracy: the decoder ensembles are also 1.2–2.0pp more accurate. The
confirmatory version runs on the sealed SNLI test of the A2/A3 trajectories.

## Exploratory: A2 pilot, rank at equal token cost (`a2_pilot.py`)

Twelve SNLI trajectories (`PREREG_AB.md` §3): alpha = 2r at both ranks (H8's
baseline used alpha/r = 1), one 4-epoch schedule with an epoch-2
mid-trajectory checkpoint, three new seeds. Ensemble = soft vote of three r = 8
seeds; single = val_selection-best of three r = 32 seeds; both cost the same
tokens. dev_test (seen in H8), so exploratory.

| backbone | checkpoint | Delta [95% CI] | D | G | S |
|---|---|---|---|---|---|
| BERT-base | epoch 2 | -1.65pp [-2.46, -0.81] | +0.44 | +1.23 | +0.86 |
| BERT-base | epoch 4 | -1.42pp [-2.21, -0.63] | +0.19 | +1.27 | +0.34 |
| Qwen-2.5-0.5B | epoch 2 | +1.22pp [+0.43, +2.03] | +1.05 | -1.96 | +1.79 |
| Qwen-2.5-0.5B | epoch 4 | +1.57pp [+0.71, +2.44] | +1.10 | -0.54 | +0.07 |

The alpha/r confound did not produce the split. At equal cost, rank 32 buys
BERT about +1.2pp over rank 8 and costs Qwen 0.5–2.0pp, and that sign decides
which pipeline wins. The expansion rule passed (all 24 checkpoints aligned,
token counts consistent).

## Confirmatory: A3, equal budget on sealed test sets (`a3_analysis.py`)

`PREREG_AB.md` §9.1. Design, budgets, second task and pass rules were committed
at `fcb98c4`, before any A3 run and before the A2 pilot was analysed. The
analysis code was committed at `ad1fe51`, and the test logits were unsealed by
`mechanism/TEST_FREEZE` (`996bee8`) after all 96 runs had finished (none failed
to train). Before the freeze, an exploratory `--dev` pass on the validation
dev_test slice agreed in sign with every cell below
(`results/a3_dev_exploratory.json`).

**Setup.** At budget B (epochs over 20k training rows, so equal tokens):
- E(B): soft vote over B/2 adapters at r = 8, 2 epochs.
- S(B): the val_selection-best of B/4 adapters at r = 32, 4 epochs.

Two replicates with disjoint seeds. SNLI test has 9,824 examples; Yahoo
Answers test has 60,000.

| task | backbone | B | Delta acc [95% CI] | NLL E+TS - S+TS [95% CI] | risk@90% E - S |
|---|---|---|---|---|---|
| SNLI | BERT-base | 4 | -2.63pp [-3.10, -2.15] | +0.060 [+0.053, +0.067] | +2.4pp |
| SNLI | BERT-base | 8 | -2.83pp [-3.34, -2.32] | +0.065 [+0.057, +0.072] | +2.4pp |
| SNLI | BERT-base | 16 | -2.76pp [-3.26, -2.24] | +0.063 [+0.056, +0.070] | +2.6pp |
| SNLI | Qwen-0.5B | 4 | +1.04pp [+0.62, +1.46] | -0.053 [-0.061, -0.045] | -0.9pp |
| SNLI | Qwen-0.5B | 8 | +1.13pp [+0.71, +1.55] | -0.059 [-0.067, -0.052] | -1.2pp |
| SNLI | Qwen-0.5B | 16 | +1.22pp [+0.80, +1.64] | -0.059 [-0.067, -0.052] | -1.3pp |
| Yahoo | BERT-base | 4 | -1.52pp [-1.71, -1.34] | +0.050 [+0.047, +0.053] | +1.4pp |
| Yahoo | BERT-base | 8 | -1.25pp [-1.43, -1.07] | +0.037 [+0.034, +0.040] | +1.2pp |
| Yahoo | BERT-base | 16 | -1.00pp [-1.19, -0.82] | +0.025 [+0.022, +0.028] | +1.1pp |
| Yahoo | Qwen-0.5B | 4 | +2.72pp [+2.52, +2.92] | -0.104 [-0.108, -0.100] | -2.7pp |
| Yahoo | Qwen-0.5B | 8 | +2.94pp [+2.75, +3.14] | -0.110 [-0.114, -0.106] | -3.0pp |
| Yahoo | Qwen-0.5B | 16 | +3.20pp [+3.00, +3.41] | -0.125 [-0.128, -0.121] | -3.6pp |

**Verdicts.** A3-P1 **REPLICATES**: all 8 predicted cells (B = 8, 16) are
SUPPORTED, and so are the 4 unpredicted B = 4 cells. B-P1 also **REPLICATES**,
8/8. Both replicates have the same sign in every cell. The greedy-subset
ensemble (secondary) agrees in every cell, and so does risk at 90% coverage.

**What it means.** At an exactly matched training budget, on a held-out test
set and on a task the project had never used, spending the budget on a pool
loses to spending it on a stronger single on BERT-base and wins on Qwen-0.5B,
in accuracy, in calibrated NLL and in selective risk.
- The Qwen single is badly overconfident: its fitted T is 2.0–2.9, against
  1.0–1.1 for the ensemble. Fitting T shrinks its NLL deficit from about
  0.33–0.47 to 0.05–0.12, but does not remove it.
- On BERT the ensemble is worse even before calibration.
- On Yahoo the BERT gap narrows as B grows (-1.5 to -1.0pp). On SNLI it does
  not.

**Limits, stated before anyone asks.**
1. Two recipes, not an optimum. S uses the corpus `n_rank` recipe (r = 32, 4
   epochs, lr 3e-4) and E the pool recipe. A single tuned for Qwen (other
   learning rates, or r = 8 at 4 epochs) may close the gap; A2 already shows
   r = 32 underperforms r = 8 on Qwen at equal epochs. The claim is about
   these two allocations of one budget, not about the best single.
2. One encoder and one decoder backbone. Which property of the backbone
   matters (family, size, pretraining) is not identified by two models.
3. Classification only; budgets up to 16 epoch-units (8 members).
4. The calibration comparison uses one temperature per arm, fitted on
   val_combine. Richer post-hoc methods are untested.


## Confirmatory: A3b, a single that tunes its recipe inside the budget (`a3b_tuned.py`)

`PREREG_AB.md` §9.2, written after the §9.1 results and committed with its
analysis script before any A3b run (`4fdf104`); runs committed at `a8c991b`.
S_tuned(16) keeps the val_selection-best of four 4-epoch configurations
(r in {8, 32} x lr in {1e-4, 3e-4}), one seed each, against the unchanged
E(16).

| task | backbone | picked (rep 1, rep 2) | Delta acc [95% CI] | NLL E+TS - S+TS [95% CI] |
|---|---|---|---|---|
| SNLI | BERT-base | c4, c4 (r32, 3e-4) | -2.64pp [-3.15, -2.13] | +0.059 [+0.052, +0.066] |
| SNLI | Qwen-0.5B | c3, c4 | +1.08pp [+0.71, +1.46] | -0.048 [-0.054, -0.043] |
| Yahoo | BERT-base | c4, c4 | -1.08pp [-1.26, -0.89] | +0.030 [+0.027, +0.033] |
| Yahoo | Qwen-0.5B | c1, c1 (r8, 1e-4) | +1.74pp [+1.59, +1.90] | -0.067 [-0.070, -0.065] |

A3-P2 **REPLICATES** (4/4 SUPPORTED) and B-P2 **REPLICATES** (4/4).
- On BERT the search keeps the fixed recipe.
- On Qwen it narrows the Yahoo gap from +3.2 to +1.7pp, but does not close it.
- No Qwen configuration reaches the ensemble even when picked on the test
  set: best 87.45 vs 88.43% on SNLI, 73.88 vs 75.59% on Yahoo.

So the §9.1 Qwen result is not an artefact of the fixed recipe, within this
search space (two ranks, two learning rates, 4 epochs).

## Confirmatory: A3c, family or size? (`a3c_analysis.py`)

`PREREG_AB.md` §9.4 (`44b3499`, before any A3c run) and §9.5 (lr check:
all healthy, lr 3e-4). A size-matched pair from opposite families,
BERT-large (encoder, 335M) and SmolLM2-360M (decoder, 362M), runs the §9.1
protocol on SNLI and Yahoo Answers. Runs committed at `49d4f12`.

| task | backbone | B=4 | B=8 | B=16 | NLL E+TS - S+TS at B=16 |
|---|---|---|---|---|---|
| SNLI | BERT-large | +24.98pp (REVERSED, see below) | -1.44pp | -1.97pp | +0.033 |
| SNLI | SmolLM2-360M | +0.14pp | +0.36pp | +0.26pp | -0.024 |
| Yahoo | BERT-large | -1.54pp | -1.14pp | -0.91pp | +0.041 |
| Yahoo | SmolLM2-360M | +0.96pp | +1.10pp | +1.13pp | -0.048 |

**Verdicts.**
- A3-P3 **REPLICATES**: of the 8 predicted cells (B in {8, 16}), 6 are
  SUPPORTED and 0 REVERSED. The two SNLI SmolLM2 cells are positive but
  unsupported.
- B-P3 **REPLICATES**, 8/8 SUPPORTED.

At matched size the encoder ensemble loses and the decoder ensemble wins or
ties, in accuracy and in calibrated NLL. The sign tracks family, not size,
for these two backbones. The decoder accuracy gain is smaller at 360M than at
0.5B on SNLI.

**Two runs failed the training-loss rule**, and both stay in their pipelines
as registered.
- `a3_snli_bertl_S_s261` (validation accuracy 0.33, chance). At B=4 it is
  replicate 2's only single candidate, so the unpredicted SNLI/BERT-large
  B=4 cell (+25pp, REVERSED) is an artefact of that one failed run.
  Replicate 1 alone gives -1.5pp. At B >= 8 the single picks a healthy
  candidate.
- `a3_yahoo_bertl_E_s208` (validation accuracy 0.10, chance) is averaged
  into replicate 1's E(16). BERT-large still loses there.
