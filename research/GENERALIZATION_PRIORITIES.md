# General-purpose CNN priorities after the 45-observation campaign

October9. This supersedes the earlier recommendation to simply confirm the
final aggregate winner as our main deliverable. That graph remains a candidate;
broader useful score/time coverage is the selection goal. This is a research
plan, not a GPU allocation or authorization to change the frozen campaign.

## Deeper findings

Offline diagnostics are retained under
`results/learning-continuation-5090-20261009-prefix45/generalization-diagnostics/`.
The new script reruns the previous transported-artifact consistency audit and
computes282 per-game/model matched-time comparisons, contribution decomposition,
leave-one-game-out sensitivity and first-layer sampling gaps. Four new tests plus
the previous25 host/artifact checks pass. No neural computation or GPU run occurs.

| Finding | Implication |
|---|---|
| At each game's Nature final checkpoint time, new44/45 has higher measured lower scores on3 games and lower on3; old quality has4 higher and2 lower. | A higher pooled final objective has not broadened fast per-game coverage. |
| No measured candidate exceeds4 of6 higher lower scores at those time allowances. | We have not yet found a convincing general fast replacement. |
| Pong supplies74.6% of new44/45's net aggregate gain over quality. | One task accounts for most of the improvement despite equal game weights. |
| Excluding Pong leaves new0.36256 versus quality0.34156, still ahead; the same effective graph remains top-scoring when any single game is omitted. | The result is not purely a Pong artifact, but these are final-score diagnostics, not cost or held-out tests. |
| About99.1% of the paired aggregate gain is in seed64174. | Discovery-seed selection risk remains substantial. |
| Connect4 one-pixel cells produce0% wins for new, quality and Nature; compact drawings and Pong reflections/inversion remain weak. | Appearance averages hide unsolved cases. |
| Legacy encoder4 allows first stride4/8 only; four sampled graphs have first kernel smaller than stride. | Some graphs never observe parts of the image, while gentler downsampling is unavailable. |
| Ten of45 observations repeat effective graphs; repeat checkpoint hashes/scores match exactly. | Duplicate trials consume training resources without adding learning-seed evidence. |
| PROTEIN sees only final score and final cost, despite four recorded checkpoints. | The search does not directly optimize the full frontier we want to deliver. |

Matched-time rule: select the candidate's **latest measured checkpoint that fits**
Nature's final mean training-time allowance in that game. Do not choose its best
score, interpolate, extrapolate or discard declines. Four checkpoints are coarse:
unmeasured performance in the remaining allowance is unknown. Comparisons use
two-seed means and do not certify significance. Different games have different
allowances; this is not a universal wall-time budget. Pong's bounds are censoring
bounds, not confidence intervals. Retain the original aggregate fronts alongside
this diagnostic, not replace them with a hand-selected new winner.

Our current generalization claim concerns **one CNN architecture trained separately
in each game**. It does not concern one policy playing every game or transfer of
pretrained weights. All41 drawings were available in training mixtures, and all
six games supplied search feedback: neither is an untouched generalization test.

## Sorted priorities

1. **Make the search target match broad frontier improvement.** Define success
   using whole observed curves, per-game time allowances, paired seed stability,
   and a declared penalty/gate for severe task/drawing regressions. Retain an
   equal-game aggregate but report coverage and lower-tail behavior beside it.
   Our3-up/3-down winner and Pong-heavy improvement demonstrate why the mean alone
   is insufficient. Do not use the absolute worst drawing as the only score:
   shared zero-score cases would flatten the objective. Compare candidate objective
   definitions offline on all existing results; freeze the new protocol before
   further search. Never mix a revised utility into the original45-row history.
   Robust performance profiles and interval reporting have a basis in
   [Agarwal et al.,2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

2. **Diagnose the weak drawings and their task semantics.** Check sampling coverage,
   slot-mixture counts, appearance/action relationships and failure trajectories
   before attributing every failure to CNN capacity. Pong's vertical reflection
   changes the relation between screen movement and unchanged action IDs; it adds
   a latent orientation/control problem, unlike a simple cosmetic texture change.
   A recurrent policy may infer it, so this is not proof of an invalid environment.
   Keep the old results and catalog unchanged; separately version any future
   control that remaps actions or separates these conditions. Compact Connect4
   tests whether the encoder can retain fine spatial information. Resolve which
   failures are perception, control ambiguity, memory or insufficient learning.

3. **Run a small, controlled architecture matrix across every game.** First complete
   C8/C16 × residual off/on with K8/s4/projection64 fixed. Existing22 is C8/off,
   23/28 C16/off, and44/45 C16/on; **C8/on is the missing corner**. It could retain
   the residual benefit with less convolution work; this is a hypothesis, not a
   speed promise. Keep quality and Nature controls, fixed per-game learners and
   full curves. Add a few narrow, information-preserving stem candidates after
   their native acceptance gates: stride2 stages or lossless space-to-depth pixel
   packing, rather than hundreds of unrelated options. Packing moves local pixels
   into channels before learned mixing; it need not erase single-pixel pieces.
   [SPD-Conv](https://arxiv.org/abs/2208.03641) is a relevant classification/detection
   precedent, not evidence that it wins RL or our native GPU timing. The current
   legacy grammar cannot express first stride2 or packing; implementation and
   independent CUDA checks are necessary before these candidates train.

4. **Repair search efficiency and conditional architecture handling.** Encode only
   active stages for the surrogate, identify effective graphs, and avoid repeated
   same-graph/same-seed training under an explicit new protocol. Keep raw proposals,
   skipped/cache identities and every failure auditable. Constrain first-layer
   sampling gaps where full pixel coverage is required; don't globally delete
   sparse sampling without documenting the tradeoff. Include cheap CNN controls
   as declared search anchors and incorporate checkpoint-curve utility/budget
   choices consistently. Early screening must be calibrated by game: previous tiny
   runs gave zero useful learning scores for several tasks. Do not pretend a
   weight-only restart is an identical continuation of optimizer/core state.
   This should improve information per GPU hour before adding many activations,
   dilation, depthwise operations or other dimensions.

5. **Confirm a broadly promising shortlist using fresh evidence.** Use fresh paired
   training seeds and episode IDs; preserve all41 drawing results and timing blocks
   with balanced run order. Selection must satisfy the predeclared coverage/curve
   criteria, not just pick the largest mean. Assess uncertainty over training
   replicas with their drawings/checkpoints kept together; drawings are not
   independent learning seeds. Future searches can reserve tasks/appearances, but
   the current six-game/full-catalog results cannot retrospectively become blind
   holdouts. No new large confirmation allocation is launched by this plan.

6. **Profile and optimize the architecture's actual bottleneck.** For the small
   shortlist, separate rollout/learner time, im2col, GEMM, activation/copy/launch
   overhead, core and environment work on the deployment GPU. The new graph adds
   only1.6% full-policy parameters on Connect4 yet42.4% aggregate training cost;
   parameter count is a poor speed guide. Apply shared backend improvements to
   Nature too, and retain a competitive native baseline. Numerical, gradient,
   graph/reload and determinism checks precede timed comparisons. Lower FLOPs or
   a fused kernel do not automatically imply better learning-time frontiers.

7. **Demonstrate portability before broad pixel-RL/SOTA claims.** The current encoder
   and all six tasks use36x44 grayscale images. After a robust local result, support
   other image shapes/channels and evaluate a frozen architecture on an established
   external pixel benchmark with competitive baselines and disclosed tuning/search
   cost. Current six environments are enough for immediate development; no new
   environment is created now. This later phase is necessary for a broad paper,
   rather than a claim restricted to these custom low-resolution tasks.

Pretraining, synthetic datasets and large activation menus remain possible later
projects. Current evidence gives more direct reasons to fix selection, sampling
and experimental efficiency first. Anti-aliasing also deserves a controlled
candidate, guided by [Zhang,2019](https://proceedings.mlr.press/v97/zhang19a.html),
but image-classification shift invariance is not automatically desirable for
control, and blurring tiny pieces could hurt. Preserve position information and
measure native overhead rather than assuming the classification result transfers.

## Reproduce the new offline diagnosis

```bash
.venv/bin/python research/analysis/generalization_diagnostics.py \
  research/results/learning-continuation-5090-20261009-prefix45 \
  --out build/generalization-diagnostics-FRESH
.venv/bin/python -m unittest research.tests.test_generalization_diagnostics
```

G240 used `/home/claude/cnn/.venv/bin/python` in the separate audit worktree.
Raw packets, retained native sources, failed allocation status and earlier
descriptive analysis are unchanged. This plan supersedes selection emphasis,
not any historical evidence or scientific qualification gate.
