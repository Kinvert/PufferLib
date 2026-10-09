# General-purpose CNN priorities after the 45-observation campaign

October9. This supersedes the earlier recommendation to simply confirm the
final aggregate winner as our main deliverable. That graph remains a candidate;
broader useful score/time coverage is the selection goal. This is a research
plan, not a GPU allocation or authorization to change the frozen campaign.

## Current operating requirements: realistic images and Nature-range curves

Kinvert's latest October9 clarification supersedes the earlier4x4-cell proposal
as the practical target. **36x44 grayscale and28x24 boards are historical
development cases, not the intended primary evidence for useful pixel RL.**
We are building a CNN for human-played games, including Atari-like tasks, with
an input path that can later serve larger real-camera images. No robotics
generalization claim follows merely from supporting RGB tensors.

1. **Training range:** use substantial paired training runs that span the useful
   region where Nature and compact CNNs compete. Do not spend the primary discovery
   budget on the much slower end just to reach IMPALA's highest scores. Calibrate
   per-game budgets on the new image contracts; old tiny-image throughput does not
   predict the new budgets. All models within a game/image condition receive the
   same declared learner schedule and decision budget. Report actual elapsed time,
   not a presumed crossover with IMPALA, which is not measured in this campaign.
2. **Curve resolution:** aim for about**eight scheduled checkpoints per training
   run**, distributed through that region, including the final checkpoint. Align
   schedules to native rollout boundaries. Choose sufficient learning budgets;
   eight nearly-zero scores are not useful resolution. Preserve declines and all
   measurements; evaluations and checkpoint overhead are separately recorded.
   The current staged campaign wrapper still caps exact stages at four, and
   bounded smoke/mini modes have their own limits. Extending the new main contract
   to eight requires updating preparation, execution limits, counts and audits,
   not just editing a cadence in a frozen INI. Keep older readers/packets unchanged.
3. **Realistic game images:** native frames should be human-readable at roughly
   **640x480-class dimensions**, with game-appropriate aspect ratios and variation
   rather than one universal image size. RGB and grayscale must be supported.
   Draw directly from game geometry at the chosen resolution, including readable
   pieces/ball/paddles/text where relevant. Do not nearest-neighbor enlarge the
   old36x44 raster and claim new visual information or realistic-resolution evidence.
4. **Explicit policy inputs:** support full-size frames and separately declared
   preprocessing/downsampling modes. Human display dimensions and policy tensor
   dimensions are distinct controls, both retained in the experiment manifest.
   Examples for investigation include640x480 and320x240 camera/game inputs and
   smaller benchmark-defined processed images. These are examples, not one
   mandatory shape or a license to silently reduce every input to36x44. Match
   preprocessing, image layout and input bytes across our CNN and every baseline.
   Hold image conditions fixed across architecture comparisons; don't let a
   proposal win by secretly receiving an easier/smaller image.
5. **Seeds are NEVER sweep coordinates.** Fixed training-replica lists,
   development/final episode suites, appearance/world assignments and the optimizer
   seed are experiment controls. The optimizer may change CNN coordinates, not
   hunt for lucky seeds. Fresh confirmation training replicas and withheld
   evaluation episodes are separate requirements. Record every assigned seed and
   use the same lists across candidate/reference pairs.
6. **Architecture and engineering:** support numeric power-of-two first strides
   **1,2,4,8** with shape-dependent buffers; keep targeted controlled architecture
   panels alongside native PROTEIN and selectively profile identified bottlenecks.
   Native C/CUDA construction/training, determinism and headless in-memory pixels
   remain the implementation direction.

### Immediate implementation order

**A. Portable image contract and realistic raster path.** Replace the specialized
36x44/channel1 assumptions in ours **and** local Nature/IMPALA/Impoola with explicit
environment height/width/channels/layout/dtype metadata. Derive convolution,
pooling, readout and workspace shapes once at construction. Permit different
game-appropriate shapes per run; fixed shapes within a run remain compatible with
preallocated buffers and CUDA graphs. Bind all of this to metadata/checkpoints,
deterministic suites and source receipts. Test portrait, landscape and square
shapes, grayscale and RGB, rather than only increasing a constant.

Add host shape/storage preflight before allocating or scheduling a GPU worker.
For context,640x480 RGB contains921,600 values versus1,584 in the old frame:
about582 times as many scalars. A2048-frame float32 tensor alone is about7.03GiB,
before rollout storage, activations, gradients and optimizer state. Investigate
compact uint8 frame storage and conversion on consumption as an explicitly
validated shared input path; don't casually scale old buffers/batches or alter
the CUDA stack. Batch/vector geometry must be calibrated consistently for each
new per-game input condition. Keep information-preserving early stages small;
resolution portability does not imply that a flat head or every old checkpoint
can be used at every size.

**B. Eight-checkpoint comparison contract.** Version the main campaign contract
for a configurable checkpoint count around eight, remove only its relevant
four-stage bound, and validate complete receipts/counts for the new schedule.
Keep bounded smoke budgets bounded. Allocate evaluation costs explicitly: going
from four to eight checkpoints doubles scheduled checkpoint evaluations, before
any seed/catalog/image-size change. Use new matched per-game calibration at the
realistic input sizes before choosing substantial main-run budgets.

**C. Broader strides and targeted small CNNs.** Qualify first strides1/2/4/8 and
the portable image path with independent GPU math/determinism checks before
learning runs. Run the controlled channel/residual family at the new practical
input contracts, keeping seeds and learners fixed. Do not assume that the old
tiny-image winner remains the best architecture or has the same relative speed.

**D. Cross-game native PROTEIN and fresh confirmation.** Use actual checkpoint
curves across the matched games/image conditions to guide broad frontier gains,
with task-regression visibility. Confirm selected graphs on new training replicas
and held-out evaluation suites. Profiling improvements are interleaved selectively,
with code/math acceptance before timing and no competing jobs during measurements.

These are preparation/implementation directions, not a new GPU launch or a
rewrite of the failed45-observation allocation. Historical low-resolution results
remain available as development evidence with their original scope.

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

The current operating requirements above take precedence: portable realistic
image input is the first implementation prerequisite. The original priority
discussion and first Q&A below remain as dated rationale; their4x4 minimum and
37-condition proposal are superseded as the practical target, not applied changes.

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

## Kinvert's questions, decisions and answers

October9 follow-up. The numbered questions below refer to the priorities above.
They record the requested direction for the next implementation. This Q&A does
not assert that new catalogs, strides, seed contracts or input shapes are already
implemented, and does not launch training. Historical packets retain their inputs.
The later clarification at the top of this document overrides the4x4-cell target:
new primary evidence must use varied, realistic game images, not tiny boards.

### 1) How would we do this?

**Answer:** give native PROTEIN one architecture-level score that reflects useful
quality throughout training across games, together with its measured training
cost. One concrete candidate protocol is:

1. Freeze per-game learner recipes, training seed lists, eligible drawings,
   checkpoint schedules, normalization anchors and time limits. Establish time
   limits from declared reference/calibration runs before candidate selection.
2. Train each shared architecture separately on every game and paired training
   seed. Evaluate every scheduled checkpoint on the assigned deterministic
   development episodes/drawings. Log process/checkpoint time, decisions, SPS,
   full config, graph, checkpoint hash and failures.
3. For each game/seed, compute the area covered by its observed score/time frontier
   within the fixed time/normalized-score rectangle: a two-objective hypervolume.
   This measures useful performance at many training costs rather than only the
   last score. Build it from actual checkpoints; no interpolation/extrapolation
   or invented neural evaluations. A rectangle left of the first observation
   contributes no demonstrated dominated area, not an asserted measured zero score.
4. Average these normalized areas equally across games and training seeds, and
   apply a predeclared penalty or eligibility rule for severe per-game regression
   against Nature and poor lower-tail drawing performance. Use a lower-tail
   summary rather than making one impossible stress drawing veto every model.
   Freeze penalty weights/tolerances before the new search; compare their behavior
   offline first. Retain Pong's lower/upper censoring analyses separately.
5. Submit that scalar score and the once-per-training-job monotonic cost to native
   PROTEIN. Report every actual per-game curve and coverage alongside the aggregate;
   the scalar is a search guide, not proof of a universal winning architecture.
6. Freeze architecture/checkpoint selection rules using development evidence.
   Confirm them on separate evaluation suites and fresh training replicas, never
   select best checkpoints using the final held-out episodes.

The supervisor/evaluator/native suggest-observe path already exists; the main
new work is the versioned aggregation/eligibility contract and adequate checkpoint
coverage. We must not replace scores inside the completed45-observation history
or mix observations from different objectives. This hypervolume-based proposal
is concrete enough to implement, but its exact penalties/time ranges still need
offline validation and a frozen experiment specification.

### 2) "Pong 1 pixel per cell is too small ... Make the smallest version ... 4X4 per cell or 8X8"

**Answer:** the cell-based one-pixel preset is **Connect4**, not Pong. Agreed on
the benchmark policy: future primary Connect4 comparisons should have a minimum
**4x4 cell pitch**, with no tiny2x2 piece preset in the primary set. A7-column,
6-row board is28x24 pixels at4x4; the usual6x6 cells produce42x36. An8x8 version
would produce56x48 and needs a larger image, for example58x48 with the current
one-pixel padding on each side.

For the existing preset table, the next primary catalog should use IDs
**0,1,3,4,5,6**: ordinary squares,4x4 pieces on6x6 cells, diameter6/4 disks,
6x6 X/O glyphs and4x4 cells. IDs**2,7,8,9** remain historical/optional stress
conditions; they include2x2 pieces,2x2 cells,1x1 cells and2x4 cells. Do not delete,
renumber or silently exclude them from the completed campaign. A separately
versioned future training/evaluation catalog is needed so all architectures use
the same declared reasonable conditions. With other games unchanged, this planned
primary set has37 conditions rather than the old41; existing readers still require
their captured old catalog until the new selection contract is implemented.

Pong has continuous paddle/ball geometry rather than board cells. Its corresponding
requirement is readable, adequately sized ball/paddle rasters, validated at each
chosen image size; simply relabeling it "4x4 cells" would not define the right test.

### 3) "Targeted things outside a normal PROTEIN sweep ... properly tabulate ... deterministically the same"

**Answer:** yes. Add a declared targeted-candidate manifest alongside native
PROTEIN proposals and run both through the same all-game preparation, native
training, deterministic evaluation and receipt audit. Targeted runs must be
identified as such, not attributed to optimizer discovery.

The initial matrix is C8/C16 × residual0/1 with K8/stride4/projection64 fixed,
plus quality and Nature controls. Within each game, hold learner/core settings,
budget/cadence, world/appearance assignments, model seeds and evaluation episode
IDs fixed across architectures. Policies take different actions, so later world
trajectories can diverge; deterministic comparison means identical seeded initial
conditions/RNG rules and repeatability, not forcing identical trajectories.
Tabulate game, drawing, training/evaluation seeds, graph, source/config/checkpoint
hashes, all checkpoint scores/bounds, times, SPS and failures. Keep5060/5090
measurements separate and preserve repeated timing blocks.

### 4) "We can increase from 4-8 to 1-2-4-8"

**Answer:** yes: expose first-stage strides**1,2,4,8** numerically through the
INI/native constructor. The current constructor asserts a minimum4; changing
only the sweep INI will fail. Update construction, shape/memory preflight and
independent CUDA forward/backward/graph checks together before training new shapes.

Smaller strides retain more spatial detail but increase feature-map work; offset
that with narrow early stages and deliberate later downsampling. For example,
a narrow stride2 stage followed by stride2 can return to the existing final
spatial scale. Reject or separately flag kernel/stride combinations with complete
blind spots when the primary search requires full pixel coverage. Adding these
options does not mean every stride1 model will be fast.

### 5) "Same seeds like7 and42 and69 ... eval ... OTHER seeds ...420 or1337"

**Answer:** adopt explicit reproducible seed roles. A reasonable development
training list is**7,42,69**, used for every architecture/game comparison. Keep
development evaluation suites fixed while selecting models. Reserve final
evaluation seeds such as**420,1337**, after checking prior experiment receipts
for use, and exclude their outcomes from search/checkpoint selection. Record
the exact per-episode assignments, not only the top-level numbers. Current panel
code generates eval suites from a base plus training-seed index; an explicit
multi-suite seed list is additional work, not already present.

We also need an independently reserved **confirmation training seed list**. Fresh
evaluation episodes alone cannot resolve the observed dependence on which seed
trained the model. Once7/42/69 influence selection, keep that role fixed and
confirm finalists using other predeclared model-training seeds. Common-looking
numbers have no statistical advantage; reproducibility and separation matter.

Keep appearance seed, model/action RNG seed, training-world RNG assignment and
evaluation world/action seed streams explicit. In the current native training
path, `base.seed` does not automatically reseed all environment spawn streams;
some use slot-based initialization. Do not claim fresh training-world seeds merely
because model seed changes. Exact evaluation adapters explicitly derive world
and policy RNG from suite seed/episode ID; this supports paired held-out episodes.

### 6) "Iteratively ... find and eliminate bottlenecks"

**Answer:** make profiling and optimization a repeated engineering loop: capture
an exclusive-GPU baseline, identify the largest actual cost, change one bottleneck,
verify math/gradients/determinism and then measure again on matched runs. Separate
rollout, learner, environment work, convolution/GEMM, patch/copy/activation traffic
and launch/synchronization overhead. Keep the before/after graph, source, compiler,
hardware, learner recipe, SPS and score/time curves in the results ledger. Fair
shared-backend optimizations also apply to Nature/other baselines. Short profiling
work must not overlap timed benchmarks or other users' GPU jobs.

### 7) "Everything is36X44 grayscale??? ... Including us ... I don't want ... restricted"

**Answer:** **yes, including ours and all local Nature/IMPALA/Impoola adaptations**
on the current six CNN games. The frame contract is height36,width44,channels1.
This is a restriction of our current native research implementation, not an
inherent limit of those CNN architecture families or all of PufferLib.

The restriction is real, not only a default: `flex.cu` starts construction from
`H=36,W=44,ci=1`; Nature has a matching static assertion and hard-coded layer
shapes; IMPALA has fixed spatial stage arrays; game observations use the same
macros. An INI resolution change alone would create inconsistent buffers/shapes.

Promote **input-shape portability to near-term infrastructure work**, ahead of
claiming a generally useful CNN and ahead of adding8x8 Connect4 cells:

- Carry explicit height/width/channel metadata from the environment to native
  encoder construction; use INI/env settings where supported. Compute convolution,
  pooling, flatten/projection and workspace sizes from it once at startup.
- Support multiple selected image sizes and grayscale/RGB layouts without silently
  resizing everything to36x44. Preserve the game state/rules and deterministic
  native in-memory raster path; no graphics window is required.
- Make our encoder and reference families consume the same images, channel/temporal
  input layout and dtype. Nature's valid convolutions require legal minimum sizes;
  compute those constraints rather than changing its architecture silently.
- Bind shape/channel/layout to metadata, checkpoint descriptors, suites, hashes
  and preflight. Flatten heads usually make weight dimensions resolution-dependent:
  configurable input shapes do not imply one checkpoint works at every size.
- Preserve the existing36x44 implementation/receipts while qualifying the new path
  on independent CUDA checks and matched GPU smoke runs. Retest speed and learning:
  current grayscale timing advantages cannot be assumed at larger RGB resolutions.

Implementation sequence now includes portable input metadata and the realistic
catalog contract, then extended strides/controlled candidates. No image resize,
renderer, checkpoint layout, native constructor or historical result is changed
by recording this Q&A; those are the next concrete code tasks.

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
