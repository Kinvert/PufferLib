# Staged learning launcher and bounded RTX 5060 validation

October 7, 2026. Implemented the real learning-control → native cross-game
architecture search → paired-reference → complete-curves workflow. Use
[NEXT_5090_LEARNING_FEEDBACK.md](../../../NEXT_5090_LEARNING_FEEDBACK.md).
No full-budget 5090 learning campaign has executed as part of this work.

## Actual local checks

All GPU work here used G240's RTX 5060, driver591.86, CUDA compiler12.8.93,
explicit sm120, float32 production callbacks and the existing runtime/NCCL.
No CUDA/system/Torch/dependency installation or production kernel change.
CPU work was scalar metadata, artifact processing or host/mock control-flow
testing; independent CNN calculations ran in CUDA.

| Allocation | Completed work | Outcome |
|---|---|---|
| Fresh real-launcher preparation | Native bridge, six normal game targets, three metadata tools, two CUDA test libraries; both full-budget 24-job panels registered | Pass; 656 evaluations per panel remain unexecuted |
| Initial independent encoder checks | Quality, Nature, three-stage residual/max/average/global-pooling fixture; 42 process/case instances, 168 native calls | Pass, raw traces/gradients/device parameters retained |
| Broader native feedback loop | Three distinct CNNs; 18 trainings, 123 drawing evaluations, 36 repeat/eager checks, 2,703 assigned episodes | Pass; 208.753s campaign, 23.010s training-process time |
| Per-proposal CUDA checks in that loop | B1/B64/B2048, two fresh workers per CNN; 42 process/case instances, 168 native calls | Pass before any candidate training |
| Current Nature numerical contract | 14 process/case instances, 56 calls, original rtol3e-4/atol3e-5 | Pass |
| Paired quality/Nature per-game-learner path | 12 trainings, 82 drawing evaluations, 24 repeat/eager checks, 1,802 assigned episodes | Pass; 119.342s campaign, 14.602s training-process time |
| Final host/scalar/mock regression suite | 68 tests, plus four actual native host optimizer tests | Pass |

Both training allocations used 65,536 decisions per job and one scheduled
checkpoint, all six games and all 41 drawings, 17 assigned episodes/16 slots.
Feedback seed65173/evaluation75173; paired controls seed66173/evaluation76173;
native full-catalog appearance seed35173. Pong uses recipe-A scalar values;
Flappy horizon64; Maze horizon256; all use 64 actors/M2048/H128/L1.
These are short execution checks, not learning/frontier experiments.

The actual proposed graphs were:

| Trial | Effective CNN | Equal-game score | Once-per-job checkpoint cost |
|---|---|---:|---:|
| happy-cat-1 | One C8 8×8/s4 layer, flatten, projection32 | 0.010836 | 6.934867s |
| mystic-tree-2 | C32 2×2/s8 + residual; C16 3×3/s1 + residual/max pool; global pooling, projection32 | 0.005052 | 7.868755s |
| swift-fox-3 | One C8 6×6/s4 layer, flatten, projection16 | 0.010444 | 6.615474s |

There were no effective-graph duplicates. Native optimizer receipts show the
default, one random proposal, then a GP proposal with two observations. The final
unused proposal replayed all three observations and reports three GP observations.
Thus different architectures actually train on every game, aggregate all drawing
outcomes and return combined quality/time feedback to native PROTEIN.
The near-zero scores do not establish usefulness or an advantage over Nature.

## Retained evidence

Small copied reports, arrays inventories, input/build hashes and process receipts
are under `receipts/`. Raw arrays/binaries/configs/checkpoints/outcomes are retained
in these ignored local packets, not recreated or overwritten:

The new launcher/feedback/current-Nature/paired-control packets are also archived
as `build/hardware-artifacts/learning-launcher-5060-20261007.SMy1arPK.tar.gz`,
SHA256 `2ce39dfa97a951a8b76c4ee8f09d11c24afbd9008f874098cde4a3ed517852a6`.
The archive is local evidence, not a committed binary dependency or a remote
launch packet. The 5090 regenerates its own builds/allocations from source.

- `build/learning-feedback/verified-launcher-20261007/`: fresh build receipts,
  graph bundle and registered real calibration/reference allocations.
- `build/learning-feedback/validation-5060-20261007/` and its separate review:
  full three-trial numerical/learning/evaluator/native-feedback evidence.
- `build/learning-feedback/nature-current-gate-5060-20261007/`: stricter Nature
  contract actually executed and audited.
- `build/learning-feedback/paired-controls-5060-20261007/` and its review:
  baseline routing, per-game learners, full layout and evaluator evidence.
- `build/connect4cnn/flex-validation-{quality,nature,three-stage}-20261007/`:
  original independent checks and their original bundle/source contract.

Initial bundles used rtol8e-4/atol6e-5 for both models. Their original receipts
are unchanged; Nature's retained arrays were also checked offline at the stricter
original Nature tolerance. The subsequent current-bundle Nature allocation above
executes and audits that stricter tolerance directly. This is not a retrospective
tolerance change. The failed unconditional H256/B2048 packet remains failed.

Two earlier preparation directories are preserved. A mocked combined-view test
exposed an omitted hardware field;
it was fixed before the final build/preparation and GPU validation. The final
combined-view regression retains all candidates, points and costs. Mocked
controller tests also verify successful stage ordering and that math failure,
missing results, a lucky seed or one unlearned game stop before search.

## Qualification

The numerical contract is [CNN_GRAPH_ACCEPTANCE.md](../../CNN_GRAPH_ACCEPTANCE.md):
independent local forwards/backward on actual float32 branches, plus full smooth
float64 graph/finite differences. It is scoped to executed encoder4/Nature H128
cases. It does not independently validate the whole policy/core/optimizer,
H256, encoder5, arbitrary unseen graphs or paper claims. Every new proposed CNN
must pass its own gate on the target hardware before training.

The staged real run uses the same full budgets/annealing within each game across
architectures and first requires informative two-seed controls on every game.
Insufficient learning stops with evidence; no dropping games or lowering the
threshold. Adaptive discovery seeds and two-seed descriptive curves still require
fresh held-out confirmation before any publication-level frontier claim.
