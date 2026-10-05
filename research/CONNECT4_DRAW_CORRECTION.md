# Connect4 draw correction — October 5, 2026

**Corrected game rules are a new benchmark revision. Do not combine their
learning curves with historical Connect4 results.** No GPU execution, training,
or new model evaluation was performed for this correction.

## Defect and evidence

Both `ocean/connect4/connect4.h` and `ocean/connect4cnn/connect4cnn.h` compared
occupied cells against decimal `4432406249472` to detect a draw. That value is
`0x40800000000`: only bits 35 and 42, the bottom cells of columns 5 and 6
(zero-based). It can terminate a game after the first player/opponent pair,
while failing to recognize a genuinely full board. With six playable cells and
one unused bit per column, full occupancy is `0xfdfbf7efdfbf`.

This is inherited environment behavior, not an encoder bug. The reconstructed
confirmation source `b2fa7787a754d36362374ac271ea6c7b23beeb25` contains the same
incorrect literal. In existing G240 state and pixel environment traces, each
4,096-transition trace contains 596 completed episodes and **23 one-decision
draws**. These counts describe an environment-test action stream, not a trained
policy's error frequency. The [retained evidence](results/connect4cnn/draw-correction-20261005/README.md)
contains those rows and full-trace hashes.

The previous state/pixel parity tests passed because both versions shared the
defect. Their terminal checks allowed zero reward without asserting full-board
draw semantics. Sanitizers check memory/undefined behavior, not game correctness.

## Changes and validation

- Replace the draw constant in both environments. No CNN kernels, observation
  format, learner settings or RNG implementation change. Opponent search also
  calls `draw`, so its behavior can change as a consequence of the corrected rule.
- Add independent coordinate-derived occupancy fixtures: a full board, empty
  board, all 42 single-hole boards and all 21 pairs of bottom cells.
- Test 32 seeded first moves in the last column: no opening may terminate.
- Construct a nonwinning 40-piece board with only one playable column; the last
  player/opponent pair must produce zero reward, one terminal/log record at
  decision 21, and a reset board. Test both observation implementations.
- Retain all ten representation checks, mixed-appearance checks, seeded traces,
  state/pixel parity, repeatability, ASan and UBSan.
- Exact evaluation and its artifact auditor reject draws before decision 21
  from the required empty-board start. GPU execution of that guard is pending.
- New measurement manifests record
  `environment_rules = connect4-full-board-draw-v2`; reports display the rules
  label. Old unversioned manifests remain readable for historical inspection but
  cannot enter the current build/execution path. CSV/RNG schema version 1 is
  unchanged; the manifest/source identity distinguishes game semantics.

The new native regression failed against the old code at `assert(draw(full))`.
After correction, the full native CPU environment/sanitizer suite and twelve
configuration/artifact checks pass. Twelve synthetic chart views plus empty-data
handling also pass. These are environment and reporting checks, not CPU neural
model tests or evidence of GPU numerical correctness. The normal native
float32 trainer build succeeds; no compiled trainer is executed.

## Consequences for the paper and the 5090

Preserve every historical result, checkpoint and provenance record. Describe
those results as the legacy native Connect4 task with the draw defect; they
remain observations of that implementation. The ranking impact is **unknown**:
different policies can encounter the erroneous termination at different rates,
so sharing the defect does not prove the comparison was unaffected.

Future matched state/pixel or CNN comparisons must use corrected rules for every
model, with fresh manifests and builds. Do not silently reevaluate old policies
under corrected rules and label the output a reproduction. Such a later study
would be explicitly labeled transfer from legacy training to corrected evaluation.
Changing the game also means the old expected scores are not acceptance targets.

The 5090 remains busy; preparation only. After safely updating the Kinvert clone,
run the CPU **environment** checks in `ocean/connect4cnn/tests/run_all.sh`, then
prepare fresh campaigns as instructed in [NEXT_5090_TASK.md](../NEXT_5090_TASK.md).
Do not repeat old campaigns or launch training. Encoder-5 GPU math/reload and
statistical-inference gates remain open.

Two separate audit findings remain follow-up work: failed training currently
prevents retaining earlier checkpoint cells in the new measurement auditor, and
the exact evaluator's worker-count repeat does not exercise parallel environment
stepping because it steps environments serially. Neither is fixed by this change.
