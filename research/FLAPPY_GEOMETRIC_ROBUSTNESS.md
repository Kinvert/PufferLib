# Flappy geometric robustness preparation

October 6, 2026. Three additional headless drawings extend FlappyCNN from four
to seven IDs: rounded bird, outlined pipes, and both together. This broadens
the local panel to six tasks and 41 drawings. No policy, GPU query/inference,
training, dataset generation or pretraining ran. Both GPU execution holds remain.

[Drawing contract](../ocean/flappycnn/REPRESENTATIONS.md) ·
[Durable evidence](results/flappycnn/geometric-preparation-20261006/README.md).

The renderer changes only image construction. Original Flappy game rules,
physics, spawns, RNG, rewards, collision geometry and terminal cap are retained.
Images stay one-channel 36×44. New shape variations aren't invertible versions
of the old raster and may change the difficulty of perception; they are not
learning evidence or reduced-FLOP claims. Default mixing remains the original
four-ID catalog; explicit `representation_mix_catalog=1` selects all seven.

## Executed checks

- Native environment/raster parity, repeated traces and ASan/UBSan pass for all
  seven fixed drawings and legacy/expanded mixed catalogs. An independent
  long-double pixel oracle covers 21 complete new-geometry fixtures, including
  tiny birds, clipping, overlap and draw order. No CPU neural model runs.
- Before/after SHA256 stream hashes match exactly for each old drawing across
  49,152 tested frames (196,608 total). Pixels were streamed into the hash tool,
  not retained as a dataset. This is sampled raster identity, not proof for
  every reachable world or GPU determinism.
- Normal float32 native builds compile for quality, Nature, IMPALA, Impoola and
  original state. Source/build/config/compiler receipts are preserved. Build
  success is not numerical, reload, fit or learning validation. The build-source
  snapshot precedes later host-test expectation edits; keep those receipts intact.
- 107 host/configuration/native-manifest tests pass in 57.908 seconds. Earlier
  reports remain: three expectations assumed the old catalog size; a subsequent
  invocation omitted process-local NCCL library paths and produced 21 launch
  errors. The final invocation sources the existing runtime helper. No dependency
  or system setting was changed and no tolerance/assertion was relaxed.
- Actual mixed six-game preparation retains 24 planned training jobs, 41 fixed
  target suites with 1,000 assigned starts each, 164 bindings and 124 native
  family/world/raster comparisons outside Connect4. Two plumbing checkpoints
  mean 328 planned evaluations; zero executed. Mixtures/learner/core/budgets are
  matched across all four families, and every checkpoint targets every drawing.

## Historical catalog compatibility

Audits now derive counts from hashed native headers captured in each packet,
instead of reinterpreting old packets using today's global catalog. Four actual
retained packets inspect offline with Flappy IDs 0–3 unchanged. Current packets
reject old Flappy binaries lacking the seven-ID metadata. Old suite v1 remains
four-ID; new v2 explicitly records seven. This preserves past evidence rather
than silently broadening its conditions.

For the current inventory, all-drawing independent training is 164 jobs per
seed; mixed training or drawing-0 training is 24 jobs per seed, followed by 164
fixed-target bindings per seed. These are different experimental questions.
Drawings are correlated conditions, not additional independent training seeds.
Retain one training-time receipt per checkpoint when evaluating many drawings.

## Next measured step

In a separately scheduled GPU window, accept the pending deterministic
evaluators and shared math/memory/reload paths before collecting learning
curves. Calibrate one shared learner recipe per game across all four frozen
encoders. Freeze the selection/held-out game and drawing split before using
results to choose a model. Report complete paired-seed score/time curves,
failed/censored cases and baseline backend efficiency. The new drawings do
not establish a cross-game Pareto advantage, certified dominance or SOTA.
