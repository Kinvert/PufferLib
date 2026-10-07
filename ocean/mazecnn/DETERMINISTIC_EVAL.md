# Deterministic native Maze evaluation

October 6, 2026. Exact adapter and GPU-free native host manifests/suite/audit
are implemented. Four normal float32 targets compile. Independent CPU game/
counter/raster/sanitizer checks and fourteen native-host/synthetic-audit tests
pass. The supervised checkpoint launcher supports GPU-free preparation and
retains failures. **No neural policy ran; GPU reset/reload/quota/repeatability/
learning remain pending. Both GPU holds remain.**

## Assigned level and episode contract

Rules `maze-native-v1` preserve original Maze movement/reward/area timeout and
level generation. Evaluation is `first-goal-or-native-area-timeout-v1`; every
assigned episode ends on its first goal or its native `2*width*height` decision
boundary. There is no administrative cap, lost survivor or complete-only mean.
Primary estimand is success fraction across all assigned IDs, alongside return,
duration and native-log-entry counts. This isn't pooled native `perf`.

Use an explicit contiguous level panel within the original table: `level_offset`
and positive `level_count`. Assignment `seed-rotated-cyclic-v1` uses
`rotation=mix64(seed XOR 0x6d617a656c657631) mod count` and
`level_id=offset+(episode_id+rotation) mod count`. A full cycle visits every
level once; a partial cycle is explicitly that fixed subset. Identity, not slot,
network/core shape, worker order or episode duration, determines the level.
The suite records actual generated width/height/horizon for every ID.

Evaluation copies the selected original level with its original spawn (1,1),
direction and maze. The game RNG starts at the shared identity-derived uint32
environment seed. No random level-selection draw is consumed at assigned start;
this is an explicit evaluation initialization, not stock training reset sampling.
The policy gets a separately derived Philox seed. Initial reward/action/logs/tick
are cleared and terminal=1 requests recurrent reset before first inference.
Neither level ID nor seed becomes a policy input.

During the episode, call unchanged original `puf_step`. Maze draws game RNG
only during terminal replacement. Both original draws are preserved and
validated; the replacement level isn't stepped. Host manifests retain the
expected two-draw terminal RNG so the auditor can check ending receipts using
the binary's actual libc semantics. Start/world/observation hashes are receipts,
not extra CNN features. Different runtimes' `rand_r` semantics need explicit
comparison; no cross-platform portability guarantee is inferred.

## Count the duplicated native log once

A native goal exactly at the area timeout logs twice. The adapter captures the
one returned reward/terminal, independent episode decisions and original start
horizon. A goal contributes success=1/return=1 once; timeout contributes zero.
It retains `native_log_entries=2` for that edge case and validates native logs/
lengths/RNG against both entries. Successful goals before timeout and zero-reward
timeouts have one native entry. Training and pooled historical scores are
unchanged. The auditor checks every assigned ID, endings, integer totals,
minimum goal distance, horizons, log multiplicity and exact initial receipts.
It doesn't replay a trained policy or certify neural inference.

## Held-out levels need training evidence

Original generation at N levels is a byte-identical prefix of a larger table
for matching `map_size`; a CPU fixture verifies this. Thus one possible future
protocol trains with `num_maps=7168`, evaluates the original full 8192 table
at IDs 7168–8191 and retains all four models' matching training configs/source.
This is a design example, not a launched or calibrated experiment. Current
Maze training recipes still use all 8192 maps and don't reserve unseen levels.

For `--purpose heldout`, suite preparation requires positive
`--training-level-count` and a disjoint panel offset. That count is a **declaration**,
not proof of which levels a checkpoint saw. Every suite marks exclusion
unverified; every audit returns `heldout_exclusion_certified=false`. The future
publication audit must verify actual captured training INI/table, matching
map-generation source, tuning/selection exposure and checkpoint provenance.
Never label an all-8192-trained checkpoint's tail-table evaluation unseen-level
generalization. Frame/trajectory splits don't create new level identities.

## GPU-free commands

```bash
# CPU environment/counter/raster only; fresh output, no neural model.
bash ocean/mazecnn/tests/run_exact_episode.sh build/mazecnn/FRESH_EXACT_GAME_ID

# Normal compilation, existing shared process-local dependencies.
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh mazecnn build/mazecnn/FRESH_DEFAULT --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPALA_CNN bash build.sh mazecnn build/mazecnn/FRESH_IMPALA --float
NVCC_ARCH=sm_120 NVCC_EXTRA=-DC4_IMPOOLA_CNN bash build.sh mazecnn build/mazecnn/FRESH_IMPOOLA --float
NVCC_ARCH=sm_120 bash build.sh maze build/mazecnn/FRESH_STATE --float

# These modes return before CUDA initialization or policy construction.
build/mazecnn/FRESH_DEFAULT eval_exact_info
.venv/bin/python ocean/mazecnn/exact_eval.py suite \
  --binary build/mazecnn/FRESH_DEFAULT --config FULL_RESOLVED.ini \
  --seed 56118 --episodes 1000 --slots 64 --representation 0 \
  --level-offset 0 --level-count 8192 --out build/mazecnn/FRESH_SUITE_ID

.venv/bin/python ocean/mazecnn/tests/test_exact_eval.py \
  --binaries build/mazecnn/FOUR_CURRENT_BUILDS
.venv/bin/python ocean/mazecnn/tests/verify_host_suites.py \
  --binary-dir build/mazecnn/FOUR_CURRENT_BUILDS --episodes 1000 \
  --out build/mazecnn/FRESH_HOST_START_ID
```

Use full matching INIs, not the partial environment overlay alone. Suites freeze
source/effective configs, actual native starts and hashes; existing output
directories are refused. Native host manifests ignore policy/core settings and
do not execute a CPU neural network. Six host tests cover all drawing worlds,
three pixel families at H32/128/512 with multiple layer counts, original state,
independent last-ID starts (65 IDs/64 slots), cyclic coverage, invalid world/
range/mixed inputs, source closure and complete/duplicate/malformed audit rows.
Synthetic receipts are audit fixtures, not measured scores. The additional
supervisor tests cover 24 family/core preparations plus the state control,
prefix-table expansion, invalid checkpoints, reservation/query rejection,
quota/summary/parameter/clock checks, immutable copied inputs/tooling and
retained timeout receipts. GPU queries and policy execution are replaced by
explicit test doubles in those supervision tests.

## Prepare an existing checkpoint without GPU use

```bash
.venv/bin/python ocean/mazecnn/exact_eval.py run --prepare-only \
  --binary MATCHING_NATIVE_BINARY --config FULL_TRAINING.ini \
  --checkpoint EXISTING_WEIGHTS.bin --family flex \
  --suite SUITE/suite.json --out build/mazecnn/FRESH_CHECKPOINT_PACKET
```

This copies the full training INI, suite and launcher tools, preserves every
policy/core setting, preflights finite float32 bytes and regenerates the suite's
exact starts with the supplied binary. It writes an isolated evaluation INI and
hash receipts. It does not construct a neural model, certify checkpoint shape,
query a GPU or measure performance. No Maze-trained checkpoint currently exists
in this preparation; supervision tests use tiny explicitly synthetic weights.

For a declared held-out suite, the supplied training INI must use exactly its
`training_level_count` prefix and the same `map_size`. Only the evaluation INI
expands `num_maps` to the suite's full table; the original training INI is kept
unchanged. A development suite instead requires identical table settings.
Binary start hashes must match the frozen full-table manifest. These checks
prevent accidental world changes but don't prove that supplied weights actually
came from that INI, or that tuning excluded those levels. Reports continue to
mark `heldout_exclusion_certified=false`.

## Native execution gate

The compiled `eval_exact` uses the original custom vector shared table, one
agent per slot, float32, a matching checkpoint/config and one GPU/no selfplay.
It evaluates in fixed-size waves, initializes actor RNG per identity, retains
the last partial wave, stops stepping completed slots and waits for every
assigned episode's native terminal. Batch loop duration is each wave's maximum
assigned native horizon. It verifies checkpoint byte count/finiteness, native
action bounds and counters, flushes every receipt and emits progress/summary.
Allocation occurs outside episode stepping. No bare native command is authorized
under the holds.

The external `run` supervisor implements the shared exclusive reservation,
GPU idle/query rejection, immutable checkpoint/config/suite/tooling preflight,
finite-byte checks, process-group timeout, monotonic process receipts and
failure retention. Successful aggregation requires exactly every assigned ID,
one native summary and matching parameter byte count. Partial failures/timeouts
retain inputs/logs/receipts and produce no successful score report. Evaluation
process seconds are separate from training time. Its GPU path is unqualified;
do not omit `--prepare-only` unless Kinvert explicitly schedules execution.

Next schedule GPU acceptance for all four encoders/state, several core sizes,
65/64 quotas, exact repeats, eager/graph, independent tail replay, goal-at-timeout,
invalid weights and reset behavior. Encoder-5 remains a separate 5090 gate.
Passing host manifests and counter tests doesn't establish any of those.

```bash
# Audit previously scheduled native receipts without GPU use.
.venv/bin/python ocean/mazecnn/exact_eval.py audit \
  --suite SUITE/suite.json --episodes RUN/episodes.csv
```

Calibrate a fair common learner and learning-scale budgets before comparative
full curves/seeds. Keep source/build/time boundaries, failed seeds, tuning costs
and losing tasks. No dominance confidence band or SOTA follows from this work.
