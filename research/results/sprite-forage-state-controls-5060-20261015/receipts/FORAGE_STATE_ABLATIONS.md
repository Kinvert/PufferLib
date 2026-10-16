# Partial-state attention controls

Kinvert requests getting partial-state learning working before changing pixel
resolution. These are native GPU state policies, not CPU neural experiments
or pixel/downsampling tests. All edits are in `/home/claude/cnn`.

## Action and observation contracts

The opt-in `FORAGE_ATTENTION` build now accepts compile-time choices:

| FORAGE_GAZE_MODE | Action heads | Meaning |
|---:|---|---|
|0|5,300|Original movement + absolute row-major gaze center|
|1|5,5|Movement + gaze noop/up/down/left/right, one cell, clipped per axis|
|2|5,20,15|Movement + separate absolute cell-X/cell-Y choices|

`FORAGE_RELATIVE_GAZE=0` preserves normalized playerX/playerY/gazeX/gazeY.
At1, the last two values become signed `(gaze-player)/(dimension-1)`.
The player world coordinates remain; gaze world position is reconstructible.
All partial cases expose exactly17 visible tile IDs +4 geometry floats.
There is no one-hot change, hidden coordinate list, new reward, diagonal
gaze action or recenter shortcut. All actions consume the same game tick;
reset starts gaze on the player and consumes no extra world RNG.

The original normal build is unchanged when FORAGE_ATTENTION is absent.
Action shapes differ between binaries and checkpoints are not interchangeable.
The old absolute/absolute path remains byte-compatible in the scoped test.

Build each fresh variant from the retained compatible core:

```bash
bash research/build_forage_attention.sh \
  build/sprite-forage-speed-5060-20261015/source \
  build/sprite-forage-controls-build-20261015-2-1 2 1 1
```

Trailing arguments are gaze mode, relative coordinates, exact evaluator.
Use a fresh output directory. The last flag copies the Forage-specific
`research/sprite_attention_eval.cu` into the captured source and enables its
existing research hook. No main core overlay or normal upstream PR is made.

## Matched first learning allocation

Seven jobs: every combination of three gaze interfaces and two coordinate
encodings, plus a full-state control. Every run starts fresh, seed7,
83,886,080 decisions,256 agents/one buffer/four threads, H64/MB2048,
H128/L1, synchronous float32. All use the same historical tuned state
learner, reward values and676-step training episode cap. LR anneals across
the same number of epochs, not a throughput-dependent clock. Eight
checkpoints occur at10,485,760-decision increments; only the final one is
used in the primary comparison.

This targets **approximately two minutes**, not exactly120 seconds.
Fixed decisions preserve deterministic training and identical LR progress.
Actual monotonic process time/SPS and native metrics are retained separately.
The head shapes necessarily change parameter counts and total action entropy;
the same entropy coefficient is not a claim of identical exploration.

| Case | Parameters |
|---|---:|
|Partial, absolute300|91,008|
|Partial, directional5|53,248|
|Partial, factoredXY|57,088|
|Full state, unused absolute300 gaze|106,368|

The full-state control has141 inputs and two heads; it is not the historical
67,968-parameter one-head vanilla policy. The six partial policies have21
inputs. Geometry encoding does not change parameter count.

## Exact matched evaluation

The new Forage evaluator explicitly seeds each episode using fixed ID and
suite seed. It resets recurrent state and records world/observation hashes,
both/all three action heads in the action hash, decisions, perf, exits,
deaths and return for every assigned episode. It never selects the first
episodes to finish. All policies receive128 IDs each at suite seeds420/1337,
16 slots,512-step cap:256 episodes per final policy. These are development
suites, not untouched publication tests. Scalar native manifests independently
enumerate starts; world fields must match across every action/observation
variant, including the full-state control. Visible-observation hashes may
differ between coordinate encodings and full/partial observation.

The evaluator is a temporary Forage-specific adaptation of the retained
sprite exact evaluator. It uses the current player_idx field, copies all
action heads with the correct stride, and hashes/validates every head.

Preceding GPU smoke: seven65,536-decision trainings, each followed by17
assigned episodes in graph/repeat/eager modes and an independently reseeded
episode16 tail replay. All28 evaluator processes/364 episode executions
match per-row exactly. The original control also matches the old binary's
65,536-decision checkpoint bytes. Six ASan/UBSan variants cover144,000
matched world steps,9,000 skin/center combinations, clipped directional
edges, factored XY positions, geometry, crop pixels and hidden-cell isolation;
seven normal environment regression suites also pass.

## Commands and scope

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/forage_state_ablations.py prepare FRESH_PACKET
.venv/bin/python research/forage_state_ablations.py smoke FRESH_PACKET
.venv/bin/python research/forage_state_ablations.py run FRESH_PACKET
```

Preparation is GPU-free scalar game/raster manifest work. Smoke/run take the
shared GPU lock, require idle status and refuse to overwrite evidence.
No systematic neural math change, dataset, downsampling,5090 launch or system
dependency change is involved. Python is process/config/artifact glue only.

Current packet: `build/sprite-forage-state-ablations-5060-20261015/`.
Preserve every weak/failed result and checkpoint. This initial allocation is
one fixed training seed, not independent confirmation or a generalization /
Pareto/publication claim. Larger claims require the predetermined seven
training replicas and new evaluation worlds; never sweep or replace seeds.

## One-hot follow-up

After completing the initial seven jobs, a separate three-case allocation
tests `FORAGE_STATE_ONEHOT=1`. Each of the17 visible cells receives13
categorical flags instead of one scalar ID; four absolute geometry floats
remain, for225 total state inputs. No hidden targets, reward labels or
nearest-object helper are added. Pixel observations are unchanged.

The build helper's sixth argument selects this opt-in flag, e.g. append1
after `gaze relative exact` to build one-hot state. The default is0, so the
existing normal/scalar contracts remain available. Three new sanitizer suites
check cell-category identity against the exact pixel crops, hidden-cell
isolation and all three action layouts. GPU smoke repeats the same exact
graph/eager/tail protocol before full learning.

`research/forage_state_encoding.py prepare|smoke|run FRESH_PACKET` reuses the
same native execution supervisor, with exactly three absolute-geometry cases.
It copies original recipes unchanged and proves every world matches its scalar
control. Inputs/parameters differ:117,120 /79,360 /83,200 parameters for
absolute300/directional5/factoredXY. Budget remains83,886,080 decisions,
seed7, with eight checkpoints and256 exactly assigned evaluation episodes.

The three-case packet is `build/sprite-forage-state-onehot-5060-20261015/`.
Both panels can be audited without executing a native process:

```bash
.venv/bin/python research/audit_forage_state_ablations.py full PACKET
```

The audit binds each one-hot case to its scalar reference's learner and
worlds. Relative/one-hot interactions and independent training replicas are
not tested by this follow-up. Keep the distinction between a useful observed
development improvement and statistically qualified generalization.
