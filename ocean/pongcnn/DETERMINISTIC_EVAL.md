# Native Pong exact-match evaluation

October 5, 2026. Exact adapter and GPU-free suite/audit commands implemented.
Four float32 native targets compile; CPU environment/counter/pixel/sanitizer and
seven initial native-host/synthetic-audit checks pass. The supervised launcher
now passes thirteen host/glue tests. **GPU inference, recurrent reset, reload,
quota/repeat/eager acceptance remain pending.**
No new Pong training or policy evaluation occurred. Both execution holds remain.

October 6: native pixel info advertises seven fixed appearances; state advertises
one. Old binaries without the field retain their five-ID interpretation. Suite
preparation/reload rejects IDs 5/6 against legacy binaries before GPU access.
Fourteen host/glue tests pass on both versions; 1,000 assigned starts per drawing
match across new build families/core sizes/state and independent tails. IDs 0–4
retain byte-identical historical CSVs. [Texture preparation receipts](../../research/results/pongcnn/texture-preparation-20261006/README.md)
are separate from prior builds. No GPU evaluator gate is lifted.

## Fixed suite and match boundaries

Rules `pong-native-v1` preserve original native Pong, opponent, rewards and
physics. This is not ALE Atari Pong. Evaluation
`first-match-terminal-or-decision-cap-v1` freezes the world, max_score, frame
skip, fixed appearance, suite seed, explicit IDs and agent-decision cap. Stock
is first to 21 points; no final paper cap/recipe has been calibrated.

The exact start seeds game RNG by episode identity, clears helper/control/log
state, and calls the original reset. One `rand_r` draw sets the initial vertical
ball direction. The actual post-reset RNG/world/image hashes come from the
selected binary's host manifest; no Python renderer or assumed ABI conversion.
Actions receive a separate identity-derived Philox seed. Neither stream seed
depends on network shape, slot or ordering. Different trajectories can consume
different subsequent game draws. Native terminal reset consumes a draw for the
replacement world's start; each new assigned identity is explicitly reseeded.

Original `puf_step` is called unchanged. It increments `tick` once per agent
decision but resets that counter after every point. Thus its ending
`episode_length` describes the final rally. The adapter's independent counters
retain **whole-match decisions**, the final rally length and each side's points.
Point rewards update independent totals; nonterminal live scores and terminal
logs are cross-checked. A terminal log is read after auto-reset, without trying
to read ending scores from the next game's live zeros. No replacement match is
stepped. First terminal on the cap decision is complete, not capped.

A decision cap stops at a native action boundary without a fake terminal, loss,
penalty or reset. Since points return early from the physics skip loop, multiplying
decisions by frame skip isn't exact frame count. The evaluator doesn't claim
physics-frame duration. Training/game code and legacy pooled-v1 data are unchanged.

## Scores and capped outcomes

Primary suite estimand is `mean-final-match-point-fraction-with-censoring-bounds-v1`.
Completed fractions are right_points/(right_points+left_points), averaged by
assigned match, not pooled over points. This is not match win rate. The raw
native float32 perf is separately recorded for completed matches only.

A capped partial fraction is not the final match fraction. At first-to-M,
right=r and left=l below M, possible final fractions range from
`r/(M+r)` to `M/(M+l)`. A completed match contributes its exact fraction to
both bounds. A capped zero-point match contributes [0,1]; its native final
perf column stays empty. Averaging these bounds over **every assigned match**
bounds the suite's eventual mean. These are deterministic censoring bounds,
**not confidence intervals**. Additional training/evaluation/frontier sampling
uncertainty needs calibrated analysis; don't certify dominance from these alone.

Also report accumulated net points through the cap, decision lengths, completed/
capped counts and known match wins. Net points is a separate bounded-return
estimand, not a substitute chosen after results. Match-win bounds retain all
caps as unknown outcomes. No capped game is dropped, assigned a convenient
win/loss or included in a misleading complete-only primary average.

Native receipts use exact hexadecimal floats, identity/world/image/RNG/action
hashes and every completed/capped ID. The external auditor recomputes bounds
from integers and uses `math.fsum` for order-independent aggregation. Native
`PONG_EXACT_BOUNDS` is an auxiliary ordinary-double running summary; audited
integer-derived episode bounds are authoritative for analysis.

## GPU-free checks and preparation

```bash
# CPU environment/pixels only; no neural policy executes.
bash ocean/pongcnn/tests/run_all.sh
bash ocean/pongcnn/tests/run_exact_episode.sh build/pongcnn/FRESH_HOST_ID

# Compilation only, using normal dependencies and process-local runtime paths.
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh pongcnn build/pongcnn/exact-default --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN \
  bash build.sh pongcnn build/pongcnn/exact-impala --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN \
  bash build.sh pongcnn build/pongcnn/exact-impoola --float
NVCC_ARCH=sm_120 bash build.sh pong build/pongcnn/exact-state --float

# Returns before CUDA device initialization.
build/pongcnn/exact-default eval_exact_info
```

Default binary dispatches quality/Nature via the matching full training INI.
Original state is eight observations with its documented information differences.
All CNNs see the same `[1,36,44]` raster. Encoder 5 keeps separate 5090 gates.

The environment harness checks both scoring sides, point resets, terminal scores,
whole-match versus final-rally length, zero/nonzero-point caps, terminal-at-cap,
independent reset RNG, dirty starts, seven rasters and 96 stock-reference action
trajectories. Independent terminal-score enumeration covers every capped score
pair for first-to-1 through first-to-21. Repeat/ASan/UBSan checks pass.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/pongcnn/tests/test_exact_eval.py \
  --binaries build/pongcnn/exact-preparation-20261005-v2
```

The initial seven tests exercise actual native host manifests across all drawings/build
families and original state worlds, independent final-ID starts, bad physics/
overflow/mixed-suite rejection, fixed source/config/hash closure, quotas below
and above a batch, wrong/duplicate/missing receipts, zero-point caps and bounds.
Synthetic rows are temporary audit fixtures, not measured policy outcomes.
Six additional launcher checks cover 24 pixel family/core settings and original
state preparation, malformed/nonfinite weights, busy/query/reservation failures,
summary/parameter/count mismatch, copied-input changes, missing/backwards clocks
and timeout retention. GPU queries and policy processes are mocked in launcher
failure/report tests; temporary finite dummy weights do not validate shapes.

For current seven-ID binaries, use a fresh directory for all four normal builds
and pass it with `--binaries` to `test_exact_eval.py`. The retained October 5
binaries exercise legacy compatibility, not the new images. The additional
test rejects new-ID suites with a legacy appearance count. To retain all native
host initialization manifests without policy execution:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/pongcnn/tests/verify_host_suites.py \
  --binary-dir build/pongcnn/NEW_FOUR_BUILDS \
  --legacy-binary-dir build/pongcnn/exact-preparation-20261005-v2 \
  --episodes 1000 --out build/pongcnn/FRESH_HOST_START_ID
```

Legacy binaries are local evidence, not tracked executables delivered by a
pull. Cross-machine verification needs those binaries or independently rebuilt
matching retained source. This doesn't substitute for GPU runtime acceptance.

Use a full resolved `FULL.ini`, e.g. a job's `config/default.ini` from the
GPU-free multi-task preparation tool. Partial `config/pongcnn.ini` alone isn't
sufficient. Output directories are immutable; regenerate after source changes.

```bash
.venv/bin/python research/prepare_pixel_robustness.py \
  --out build/pixel-robustness/FRESH_PANEL_ID

# 4096 is a development fixture cap, not a justified confirmation cap.
.venv/bin/python ocean/pongcnn/exact_eval.py suite \
  --binary build/pongcnn/exact-default --config FULL.ini --seed 56111 \
  --episodes 1000 --slots 64 --representation 0 --max-decisions 4096 \
  --out build/pongcnn/FRESH_SUITE_ID
```

The suite stores source/effective manifest INIs, physics, cap, estimand, IDs,
actual native starts, binary identity and hashes. All seven fixed appearances
can be prepared separately. Host manifests construct no policy and prove no
GPU batch/recurrent-reset behavior. No pretraining dataset is generated.

## Native execution and remaining acceptance

`BINARY eval_exact --headless` is implemented for the native CPU environments.
It uses the normal isolated full-INI path and `base.load_model_path`; no positional
checkpoint argument. `[eval_exact]` requires `seed`, `episodes`, `episode_offset`,
`slots`, `max_decisions`, `output`. Fixed appearance is required even for mixed
training. Preserve the matching encoder/core and three-action head. Do not load
Connect4/Flappy checkpoints. Runtime requires float32, no selfplay, one GPU and
one agent per slot; it validates weights, actions, counters and exact quotas.

**Don't run the raw mode under either execution hold.** The supervised launcher
now implements idle/reservation checks, finite checkpoint preflight, frozen
inputs/tooling, native manifest regeneration, process-group timeout, monotonic
evaluation timing and summary/parameter/CSV audits. Pending GPU acceptance
is explicit in every result. Partial/time-out failures retain rows and logs
without a successful aggregate/report. The independent per-match audit supplies
reported bounds; native auxiliary floating-point totals never override it.

```bash
# GPU-free preparation using existing Pong weights and matching full training INI.
# The binary host manifest runs; no GPU query or neural policy runs.
.venv/bin/python ocean/pongcnn/exact_eval.py run --prepare-only \
  --binary build/pongcnn/exact-default --config RESOLVED_TRAINING.ini \
  --checkpoint PONG_CHECKPOINT.bin --family flex --suite SUITE/suite.json \
  --out build/pongcnn/FRESH_PREPARED_CHECKPOINT_ID
```

Preparation preserves all policy/core fields and training appearance metadata,
applies only the suite's fixed evaluation drawing and retains immutable copies.
It checks finite float32 bytes and host starts, **not parameter shape or model
math**. Native execution checks exact weight count before inference. Prepared
directories cannot be reused as completed runs. After an explicitly scheduled
GPU window, use the same command without `--prepare-only` and a fresh output;
`--timeout` bounds the whole native process, and `--eager` disables graph capture.
Original state uses `--family state`, matching state INI/binary and a state suite.
No successful partial suite, silent retry or old pooled score substitution.

Then, after explicit scheduling, GPU acceptance must cover all four encoders,
original state and multiple cores; 65 IDs with 64 slots; exact repeats, eager/
graph, independent tail replay, zero/nonzero-point caps, terminal-at-cap and
malformed weights. No CNN numerical claim follows from compilation/host tests.
Calibrate cap/common learning recipe and freeze matched paired-seed checkpoint
frontiers before comparative data; keep paper confirmation/inference gates open.

```bash
# Audit a previously scheduled native result, without GPU use.
.venv/bin/python ocean/pongcnn/exact_eval.py audit \
  --suite SUITE/suite.json --episodes RUN/episodes.csv
```

This checks assigned identities, starts, ending counters and score bounds, not
runtime/checkpoint source provenance or SOTA. Existing historical Pong results
remain separately labeled; this work hasn't produced new learning evidence.

[Retained native/host preparation receipts](../../research/results/pongcnn/exact-preparation-20261005/README.md).
[Retained supervised launcher tests and source](../../research/results/pongcnn/exact-launcher-20261005/README.md).
