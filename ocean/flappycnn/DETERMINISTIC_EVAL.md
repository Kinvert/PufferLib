# Exact deterministic Flappy evaluation

October 5, 2026. Native adapter and audit tooling implemented; four float32
targets compile and GPU-free spawn/episode/audit checks pass. **GPU inference,
quota/reload/recurrent-reset/eager-graph repeatability remain unvalidated.**
This document does not lift the 5060/5090 GPU execution holds.

October 6 follow-up: `run --prepare-only` now freezes an existing checkpoint,
matching full INI, suite, effective configuration and launcher tools without
querying a GPU. Twenty-one host/synthetic supervision tests pass, including
24 pixel family/core configurations and the original state control. Three
retained pre-fix regressions demonstrate that the older loader accepted changed
source/manifest INIs and rehashed physics drift. New suites hash both INIs and
reload checks their exact configuration relationship. Historical suites remain
readable only with matching retained source and reconstructed manifest settings;
their missing original manifest hash is explicitly reported. No native
environment/encoder/learner or historical evidence was changed.

## Contract

One assigned episode ID determines environment RNG and policy Philox RNG,
using the versioned identity mapping shared with Connect4. Resets begin with
the same bird state and seeded three-pipe layout for every model. Rendering
has its own fixed ID, consumes no game RNG, and never changes physics. The
suite freezes all 16 native geometry/physics/reward keys and the native cap.

Inference stays C/CUDA. Episodes run in waves with no replacement games:
finished slots stop stepping, last-wave padding is never scored, every assigned
episode is scored once, and each new episode supplies terminal=1 to the
production recurrent-reset path. Policy/core sizes come from the supplied
matching full training INI; no H128/L1 or encoder substitution is made.

Native Flappy already resets on crash or `max_steps` (stock 4096). At that
limit it applies the existing crash reward. This protocol **preserves** those
semantics; it doesn't convert the cap into a reward-free truncation or drop
long episodes. `cap_reached` means the episode ended at the limit, not proof
that the cap was the only terminal cause. Pipes, clipped pipes/20 perf, return
and episode length are captured from the ending episode's log before that
auto-reset loses the live score. Score/perf are not match wins.

Native validates each starting/terminal state, accumulated return and cap;
the external audit requires the complete assigned ID set, expected per-episode
environment/action seeds, frozen world/image/start-RNG hashes, cap accounting
and matching native completion/weight-size receipts. Extra/missing/duplicate
episodes or changed inputs fail. Partial files and failure timing remain
recorded; a failed evaluation gets no partial-episode mean reported as success.

## Native modes and GPU-free checks

- `BINARY eval_exact_info` identifies game/default encoder/precision/rules
  without entering CUDA device discovery or creating a policy.
- `BINARY eval_exact_manifest FULL.ini` runs native initialization/rasterization
  only and prints episode IDs, seeds and initial world/image/RNG hashes. This
  uses the actual compiled host math instead of guessing floats in Python.
- `BINARY eval_exact --headless` executes saved-policy GPU inference and must
  only run when scheduled. Ordinary `eval` remains the historical pooled-v1
  path; adding this adapter doesn't upgrade its old scores.

Build in a fresh directory using existing process-local dependencies:

```bash
source ocean/connect4cnn/runtime_env.sh
mkdir -p build/flappycnn/FRESH_EXACT_ID
NVCC_ARCH=sm_120 bash build.sh flappycnn build/flappycnn/FRESH_EXACT_ID/default --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN \
  bash build.sh flappycnn build/flappycnn/FRESH_EXACT_ID/impala --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN \
  bash build.sh flappycnn build/flappycnn/FRESH_EXACT_ID/impoola --float
NVCC_ARCH=sm_120 bash build.sh flappy build/flappycnn/FRESH_EXACT_ID/state --float

bash ocean/flappycnn/tests/run_all.sh
.venv/bin/python ocean/flappycnn/tests/test_exact_eval.py \
  --binaries build/flappycnn/FRESH_EXACT_ID
```

Nature and Flex are selected by INI from the default binary. Use the machine's
actual GPU architecture for compilation; `sm_120` is appropriate for these
5060/5090 hosts. Builds and the above tests execute no neural model.

The simulation harness independently checks reset seeds/pipe layouts and
terminal/cap/crash/pass reward accounting, including UINT32 boundaries,
dirty prior states and post-reset score loss. Existing state/pixel parity and
ASan/UBSan pass. Host manifests from default/IMPALA/Impoola match byte-for-byte;
state produces the same world/RNG with different observation hashes. Ten
suite/audit tests include core preservation, independent tail episodes,
invalid native input/weight rejection and synthetic process failure/completion
fixtures. Synthetic audit outcomes are **not** GPU evaluation evidence.

## Prepare a frozen suite without GPU execution

Use a full resolved INI, not the game INI alone. For preparation, generate one:

```bash
.venv/bin/python research/prepare_pixel_robustness.py \
  --environments flappycnn --out build/pixel-robustness/FRESH_FLAPPY_ID

.venv/bin/python ocean/flappycnn/exact_eval.py suite \
  --binary build/flappycnn/FRESH_EXACT_ID/default \
  --config build/pixel-robustness/FRESH_FLAPPY_ID/flappycnn/r0/flex_quality-s53111/config/default.ini \
  --seed 56111 --episodes 1000 --slots 64 --representation 0 \
  --out build/flappycnn/FRESH_SUITE_ID
```

The suite contains complete native host spawn receipts plus hashes of its
source INI, manifest INI and native binary. Configuration receipt version 2
checks both file bytes and the manifest's derivation from the frozen source,
world, appearance, quota, identity seeds and native cap. All pixel architectures can reuse this suite;
their run preflight regenerates and compares native starts before touching
the GPU. For original state Flappy, create a separate state suite from its
matching binary/INI with the same suite seed/IDs/physics. Its world hashes
should match, while image/type hashes differ. State has no appearance selector.

Each fixed representation (0–3) gets its own suite. Changing slots leaves
episode identities/spawns unchanged; retain the inference batch size when
testing byte-level repeatability. Training mixed appearances and evaluating
fixed IDs is a declared transfer experiment, not implicit reproduction.

Legacy version-1 suites did not record `manifest_ini_sha256`. Reload verifies
the retained source's original hash, matching source/world and all reconstructed
manifest settings; it reports
`legacy-derived-config-no-original-manifest-hash`. It doesn't add an original
hash to historical files or retroactively call those suites version 2. Keep
old scores/source receipts separate; create fresh suites for new measurements.

## GPU-free existing-checkpoint preparation and offline audit

```bash
.venv/bin/python ocean/flappycnn/exact_eval.py run --prepare-only \
  --binary MATCHING_NATIVE_BINARY --config FULL_RESOLVED_TRAINING.ini \
  --checkpoint EXISTING_FLAPPY_WEIGHTS.bin --family flex \
  --suite SUITE/suite.json --out build/flappycnn/FRESH_CHECKPOINT_PACKET

.venv/bin/python ocean/flappycnn/exact_eval.py audit \
  --suite SUITE/suite.json --episodes PREVIOUSLY_SCHEDULED_RUN/episodes.csv
```

The first command never queries a GPU or executes a policy. It checks finite
float32 checkpoint bytes, copies immutable inputs/tooling, preserves all
policy/core settings and regenerates actual native starts using the supplied
binary. Byte checks aren't shape/math or training-provenance certification.
GPU acceptance remains false, and no episode score/report is fabricated.
Using historical trained checkpoints avoids new training for their first
scheduled evaluator acceptance; those files aren't distributed in Git.

The offline audit requires all assigned IDs, validates cap/score/perf/return
receipts and marks policy runtime uncertified. Floating aggregates use
`math.fsum` so CSV completion order doesn't change the reported mean. Synthetic
large cancelling-return fixtures test aggregation only, not valid gameplay.

Additional supervision tests retain failed/partial evidence, reject reservation
conflicts, empty/malformed/nonfinite weights, failed/empty GPU queries, changed
effective/copied/source inputs and tools, missing/backwards clocks and bad
quota/summary/parameter totals. No successful report exists after failure.
All GPU queries/processes in those tests are explicit mocks. Encoder-5 remains
subject to its separate 5090 numerical gate.

## Scheduled saved-policy GPU acceptance

**Do not execute now.** When Kinvert schedules it, supply actual matching
checkpoint/training INI paths; the placeholders below are deliberate. Never
use Connect4/Pong weights with a different Flappy action head.

```bash
.venv/bin/python ocean/flappycnn/exact_eval.py run \
  --binary build/flappycnn/FRESH_EXACT_ID/default \
  --config /absolute/path/to/full/resolved-training.ini \
  --checkpoint /absolute/path/to/flappy-checkpoint.bin --family flex \
  --suite build/flappycnn/FRESH_SUITE_ID/suite.json \
  --out build/flappycnn/FRESH_EVALUATION_ID --timeout 600
```

The runner checks a shared benchmark lock and GPU idleness, never interrupts
other jobs, retains effective/supplied INIs, binary/checkpoint/suite hashes,
copied tool/source/manifest hashes, start receipts, native log/per-episode CSV, process-monotonic duration and
failure status. Input changes invalidate the result. Build/host/evaluation
time are outside training timing. Timeout is a declared external failure
limit, not a game cap. It has not been runtime-calibrated for these models.

First acceptance uses 65 episodes/64 slots: repeat twice with fresh output
directories, compare per-episode receipts; repeat with `--eager`; replay the
final episode independently using the same ID and 64 slots. Test existing
quality/Nature/IMPALA/Impoola and state checkpoints with their real shapes.
Where absent, schedule bounded train/reload canaries before evaluations.
Verify cap/crash/long-episode cases, recurrent reset, malformed weights and
configuration rejection. Preserve every failed case. Encoder-5 testing is
still gated on its independent 5090 numerical acceptance.

After runtime acceptance, a matched learning panel can retain all checkpoints
and evaluate each on frozen fixed-ID suites. Common learning recipes/budgets,
baseline backend review, paired training seeds, frontier uncertainty and final
confirmation design remain necessary; this evaluator alone cannot establish
multi-environment superiority or SOTA.
