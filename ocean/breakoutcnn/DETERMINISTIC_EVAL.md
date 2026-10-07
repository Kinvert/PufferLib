# Breakout exact episode evaluation

October 5, 2026. Native C/CUDA adapter, host suite/audit tooling and supervised
checkpoint launcher implemented.
State/default/IMPALA/Impoola float32 binaries compile. CPU simulation, sanitizer,
five-appearance manifests and receipt auditing pass. **GPU policy execution,
recurrent reset, checkpoint reload and eager/graph repeatability remain untested.**
Both machine execution holds remain; these instructions do not schedule GPU work.

## Episode and score contract

Game rules are `breakout-native-v1`: unchanged original native Breakout, not ALE.
Evaluation is `first-terminal-or-frame-cap-v1`. Freeze the world configuration,
frame skip, appearance, suite seed, explicit episode IDs and a physics-frame cap
before collecting policy results. The cap is an evaluation control, not a new
training environment setting. No empirically calibrated final cap exists yet.

Each episode seeds the original `rand_r` game stream from its identity using
the same versioned mapping as Connect4/Flappy. Reset consumes no randomness;
the first launch uses the first stream draw. Policy categorical sampling gets
a separate identity-derived Philox seed. Neither seed depends on encoder/core
size, slot or worker order. Different policy trajectories can consume different
numbers of subsequent game draws; common starts don't make trajectories equal.

One episode per active slot, one policy action per frame-skip decision. The
evaluation-only step helper uses original `step_frame` physics and stops on the
first terminal frame. Native `step_frame` still writes the ending log and resets
the world; the helper reads that log, includes the terminal reward, and prevents
the remaining skip frames from entering the replacement game. Training's
`puf_step` is untouched and still performs its historical continuation.

At the frame cap, stop even inside a frame-skip decision. Keep the accumulated
score/return and live state; do not invent a loss, game terminal, crash penalty,
reset or completed-game log. A real terminal on the cap frame is completed,
not capped. Report both action decisions and actual physics frames.

The primary bounded estimand is **score accumulated through the first native
terminal or the common frame cap**, averaged over all assigned episodes. Also
report normalized score, completed/capped counts and durations. Stock maximum
score is 864; return is the sum of brick points, matching score. A capped score
does not determine eventual full-game score. Never drop capped games or report
the complete-only average as the primary comparison. Missing/failed receipt
rows make the assigned suite incomplete; no successful partial aggregate.

The native adapter allocates waves without replacement games; the last wave's
dummy slots aren't counted or stepped. Every real ID appears exactly once,
with world/observation/RNG starts, action digest and ending receipts. Initial
terminal flags reset the recurrent core. Actual GPU independence/repeatability
still needs acceptance, including partially filled batches.

## Preparation and host checks

Use the normal dependencies/build paths. The runtime helper sets process-local
paths to existing NCCL/driver libraries; it installs or changes nothing.

```bash
# CPU environment/raster checks only, no neural models.
bash ocean/breakoutcnn/tests/run_all.sh
bash ocean/breakoutcnn/tests/run_exact_episode.sh build/breakoutcnn/FRESH_HOST_ID

# Compilation only. Default binary dispatches quality/Nature by training INI.
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh breakoutcnn build/breakoutcnn/exact-default --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPALA_CNN \
  bash build.sh breakoutcnn build/breakoutcnn/exact-impala --float
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS=-DC4_IMPOOLA_CNN \
  bash build.sh breakoutcnn build/breakoutcnn/exact-impoola --float
NVCC_ARCH=sm_120 bash build.sh breakout build/breakoutcnn/exact-state --float

# Host-only identity and manifests return before any CUDA device initialization.
build/breakoutcnn/exact-default eval_exact_info
```

Prepare a full resolved INI with the GPU-free multi-task tool; a partial game
INI alone isn't sufficient. Pick the job's `config/default.ini` as `FULL.ini`.

```bash
.venv/bin/python research/prepare_pixel_robustness.py \
  --out build/pixel-robustness/FRESH_PANEL_ID

# 4096 is a development fixture cap here, not a justified paper cap.
.venv/bin/python ocean/breakoutcnn/exact_eval.py suite \
  --binary build/breakoutcnn/exact-default --config FULL.ini \
  --seed 56111 --episodes 1000 --slots 64 --representation 0 \
  --max-frames 4096 --out build/breakoutcnn/FRESH_SUITE_ID
```

The suite stores the original INI, native manifest INI, `episodes.csv`, physics,
cap, binary identity/hashes and purpose. Output directories cannot be overwritten.
Image labels/datasets aren't produced. `manifest.ini` preserves the supplied
policy structure, but no policy is constructed by the host manifest command.
All five appearance IDs can be prepared separately. Freeze IDs/purpose/cap
before results; choose held-out games/appearances independently of selection.

```bash
# Existing local preparation targets; select another binary directory if needed.
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/breakoutcnn/tests/test_exact_eval.py \
  --binaries build/breakoutcnn/exact-preparation-20261005
```

Twelve tests exercise real native host manifests for all pixel build families,
all appearances and original state worlds; independent final-ID replay;
invalid physics/overflow/mixed-evaluation rejection; frozen-source hashes;
exact quotas below/above a batch; and corrupt/missing/duplicate/capped synthetic
receipt rejection. Launcher checks add 24 pixel family/core configurations and
one original-state configuration in preparation-only mode, busy/query/reservation
refusal, malformed/nonfinite weights, partial timeout preservation, summary/
parameter-count checks and immutable input/process-clock audits. Synthetic
process outcomes test glue, not learning or GPU math. Configuration preservation
does not validate every supplied architecture's math or checkpoint shape.

## Native runtime and remaining gates

`BINARY eval_exact --headless` is implemented. The checkpoint comes from
`base.load_model_path` in the effective INI, not a positional argument. It uses the binary's
normal full-INI loading path (an isolated job working directory with
`config/default.ini`/`config/breakoutcnn.ini`) and `[eval_exact]` keys:
`seed`, `episodes`, `episode_offset`, `slots`, `max_frames`, `output`.
Keep the matching training policy configuration and the three-action output
head; no Connect4/Pong weights. Fixed appearance is required during evaluation,
even for mixed-appearance training. State uses the original 118 observations.
Runtime rejects BF16, CUDA environments, selfplay, invalid actions/physics,
bad accounting, nonfinite/mismatched weights and existing CSV paths.

Use `exact_eval.py run` as the supervised single-checkpoint launcher. It retains
the matching training INI/encoder/core, checks finite float32 weights before
querying the GPU, regenerates the suite's host starts from the selected binary,
and freezes an isolated effective INI. The native policy constructor performs
parameter-count compatibility checks during scheduled execution; preparation
alone cannot certify weight/architecture compatibility.

`--prepare-only` writes those configs and receipts without GPU queries or
policy execution. It produces `status=prepared-not-executed`, no episode scores
and no `REPORT.md`; it cannot be resumed in place because overwrite is refused.
Use a fresh output ID for a later scheduled run. A real Breakout checkpoint and
its matching resolved training INI are required; no dummy weights in campaigns.

```bash
# GPU-free preparation for an EXISTING Breakout checkpoint.
.venv/bin/python ocean/breakoutcnn/exact_eval.py run \
  --binary BINARY --config TRAINING_RESOLVED.ini --checkpoint WEIGHTS.bin \
  --family flex --suite SUITE/suite.json --out build/breakoutcnn/FRESH_PREP_ID \
  --prepare-only
```

For the frozen quality model use `--family flex`; Nature `nature`, compiled
IMPALA `impala`, compiled Impoola `impoola`, original state `state`. Other
families must match the selected binary/INI. Mixed-training appearance settings
are recorded separately; evaluation uses the suite's fixed drawing. Encoder 5
retains the separate 5090-only numerical/runtime gates.

Only after explicit scheduling, omit `--prepare-only` with a fresh output ID.
The launcher takes the shared hardware reservation, refuses existing compute
processes and query errors, records hardware, and runs native inference in a
bounded process group. Use `--timeout SECONDS` and optionally `--eager`. It saves
monotonic whole-process evaluation time, which is not training time or SPS.

Failures retain `result.json`, native logs/process receipts and any partial CSV;
no partial mean or successful report. Successful execution requires exactly all
assigned IDs, a matching native summary/weight count and unchanged binary,
checkpoint, original/effective/copied INIs, suite/manifest and launcher sources.
Python glue snapshots are retained separately from compiled-native build
provenance; the native binary hash must be linked to its separate build/source
receipts. Even a successful checkpoint report does not qualify evaluator GPU
repeatability or establish frontier superiority. Both execution holds remain.

After explicit scheduling, required acceptance includes all four encoders,
state and multiple policy/core sizes; 65 IDs with 64 slots; exact repeated
CSV equality; eager/graph equality; independent last-episode replay; cap inside
skip; natural terminal on cap; long and early-ending games; malformed weights;
and final output/weight/source audits. Encoder 5 keeps its separate 5090 gates.
Then calibrate cap and common learning recipe/budgets, preserving failures and
contrary results, before a matched multi-seed frontier campaign.

```bash
# GPU-free audit of an existing native CSV, after a scheduled run.
.venv/bin/python ocean/breakoutcnn/exact_eval.py audit \
  --suite SUITE/suite.json --episodes RUN/episodes.csv
```

This checks exact quotas/identity/start/score/frame/cap receipts. It explicitly
does not certify runtime/checkpoint provenance or architecture superiority.
No Breakout policy evaluation or learning result has been produced by this work.
