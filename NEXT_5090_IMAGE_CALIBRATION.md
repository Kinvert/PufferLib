# 5090: fixed CNN calibration, no coding required

Pull the delivered `Kinvert/PufferLib:cnn-research` revision. This run compares
Nature, our locked quality architecture and a compact spatial CNN on all six
realistic RGB games/all41 drawings. It prepares the next architecture sweep;
**it does not start PROTEIN or certify a winner.** All code/configuration is
provided. Do not edit sources/INIs or substitute an older sweep launcher.

## Get an isolated checkout

From your existing Kinvert/PufferLib checkout, preserving its retained campaigns:

```bash
IMAGE_EXISTING_CHECKOUT="$PWD"
git remote get-url origin
# It must identify Kinvert/PufferLib. Never use official upstream.
git fetch origin cnn-research
git worktree add --detach ../cnn-5090-image-calibration origin/cnn-research
cd ../cnn-5090-image-calibration

# Reuse the existing uv Python3.12 venv; do not resolve the interpreter symlink.
if [[ -x "$IMAGE_EXISTING_CHECKOUT/.venv/bin/python" ]]; then
    ln -s "$IMAGE_EXISTING_CHECKOUT/.venv" .venv
fi
mkdir -p build/connect4cnn
if [[ -f "$IMAGE_EXISTING_CHECKOUT/build/connect4cnn/runtime-5090.sh" ]]; then
    cp "$IMAGE_EXISTING_CHECKOUT/build/connect4cnn/runtime-5090.sh" build/connect4cnn/
fi
# Share the original reservation path with other jobs/checkouts.
mkdir -p "$IMAGE_EXISTING_CHECKOUT/build/connect4cnn"
touch "$IMAGE_EXISTING_CHECKOUT/build/connect4cnn/hardware-benchmark.lock"
ln -s "$IMAGE_EXISTING_CHECKOUT/build/connect4cnn/hardware-benchmark.lock" \
    build/connect4cnn/hardware-benchmark.lock
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS NVCC_FLAGS NVCC_EXTRA
```

If that new worktree path already exists, preserve it and choose a fresh path.
No checkout reset or cleanup is needed. Reuse the existing CUDA/NCCL installation
and runtime helper. Do not install/change drivers, CUDA, Torch or NCCL. The
wrapper creates a uv Python3.12 venv with NumPy2.5.3 only when no venv exists;
an existing venv is not recreated or upgraded. An unavailable runtime/library
path is a setup failure to report, not permission to alter the system.

## Prepare (GPU-free)

```bash
IMAGE_CALIBRATION_DIR="build/image-calibration-5090-$(date -u +%Y%m%dT%H%M%SZ)"
bash research/run_image_calibration_5090.sh prepare "$IMAGE_CALIBRATION_DIR" \
    > "$IMAGE_CALIBRATION_DIR.prepare.log" 2>&1
bash research/run_image_calibration_5090.sh inspect "$IMAGE_CALIBRATION_DIR"
```

Preparation compiles six float32 RGB trainers and five independent CUDA-check
library pairs, freezes per-game host metadata, the three model configurations,
learner overlays and seed roles. It executes no neural policy or GPU query.
Keep the printed directory; subsequent commands use exactly that directory.

| Model | Frozen encoder |
|---|---|
|quality-reference|One C16/kernel7/stride4 stage; spatial flatten; projection64.|
|spatial-control|Two C8 stages, kernels8/3, strides4/2, pooling after each; spatial flatten; projection64.|
|nature-cnn|32/64/64 valid convolutions, kernels8/4/3, strides4/2/1; common-H128 flattened readout.|

All models share the same H128/L1 recurrent core and per-game learners. Nature's
head is a declared adaptation, not the exact original DQN policy. RGB frame
sizes are576x672 Connect4,640x620 Pong,640x420 Flappy,330x576 Breakout and616x616
Snake/Maze (height x width). There is no resize. Snake/Maze retain local visibility.

Scalar optimizer settings come from each game's current `compare.ini`. The new
explicit geometry is16 actors/one buffer/two CPU workers, horizon16/minibatch16;
it is smaller than historical64-actor/H32/M2048 comparisons. It stays identical
across models and phases within each game. This is a learning-calibration recipe,
not already tuned proof of maximum performance. No learner or seed is swept.

## Run when the 5090 is idle

```bash
bash research/run_image_calibration_5090.sh run "$IMAGE_CALIBRATION_DIR" \
    > "$IMAGE_CALIBRATION_DIR.run.log" 2>&1
```

Run once. GPU reservation/idle checks refuse competing workloads. Do not relaunch
an existing `execution/` directory, retry a failed allocation, shorten horizons,
remove a game/drawing, select better seeds or edit a threshold to obtain a pass.

The supervisor aims for12 hours from this explicit run, including numerical
checks, probe, learning calibration and reporting reserves. Build preparation is
separate. Native process deadlines are enforced; reserves are empirical estimates,
not a guarantee against slower learned episodes or expensive offline array audits.
The result records actual elapsed time and any window overrun.

1. Every fixed architecture receives fresh5090 independent CUDA checks on all
   five input shapes atB1/4/16 before any policy training. No5060 numerical packet
   substitutes for hardware-matched evidence. Failed checks stop the run.
2. The timing probe runs18 jobs: three models x six games, training seed7,
   65,536 decisions, one final checkpoint, all41 drawings, five exact episodes
   per drawing and16 evaluator slots. Repeat/eager checks are mandatory. This
   probe estimates execution costs; it is not a learning/frontier result.
3. Raw receipt audit and SPS/curve export complete before budget admission.
   The slowest measured model in each game sets one shared decision budget for
   all three models/all three replicas. Training budgets range2,097,152 to
   13,631,488 decisions, aligned to the native rollout and eight checkpoints.
   A factor-two cost margin and reporting/archive reserve are retained.
4. If a complete minimum-budget panel cannot fit, the supervisor stops with
   `stopped-insufficient-time` and `ADMISSION.json`. It does not shrink learning
   below2M or drop evaluation conditions to fill the window.
5. If admitted, the panel runs54 jobs: three models x six games x training
   seeds7/42/69. Evaluation seeds7007/7008/7009, appearance17 and all41 drawings
   are fixed. Each job saves/evaluates its actual initial policy plus eight
   trained checkpoints. There are3,321 drawing/checkpoint evaluations and
   108 repeat/eager checks, plus their exact assigned episodes.
6. All results/declines are audited and exported. The existing quality/Nature
   learning gate still needs at least one of those two fixed controls to qualify
   each game, with that control passing every development replica.
   The spatial control is visible but cannot rescue an unlearned reference pair.
   An unlearned game yields `stopped-unlearned-games`; no Nature anchors or
   architecture search are launched. Confirmation seeds remain untouched.

## Read/report the result

Start with `$IMAGE_CALIBRATION_DIR/execution/RESULT.json`. Then read:

- `execution/ADMISSION.json`: actual probe costs, chosen per-game budgets or why
  the minimum complete panel did not fit.
- `execution/probe-report/REPORT.md`: process SPS, raw native SPS/uptime bins,
  all-drawing scores and interactive curves. It distinguishes process timing
  from the trainer's own clock.
- `execution/calibration-report/REPORT.md`: every initial/trained checkpoint,
  game/drawing/seed and measured cost, if the full panel completes.
- `execution/learning-eligibility.json`: paired final-minus-initial gains and
  unresolved games. Eligibility is not superiority or SOTA.
- `*-failure-review/REPORT.md` or native stage logs: completed audited prefix,
  explicit missing data and retained failure, if execution stops unexpectedly.

Exit0 means the calibration completed and passed its learning gate. Exit2 means
a deliberate insufficient-time or unlearned-game stop. Other nonzero statuses
are failures. None means that the CNN won a Pareto comparison. Preserve all logs,
raw arrays, weights and source/build receipts on this box.

The wrapper automatically creates a **compact summary archive** under
`build/hardware-artifacts/`, with a checksum. It contains configs/receipts/plots
and excludes huge numerical arrays, checkpoints and binaries. It is not a
self-contained independent numerical replay. To package retained results again
without executing any policy:

```bash
bash research/run_image_calibration_5090.sh handoff "$IMAGE_CALIBRATION_DIR"
```

Send G240 the final status, elapsed time, report paths, admission/eligibility
summary and printed archive filename/SHA256. Do not push code, write another
continuation launcher, start PROTEIN, use confirmation seeds or run an automatic
second campaign. G240 will analyze the actual calibration before the next step.
