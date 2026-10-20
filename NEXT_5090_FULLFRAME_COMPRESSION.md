# RTX5090: full-frame compression acceptance packet

October19,2026. First increment only: three fixed full-frame visual pipelines,
encoder weight import/export, numerical checks and short deterministic training.
No overnight sweep, dataset generation or pretrained-learning claim.

## Delivery status

October20local result:120numerical workers and144finite-difference probes pass.
The first device memcheck failed to initialize its WDDM debugger interface under
WSL and exited99 with2tool errors. No training/eval/import smoke has executed.
This does not qualify the memory gate or a fully validated handoff. Preserve the
failed v6 packet; do not bypass the gate or change system/CUDA settings.

Implementation is under local validation. **It has not been committed or pushed
by this task yet.** The remote machine must receive the final tested revision
before using these commands. Do not interpret the older sprite/image sweep
handoffs as an instruction to start this packet automatically. The final local
result report will identify exact source/config/build evidence and gate status.

All edits belong in `/home/claude/cnn` on the5060 host. The5090 agent should only
pull, prepare, execute and report; no coding or package installation is required.
Use existing CUDA/NCCL and the checkout's uv Python3.12 venv. No system changes.

## What runs

Full native RGB Flappy640H×420W and Pong640H×620W. Every model receives the same
raw scene and keeps the same game actions. Internal pipelines:

| ID | Visual processing |
|---|---|
|0|Learned4/s4 C8,3/s2 C16,3/s2 C16,adaptive4x4,projection128|
|1|Area4 then3/s1 C8,3/s2 C16,3/s2 C16,adaptive4x4,projection128|
|2|Area4 then valid Nature8/s4 C32,4/s2 C64,3/s1 C64,projection128|

All use ReLU, FP32 and MinGRU H128/L1. Model2 is resized RGB Nature/H128, not
original DQN or the earlier full-input Nature benchmark. Source pixels remain
full frames; this uses no tile/object lookup or atlas shortcut.

Math covers declared small/odd and actual game shapes, B1/3/16 as specified,
all outputs/encoder gradients/input derivatives, signed and zero fixtures,
GPU finite differences and graph/process repeats. It is not a proof of arbitrary
architectures or independent full-learner mathematics.
Inputs include RGB ramps, translated edges/dots, clipped colors and zero frames;
they contain no game labels. Signed cases refer to weights/upstream gradients;
raw RGB observations always remain in[0,1].

Smoke trains3models ×2games ×2same-seed repeats =12jobs,8192decisions/job,
seed7. Checkpoints4096/8192 must match across repeats. Exact evaluation uses
2assigned development episodes per policy, suite420, graph/repeat/eager and
one/two slots,256decision cap:24evaluation processes/48episode executions.
Encoder import checks include valid roundtrips and deliberate rejections.
This is plumbing/correctness evidence, not enough training for quality comparisons.

## Commands after the tested revision is published

From the research checkout on5090:

```bash
git pull --ff-only origin cnn-research
# Reuse the existing machine-local runtime helper when present.
if [ -f build/connect4cnn/runtime-5090.sh ]; then source build/connect4cnn/runtime-5090.sh; fi
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash research/fullframe_compression.sh prepare --out build/fullframe-compression-v1-5090-01
bash research/fullframe_compression.sh inspect --packet build/fullframe-compression-v1-5090-01
bash research/fullframe_compression.sh host-check --packet build/fullframe-compression-v1-5090-01
.venv/bin/python -m unittest discover -s research/tests -p test_fullframe_compression.py -v
bash research/build_image_probe.sh flappycnn 640 420 3 build/fullframe-compression-v1-5090-01/raster-flappy
build/fullframe-compression-v1-5090-01/raster-flappy config/flappycnn.ini build/fullframe-compression-v1-5090-01/raster-flappy.csv > build/fullframe-compression-v1-5090-01/raster-flappy.json
bash research/build_image_probe.sh pongcnn 640 620 3 build/fullframe-compression-v1-5090-01/raster-pong
build/fullframe-compression-v1-5090-01/raster-pong config/pongcnn.ini build/fullframe-compression-v1-5090-01/raster-pong.csv > build/fullframe-compression-v1-5090-01/raster-pong.json
```

Preparation compiles from captured sources without querying or executing a GPU.
The unit/raster commands also use no GPU or CPU neural model. Require every
command above to succeed before running math; preserve sanitizer failures.
Use a fresh output name. If another job owns the GPU, wait for it; do not stop it.
The run takes the shared benchmark lock and checks for foreign GPU processes.
Wait for the other campaign's supervisor to finish as well: a brief idle gap
between its jobs is not an exclusive allocation. The new Python glue uses the
standard library; install no Torch/CUDA packages. If the runtime helper cannot
find existing NCCL, use that machine's already-working CUDA/NCCL paths instead
of installing or changing the stack.

```bash
bash research/fullframe_compression.sh math --packet build/fullframe-compression-v1-5090-01 --hardware 5090 > build/fullframe-compression-v1-5090-01/math-supervisor.log 2>&1
bash research/fullframe_compression.sh smoke --packet build/fullframe-compression-v1-5090-01 --hardware 5090 > build/fullframe-compression-v1-5090-01/smoke-supervisor.log 2>&1
bash research/fullframe_compression.sh report --packet build/fullframe-compression-v1-5090-01
```

Run smoke only after math exits successfully. Each stage has a900second cap;
individual training/eval caps45/30seconds. No automatic retries or restarting
completed/failed stage directories. Keep all failed receipts. Commands block
until their short packet finishes; redirect logs and do other work rather than
following the trainer TUI. A failed gate needs diagnosis on the coding host.

Read `REPORT.md`, `REPORT.json`, `stages/math.json`, `stages/smoke.json` and any
`stages/*.failure.json`. Report processSPS as startup-dominated; preserve every
checkpoint and negative/capped episode. Do not infer learning superiority from
these8K-decision runs. Raw arrays/weights stay under ignored build storage.
The native metrics are binned means, retained under each training row's`native`
field. Do not divide final decisions by a binned uptime to claim native-average
throughput. None of the required commands depends on optional encoder timing.

## Optional encoder timing, after acceptance

The timing worker is implemented and builds on5060; GPU measurements are still
qualification-pending. It compiles against the packet's existing captured native
sources and records its own small source/build supplement. It never changes or
rebuilds the accepted trainer/math binaries. Preparation is GPU-free:

```bash
bash research/fullframe_compression.sh prepare-timing --packet build/fullframe-compression-v1-5090-01
```

Only after both required stages pass, with exclusive GPU access:

```bash
bash research/fullframe_compression.sh timing --packet build/fullframe-compression-v1-5090-01 --hardware 5090 > build/fullframe-compression-v1-5090-01/timing-supervisor.log 2>&1
bash research/fullframe_compression.sh report --packet build/fullframe-compression-v1-5090-01
```

This is60fresh worker processes:two games,three models,B1/B16,five fixed
replicas each. Each records CUDA-event spans after20warmup/100timed iterations
for forward and forward+backward. All models receive identical native reset
frames from fixed seeds7through7+B-1. Frames exist only in memory; no dataset is
collected. Pinned input H2D, internal area/pooling and encoder projection are
included. Rendering, allocation, recurrent core, optimizer and checkpoint writes
are excluded. Execution is eager, not CUDA graphs. Keep the five samples and
their median/range; do not equate this microbenchmark with trainingSPS.
Stage cap300seconds,worker cap30seconds. No automatic retry/long training.

## Return evidence

Provide the source revision, source-manifest SHA256, GPU/driver identity, gate
statuses, model parameter counts, all12training durations/SPS, exact-checkpoint
repeat result, all24evaluation comparisons, and valid/rejected import counts.
Keep5060 and5090 measurements separate. The root report is generated by offline
receipt/hash inspection, not by launching more models.

Export an accepted final encoder when needed:

```bash
bash research/fullframe_compression.sh export-encoder --packet build/fullframe-compression-v1-5090-01 --game flappycnn --model 0 --checkpoint build/fullframe-compression-v1-5090-01/smoke/flappycnn-m0-r0/checkpoints/flappycnn/happy-cat-1/0000000000008192.bin --out build/fullframe-compression-v1-5090-01/export-for-transfer
```

This export is only a short-RL initialization artifact. No visual pretraining
has happened. Later experiments must select data/splits/compute and compare
scratch/frozen/fine-tuned learning under a separate bounded specification.
