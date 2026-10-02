# FlappyCNN quality model with stock learner settings

September 26, 2026. Kinvert requested one learning-scale pilot on G240's RTX 5060, using our best current network and otherwise general stock Flappy hyperparameters. No architecture search or reference-model comparison is included.

Completed once outside the sandbox: [stock-pilot.DN7C2JF6 report](results/flappycnn/stock-pilot.DN7C2JF6/REPORT.md), 19.923M decisions in 56.12 process seconds, final mean 52.25 pipes and clipped perf 0.9759. All 19 checkpoint hash/count/finiteness checks and executed-stock-setting checks pass. The earlier blocked attempts remain preserved. Do not relaunch this historical pilot automatically.

Use the existing Connect4-selected `flex_quality` encoder (ID 4): one 16-channel 7×7 convolution with stride 4, flatten, projection width 64, followed by its H128/L1 recurrent core. This is the quality model with the strongest five-seed final mean among our three frozen choices; it is not a proven universally best model or a Flappy-selected architecture. The stock Flappy core is H64/L2; retaining our complete selected network is the architecture exception to stock settings.

All environment, vectorization and learner settings come from `config/default.ini` plus `config/flappycnn.ini`, which preserves native `config/flappy.ini` values. No `compare.ini` is loaded. In particular: 2,048 agents, four buffers, eight CPU environment workers, horizon 64, minibatch 16,384, learning rate 0.01 with stock annealing, gamma 0.995, GAE 0.92, entropy 0.003, value coefficient 1.5, replay ratio 1, gradient norm 1.5 and stock async=1. Float32, one GPU, default learner seed 73, fixed pixel appearance 0. Initial weights are random; no encoder pretraining or prior-game checkpoint transfer.

Request 20,000,000 decisions exactly as stock does. Native training rounds this down to 152 complete rollouts, **19,922,944 actual decisions**. Preserve this distinction in SPS calculations. Retain a checkpoint every eight rollouts (1,048,576 decisions) plus the final checkpoint. Recording cadence, run/artifact paths and disabling the default embedded 10,000-episode evaluation are operational changes, not learner tuning.

After training, evaluate the final checkpoint separately with action seed 10073, 64 native slots and 256 requested completed episodes. This uses the existing pooled-v1 evaluator; actual counts may exceed the request. Native environment RNG remains slot based, so a separate action seed does not establish an independent environment-seed allocation. This is a learning pilot, not paper confirmation. Report raw mean pipes passed (`score`), clipped pipes/20 (`perf`), episode return/length, full curves, parameters, VRAM, actual decisions, whole-process training time/SPS and native uptime/SPS. Preserve failures and declines.

Run from the checkout in a session that can access the idle 5060:

```bash
NVCC_ARCH=sm_120 NVCC_EXTRA='--threads 1' bash ocean/flappycnn/stock_pilot.sh
```

The native Bash runner builds through the normal shared paths, verifies source hashes, rejects GPU query failures/competing compute jobs, trains once with a 3,600-second maximum, and evaluates once with a 300-second maximum. Both timeouts retain incomplete logs and checkpoints. It writes fresh `build/flappycnn/stock-pilot.*` receipts, resolved config files, exact commands, native metric histories and status. Do not actively watch the training or launch another pilot while it is running. `completed_pending_audit` only records successful process exits; inspect executed settings, finite weights, completion counts and metrics before claiming results. No W&B upload occurs.

`--prepare-only` saves the recipe without GPU work. If status is `blocked_gpu`, no learner has started; running the command later creates a new recorded attempt and does not erase the blocked receipt. GPU-access failure in the managed shell requires a session with device access; do not change CUDA, drivers or system configuration.

## Original stock state control

Kinvert subsequently authorized training the original native game to measure vanilla PufferLib throughput and performance. `stock_pilot.sh --state` selects `ocean/flappy/flappy.h` and copies `config/flappy.ini` unchanged, retaining the original six state observations and stock H64/L2 network. No CNN or synthetic-pixel observation code is used by that binary. The same float32 build, stock learner/vectorization/game settings, train seed 73, requested 20M decisions, checkpoint cadence and final evaluation action seed 10073 / 64 slots / 256 requested episodes apply. No learner/core tuning or fresh-seed search.

```bash
NVCC_ARCH=sm_120 NVCC_EXTRA='--threads 1' bash ocean/flappycnn/stock_pilot.sh --state
```

This is a stock-state versus selected-pixel **whole-training** comparison, with native adjusted SPS reported separately. State observes velocity and offscreen pipes, and H64/L2 differs from the pixel network's H128/L1; equal learner settings do not isolate the causal cost of the pixel encoder. Both runs are single-seed development pilots under pooled-v1 evaluation, not a final generalization or statistical-superiority result. The launcher accepts `--state --prepare-only` for isolated config preparation.

Completed once as `stock-pilot.tgKBYrNs`: [audited state/pixel comparison](results/flappycnn/stock-pilot.tgKBYrNs/REPORT.md). State: 9.73 process seconds / 2.048M SPS / 48.53 mean pipes / clipped perf 0.9472; pixels: 56.12 seconds / 355K SPS / 52.25 pipes / perf 0.9759. All 19 state checkpoints and actual executed-stock-setting checks pass. Do not automatically relaunch either historical pilot.
