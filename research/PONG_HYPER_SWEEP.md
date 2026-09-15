# Pong: frozen architectures, training-hyperparameter discovery

Authorized September 14, 2026: run the Pong experiments with our existing CNN architecture locked, searching training hyperparameters. This is development work toward the [paper evidence standard](PAPER_PLAN.md#required-evidence-standard-defend-the-claim-like-a-thesis). It does not complete fresh-seed confirmation or establish an advantage over Nature.

## Fixed models and game

Our candidate is the existing Connect4-selected **quality** model: one 16-channel SAME 7×7/stride-4 convolution, flatten readout, projection 64. Hidden size **128**, **one recurrent layer**. All these settings stay fixed throughout the sweep. Nature retains its adapted three-convolution architecture; IMPALA retains its 15-convolution residual backbone; Impoola retains that backbone with global average pooling. Their architectures and the common recurrent core also stay fixed.

All four receive identical **1×36×44** PongCNN pixels, score bars, game physics, scripted opponent, frame skip 8, three discrete actions, and float32 precision on G240's RTX 5060. Fixed 64 agents, two CPU threads, horizon 32, minibatch 2,048, synchronous rollout and CUDA graphs. Original state Pong is a plumbing control only and is not part of this pixel-encoder hyperparameter search.

Total parameter counts are 160,224 / 138,016 / 269,984 / 151,200 for ours / Nature / IMPALA / Impoola. Each has 512 fewer parameters than its Connect4 counterpart because the action head has three actions instead of seven; this is not a searched architecture change. PongCNN is PufferLib's native game, not ALE Pong. `perf` means episode-averaged fraction of points won, not match win rate. The original environment's `episode_length` logs the last rally's decision count, since the tick counter resets between points; do not interpret it as full-match duration.

## Search protocol

Native **PROTEIN** selects the following settings, identically available to every family:

| Setting | Range |
|---|---|
| Total agent decisions | 2,097,152–13,312,000 requested; native rollout rounding recorded |
| Learning rate | 0.0001–0.1 |
| Entropy coefficient | 0.00001–0.01 |
| Gamma | 0.95–0.9995 |
| GAE lambda | 0.9–0.995 |
| Replay ratio | 1–4 |
| PPO clip | 0.1–0.5 |
| Value-loss coefficient | 0.1–5 |
| Maximum gradient norm | 0.1–1.5 |

Exact distributions/defaults are in [sweep.ini](../ocean/pongcnn/sweep.ini). No `policy`, `env` or vectorization dimensions may enter the search. The runner removes inherited default search dimensions in isolated configs, without rewriting repository defaults. Model construction, training and optimization remain native C/CUDA. Bash handles preparation/launch; existing-venv Python handles validation/reporting and external W&B only. No `src/`, CUDA-stack or package changes are needed.

Each family gets a **24-completed-trial maximum** and a **3,600-second native sweep-process cap**, serially in the initial order ours, Nature, IMPALA, Impoola. Maximum 96 completed trials and four hours of native sweep time; builds, evaluation and uploads are additional. The cap includes optimizer/worker overhead. A running trial can be interrupted by the resource cap; retain partial artifacts and report it as incomplete, never as a completed score. Search defaults begin at 13.312M decisions for every family. The native `max_suggestion_cost` controls predicted costs, not a hard per-trial timeout.

These are equal **resource ceilings**, not necessarily equal consumed tuning compute or equal trial counts. Record the actual cost and completed/failed/interrupted runs. Different model speeds and the fixed family order make this an initial development comparison, not the final equally tuned, interleaved confirmation. No best-of-family score here establishes statistical dominance.

## Measurement and separation from confirmation

- Training seed **9173** for discovery. PROTEIN consumes native downsampled training `env/perf` curves against adjusted native time. The seed is not swept and these trials are not independent training-seed replications.
- Reload **every completed final checkpoint** with its exact saved native config and evaluate **256 requested matches** on separate development seed **29173**, after that family's timed search. Record actual game counts, point fraction, point-score difference, finite weights, parameter count, checkpoint hash, errors and evaluation wall time. This validation can inform selection; it is not an untouched final test.
- Preserve all native per-trial curves, `SPS`, `uptime`, `agent_steps`, loss and environment metrics, final observations and effective hypers. Native final stdout rounds score/cost; reports label that timing boundary. Do not label final-point Pareto flags as a full uncertain held-out learning-curve frontier.
- W&B **[kinvert-k/cnn3](https://wandb.ai/kinvert-k/cnn3)**, friendly adjective/noun/trial names. Upload each completed family after its evaluations, outside native sweep timing. Groups include campaign and family; short canaries remain distinguishable by their budgets and campaign receipts. Native and `eval/*` metrics are separate.
- Archive source/config snapshots and SHA-256 receipts, build commands and binary hashes. The campaign uses copied reporting scripts so later edits do not change its analysis mid-run. GPU availability is checked before each family and evaluation; this is not continuous exclusive-host monitoring.

After inspecting discovery, freeze finalists and their budget/learning schedules, specify an adequately powered paired-seed comparison, and collect fresh confirmation data. Fair baseline tuning/implementation audits, low-cost baseline variants, frontier uncertainty, and untouched-task evidence from the paper plan remain open.

## Initial validation and budget calibration

Five-model GPU canary `canary.w5tqV8nE`: state/ours/Nature/IMPALA/Impoola each trained 65,536 decisions, saved/reloaded the final checkpoint, and completed separate evaluation. All had zero evaluated point fraction; this establishes plumbing only.

Quality-model pilot `pilot.p0ydqKv4`: **13,312,000 decisions, 61.82 seconds whole training-process wall**, unchanged common learner recipe (.001 learning rate, replay 1, and the fixed model above). Final evaluation was **0% point fraction**, score −21, 256 matches. Thus the common Connect4 training recipe has not transferred successfully, and this timestep range is a development starting range rather than an established adequate Pong learning budget. Preserve this negative result; do not claim the sweep will succeed.

The first native sweep canary `hypers.l6li6thO` completed two ours trials, then its audit rejected a quoted-versus-unquoted `None` config value. The check now normalizes the native writer's quote removal; original failed receipts are retained. Focused CPU tests cover model/core locking, forbidden environment/search dimensions, identical family search spaces, bounds and explicit compiled reference identity, alongside existing sidecar regressions.

Corrected canary **`hypers.hF7MhfRo`** completed two trials per family: **eight native train/reload/evaluations and eight online W&B uploads**, with one verified architecture per family. The four new CPU safeguard tests and twelve existing sidecar/config regression tests passed. This validates the search infrastructure; canary scores remain excluded from learning comparisons.

## Commands

Active campaign: **`build/pongcnn/hypers.jwyApeXS`**, originally tmux `pong-cnn3-hypers-20260914`, launcher log `build/pongcnn/cnn3-hypers-launch.log`. Started September 14 after the online canary passed. Small initial evidence is archived under [results/pongcnn](results/pongcnn/); full source/checkpoint binaries stay local in `build/`.

September 14 status/recovery: ours and Nature each completed 24 native training trials (437.21 / 614.65 seconds of sweep wall). Ours completed all 24 evaluations/uploads. One Nature evaluation, `sweep_1789436514815_0004`, hit the predeclared 180-second cap; 23 others completed. The strict reporting step halted the campaign before IMPALA/Impoola and before Nature upload. This timeout remains a failed evaluation, with no selectively extended retry and no invented score.

Resumed in **`pong-cnn3-resume-20260914`**, log **`build/pongcnn/cnn3-hypers-resume.log`**, using [resume_hypers.sh](../ocean/pongcnn/resume_hypers.sh). It uploads Nature with the failure retained, then runs only the two unstarted families using their original binaries/configs and one-hour native search caps. Recovery source/binary hashes and original failure/report receipts are in `recovery-eval-timeout/`. No completed training is repeated. Final reporting explicitly permits recorded evaluation failures while keeping missing-evaluation, model/config, finite-weight and checkpoint audits strict. Future full discovery runners use the same policy; canaries still require successful evaluations. Full results remain pending and cannot be presented as a failure-free confirmation.

```bash
# CPU configuration validation only.
bash ocean/pongcnn/sweep.sh --prepare-only
.venv/bin/python -B ocean/pongcnn/tests/test_sweep.py

# Two tiny trials per family, reload/eval/upload plumbing only.
NVCC_ARCH=sm_120 bash ocean/pongcnn/sweep.sh --canary --wandb online

# Authorized bounded development search, launch detached and leave running.
NVCC_ARCH=sm_120 bash ocean/pongcnn/sweep.sh --wandb online --project cnn3
```

Every invocation creates `build/pongcnn/hypers.XXXXXXXX/`. Read `status.txt`, each family `status.txt` / `summary.json` / `REPORT.md`, and raw receipts when checking results. Do not launch a duplicate or actively poll a long run. GPU commands require execution outside the sandbox with the existing process-local runtime helper; no Docker or system modifications.
