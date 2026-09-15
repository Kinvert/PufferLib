# Connect4CNN frozen confirmation v1

Completed and audited: `confirm.ol9tcj5k`, 30 jobs and 390 checkpoint evaluations, no failures. Results and the complete uncertain Pareto curves are in [CONFIRMATION_RESULTS.md](CONFIRMATION_RESULTS.md). The specification below is retained as the pre-result protocol.

Specified September 14, 2026, before inspecting confirmation results. User authorized item 1 of the paper plan: freeze existing architectures and confirm them across fresh paired seeds with equal learning-curve measurement. This is a fixed comparison, not a new architecture sweep.

The machine-readable specification is [confirmation.json](../ocean/connect4cnn/confirmation.json). Every campaign archives its exact manifest, effective learner config, native source and binaries' hashes, model order, commands, GPU snapshots, resolved per-run settings, checkpoint hashes, evaluation logs, and reports. Full source and checkpoints remain locally under its build directory. Manifest and source hashes freeze the current working tree even though it has uncommitted research work; a recorded commit alone is not treated as sufficient provenance.

## Frozen panel

| Variant | Frozen encoder | Total parameters | Development selection |
|---|---|---:|---|
| flex_quality | 16 channels, one SAME 7×7/s4 conv, flatten, projection 64 | 160,736 | brave-comet-51: maximum training score |
| flex_fast | 32 channels, one SAME 4×4/s4 conv, flatten, projection 64 | 261,856 | cosmic-wolf-116: fastest training trial with score ≥80% |
| flex_small | 8 channels, one SAME 8×8/s4 conv, flatten, projection 64 | 109,768 | clever-otter-60: smallest model with training score ≥85% |
| nature_cnn | Existing adapted Nature: 3 convolutions, kernels 8/4/3, channels 32/64/64 | 138,528 | Fixed reference |
| impala_cnn | Existing adapted IMPALA: 15 convolutions in 3 residual stages | 270,496 | Fixed reference |
| impoola_cnn | Same IMPALA backbone with GAP readout | 151,712 | Fixed reference |

These names describe their development-selection role; they do not assert that a configuration will retain that ranking after confirmation. No parameters or layers are added or tuned. All three Flex encoders feed projection 64 into the same hidden-128, one-layer recurrent core through the existing output projection. References retain their existing adaptations.

## Common training and evaluation

- **Training seeds:** 173, 174, 175, 176, 177 for each model. Five independent training seeds, 30 jobs total.
- **Evaluation seeds:** 20173, 20174, 20175, 20176, 20177 respectively. Each training/evaluation pair is shared across the six models. Existing development seeds 73–75 and evaluation seeds 10073–10075 are excluded.
- **Training budget:** exactly 13,312,000 agent decisions per job, 399,360,000 total planned decisions. This is 6,500 rollout batches of 2,048 decisions, giving 13 equally spaced checkpoints of 1,024,000 decisions each. The small increase from the earlier 13.279M reference budget makes the schedule exactly divisible and applies to every model.
- **Learner:** frozen effective defaults plus `compare.ini`, same 64 agents, horizon 32, minibatch 2,048, replay ratio 1, learning rate .001 and annealing over the full budget, fixed hidden-128/one-layer core, synchronous rollout and CUDA graphs. All effective non-encoder settings must match, not merely the settings listed in this paragraph.
- **Environment:** unchanged 7×6 Connect4 game, scripted opponent, 1×36×44 synthetic pixel input, same observation/action semantics.
- **Evaluation:** 1,024 requested games per checkpoint, recording actual batched game count, return, win rate, and exact checkpoint hash. Evaluation runs after each training job and cannot affect that job's training timing. Total planned checkpoint evaluations: 390. Episodes and checkpoints are not independent training seeds.
- **Precision/hardware:** float32, one RTX 5060 on G240. No system CUDA, driver or dependency changes. All model binaries built before training starts; all three Flex variants share one binary with numeric construction settings.
- **Order:** fixed cyclic rotation of the six-model order across the five seed blocks, archived before training. Serial jobs avoid GPU sharing between candidates. GPU snapshots are taken before each job. Rotation reduces order bias but is not perfect six-position balance with five seed blocks.
- **Timeouts/failures:** training cap 2,400 seconds per job; evaluation cap 60 seconds per checkpoint; build cap 300 seconds per binary. Process groups are cleaned up on failure/timeout. Preserve failed runs and report them; no unrecorded retries or seed replacement.

## Scoring, timing, and W&B

The primary observations are held-out checkpoint win rates versus full training wall time and versus agent decisions. Full process timing includes startup and checkpoint writes; per-checkpoint times use checkpoint modification time relative to the recorded process launch and are approximate. Native SPS/uptime use the trainer's adjusted timer and are reported separately. Search/compilation/evaluation/W&B upload costs are outside training-job timing. No GPU utilization or native SPS claim is based on the canary.

W&B project: `kinvert-k/cnn2`, one group per confirmation campaign and one friendly-named run per model/seed. Native training keys retain `SPS`, `uptime`, `agent_steps`, `env/perf`, and losses. Held-out curves use separate `eval/perf`, `eval/score`, and `eval/games` keys. `eval/perf_vs_time` plots held-out performance against `eval/train_wall_s`. Training and evaluation observations are merged by actual step in increasing order, so late-uploaded earlier checkpoint results are not dropped or mislabeled as training performance.

The external CPU sidecar uploads a completed job **between** training jobs and exits before the next training job starts. SDK/network work therefore does not run concurrently with timed training. Upload failures are recorded separately from native failures, and saved comparison jobs can be uploaded later without retraining. No reference family receives adaptive tuning in this confirmation; the equally tuned comparison remains a later paper action.

## Canary and execution

If another compute process is using the GPU, launch `bash ocean/connect4cnn/confirm_when_idle.sh` in tmux with `NVCC_ARCH=sm_120`. This waits up to one hour for the compute-process list to clear, then invokes the same frozen protocol. Waiting is outside the experiment timing. It never terminates another job and is not a GPU reservation; inspect recorded per-job GPU snapshots for later contention before accepting timing comparisons.

Canary settings: all six models, training seed 9173, evaluation seed 29173, 65,536 decisions, four checkpoints, 128 requested evaluation games, 120-second training cap. This validates build/selection, matching resolved settings, all saved parameter counts/finiteness, reload/evaluation, friendly W&B logging, and completion receipts. Canary results are tagged and grouped separately and excluded from learning conclusions.

```bash
# Run outside the sandbox from the checkout. No installs or Docker required.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/confirm.sh --canary

# Fixed full protocol; launch detached after a successful canary and GPU check.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/confirm.sh
```

`--confirmation` rejects individual overrides of the locked budget, seeds, selected variants, recipe, checkpoint count, and evaluation count. Changes require an explicit new manifest/protocol version rather than silently modifying the running experiment. The sidecar destination/mode remains selectable. CPU configuration/history checks are in `tests/test_confirmation.py`.

## Analysis after completion

Canary validation completed September 14: `confirm-canary._0y3shor` passed all six jobs, 24 checkpoint evaluations, parameter/finiteness/config checks, and six online W&B uploads. Sidecar payload checks confirmed monotonic actual-step history, native SPS/uptime/training performance, and all separate held-out points. The preceding `confirm-canary.hdq_fms8` failed before training for the three Flex variants because their newer native INI keys were missing; references succeeded. The corrected launcher provides isolated effective configs for every job. Both attempts are preserved under `research/results/connect4cnn/`; these short runs provide no learning-quality evidence.

Compare all five curves for every fixed architecture with the same checkpoint density, including unsuccessful learning outcomes. Publish per-seed curves and observed family/combined frontiers with uncertainty. Retain the frozen panel; do not report only the best confirmation seed or choose new architectures from these results. Report threshold non-achievement and the checkpoint resolution of crossing estimates. A 9M checkpoint within the frozen 13.312M schedule is not an independently trained 9M-budget model.

This confirmation tests reproducibility of the current Connect4 tradeoff. It does not establish unseen-environment architecture transfer, policy transfer, universal optimality, or equally optimized reference frontiers. The remaining research actions are in [PAPER_PLAN.md](PAPER_PLAN.md).
