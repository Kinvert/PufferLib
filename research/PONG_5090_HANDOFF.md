# RTX 5090 agent handoff: fixed-architecture Pong comparison

**Historical task, completed. Do not relaunch from this document.** See [PONG_5090_RESULTS.md](PONG_5090_RESULTS.md), [the received audit](PONG_EVALUATION_AUDIT_RESULTS.md), and [current 5090 instructions](../NEXT_5090_VALIDATION.md).

## Objective and authorization

Work in `~/Git/ml/cnn-5090`, Kinvert's PufferLib fork, branch `cnn-research`. Implement the small amount of comparison tooling still needed, validate it, and run the bounded experiment below on the RTX 5090. This is native **PufferLib PongCNN**, not ALE/Atari Pong. Compare the existing four architectures trained from fresh initialization, with their shapes and recurrent cores locked. Do not start another architecture search or Connect4 representation experiment.

The purpose is to measure learning reliability and the complete observed wall-time/score tradeoff on a second game. This is a five-seed replication stage, not a powered final superiority/SOTA test. Freeze the protocol before full runs; never add seeds until significance or tune against these new evaluation results.

## Existing checkout and constraints

The successful Connect4 5090 run used revision `9b829e071f6d3d56064465dc8284a4b0f03c82a4`, RTX 5090, Ryzen 9 9950X3D, CUDA compiler 13.1.115 and driver 580.105.08. Its run directory was `build/connect4cnn/compare.vtew3n12/`. Preserve that experiment and the working runtime setup.

```bash
cd ~/Git/ml/cnn-5090
git status --short
git rev-parse HEAD
git remote -v
source build/connect4cnn/runtime-5090.sh
```

Inspect existing modifications before editing; do not reset or overwrite them. No new pull is required to follow this handoff: the source references below exist at the recorded revision. Record any later revision/diff separately. Push only to Kinvert-owned repositories if requested; never to PufferAI/PufferLib. This handoff requests local implementation and execution, not a push.

Read `AGENTS.md`, `START_HERE_5090.md`, `research/PAPER_PLAN.md`, `research/PONG_HYPER_SWEEP.md`, `ocean/pongcnn/README.md` and the relevant existing runners. Use direct `apply_patch` edits. Keep construction/training/evaluation native C/CUDA, orchestration in Bash, and existing Python/NumPy for reporting only. Avoid changes to `src/`, encoder kernels, initialization or environment rules. No Docker, system CUDA/driver changes, global packages, Torch installation, or modifications to the F-Zero environment. Reuse the local reporting venv; if missing, follow `START_HERE_5090.md` using `uv venv --python 3.12`.

**Known portability issue:** `ocean/pongcnn/canary.sh` and `sweep.sh` hardcode `/usr/lib/wsl/lib/nvidia-smi`. Replace that lookup with PATH-first discovery and the WSL fallback, following `ocean/connect4cnn/compare.py`. Do not fake GPU availability or create system symlinks. The Connect4 comparison runner is not a Pong runner: do not use its seven-action assumptions or call Pong `perf` a win rate.

## G240 development results and recipe choice

G240 discovery `build/pongcnn/hypers.jwyApeXS` finished with evaluation failures retained. Ours/Nature completed 24 training trials each; IMPALA/Impoola completed eight each before their one-hour search caps. There were two failed evaluations (Nature trial 5, IMPALA trial 7). Every family found a setting above 99% evaluated point fraction. This was training seed 9173 and development evaluation seed 29173, not independent confirmation.

Because tuning coverage differed, do not compare independently selected best-of-24 and best-of-eight runs as equally tuned winners. Instead use **two shared recipes**, crossed with every model. These were development trials 5 and 7, and G240 verified their eight learner values match across all four families. Recipe A favored IMPALA/Impoola; recipe B favored ours/Nature in this discovery seed. Selection is explicitly data-informed and does not represent exhaustive baseline tuning. Do not silently discard the failed evaluations from that history.

Start each job from isolated configs merged from the captured `config/default.ini` and `ocean/pongcnn/compare.ini`; apply the following exact overrides. Remove all sweep dimensions and run native `train`, not `sweep`. Do not copy a saved training metrics history as a live configuration.

| Training key | Recipe A: development trial 5 | Recipe B: development trial 7 |
|---|---:|---:|
| learning_rate | 0.0164826009 | 0.0840475857 |
| ent_coef | 3.94817544e-05 | 7.57243834e-05 |
| gamma | 0.969679892 | 0.989680648 |
| gae_lambda | 0.979004204 | 0.903716803 |
| replay_ratio | 1.91617525 | 3.57716227 |
| clip_coef | 0.297549665 | 0.193612725 |
| vf_coef | 1.75164807 | 3.41555738 |
| max_grad_norm | 0.538654029 | 1.25941002 |

Preserve those floating replay values and record the native effective update counts/semantics; do not independently round them. Other settings stay common: `anneal_lr=1`, `min_lr_ratio=0`, `vf_clip_coef=.2`, `anneal_ent_coef=0`, `momentum=.5`, `vtrace=0`. All models within a recipe get identical learner settings, observations, seeds, schedules and evaluation rules.

## Locked models and task

| Family | Encoder selection | Fixed definition | Total Pong parameters |
|---|---|---|---:|
| flex_quality | `policy.encoder=4` | One 16-channel SAME 7x7/stride-4 convolution, no pool/residual, flatten, projection 64 | 160,224 |
| nature_cnn | `policy.encoder=2` | Existing adapted Nature implementation | 138,016 |
| impala_cnn | per-build `-DC4_IMPALA_CNN`, `policy.encoder=0` | Existing adapted IMPALA implementation | 269,984 |
| impoola_cnn | per-build `-DC4_IMPOOLA_CNN`, `policy.encoder=0` | Existing adapted Impoola implementation | 151,200 |

For ours explicitly set `cnn_depth=1`, `cnn_channels_1=16`, `cnn_kernel_1=7`, `cnn_stride_1=4`, `cnn_pool_1=0`, `cnn_residual_1=0`, `cnn_global_pool=0`, `cnn_projection=64`. All families keep `hidden_size=128`, `num_layers=1`. Checkpoint parameter totals include policy/value heads and the recurrent core, not just the CNN.

All jobs use the existing float32 1x36x44 observation, three actions, score bars, scripted opponent, frame skip 8 and unchanged `compare.ini` environment settings. Common rollout settings: 64 agents, two CPU threads, one buffer/policy, horizon 32, minibatch 2,048; GPU 1, synchronous, CUDA graphs enabled, self-play off. Train new weights per job; never load Connect4 weights or another training seed's checkpoint.

## Frozen run matrix and timing

- Four families x two recipes x **five training seeds: 31001, 31002, 31003, 31004, 31005** = **40 full training runs**. Verify these seeds have not been used in local Pong discovery before freezing; if they have, choose and document an unused five-seed block before any new results exist.
- Every run: **4,194,304 decisions**, eight checkpoints at increments of **524,288**. With a 2,048-decision rollout this is 2,048 epochs and `base.checkpoint_interval=256`; verify against the actual trainer before launch.
- This deliberately replaces discovery's variable 2.1M–13.3M budgets with one matched 4.194M budget. Linear learning-rate schedules end at that full budget. Early checkpoints are full-run learning-curve points, not independently optimized short-budget runs.
- Evaluation seeds paired with training seeds: **41001–41005**, shared across models/recipes for each training-seed block. Each checkpoint receives **512 requested full matches**, recording actual completions. Expected complete matrix: **320 checkpoint evaluations**.
- Native Pong `perf` is episode-averaged **fraction of points won**, not percentage of matches won. Keep point-score difference alongside it.
- Train sequentially with no other GPU jobs. Precompute a balanced, deterministic family/recipe order within seed blocks; save the exact order before launching so each model is not always first or last.
- Full train timeout **1,800 seconds per job**; evaluation timeout **300 seconds per checkpoint**; no automatic retraining or longer-timeout retries based on poor results. Save incomplete/censored attempts and continue unrelated valid jobs. A numerical/configuration failure requires investigation, not blind continuation.
- Collect checkpoint timestamps during uninterrupted training. Evaluate saved checkpoints afterward so evaluation does not contaminate training wall time. Keep full-process wall/SPS separate from native adjusted uptime/SPS. Include startup and checkpoint writes in process wall; report builds, evaluation and complete campaign elapsed time separately.

The caps are safeguards, not expected durations. Estimate normal total runtime from canary throughput and evaluation cost, with explicit uncertainty; do not reuse Connect4's 18-minute figure for this larger experiment.

## Implementation and validation before full launch

There is no ready-made full Pong confirmation command at the recorded revision. Build a small dedicated runner, reusing `ocean/pongcnn/canary.sh`, its config/compiled-family conventions, and the receipt/timing ideas in `ocean/connect4cnn/compare.py`. Keep the existing Connect4 runner backward compatible. Provide `--prepare-only`, `--canary`, full execution, and receipt-based resume that never repeats successful jobs. Do not use `resume_hypers.sh`: it is specific to the old G240 discovery.

Before launch, save a versioned `protocol.json` and complete `jobs.json`, exact resolved INIs, source snapshots including untracked runner files, Git revision plus dirty diff, source/config/binary hashes, runtime helper, build commands and host/GPU/compiler metadata. If possible make a local commit of the validated new runner and frozen protocol, preserving unrelated work; no push is required. Captured file hashes are mandatory even if the checkout is dirty.

Run the existing Pong CPU validation and focused configuration tests:

```bash
bash ocean/pongcnn/tests/run_all.sh
.venv/bin/python -B ocean/pongcnn/tests/test_sweep.py
```

Use `NVCC_ARCH=sm_120`, `OPENBLAS_NUM_THREADS=1`, the existing runtime helper and `build.sh ... --float`. Build separate reference binaries using per-command flags, not exported global encoder flags. Reuse the encoder numerical/gradient and repeatability audits already passed on this 5090 if source/build identity is unchanged; otherwise rerun relevant tests from `START_HERE_5090.md`. CPU checks do not establish GPU correctness.

New canary: **four models x both recipes**, seed 51001, eval seed 61001, 65,536 decisions, four checkpoints, 32 evaluation matches per checkpoint. Use 120-second train/eval caps. Validate all 32 reload/evaluation attempts, finite weights/outputs, expected parameter counts, correct environment/action shape, exact checkpoint schedule, source/config identity, timers and recipe locking. Scores at this short budget are not a learning gate. If a canary evaluation hangs, fix or document the failure before starting the full campaign; do not alter full recipes based on canary scores.

Record representative same-host same-seed repeatability for any changed native path. Add meaningful runner tests for recipe/model locks, Pong labels, expected counts and resume/failure handling. Do not rewrite or optimize encoder kernels as part of this experiment.

Once canary/audits pass and the full protocol is frozen, start the full matrix detached in tmux, check startup once, and leave it running. **Do not actively watch or repeatedly poll the run.** Do not launch training on G240. A rerun request is not implied by a statistical interval crossing zero.

W&B: reuse existing external sidecar capability only if available and working; use project `cnn3`, a distinct `pong-5090-replication` group, friendly names, native SPS/uptime/agent_steps/env metrics and separately named evaluated point fraction. Upload outside timed training. Local receipts remain authoritative, and missing W&B credentials must not block the experiment or prompt a system dependency install. Preserve failed uploads for later retry without retraining.

## Analysis and interpretation

Before full results, record the primary comparison as **ours versus Nature within each shared recipe**, at the 95% evaluated point-fraction target. Report the fraction of seeds reaching the target, first observed checkpoint crossing time and its checkpoint-resolution limit. Non-achieving seeds are censored; never drop them or average only successful runs. Keep first crossing separate from sustained performance and retain later declines. This five-seed stage estimates reliability/variance; it does not meet a pre-established power guarantee.

Report all checkpoint points, full per-seed curves, mean time/score and steps/score curves, and observed Pareto sets for each recipe. Use paired training-seed bootstrap intervals, resampling whole seed blocks across models/checkpoints/recipes; evaluation matches are not independent training-seed replications. Label pointwise intervals as pointwise; they do not establish frontier-wide dominance. Include raw observations so any frontier can be reproduced. Do not interpolate unsupported perfect scores, combine the best seeds, or interpret an evaluation failure as zero performance.

A combined envelope across the two recipes may be shown as a **descriptive two-recipe envelope**, but the separate shared-recipe results are primary. Selecting a winner on these new seeds does not establish an unbiased best-tuned comparison. No SOTA or universal encoder claim follows from this experiment. Pong was excluded from architecture selection but has already been used for learner development; distinguish architecture transfer from untouched-task evaluation.

Keep the earlier discovery's unequal coverage and two evaluation failures in context. Report that the original common .001-LR Pong pilot failed to learn while alternative learners worked. A fast implementation of our model versus slow reference code is not, by itself, proof of architectural superiority; current native backend limitations remain part of the paper fairness audit.

## Deliverables and transfer back

Write one `REPORT.md` with exact reproduction commands, protocol/source IDs, totals and failures, per-model/per-recipe final means and paired uncertainty, target-attainment results, full time/step frontiers, timing definitions and limitations. Include CSV/JSON data for every checkpoint, raw native metrics/eval logs, resolved configs, source/build/runtime/hardware hashes, and a machine-readable audit report. Give actual training-process sum, total evaluation time and overall elapsed time separately. Keep checkpoints and executable binaries on the 5090 for independent rechecks.

Save the immutable original receipts under a unique `build/pongcnn/confirm-5090.*` directory. Copy small evidence and authored analysis into a unique `research/results/pongcnn/` directory if committing locally, preserving raw contents and keeping weights/build outputs out of Git. Do not overwrite earlier Connect4 artifacts or reports.

The existing packager can be reused if the new runner emits `protocol.json`, `jobs.json` and `results.csv`; inspect its inclusions and make sure new receipt formats/runtime metadata are retained:

```bash
bash ocean/connect4cnn/package_results.sh build/pongcnn/confirm-5090.ACTUAL_RUN_ID
```

It prints the real archive path and SHA256. Include new runner/source files and the runtime helper inside the evidence directory before packaging, so G240 can reproduce and audit them without a push. Transfer the printed archive path (replace the placeholder):

```bash
scp build/hardware-artifacts/ACTUAL_ARCHIVE.tar.gz claude@g240:/home/claude/cnn/build/hardware-artifacts/
```

Return the archive SHA256, exact local code/source identity, concrete rerun/resume commands, actual job/evaluation completion counts, total runtime and a short factual result. If some runs fail or targets are missed, report that directly and preserve them.
