# RTX 5090 handoff: validate and search expanded native CNNs

Start at [NEXT_5090_TASK.md](NEXT_5090_TASK.md) for current scope and executable
preparation. This document supplies the detailed future architecture-search gates.

October 2: GPU execution is currently deferred at Kinvert's request. Prepare and
review only. [Measurement groundwork](research/CLAIM_PIPELINE.md) now covers the
separate locked-model exact-evaluation/timing canary; it does not authorize
discovery or full confirmation. The commands below are for a later scheduled
GPU session, after the applicable validation gates.

September 29, 2026. Owner: Kinvert. Work in the existing `~/Git/ml/cnn-5090` checkout. This is the current encoder-5 task. Historical hardware, Pong, and appearance campaigns are complete; preserve their source, logs, checkpoints, and unfavorable results. G240's RTX 5060 is **not** the execution host for this task.

The goal is a reproducible first architecture search on Connect4CNN, not a claim that an encoder beats Nature. Encoder 5 adds 1–4 stages, configurable channels/kernel/stride/dilation/pooling/residual repetitions, four readouts, and ReLU/SiLU/GELU/learned PReLU/learned rational activations. It uses the existing native C/CUDA learner and PROTEIN optimizer. The recurrent core remains H128/L1. Read [the grammar and limitations](research/FLEX2_CNN_SWEEP.md).

## 1. Update without losing 5090 work

Use only `Kinvert/PufferLib` as `origin`; never push to official PufferLib. Inspect local changes and divergence before merging or pulling:

```bash
cd ~/Git/ml/cnn-5090
git status --short
git remote -v
git fetch origin cnn-research
git log --oneline --left-right HEAD...origin/cnn-research
```

If clean and strictly behind, fast-forward with `git merge --ff-only origin/cnn-research`. If diverged or dirty, preserve the 5090's local commits and artifacts, then reconcile in a separate worktree/branch; do not reset. Record the actual commit and working diff used for each result. Confirm these files arrived: `ocean/connect4cnn/flex2.cu`, `sweep_flex2_fast.ini`, `sweep_flex2_stage2.ini`, `tests/test_flex2.py`, and this handoff.

## 2. Reuse the working local runtime

Reuse the 5090's Python 3.12 `.venv`, CUDA toolkit, NCCL and `build/connect4cnn/runtime-5090.sh`. For a fresh venv only, run `uv venv --python 3.12 .venv` and install NumPy 2.5.3; online W&B additionally needs the pinned `wandb==0.21.4`. No Torch, Docker, driver, CUDA or global-library changes. Run GPU commands outside an agent sandbox and only when the 5090 is idle.

```bash
source build/connect4cnn/runtime-5090.sh
export NVCC_ARCH=sm_120
export OPENBLAS_NUM_THREADS=1
nvidia-smi
.venv/bin/python -B ocean/connect4cnn/tests/test_sweep_tools.py
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_fast.ini --max-runs 3 --canary --prepare-only --wandb disabled
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_stage2.ini --max-runs 3 --canary --prepare-only --wandb disabled
```

Preparation writes isolated configs and source SHA256 receipts without building or using the GPU. Inspect both printed `protocol.json` and resolved INI files: encoder 5, depths 1/2, 7/8 active search dimensions, one GPU, representation 0, and H128/L1. These `--canary` preparations record a 32,768-decision plumbing budget; the normal recipes specify a fixed 13,312,000 decisions. The broad `sweep_flex2.ini` is a grammar example; its 17-dimensional/12-trial search is not this launch plan.

## 3. GPU acceptance gates before discovery

Save command output and failures. The new activation/dilation/readout kernels and their shared-kernel changes have not yet passed GPU validation on this revision. Build and run the native CUDA test library, then the older Nature/Flex regressions:

```bash
bash ocean/connect4cnn/tests/build_encoder_test.sh test_flex2
.venv/bin/python ocean/connect4cnn/tests/test_flex2.py --library build/connect4cnn/test_flex2.so --report build/connect4cnn/flex2-math-5090.json
bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature
.venv/bin/python ocean/connect4cnn/tests/test_nature.py --library build/connect4cnn/test_nature.so
.venv/bin/python ocean/connect4cnn/tests/test_flex.py --library build/connect4cnn/test_nature.so
```

Require all cases to pass, including all five activations, parameter/coefficient finite differences, dilation, odd padding, adaptive pooling, residuals, and eager/graph repeatability. The October 2 harness adds an independent float64 full-forward/all-parameter-gradient oracle, 69 direct boundary/operator checks and 14 whole-encoder fixtures. Read [the verification contract](research/FLEX2_VERIFICATION.md) for preserved evidence, fixed tolerances and a fresh-report command sequence. Reports cannot be overwritten; choose a new path for a rerun. A failure blocks the sweep. Do not waive a failed case or switch precision. This strengthened suite is compiled but still awaits its first 5090 execution.

Next run **one short native PROTEIN canary per panel**. Each builds the real trainer and attempts exactly three 32,768-decision trials:

```bash
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_fast.ini --max-runs 3 --canary --timeout 3600 --wandb disabled
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_stage2.ini --max-runs 3 --canary --timeout 3600 --wandb disabled
```

For each printed `Sweep: ...` directory, require `finished.json` status `ok`, three rows in `results.csv`, unique effective architectures, finite model weights, expected checkpoint step counts, and plausible measured SPS/VRAM. Preserve failures. Checkpoint reload and same-seed repeatability are separate gates: select a resolved trial INI under `metrics/connect4cnn/`, evaluate its corresponding saved checkpoint using **that trial's policy config** in an isolated config directory, and repeat a fixed short train/eval pair at the same seed. Require identical checkpoint bytes and native evaluation record on the same 5090. Do not evaluate a searched checkpoint with the campaign's unswept default architecture. Record exact commands, config and checkpoint hashes. Existing `appearance_canary.sh` shows the native train/reload/checkpoint pattern, but does not test encoder 5.

If math, legacy regression, memory, reload, or repeatability fails, fix and revalidate before a longer search. A successful three-trial canary proves plumbing, not learning or a Pareto advantage. Use its actual process time and VRAM to estimate the full campaign, and keep the GPU exclusive while timing it.

## 4. Run the two discovery panels after the gates pass

Panel A searches seven controls in a one-stage CNN: projection, readout, first-stage channels/kernel/stride/dilation/activation. Panel B fixes the existing 16-channel 7×7/stride-4 first stage and searches eight second-stage/readout controls, including pooling and one residual repetition. Both fix Connect4 appearance 0, learner/core settings, and 13.312M decisions per trial. First-stage stride is at least four and channels at most 32, avoiding the unbounded stride-one/full-width patch buffers in the grammar example. The panel recipes request **64 trials each**, serially. Do not mix trial scores across panels without recording that they are independent searches.

Use the existing W&B sidecar only if the pinned package and credentials are ready; online upload adds CPU/network work to timed campaigns. `--wandb disabled` still saves native metrics, configs, source hashes, checkpoints, CSV and report locally. Give each process a sufficiently large whole-campaign timeout; the 600-second `max_suggestion_cost` in the recipe is a PROTEIN cost model, not a per-trial timeout. For example, after estimating from the canary:

```bash
mkdir -p build/connect4cnn
tmux new-session -d -s cnn-flex2-fast 'bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_fast.ini --timeout 86400 --wandb disabled > build/connect4cnn/flex2-fast-launch.log 2>&1'
```

Run the second panel serially after the first completes and the GPU is idle, replacing the recipe with `sweep_flex2_stage2.ini` and using a distinct launch log/session. Verify startup once, then leave the runs alone. Do not relaunch a slow or completed campaign. Retain any timeout, failed graph, duplicate-proposal count or incomplete trial; `finished.json` must say `ok` and `completed=64` for full success. A wall cap can truncate the search and must be reported as such.

## 5. Report all outcomes and choose follow-up candidates

Keep the full `results.csv`, `REPORT.md`, `sweep.log`, resolved trial INIs, checkpoints, `protocol.json`, source hashes, validation reports, host/GPU/compiler facts and exact commit. For each campaign run:

```bash
.venv/bin/python research/audit_sweep_coverage.py build/connect4cnn/sweep.EXACT_ID/results.csv --long-budget 8000000
```

The report should include completed/failed/duplicate counts, distinct active graphs, architecture-control coverage, parameters, VRAM, process wall time, native SPS, performance versus wall-clock Pareto points, and every training curve. Select candidates using discovery data only; preserve the reference winners and weak configurations. A later frozen evaluation must use fresh paired seeds, independent held-out episodes, the same learner budgets, and Nature/IMPALA/Impoola controls before any comparative claim. This search does not repair Pong's pooled evaluator or establish cross-task generality. Do not call the best training score a held-out win rate or claim SOTA from this search.

Return a compact status/results table and the exact paths to the 5090 evidence; keep large artifacts there until reviewed. Do not push 5090-local evidence or changes upstream without Kinvert's authorization.
