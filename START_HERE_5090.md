# 5090 handoff: build, validate, train and report the CNN comparison

This file is the complete starting brief for an agent or person with **no prior conversation context**. Work in this clone; follow [AGENTS.md](AGENTS.md). The owner is **Kinvert**. Publish only to a Kinvert-owned repository when authorized; official `PufferAI/PufferLib` is not the destination.

**October 7 current entry point: [NEXT_5090_FEEDBACK.md](NEXT_5090_FEEDBACK.md).**
Kinvert authorized the research delivery and its bounded native cross-game
feedback canary. Follow that two-command handoff. Earlier execution holds below
are historical; larger campaigns and encoder5 are not scheduled by this update.

**October 2 prior entry point: [NEXT_5090_TASK.md](NEXT_5090_TASK.md).** It gives
the first GPU-free preparation commands and the ordered future validation gates.
Use this file's runtime/venv information as needed; the old hardware campaign
commands in sections 4–7 below are historical and must not be relaunched.

## What we are doing

**October 2 execution hold:** Kinvert currently wants groundwork without occupying
the 5090. Do not run GPU tests, canaries or searches from the commands below now.
Read [CLAIM_PIPELINE.md](research/CLAIM_PIPELINE.md) for the new native evaluation,
timing and artifact-audit preparation. When GPU work is scheduled again, retain
the encoder math/legacy/reload gates; full-frontier confirmation also requires
inference calibration. Earlier discovery instructions are not an automatic launch.

**Current task, September 29:** follow [NEXT_5090_FLEX2_SWEEP.md](NEXT_5090_FLEX2_SWEEP.md). Validate expanded encoder 5, run bounded native canaries, and then search two controlled architecture panels on this RTX 5090. The appearance task in [NEXT_5090_VALIDATION.md](NEXT_5090_VALIDATION.md), the hardware comparison, and the Pong campaigns are historical context; their launch commands below are not instructions to repeat them. The planned full-frontier confirmation still needs separate measurement/evaluation/inference gates.

We are researching efficient pixel-input CNNs in native PufferLib 5.0. Compare our existing quality CNN with adapted Nature, IMPALA and Impoola. All models train from scratch; none is pretrained. Keep the full observed frontier and identical per-panel learner settings; preserve outcomes where references win.

The task is unchanged 7-column × 6-row Connect4 against the same scripted opponent, rendered as grayscale 1×36×44 pixels. Each cell occupies a solid 6×6 square (`env.representation=0`). All four models use the same hidden-128, one-layer recurrent core and learner recipe. Ours has one 16-channel 7×7/stride-4 SAME convolution and projection 64. Exact fixed selections are in `ocean/connect4cnn/confirmation.json`; the recipe is `ocean/connect4cnn/compare.ini`.

G240 is a **different machine with an RTX 5060**. Its old Pong campaign is finished; do not restart it or assume its paths exist here. Historical results and limitations are in [BENCHMARK_STATE.md](research/BENCHMARK_STATE.md). We have an encouraging Connect4 time frontier, not a proven general-purpose SOTA CNN. Report actual measurements, including losses and failures.

## 1. Get the right checkout

```bash
git clone --branch cnn-research git@github.com:Kinvert/PufferLib.git cnn-5090
cd cnn-5090
git remote -v
git rev-parse HEAD
git status --short
```

If already cloned, fetch `origin` and inspect changes before updating; do not reset someone else's work. Record the revision and any local changes. This branch's normal `origin` is `Kinvert/PufferLib`. Use a separate directory from existing F-Zero/PufferLib training checkouts.

## 2. Inspect and reuse the existing GPU environment

This machine has already trained PufferLib workloads. Locate its working toolkit and NCCL rather than installing or upgrading system software. Inspect the existing launch script if useful, without modifying that checkout.

```bash
command -v uv
command -v nvcc
command -v clang
command -v ccache
command -v nvidia-smi
```

Run `nvidia-smi` (or `/usr/lib/wsl/lib/nvidia-smi` on WSL if missing from PATH) outside the agent sandbox. Confirm the GPU is the 5090 and has no competing compute workload. Missing `/dev/nvidia*` inside a WSL sandbox is not proof that the GPU is unavailable. Native Linux does not require the WSL path. Do not use Docker or modify CUDA, drivers, cuDNN, global packages or another venv.

The build needs an existing CUDA toolkit supporting `sm_120`, clang, ccache, normal Linux development libraries, and NCCL. Set these **in this shell only**, using actual existing paths:

```bash
# Omit CUDA_HOME if the existing toolkit is already /usr/local/cuda.
export CUDA_HOME=/actual/existing/cuda/toolkit

# Package directory containing include/nccl.h and lib/libnccl.so*.
export NCCL_ROOT=/actual/existing/venv/lib/python3.12/site-packages/nvidia/nccl
```

Those are placeholders, not G240 paths to copy literally. Find NCCL by inspecting the known working training venv's `site-packages/nvidia/nccl`; do not activate or change that environment. If a required dependency is actually absent, report precisely what is missing before making system changes. The native build downloads the shared Raylib release into this checkout if absent; there is no per-environment renderer setup.

## 3. Create this clone's local reporting venv

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'numpy==2.5.3'
uv pip check --python .venv/bin/python
.venv/bin/python --version
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
export OPENBLAS_NUM_THREADS=1
```

Reuse a compatible existing `.venv` in this clone rather than overwriting it. NumPy's pin matches `research/requirements.lock`. Native training/search is C/CUDA; Python here runs validation and reporting. **Torch, LanceDB and downloaded research papers are unnecessary for this job.** The fixed hardware runner disables W&B, so neither its package nor credentials are required. Do not install the entire research dependency set just to train.

## 4. Run cheap configuration checks, then build and validate on GPU

From the repository root:

```bash
.venv/bin/python -B ocean/connect4cnn/tests/test_hardware.py
.venv/bin/python -B ocean/connect4cnn/tests/test_confirmation.py
.venv/bin/python -B ocean/connect4cnn/tests/test_sweep_tools.py
bash ocean/connect4cnn/hardware_compare.sh --print-command

# Builds four separate float32 binaries, then short train/save/reload/eval jobs.
bash ocean/connect4cnn/hardware_compare.sh --canary
```

Use the outside-sandbox execution path for build/GPU commands. The canary prints `Comparison: /.../build/connect4cnn/compare.XXXXX`. In **that exact directory**, inspect `finished.json`, `jobs.json` and `REPORT.md`. Require status `ok`, four successful jobs and 16 evaluations before the full run. Short canary scores and SPS are plumbing evidence, not speed or learning results. The normal build automatically creates the shared Raylib directory.

For numerical/gradient checks on this GPU, use the existing native test harnesses and independent NumPy references:

```bash
bash ocean/connect4cnn/tests/run_all.sh
bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature
.venv/bin/python ocean/connect4cnn/tests/test_nature.py --library build/connect4cnn/test_nature.so
.venv/bin/python ocean/connect4cnn/tests/test_flex.py --library build/connect4cnn/test_nature.so
bash ocean/connect4cnn/tests/build_encoder_test.sh test_impala
.venv/bin/python ocean/connect4cnn/tests/test_impala.py --library build/connect4cnn/test_impala.so
```

These checks include environment parity/pixels, forward/gradient comparisons and repeatability checks supported by the harnesses. Preserve their outputs under `build/connect4cnn/validation-5090/` or equivalent and record failures. Same-device repeatability is distinct from bitwise matching the 5060. Avoid changing precision or math settings to make a failing test pass unnoticed.

An individual native build, when diagnosing a failure, follows normal PufferLib style:

```bash
bash build.sh connect4cnn build/connect4cnn/diagnostic --float
```

Use the comparison runner for the actual experiment: it supplies the correct family selection and isolated resolved configs. Do not benchmark `diagnostic train` with unrelated checkout defaults. `--cu` is a GPU environment implementation selector and is not needed here.

## 5. Run the actual fixed comparison and leave it alone

```bash
mkdir -p build/connect4cnn
tmux new-session -d -s cnn-hardware 'bash ocean/connect4cnn/hardware_compare.sh --full > build/connect4cnn/hardware-launch.log 2>&1'
```

If tmux is unavailable, running `bash ocean/connect4cnn/hardware_compare.sh --full` in a persistent terminal is sufficient. Verify startup once and do other work; **do not actively watch or repeatedly poll training**. Do not start a second copy when the first is merely slow.

This trains four fixed models serially, seed **173**, **13,312,000 agent decisions each**, saving **13 checkpoints** each. Evaluation uses seed **20173** and **1,024 requested games per checkpoint**. Success means four jobs and **52 evaluations**. Training timeout is 2,400 seconds per model; each evaluation has a 60-second cap. Report actual time, since 5090 runtime is not yet measured. Never silently change the budget, core, recipe, representation or precision for a faster result.

At a requested status check or completion, find the exact run from the launcher:

```bash
run_dir=$(sed -n 's/^Comparison: //p' build/connect4cnn/hardware-launch.log | head -n 1)
cat "$run_dir/finished.json"
cat "$run_dir/REPORT.md"
```

`finished.json` is written at completion; absence is not success. If the process exits earlier, inspect the launcher and per-job build/train/eval logs. Preserve all failed attempts. Fix infrastructure if appropriate, but label any retry explicitly rather than replacing unfavorable results.

## 6. Tabulate and explain results

The runner **already writes the tables**: `REPORT.md` has final per-model results and all checkpoint evaluations; `results.csv` contains 52 rows for a fully successful full run. `research/EXPERIMENT_LOG.md` gets an automatic entry. Select each model's **final 13.312M-step row** for the compact summary, not its most favorable earlier score. Retain the whole curves for frontier analysis.

Return this four-row table, plus failures separately:

| Model | Decisions | Held-out wins | Train wall seconds | Process SPS | Native average SPS | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Ours quality | | | | | | |
| Nature | | | | | | |
| IMPALA | | | | | | |
| Impoola | | | | | | |

Include the Git revision, actual GPU/CPU, CUDA compiler/driver, run directory, test outcomes, and completion counts. `protocol.json`, `host.json`, `gpu.txt`, `cpu.txt`, `cuda-compiler.txt`, resolved INIs and native metric histories provide these facts. `train_process_wall_s` includes startup/checkpoint writes; `native_avg_sps` uses the trainer's adjusted timer. Never mix those definitions.

Within the 5090 run, compare all four models fairly. For 5090-versus-5060 ratios, use matching model/seed/config/step rows from the **same source revision and preset** on both hosts: wall-time speedup = 5060 seconds / 5090 seconds. G240's same-revision run must wait until Pong finishes. Historical 5060 confirmation data in `research/results/connect4cnn/confirm.ol9tcj5k/` used older captured source; label any comparison to it as historical, and do not compare this one seed to its five-seed mean as though they were paired.

One seed is an initial hardware measurement, not proof of score superiority. CPU/toolchain differences also prevent automatically attributing all whole-machine speedup to the GPU alone. Do not claim SOTA from this run.

## 7. Package the evidence for the G240 project

```bash
bash ocean/connect4cnn/package_results.sh "$run_dir"
```

This creates a `.tar.gz` under `build/hardware-artifacts/` containing reports, raw logs/metrics, resolved configs, source snapshots and provenance. It excludes full checkpoints, compiled binaries and W&B caches. Keep the original local run and checkpoints until the results are verified. Transfer the archive using the user's existing file-transfer method; do not publish it or push to upstream without authorization.

Final handoff: the table above, brief interpretation with limitations, the archive path, and the exact command/revision used. Further architecture search, Pong experiments, dilation implementation and system tuning are outside this initial 5090 job.
