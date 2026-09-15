# Run the fixed CNN comparison on the RTX 5090

For a fresh session with no conversation context, follow **[START_HERE_5090.md](../START_HERE_5090.md)**. It includes the full setup, CPU/GPU tests, build/train instructions, expected completion counts, result-table definitions and evidence packaging.

Purpose: measure ours, Nature, IMPALA and Impoola on the 5090 using one fixed recipe. No new architecture search. Use the **Kinvert/PufferLib** fork, branch **cnn-research**; never push this research to PufferAI/PufferLib. Record the exact `git rev-parse HEAD` on both machines.

## Checkout and existing dependencies

After this branch is pushed to the Kinvert fork, clone into a new directory so existing 5090 training checkouts stay intact:

```bash
git clone --branch cnn-research https://github.com/Kinvert/PufferLib.git cnn-5090
cd cnn-5090
git rev-parse HEAD
```

Reuse the 5090 machine's existing working CUDA/NCCL toolchain. The native build needs `nvcc` with `sm_120` support, clang, ccache, the normal Linux build libraries, and NCCL headers/libraries. `build.sh` obtains shared Raylib in the checkout if absent. This is CPU environment simulation with a GPU learner; do not pass `--cu`. No Docker, driver/CUDA upgrades, Torch installation, or global environment changes are part of these instructions.

Create a local reporting venv using **uv venv**. The comparison needs only NumPy; the full research/search dependency set is unnecessary. The NumPy pin below matches `research/requirements.lock`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'numpy==2.5.3'
```

Set `CUDA_HOME` only if the existing toolkit is elsewhere than `/usr/local/cuda`. If NCCL is already in a working training venv, point `NCCL_ROOT` at that package directory (containing `include/nccl.h` and `lib/`), for example:

```bash
# Replace this example with the existing location on the 5090 machine.
export NCCL_ROOT=/path/to/existing/venv/lib/python3.12/site-packages/nvidia/nccl
```

The runtime helper reads those files and sets process-local paths. It also discovers NCCL in this checkout's `.venv` if present; the G240 fallback path is not assumed to exist on the 5090. GPU inspection now finds `nvidia-smi` in PATH, with a WSL fallback. These steps have not yet been executed on the separate 5090 host.

## Run

First validate build, training, saved-model reload and evaluation:

```bash
bash ocean/connect4cnn/hardware_compare.sh --canary
```

Then run the actual comparison, optionally detached:

```bash
mkdir -p build/connect4cnn
tmux new-session -d -s cnn-hardware 'bash ocean/connect4cnn/hardware_compare.sh --full > build/connect4cnn/hardware-launch.log 2>&1'
```

The full preset trains **four models × one seed (173)** for **13,312,000 decisions each**, with 13 checkpoints and 1,024 requested evaluation games per checkpoint on seed 20173. It uses the frozen quality model, the common H128/L1 core, original Connect4 representation 0, and float32. Each training job has a 2,400-second safety cap. Actual 5090 runtime is unmeasured. W&B is disabled for this initial local comparison. `--print-command` displays the full command without using the GPU.

The runner refuses existing compute processes before builds and each training job. These checks are not a GPU reservation; keep other jobs off that GPU during measurement. Let the run finish unattended. Results appear in a new `build/connect4cnn/compare.*` directory; the launcher prints its exact path.

## Read the results correctly

- `REPORT.md` and `results.csv`: full training-process wall time/SPS, native adjusted uptime/SPS, evaluation results, steps, parameter counts and VRAM. Compare the same timing column across hosts. Canary timing is not the measured learning-scale result.
- `protocol.json`, `recipe.ini`, `commands.jsonl`, resolved per-job INIs, source/config/binary hashes: exact experiment provenance. `host.json`, `cpu.txt`, `cuda-compiler.txt` and GPU snapshots record the machine/toolchain context. Raw metrics and checkpoints remain local.
- **Run the same revision and preset on G240 after Pong finishes** for a controlled cross-host comparison. Differences in CPU, software and GPU make this a whole-machine training comparison unless those other factors are controlled. Do not attribute every speed difference solely to the GPU.
- The older 5060 confirmation is a useful historical reference, but it used the [reconstructed measured source](BENCHMARK_STATE.md), commit `b2fa7787a754d36362374ac271ea6c7b23beeb25`. Current code includes later observation and reporting work. Do not label old-versus-new timings as an exact same-source hardware comparison.
- One paired training seed is an initial speed check, not confirmation of score superiority. Repeat matched runs and use the [paper evidence standard](PAPER_PLAN.md#required-evidence-standard-defend-the-claim-like-a-thesis) for a publishable claim. The full six-model/five-seed protocol remains available through `confirm.sh` if separately scheduled.

Keep reports, configs, commands and hashes when transferring results back. The research branch includes source, authored notes and archived small receipts; downloaded papers, virtual environments, search caches, build binaries and full checkpoints are ignored. The current Pong campaign runs from its captured binaries/reporting files and is unaffected by this checkout being committed or pushed.
