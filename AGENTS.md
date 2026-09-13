# Working preferences

- Edit files directly using `apply_patch` whenever possible. Do not use Python heredocs (`python << 'PY'`) or scripts merely to create or rewrite text files.
- Reuse PufferLib's normal build paths and shared dependencies. Do not create per-environment Raylib installations or header copies; generate synthetic pixel observations directly in the native buffer, independently of the human viewer.
- When changing `src/`, follow the surrounding PufferLib coding style and existing interfaces. Keep changes small, simple, and straightforward; avoid unrelated refactors or speculative abstractions. Keep experimental CNN code outside the core where the existing custom-encoder pattern permits. Preserve determinism and speed: inspect allocation, synchronization, RNG, and reduction-order effects, and validate numerical correctness, seeded repeatability, and relevant performance before claiming an improvement.
- The user is Kinvert and owns the `Kinvert/PufferLib` fork. Use that fork as `origin` and official `PufferAI/PufferLib` as `upstream`.
- Keep work local until it is worthwhile and clean. Do not push to official PufferLib. Any future publication should use the Kinvert fork and follow the user's authorization; upstream contributions must be clean and ready for review.
- Use `uv venv` for Python environments. Research tools target Python 3.12; install from `research/requirements.lock` and verify compatibility before upgrading LanceDB or its dependencies.
- CUDA-related inspection is allowed, but do not modify the system CUDA stack, GPU drivers, cuDNN, system libraries, or global configuration. Basic Torch work is allowed only when strictly contained within `/home/claude/cnn/.venv`; do not alter other environments. The current paper conversion/search tools run on CPU and do not require Torch.
- For CNN research, start at `research/README.md` and its linked reports. Use `rg` for exact code references and `.venv/bin/python research/search.py query "..."` for local paper/code retrieval. Check `status` and run `build` after source changes; inspect cited files/PDFs before treating search hits as evidence. The original pasted research is an unverified transcript; preserve it verbatim.

## GPU access on G240 (WSL)

- This host uses WSL. The GPU was verified outside the sandbox on 2026-09-12 as an RTX 5060 with 8 GB VRAM. The user's RTX 5090 training machine is separate.
- **Do not infer that GPU access is unavailable because `nvidia-smi` is absent from PATH or `/dev/nvidia*` is missing.** Use `/usr/lib/wsl/lib/nvidia-smi` through the tool's `require_escalated` execution path. No Docker, device-file creation, driver installation, or system CUDA changes are needed.
- Existing toolkit: `/usr/local/cuda` (CUDA 12.8); compiler `/usr/local/cuda/bin/nvcc`. WSL driver libraries are under `/usr/lib/wsl/lib`. `NVCC_ARCH=sm_120` is the explicit compile target for this RTX 5060 when sandboxed `-arch=native` cannot query it.
- Follow normal PufferLib builds and `train.gpus=1`. GPU training/runtime checks must run outside the sandbox with tool escalation. CPU environment simulation still uses a GPU learner; `--cu` means a CUDA **environment implementation**, and is not required for Connect4CNN's CPU environment plus CUDA CNN.
- Read-only working references: `/home/claude/5c-admiral/config/admiral.ini`, `/home/claude/5c-admiral/build.sh`, and `/home/claude/fzero/PufferLib-5.0/ocean/fzero_combat_jack/runtime_env.sh`. F-Zero's launch scripts set `LD_LIBRARY_PATH` to the NCCL wheel's `lib` directory before launching the native binary.
- An existing NCCL package is available at `/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl`. Its headers/libraries may be read for this build; do not modify that environment. This project's research `.venv` does not currently contain NCCL or Torch. Set compiler/library paths for the individual build/run; do not change shell startup files or global settings.
- Check GPU load before starting a bounded run. Save logs/checkpoints locally. Do not interfere with other training jobs or claim determinism/performance until the corresponding tests actually pass.
- Do not actively watch long training runs or spend turns polling them. Let the runner save results automatically; check status when the user asks or when results are needed for authorized work.

Verified GPU inspection command (run with tool escalation):

```bash
/usr/lib/wsl/lib/nvidia-smi
```

Connect4CNN now has a verified native float32 GPU workflow. Run from this checkout, with tool escalation for builds/GPU execution:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh connect4cnn build/connect4cnn/train --float
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh
.venv/bin/python ocean/connect4cnn/tests/test_encoder.py --library build/connect4cnn/test_encoder.so
bash ocean/connect4cnn/smoke.sh
```

The runtime helper only sets process-local paths and reads existing NCCL files. CUDA stubs are used only for linking, never as runtime driver libraries. The smoke runner explicitly sets `train.gpus=1`, stores fresh local artifacts, and enforces time limits. Read `ocean/connect4cnn/README.md` for verified results and remaining limitations; CPU-only reference checks are not GPU validation.

For matched **original state Connect4 versus pixel CNN** comparisons, use `NVCC_ARCH=sm_120 bash ocean/connect4cnn/compare.sh` outside the sandbox. Common settings live in `ocean/connect4cnn/compare.ini`; do not tune the two environment INIs independently for a common-recipe claim. The runner builds separate binaries and writes `REPORT.md`, `results.csv`, exact commands, source snapshots/hashes, logs, and checkpoints under a fresh `build/connect4cnn/compare.*` directory. Use `--help` for budgets, seeds, and evaluation options. Implemented candidates are `state`, `tiny_cnn`, `nature_cnn`, `impala_cnn`, and `impoola_cnn`; select with `--variants`. Defaults remain state/tiny. Use `--stock --timeout 600` for the separately labeled stock training configuration in float32; it rejects custom training recipes/budgets and uses the common held-out evaluation setup.

Track performance across changes in `research/EXPERIMENT_LOG.md`. The comparison runner appends results automatically; use `--note` to state what changed or why the run exists. Keep SPS (whole-process and native timing boundaries explicitly distinguished), win rate, score, timesteps, seeds, parameters, VRAM, hardware, and source/config/build provenance. Preserve old results and failed runs. For experiments outside the runner, add equivalent records rather than reporting numbers only in chat.

Before committing results, copy their small reports, CSVs, configs, evaluation outputs, and provenance into `research/results/connect4cnn/<run>/` and point the experiment-log links there. Follow that archive's README. Full checkpoints/build outputs and research caches remain ignored; links only into local `build/` do not preserve evidence in a fresh clone.

Label baselines precisely: `state` uses the original PufferLib architecture with a modified common configuration, not untouched stock tuning. `tiny_cnn` is the custom single-convolution encoder. `nature_cnn` preserves Nature's three convolution sizes/widths, adapting grayscale input and projection to this game/core. `stock_state` preserves stock training settings in float32, with common evaluation controls. The stock baseline is now measured (98.87% mean wins); the earlier tiny CNN result did not beat it. Consult the policy/hyperparameter table and dated entries in `research/EXPERIMENT_LOG.md` before describing results; distinguish matched learner settings from unequal parameter counts and separately configured baselines.

Nature checks use `NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature`, then source the runtime helper and run `OPENBLAS_NUM_THREADS=1 .venv/bin/python ocean/connect4cnn/tests/test_nature.py --library build/connect4cnn/test_nature.so` outside the sandbox. The runner selects Nature with a per-build `C4_NATURE_CNN` definition; do not alter the tiny encoder or export that definition globally. Float32 forward/all parameter gradients and eager/graph repeatability passed, and two short same-seed training runs produced identical checkpoints. BF16 remains unvalidated.

IMPALA/Impoola share `ocean/connect4cnn/impala.cu`, selected with per-build `C4_IMPALA_CNN` / `C4_IMPOOLA_CNN`. Both use original TensorFlow SAME pooling alignment; GAP is the only readout change. Use `build_encoder_test.sh test_impala` and `test_impala.py --library build/connect4cnn/test_impala.so` with the same runtime helper/CPU-thread setting as Nature. Both passed float32 numerical, max-pool edge/tie, and determinism checks, including identical same-seed training smoke checkpoints. The initial implementation ran around 10,000 SPS and 3.5 GB GPU memory; use a suitable per-process timeout (2,400 seconds for the first full runs) and check current measurements before scheduling large comparisons.

## Local LanceDB search for future agents

Run these commands from `/home/claude/cnn`. Use the existing Python 3.12 `.venv`; do not reinstall or upgrade packages just to search.

```bash
# Check whether the index matches current files.
.venv/bin/python research/search.py status

# Default: hybrid keyword + semantic search across papers, notes, and code.
.venv/bin/python research/search.py query "impoola benchmarks" --limit 5

# Restrict results to original paper conversions.
.venv/bin/python research/search.py query "impoola benchmarks" --kind paper --limit 5

# Exact identifiers: full-text search, without embedding inference.
.venv/bin/python research/search.py query "create_custom_encoder" --mode fts --kind code

# Conceptual code search when the exact identifier is unknown.
.venv/bin/python research/search.py query "recurrent state reset on terminal" --mode vector --kind code

# Structured results for tooling; --kind also accepts note.
.venv/bin/python research/search.py query "encoder core compute allocation" --kind note --json

# Refresh after changing code, notes, or paper conversions; reuse cached vectors.
.venv/bin/python research/search.py build

# Validate paper hashes and keyword/vector/hybrid retrieval with source ranges.
.venv/bin/python -u research/verify_library.py
```

- `hybrid` is the default mode; use `fts` for keywords/identifiers and `vector` for conceptual similarity. `--limit` controls returned passages. Do not assume the highest-ranked hit is authoritative.
- Results identify the source path, line range, kind, and paper URL where available. Open the cited source lines before summarizing; check `research/sources/<slug>/paper.pdf` for figures, equations, and tables. Authored notes and the original conversation are not primary paper evidence.
- Search checks content hashes and omits stale hits. Run `build` when `status` reports changed/new/deleted files, including after editing this document. Avoid `build --rebuild` unless deliberately regenerating every embedding.
- The local database is `research/.lancedb/`; the CPU embedding cache is `research/.models/`. Normal queries use cached model files and no external embedding service. `build`/`doctor` can download model files if missing. Do not change the system CUDA stack to operate search.
- Indexed content includes repository source/config/scripts/Markdown and `research/papers/*.md`. It excludes other PufferLib clones, `vendor/`, `resources/`, raw PDFs/HTML, binaries, and files over 2 MB. Use `rg` or inspect those files directly when relevant.
- If a managed read-only sandbox blocks cache access or stalls a search subprocess, request tool escalation for this narrowly scoped local command. Do not work around it by modifying global packages or system settings.
- Verified query: `impoola benchmarks` finds the Impoola paper's Procgen evaluation, Figure 5 comparisons, normalization constants, and hyperparameter tables. Full setup and conversion instructions are in `research/README.md`.
