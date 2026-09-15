# Current RTX 5090 task: appearance reproducibility and evaluator readiness

September 15, 2026. Owner: Kinvert. Work in `~/Git/ml/cnn-5090`; start with `START_HERE_5090.md` for existing-toolchain/venv setup. This supersedes the old hardware/Pong launch instructions. Those campaigns are finished and their unfavorable/missing results must remain intact.

## Update safely and retain the existing machine's work

Only use `Kinvert/PufferLib`, branch `cnn-research`; never push to official PufferLib. Fetch and inspect local status/history first. The 5090 previously ran local revision `1d9e9e0dca88706c7a8495644c8dc584a4446f11`, which is newer than G240's historical launch base and must not be discarded with a reset. If clean and behind, fast-forward. If diverged, reconcile/review the local agent's commits in an isolated worktree; do not overwrite local binaries, checkpoints, runtime helper or reports. A pull is not proof of the right version: verify these files exist:

```bash
git status --short
git fetch origin cnn-research
git log --oneline --left-right HEAD...origin/cnn-research
git diff --stat HEAD...origin/cnn-research
git rev-parse HEAD
test -f ocean/connect4cnn/appearance.h
test -f ocean/connect4cnn/appearance_canary.sh
```

The final two checks are after updating. Record the actual revision and working diff used. If the new files are not yet on origin, report that delivery is pending; do not substitute an old full-run command. G240's received archive sources are evidence, not patches to apply blindly.

## Existing dependencies and bounded validation

Reuse the existing local Python 3.12 `.venv` and the working 5090 runtime helper. For a new checkout, use `uv venv --python 3.12 .venv` and the small reporting dependency setup in `START_HERE_5090.md`. Do not overwrite an existing venv or install Torch/system CUDA. The canary needs NumPy only for reporting; native construction/training is C/CUDA. No Docker, driver/cuDNN changes or global shell changes.

```bash
source build/connect4cnn/runtime-5090.sh
export NVCC_ARCH=sm_120
export OPENBLAS_NUM_THREADS=1
bash ocean/connect4cnn/appearance_canary.sh --prepare-only
bash ocean/connect4cnn/appearance_canary.sh
```

If the existing helper is absent, set process-local `CUDA_HOME`/`NCCL_ROOT` to this host's existing installations and source `ocean/connect4cnn/runtime_env.sh`. Run builds/GPU commands outside the sandbox. The canary refuses competing GPU compute processes; do not stop other jobs. The first command prepares isolated configs only. The second creates a different fresh run directory, runs CPU sanitizer/parity checks, builds all four encoders for both games, and performs the bounded tests. Do not actively monitor; inspect once finished or on failure.

Expected successful output: `build/connect4cnn/appearance.<id>/status.txt` says `ok`; `REPORT.md` and `results.csv` contain **16 short training runs and 16 evaluations**. All four encoders on both games receive 16,384 decisions with two checkpoints. The two repetitions change only CPU worker count (one/two) and output paths; their checkpoint bytes and native evaluation records must match. Weight count/finiteness, unchanged checkpoint hashes after reload, and source checks must pass. A timeout is a failure, never a zero score. Retain its logs and do not retry based on scores. Training and evaluation have 60s/30s process limits with a forced-kill grace; each build has a 300s limit.

This exact canary passed on G240: [receipt](research/results/connect4cnn/appearance.xRxsamIZ/REPORT.md). It proves same-host reproducibility for the tested panel, not cross-GPU bitwise equality, useful learning, steady-state SPS or score superiority. All encoder math is unchanged; existing float32 forward/gradient tests remain applicable. Re-run the relevant numerical harness if reconciling the 5090 branch changes encoder math.

## Appearance settings and fairness

```ini
[env]
representation = 0
representation_mode = 1
representation_seed = 12345
```

Mode 0 means one fixed appearance ID. Mode 1 deterministically chooses an ID for each native slot at initialization and keeps it for the slot's lifetime, including match resets. Connect4 has ten IDs; Pong has five. Appearance uses an explicit independent seed, not `base.seed`, wall time, process ordering or the game RNG. Native game RNG initialization remains the existing slot-based stream; changing `base.seed` changes model/action randomness but does not create new environment initial RNG streams. Record this limitation when defining evaluation seeds.

Keep representation mode/seed, total slots and their ordering identical across all models in a panel. The canary records the 64-slot assignment CSVs and saved configs. Seeded assignments need not balance counts. Do not choose a different easy appearance per model. Evaluate mixed-trained weights separately on every fixed ID to measure per-appearance robustness; pooled mixed performance can hide failures. Fixed-only training, mixed training and unseen-appearance transfer are separate experiments. See [Connect4 presets](ocean/connect4cnn/REPRESENTATIONS.md) and [Pong presets](ocean/pongcnn/REPRESENTATIONS.md).

## Next implementation gate: exact episode evaluation

**Do not launch the long full-frontier confirmation yet.** The mixed-appearance path is tested, but the current evaluator is pooled v1. Pong's original eight timeouts remain unresolved; high simulator SPS plus few completed matches is not a hung GPU. New appearance options do not fix evaluation sampling. Read [the audit](research/PONG_EVALUATION_AUDIT_RESULTS.md) and [full-frontier protocol](research/CONNECT4_CLAIM_PROTOCOL.md).

The useful next engineering task is a separately versioned native evaluator using saved checkpoints:

1. Assign unique episode IDs, explicit environment/policy seed mapping and exact per-slot quotas. Stop collecting from finished slots. Verify recurrent resets and inactive-slot isolation; don't assume the existing action RNG automatically reseeds per episode. Retain per-episode representation ID and completed/missing/truncated status. Preserve old v1 records unchanged.
2. Connect4 should terminate within 21 decisions from the empty board. Treat an over-bound match as a correctness failure; never convert it into a draw. Pong needs a whole-match decision counter, since `tick` currently resets every point.
3. Calibrate a Pong per-match decision cap using existing development checkpoints and duration-only information under a recorded resource ceiling; freeze it before comparative quality evaluation. At the cap, retain unknown completion with bounds, not a fabricated win/loss/point fraction. Distinguish episode truncation from an infrastructure/process timeout. Do not change Pong physics, rewards, max score or action semantics to make it finish.
4. Test exact quotas/unique IDs, cap versus terminal precedence, all-complete/all-truncated/mixed cases, reproducible resets/action streams and checkpoint immutability with small fixtures. Then use the original saved successful and timed-out checkpoint panel for bounded validation, preserving all attempts. Do not retrain models to test an evaluator.
5. Record completed episode count, truncation fraction, score definition/bounds, duration and versioned source/config/checkpoint hashes. Plain Pong `perf` is point fraction, not match win rate. A complete-case mean alone cannot resolve censored outcomes.

Keep changes small and native; use the existing environment/custom-encoder interfaces. Training and unrelated environments should keep their prior behavior when new evaluation mode is disabled. Coordinate/reconcile this evaluator work with G240 before editing shared core paths in both checkouts.

## Final experiment and reporting boundaries

The planned primary Connect4 comparison uses fixed representation 0, one identical learner recipe for all four encoders, 13.312M decisions each and the whole observed frontier. Mixed appearances are a separate robustness panel; don't silently replace that primary comparison. Checkpoint publication receipts, exact allocation, baseline fairness, simultaneous frontier analysis and a justified frozen seed count remain prelaunch gates. Forty seeds is only a resource estimate, not a fixed requirement.

For the canary, return revision, host/compiler/driver, run directory, pass/failure counts, paired byte-equality results and the compact table. Keep large weights/binaries on the 5090 and retain small source/config/log/hash receipts. No new SCP step or transfer is required to complete this task. No long sweep, automatic campaign rerun or SOTA claim follows from passing validation.
