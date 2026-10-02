# RTX 5090 mixed-appearance validation — completed

## Result

**PASS: 16/16 training runs, 16/16 saved-checkpoint evaluations, no unexpected
failures or timeouts.** End-to-end canary time, including CPU tests and eight
builds, was **54.925 seconds**. Prepare-only produced a separate untouched setup.

Measured revision: `c49ad33e9ab899b8f08198d4a154f532cfebe1d2`, fetched from
Kinvert/PufferLib `cnn-research`. Hardware: RTX 5090, Ryzen 9 9950X3D, CUDA compiler
13.1.115, driver 580.105.08, float32, explicit `sm_120`. Existing Python 3.12.3 /
NumPy 2.5.3, CUDA/NCCL, Raylib and compiler cache were reused. No installations or
system changes were made.

The two histories diverged. Validation used a detached, isolated worktree:
`/home/keith/Git/ml/cnn-5090/build/appearance-validation-c49ad33e`.
There were no tracked source modifications in this worktree. The only untracked
dependency entry was the symlink to the original shared Raylib directory.
The main checkout remains at local commit
`1d9e9e0dca88706c7a8495644c8dc584a4446f11`; it was not reset or merged.
Its existing tracked diff and untracked file hashes were verified unchanged
after the canary. Original runtime helper, checkpoints and experiment results
remain in place. Use the isolated worktree for this new revision until the
branches are deliberately reconciled.

## Compact validation table

Each row comprises two training/evaluation runs, changing CPU workers from one
to two. Each training run received 16,384 decisions and wrote checkpoints at
8,192 and 16,384 decisions. Each pair's two checkpoint files matched byte-for-byte;
the final native evaluation records also matched exactly.

| Game | Model | Parameters | Train/eval runs passed | Paired checkpoints equal | Evaluation games, requested 64 |
|---|---|---:|---|---|---:|
| Connect4CNN | Ours quality | 160,736 | 2 / 2 | 2 / 2 | 290 |
| Connect4CNN | Nature | 138,528 | 2 / 2 | 2 / 2 | 297 |
| Connect4CNN | IMPALA | 270,496 | 2 / 2 | 2 / 2 | 307 |
| Connect4CNN | Impoola | 151,712 | 2 / 2 | 2 / 2 | 304 |
| PongCNN | Ours quality | 160,224 | 2 / 2 | 2 / 2 | 65 |
| PongCNN | Nature | 138,016 | 2 / 2 | 2 / 2 | 65 |
| PongCNN | IMPALA | 269,984 | 2 / 2 | 2 / 2 | 66 |
| PongCNN | Impoola | 151,200 | 2 / 2 | 2 / 2 | 64 |

All **32 checkpoint files** were finite and had expected parameter counts. All
**16 checkpoint-pair comparisons** and **eight evaluation-pair comparisons**
passed. Hash checks after evaluation verified checkpoint immutability. An
independent audit compared resolved native INIs: paired differences were exactly
worker count and checkpoint/log output paths. Appearance mode 1, seed 12345 and
64 slots were preserved. Training seed was 56173 and evaluation action seed 66173.
The two assignment CSVs matched G240's recorded deterministic mappings; this is
not a claim of cross-GPU checkpoint identity.

CPU checks passed: Connect4 all ten representations, mixed seeds, pixel/reset
fixtures, 4,096-transition native parity per panel, invalid settings, ASan/UBSan;
Pong all five representations, mixed seeds, 49,152-transition parity per panel,
pixel/reset/reward fixtures, invalid settings, ASan/UBSan. Three GPU-discovery and
query-failure regression tests also passed. The core math and four encoder source
hashes match the previous 5090 numerical audit; no encoder math was changed.

These are reproducibility/plumbing checks. They do not establish learning,
steady-state throughput, cross-device bitwise equality or appearance robustness.
The raw short-run scores are retained, including zero evaluated performance for
all Connect4 canaries and all Pong models except Impoola's 0.071% point fraction.

## Exact-episode evaluator assessment and next work

**The current evaluator still pools completed matches and overshoots its target.**
This canary directly demonstrates it: Connect4 requested 64 but returned
290–307 games after a rollout. A check only at the outer evaluation-loop boundary
cannot enforce per-slot quotas. The appearance feature does not fix this.

The next implementation should be an opt-in native v2 evaluator with per-decision
quota enforcement, unique episode IDs, explicit game/policy seed mappings,
inactive-slot isolation, correct recurrent resets, and episode records captured
before automatic reset. Game RNG currently starts from native slot IDs;
`base.seed` alone is not independent per-episode environment seeding. Preserve
the independently assigned appearance across game reseeding/resets.

Connect4 exceeding 21 decisions must be a correctness error. Pong needs a separate
whole-match decision counter and a duration-only cap calibration on saved
development checkpoints before the cap is frozen. Preserve terminal/cap precedence,
truncation versus process-timeout distinctions and bounds for unknown outcomes.
Validate with small fixtures and saved successful/timed-out checkpoints; do not
retrain merely to test evaluation. Coordinate one owner of shared core changes
with G240 before implementation in both checkouts.

The detailed source assessment and acceptance plan are in
[EVALUATOR_READINESS.md](results/connect4cnn/appearance.LA89uk8w/EVALUATOR_READINESS.md).
No evaluator implementation or long campaign was started. Original Pong
long-match/truncation limitations remain unresolved. Full-frontier confirmation
also still needs checkpoint-publication timing, cadence/overhead validation,
simultaneous inference calibration and a justified frozen seed count.

## Evidence and reproduction

- [Native report with all 16 rows](results/connect4cnn/appearance.LA89uk8w/REPORT.md)
- [Independent paired/configuration audit](results/connect4cnn/appearance.LA89uk8w/independent-audit.json)
- [CSV results](results/connect4cnn/appearance.LA89uk8w/results.csv)
- [Source identities](results/connect4cnn/appearance.LA89uk8w/source.sha256)
- [Reconciliation and dependency receipts](results/connect4cnn/appearance.LA89uk8w/validation-5090/)

Full local run (including executable binaries and checkpoints):
`build/appearance-validation-c49ad33e/build/connect4cnn/appearance.LA89uk8w/`.
Prepare-only run: `appearance.nDTeijT4` in the same directory.
Small receipts are archived under `research/results/connect4cnn/appearance.LA89uk8w/`.

Executed from the isolated checkout:

```bash
source build/connect4cnn/runtime-5090.sh
export NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1
bash ocean/connect4cnn/appearance_canary.sh --prepare-only
bash ocean/connect4cnn/appearance_canary.sh
```

These are reproduction commands, not instructions to repeat this completed
canary automatically. No SCP, push, architecture search or historical campaign
rerun was performed.
