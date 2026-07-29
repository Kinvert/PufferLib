# Dogfight Native Sweep: 5090 Handoff

## Intent

Use the current Dogfight experimental branch on the 5090 machine for the
native Protein screening sweep. The g240/5060 machine is the build, test, and
canary reference only; do not launch `max-runs=1000` there.

The branch must remain based on PufferLib commit `179325e9`, contain no
PufferLib core changes, be clean, and be pushed to the user's PufferLib fork
before this runbook is used.

## Clone and Environment

```bash
git clone <YOUR_PUFFERLIB_FORK_URL> PufferLib-dogfight5c
cd PufferLib-dogfight5c
git checkout dogfight5c-robocode-stage10
uv sync
source .venv/bin/activate
```

The launcher prefers the clone-local `.venv`. `PUFFER_VENV` is only needed when
the environment lives elsewhere.

Authenticate W&B before the unattended launch and ensure project `df42` is
writable. Do not put an API key in the repository.

## TDD and Build Gate

```bash
python -m pytest -q ocean/dogfight/tests

CUDA_HOME=/usr/local/cuda ./build.sh dogfight

CUDA_HOME=/usr/local/cuda \
  bash ocean/dogfight/build_eval.sh
```

Both `./puffer` and `build/puffer-dogfight-eval` must exist. The complete tests
must pass before training.

The verified g240 reference gate was:

```text
Dogfight tests: 85 passed, 3 skipped
trainer build:   succeeded
eval build:      succeeded
```

## GPU and Command Preview

```bash
nvidia-smi

bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --dry-run --max-runs 1
```

The preview must contain native 5c syntax:

```text
sweep.max_runs=1
```

It must not contain the obsolete spaced native form `--max-runs 1`.

## 5090 Structural Canary

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --max-runs 1 --timesteps 8388608
```

Wait for the generated experiment manifest to report:

```json
{
  "status": "completed",
  "exit_code": 0,
  "uploaded_runs": 1
}
```

Confirm that one trial appears in W&B project `df42`, its history uses
`agent_steps`, and the experiment contains a resolved native INI plus initial
and final checkpoints.

The reference no-name canary was
`df42-native-sweep-20260729-102730`: native exit code 0, 8,388,608 steps, and
one W&B upload. Its post-sweep fixed evaluator completed 6 cells, rejected the
candidate, reported no evaluation failures, and left the experiment status
`completed`. Stage 0 had minimum perf `0.046875`, mean perf `0.0663052`, and
maximum absolute signed bias `42.275864`.

Identical repeated canaries produced identical fixed metrics and checkpoint
SHA-256:

```text
fbf5dc6cfd8477382596c315797c7848465c4da3ccc018468802d74164d0b868
```

The current logging payload is also within the native limit: 30 explicit
Dogfight values plus PufferLib's core `n` value is 31, below 32.

## 5090 Calibration Gate

No explicit tmux session or experiment ID is needed:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh --max-runs 16
```

The launcher generates local names automatically. Those names do not define
W&B metric keys or chart axes.

Run 16 trials first and expand to at most 32 if needed for a useful sample.
Apply the post-sweep fixed evaluator to those candidates, then measure whether
native Protein score ranks candidates consistently with fixed-stage results.
Do not infer this correlation from the one-run structural canary.

Resume or reattach to an existing experiment without reconstructing its
arguments:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --screen-existing EXPERIMENT_DIR
```

## Full-Cap Screening Sweep

Launch the full cap only after the 5090 calibration demonstrates useful
correlation:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh --max-runs 1000
```

The committed screening profile uses:

- `671,088,640` steps per trial.
- The full `134,217,728`-step native acquisition school.
- Progressive spawn completion swept from `402,653,184` to `805,306,368`.
- One official frozen bank with historical fraction swept from `0.05` to
  `0.20`.
- Frontier fraction swept from `0.45` to `0.80`.
- Checkpoint interval `384`, approximately one snapshot per 100M agent steps.

At the measured 5060 throughput, the cap is approximately three to four
single-GPU days. Re-measure on the 5090 rather than assuming the same duration.

## Evidence and Promotion

Each generated experiment directory contains:

- `manifest.json`: source revision, input hashes, exact native command, and
  lifecycle status.
- `config/default.ini` and `config/dogfight.ini`: exact staged native inputs.
- `logs/dogfight/*.ini`: resolved per-trial configs and metrics.
- `checkpoints/dogfight/<run_id>/*.bin`: native pool/final checkpoints.
- `wandb_uploaded.json`: idempotent upload record.
- `sweep.log`: native Protein results and failures.

Training `env/perf` ranks screening candidates only. It is not mastery.
Promote candidates by training the full 12.08B-step target configuration from
random initialization and then applying the fixed-seed, mirrored stage 0-10
matrix. Every stage must reach at least 90%; timeout-derived wins do not count.
