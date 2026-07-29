# Dogfight Test System

This directory starts red by design. The imported C regression fixtures define
the contracts that the Phase 1 native scaffold must satisfy; they require
`ocean/dogfight/dogfight.h`, which has not been ported yet.

## Provenance

The C tests are copied unchanged from:

```text
/home/claude/dogfight5
10da7b5abac0f67f69f302c85511baf99a22446c
```

Their behavior authority is Dogfight 3 at
`5a3effce470ce19fd6c4c6385a515853a4bfc0cb`. Three donor
source-compatibility tests explicitly compare against
`171482a9b9889bebaa427ab658c95c0fc391c031`, but the embedded numerical C
fixtures do not record their generating oracle SHA. See
`FIXTURE_PROVENANCE.json`; unpinned numerical fixtures are quarantined rather
than silently attributed to that older revision.

Dogfight 4 revision `21281f623abeb44c525b070cdd20b17519470195`
supplies test scenarios only. Its thresholds and physics values are not
oracles.

## Test tiers

### Every edit

- fixed-state opponent-aware observations;
- one-step and 12-step physics/reward traces;
- action bounds and control polarity;
- terminal rewards and immediate-reset alignment;
- repeated state/trajectory determinism;
- finite observations and aircraft state.

### Before training

- all every-edit tests;
- curriculum spawn geometry and episode caps;
- long integration, oscillation, smoothness, speed, climb, glide and turn
  envelopes;
- energy ordering and G limits;
- autopilot recovery from bank, pitch, inverted, dive, speed and altitude
  disturbances.

### Major physics or controller changes

- combat recovery matrix;
- full observation edge/crossover suite;
- AutoAce tactical behavior and firing discipline;
- visible sustained-turn and recovery evaluation through `DISPLAY=:0`.

The long-horizon flight scenarios will be adapted from Dogfight 4 only after
Dogfight 3 supplies approved expectations. Tests that merely print `FAIL` must
be converted to nonzero hard failures before becoming gates.

## Candidate commands

These commands came from the older Dogfight repositories. Future agents must
try them against the current branch, bound expensive attempts, inspect current
5c CLI/config behavior, and record the command that actually works. They are
starting points, not gospel.

Dogfight 5 test command:

```bash
.venv/bin/python -m pytest ocean/dogfight/tests -q
```

Dogfight 5 bounded training canary:

```bash
python -m pufferlib.pufferl train dogfight --train.gpus 1 --train.total-timesteps 5000000
```

Dogfight 5 bounded sweep:

```bash
timeout 300s bash -lc 'source .venv/bin/activate && python -m pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1 --sweep.max-runs 2'
```

Current 5c checkpoint-loaded display candidate, synthesized from the current
Nethack eval form and Dogfight 4 display setup:

```bash
DISPLAY=:0 timeout 20s ./puffer eval dogfight --load-model-path=latest
```

An explicit checkpoint path is preferred over `latest` for recorded gates.

## Verified stock-core TDD loop (2026-07-28)

The commands below were executed successfully in `/home/claude/5c-research`.
They are evidence for the pinned 5c checkout, not promises about future
PufferLib CLI compatibility. Future agents must rerun the tests and use each
script's `--dry-run` output before starting an expensive job.

Activate the established environment:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
```

Build and test:

```bash
./build.sh dogfight
bash ocean/dogfight/build_eval.sh
python -m pytest ocean/dogfight/tests -q
```

Verified result:

```text
46 passed, 3 skipped
```

Print or run the proven stock-core Phase 2 profile:

```bash
bash ocean/dogfight/train_reproduction.sh --dry-run RUN_ID 1048576 0
bash ocean/dogfight/train_reproduction.sh RUN_ID 1048576 0
```

`FIXED_STAGE` defaults to `0`; pass `-1` only when intentionally testing the
environment-managed curriculum. The script's `ent_coef=0.02` reproduces the
known profile and is not yet an accepted long-run production value.

Run matched headless and human-visible evaluation:

```bash
bash ocean/dogfight/eval_checkpoint.sh headless CHECKPOINT 2 64 42
DISPLAY=:0 bash ocean/dogfight/eval_checkpoint.sh visible CHECKPOINT 2 64 42
```

Both modes deliberately use:

- two evaluation environments and two vector buffers;
- one trainable-policy slot against the scripted opponent;
- the same explicit seed, stage, and episode target;
- disabled self-play and frozen banks;
- disabled role, domain, vertical, and rehearsal randomization.

The visible command terminates after the requested completed-episode count.
It is therefore a numerical evaluation with human observation, not an
unbounded render session.

Full evidence and the explanation of the former false stage-2 result are in
`ocean/dogfight/EVAL_CONTRACT_EVIDENCE_20260728.md`.
