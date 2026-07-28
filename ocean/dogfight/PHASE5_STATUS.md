# Phase 5 Status

Status: IN PROGRESS

This file records dated evidence, not permanent command-line gospel. Future
agents must try these commands against the current checkout and adjust them if
PufferLib changes.

## 2026-07-28: native self-play preflight slice

Implemented generic native safeguards in `src/pufferl.cu`:

- reject native self-play when `base.async != 0`;
- reject mismatched primary/frozen hidden size or layer count before
  `create_pufferl`;
- preserve normal non-self-play training behavior.

Exact frozen-row exclusion and atomic opponent rotation are not implemented
yet. Native self-play must not be trusted or used for a release run until those
remaining Phase 5 contracts pass.

## Build

Working command:

```bash
source /home/claude/PufferLib/.venv/bin/activate
./build.sh dogfight
```

Result: PASS. The build produced `./puffer`.

## Tests

Working affected-suite command:

```bash
/home/claude/PufferLib/.venv/bin/pytest -q \
  ocean/dogfight/tests \
  tests/test_native_selfplay_contract.py \
  tests/test_league.py
```

Result: `39 passed, 3 skipped`.

Repository-wide `pytest -q` did not collect because this environment lacks
unrelated optional dependencies including JAX, HeavyBall, pandas, Cython, and
the point-linear-max reference module. This was not treated as a Dogfight
failure.

## Warm-start smoke training

The native CLI accepts overrides as `section.key=value`. Do not assume the
Python CLI's `--key value` spelling applies to `./puffer`.

Working command:

```bash
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
timeout 180s ./puffer train dogfight \
  base.run_id=phase5_preflight_20260728_c \
  base.seed=42 \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.async=0 \
  base.cudagraphs=-1 \
  base.checkpoint_interval=1 \
  vec.total_agents=1024 \
  vec.num_buffers=2 \
  vec.num_threads=8 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  train.total_timesteps=262144 \
  selfplay.enabled=0 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1
```

Result: PASS, four epochs and 262,144 steps.

Artifact:

```text
checkpoints/dogfight/phase5_preflight_20260728_c/0000000000262144.bin
SHA-256 a605930ed6a6b0b0e90e252b4a0f27485c4a411158b81c000faf0930e987a30f
```

## Numerical evaluation

Working command:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase5_preflight_20260728_c/0000000000262144.bin \
  base.num_games=4096 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Result: `perf=0.8958`, `score=0.774`. The accepted Phase 4 comparison was
`perf=0.8980`, `score=0.778`, so no meaningful regression was observed.

## Human-visible evaluation

Working command:

```bash
DISPLAY=:0 timeout 30s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase5_preflight_20260728_c/0000000000262144.bin \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Mechanical result: PASS; the expected timeout exit status was `124`.

Human result: PASS. The user reported that the flight "looks ok."
