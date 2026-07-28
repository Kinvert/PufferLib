# Phase 7 Self-Play Runtime Status

Date: 2026-07-28

Branch: `dogfight5c`

Audited upstream base:
`ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`

## Current result

Dogfight keeps both self-play layers:

- `selfplay.mode=native` uses PufferLib's official in-process FIFO,
  checkpoint insertion, uniform opponent sampling, and synchronous rotations.
- `selfplay.mode=coordinator` keeps the Dogfight persistent manifest, PFSP/Elo
  evidence, fixed anchors, both-seat gates, and atomic promotion. It selects an
  external checkpoint and pins it for one native training segment.
- `selfplay.mode=off` disables training-only frozen banks.

The coordinator was not deleted or replaced. It remains in
`ocean/dogfight/selfplay_coordinator.py`.

Configs without `selfplay.mode` retain upstream compatibility:
`selfplay.enabled=1` resolves to native and `enabled=0` resolves to off.
This is how the pinned Robocode and Chess INIs continue to select official
PufferLib self-play.

These commands are dated evidence from this branch, not permanent API
documentation. Future agents must try them, inspect the current CLI/config
contract, and adjust them if 5c has changed. Do not take them as gospel.

## TDD release gate

Observed working command:

```bash
cd /home/claude/5c-research
source /home/claude/PufferLib/.venv/bin/activate
PYTHONPATH=. pytest -q \
  ocean/dogfight/tests \
  tests/test_native_selfplay_contract.py \
  tests/test_native_selfplay_mode_contract.py \
  tests/test_native_selfplay_seed_contract.py \
  tests/test_native_simple_selfplay_contract.py \
  tests/test_selfplay.py
```

Observed result:

```text
63 passed, 3 skipped in 29.92s
```

The new mode contract was written and observed red before implementation:
four tests failed. It then passed after the runtime and INI changes.

## Build

Observed working command:

```bash
cd /home/claude/5c-research
source /home/claude/PufferLib/.venv/bin/activate
./build.sh dogfight
```

Observed result: `Built: ./puffer`.

The build retained existing narrowing warnings in
`ocean/dogfight/dogfight_render.h`; it had no compile or link failure.

## Official native self-play smoke

Observed working command:

```bash
cd /home/claude/5c-research
source /home/claude/PufferLib/.venv/bin/activate
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib \
  ./puffer train dogfight \
  base.run_id=phase7_native_mode_smoke_20260728_a \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  train.total_timesteps=1048576 \
  base.checkpoint_interval=1 \
  selfplay.opp_timeout_steps=262144
```

Observed evidence:

- `selfplay/mode=native`
- bank 0 seeded from the current checkpoint with `external=0`
- four synchronous swap events, generations 1 through 4
- final dashboard throughput about `624.5K SPS`
- no NaN/Inf
- final checkpoint:
  `checkpoints/dogfight/phase7_native_mode_smoke_20260728_a/0000000001048576.bin`

This is a smoke artifact, not a promoted league candidate.

## Human-visible evaluation

Observed working command:

```bash
cd /home/claude/5c-research
source /home/claude/PufferLib/.venv/bin/activate
DISPLAY=:0 \
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib \
  timeout 25s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase7_native_mode_smoke_20260728_a/0000000001048576.bin \
  env.curriculum_enabled=0 \
  env.fixed_stage=0
```

Observed result:

- the checkpoint loaded;
- raylib created a 1280x720 window on `DISPLAY=:0`;
- the user reported: `looked good`;
- exit status 124 is expected because `timeout 25s` closes the interactive
  renderer.

## Coordinator-mode smoke

Observed working command:

```bash
cd /home/claude/5c-research
source /home/claude/PufferLib/.venv/bin/activate
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib \
  ./puffer train dogfight \
  selfplay.mode=coordinator \
  base.run_id=phase7_coordinator_mode_smoke_20260728_a \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.load_enemy_model_path=checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
  train.total_timesteps=262144 \
  base.checkpoint_interval=1 \
  selfplay.eval_games=0
```

Observed evidence:

- `selfplay/mode=coordinator`
- bank 0 seeded from the Phase 3 anchor with `external=1`
- no native swap event; coordinator mode forced
  `selfplay.opp_timeout_steps=0`
- final dashboard throughput about `548.0K SPS`
- no NaN/Inf

Coordinator mode requires `base.load_enemy_model_path` and
`selfplay.eval_games=0`. The external coordinator owns opponent selection,
both-seat evaluation, evidence collection, and promotion between bounded
training segments.

## Native reference behavior

The concise Robocode/Chess comparison and mode contract are in
`NEXT_AGENT_HANDOFF.md` under `Runtime self-play choice`.

At the audited revision:

- Robocode uses PufferLib's native two-slot routing, a 10% frozen fraction, a
  memory-only size-100 uniform FIFO, a 100M-step timeout, and optional final
  pool evaluation.
- Chess uses the same native FIFO layer with randomized colors, a 10% frozen
  fraction, a size-500 pool, and a 4B-step timeout.
- Neither environment supplies the coordinator's persistent manifest, PFSP,
  Elo, fixed-anchor, both-seat, or atomic-promotion guarantees.
