# Phase 6 Status

Status: COMPLETE

This file records evidence from 2026-07-28. Commands are dated working
examples, not permanent CLI gospel. Future agents must try them against the
current checkout and adjust them if PufferLib changes.

## Scope

Phase 6 follows `NEXT_AGENT_HANDOFF.md`'s "Simple native self-play" scope:

- one frozen bank;
- a modest historical-battle fraction;
- a uniform FIFO checkpoint pool;
- current-vs-current and current-vs-history metrics;
- checkpoint generation logging;
- a stable one-GPU learning baseline.

Persistent manifests, PFSP/top-N selection, promotion gates, and native anchor
league management remain Phase 7 work.

## Implementation

Native training now aggregates and emits separate cohort metrics before the
main environment log is cleared:

```text
selfplay/current_vs_current/*
selfplay/current_vs_history/*
```

The console evidence line includes cohort episode counts, slot scores, and draw
rates. Pool metrics also record:

```text
pool/current_vs_current_battles
pool/current_vs_history_battles
pool/current_trainable_rows
pool/frozen_rows
pool/generation_0
pool/swap_truncations
```

At `vec.total_agents=1024` and `vec.frozen_bank_pct=0.10`, the pinned allocator
creates 512 two-agent battles: 462 current-vs-current battles and 50
current-vs-history battles. Therefore 974 rows are current/trainable and 50
rows are frozen/excluded.

## Build and tests

Working build:

```bash
source /home/claude/PufferLib/.venv/bin/activate
./build.sh dogfight
```

Result: PASS.

Working affected suite:

```bash
/home/claude/PufferLib/.venv/bin/pytest -q \
  ocean/dogfight/tests \
  tests/test_native_simple_selfplay_contract.py \
  tests/test_native_frozen_exclusion_contract.py \
  tests/test_native_selfplay_contract.py \
  tests/test_league.py
```

Result: `49 passed, 3 skipped`.

## Accepted one-GPU FIFO baseline

Working command:

```bash
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
timeout 300s ./puffer train dogfight \
  base.run_id=phase6_fifo_baseline_16m_20260728_a \
  base.seed=42 \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.async=0 \
  base.cudagraphs=-1 \
  base.checkpoint_interval=32 \
  vec.total_agents=1024 \
  vec.num_buffers=2 \
  vec.num_threads=8 \
  vec.num_frozen_banks=1 \
  vec.frozen_bank_pct=0.10 \
  vec.frozen_bank_hidden_size=64 \
  vec.frozen_bank_num_layers=3 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  train.total_timesteps=16777216 \
  selfplay.enabled=1 \
  selfplay.seed=42 \
  selfplay.max_size=8 \
  selfplay.eval_games=0 \
  selfplay.opp_timeout_steps=2097152 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1
```

Result:

```text
training steps:                  16,777,216
metric windows:                 64
current-vs-current episodes:    37,984
current slot 0 score:           0.5033
current slot 1 score:           0.4967
current draw rate:              0.0913
current-vs-history episodes:    4,168
learner slot 0 score:           0.5129
history slot 1 score:           0.4871
history draw rate:              0.0835
rotations:                      8
swap truncations:               400
non-finite scan:                none
final clean-fight rate:         0.820
final mastered stage:           0
```

The current-vs-current aggregate is side-balanced. Historical battles train
current slot 0 only; frozen slot 1 rows remain outside priority construction
and every learner loss.

Artifact:

```text
checkpoints/dogfight/phase6_fifo_baseline_16m_20260728_a/0000000016777216.bin
SHA-256 c2458b0c5fb05ad471761e24a544e6ba6af3511ffe721fbd2cea075412fb93bb
```

## Fixed-snapshot matches

Native match shape:

```bash
./puffer match dogfight \
  base.load_model_path=PATH_TO_CANDIDATE \
  base.load_enemy_model_path=PATH_TO_FIXED_SNAPSHOT \
  base.num_games=16384 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=STAGE
```

Candidate against its fixed Phase 4 parent:

```text
stage 0: learner=0.521, parent=0.479, draw=0.088
stage 1: learner=0.507, parent=0.493, draw=0.102
stage 3: learner=0.519, parent=0.481, draw=0.086
```

Candidate against the immutable Phase 3 anchor:

```text
stage 0: learner=0.522, anchor=0.478, draw=0.087
stage 1: learner=0.508, anchor=0.492, draw=0.103
stage 3: learner=0.519, anchor=0.481, draw=0.086
```

Parent self-match at stage 0 was `0.497/0.503`; parent self-match at stage 3
was `0.503/0.497`. The candidate gains are therefore not explained by the
observed seat bias.

## Numerical and human-visible evaluation

Working numerical command:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase6_fifo_baseline_16m_20260728_a/0000000016777216.bin \
  base.num_games=4096 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Result: `perf=0.9308`, `score=0.836`. The Phase 4 accepted comparison was
`perf=0.8980`, `score=0.778`.

Working visible command:

```bash
DISPLAY=:0 timeout 30s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase6_fifo_baseline_16m_20260728_a/0000000016777216.bin \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Mechanical result: PASS; timeout exit status `124` was expected.

Human result: PASS. The user reported that the flight "looks fine."

## Phase 6 release decision

PASS. The accepted baseline is stable, side-balanced, free of observed
non-finite values, improves against fixed snapshots, advances the curriculum,
and preserves visible flight quality.
