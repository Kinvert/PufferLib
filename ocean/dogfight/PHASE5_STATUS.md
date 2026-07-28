# Phase 5 Status

Status: COMPLETE

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

## 2026-07-28: exact frozen-row exclusion

Implemented:

- priority buffers contain only the explicit number of primary-policy rows;
- grouped priority reduction reads the primary prefix within every vector
  buffer and never reads a frozen trajectory;
- sampled compact indices are mapped to physical primary rows only after
  importance weights are computed;
- a trainable-row mask gates policy, value, entropy, KL, clip-fraction, and
  total-loss terms as defense in depth;
- Dogfight slot 1 now advertises policy bank 1, matching the vector routing
  contract already used by Robocode.

The focused contract makes one million total weighted draws across
`prio_alpha=0` and `prio_alpha=1`. No draw maps to a frozen row, and arbitrary
changes to frozen advantages do not change the primary priority domain.

Affected-suite command:

```bash
/home/claude/PufferLib/.venv/bin/pytest -q \
  ocean/dogfight/tests \
  tests/test_native_frozen_exclusion_contract.py \
  tests/test_native_selfplay_contract.py \
  tests/test_league.py
```

Result: `43 passed, 3 skipped`.

The non-pool regression run used the preceding warm-start command with
`base.run_id=phase5_frozen_exclusion_nonpool_20260728_a`. Its final checkpoint
was byte-identical to the pre-change Phase 5 checkpoint:

```text
SHA-256 a605930ed6a6b0b0e90e252b4a0f27485c4a411158b81c000faf0930e987a30f
```

Controlled one-bank smoke command:

```bash
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
timeout 180s ./puffer train dogfight \
  base.run_id=phase5_frozen_exclusion_pool_20260728_b \
  base.seed=42 \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.async=0 \
  base.cudagraphs=-1 \
  base.checkpoint_interval=1 \
  vec.total_agents=1024 \
  vec.num_buffers=2 \
  vec.num_threads=8 \
  vec.num_frozen_banks=1 \
  vec.frozen_bank_pct=0.5 \
  vec.frozen_bank_hidden_size=64 \
  vec.frozen_bank_num_layers=3 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  train.total_timesteps=262144 \
  selfplay.enabled=1 \
  selfplay.eval_games=0 \
  selfplay.opp_timeout_steps=100000000 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1
```

`selfplay.opp_timeout_steps=100000000` intentionally prevents opponent
rotation. This run validates frozen-bank routing and exclusion only; it does
not validate the still-pending atomic generation swap.

Artifact:

```text
checkpoints/dogfight/phase5_frozen_exclusion_pool_20260728_b/0000000000262144.bin
SHA-256 d78f8332270fdabeb53fc7bd908a2de71125431ad08176139d57927ab22be88c
```

Independent fixed-stage-0 evaluation: `perf=0.8948`, `score=0.772`.

Human-visible `DISPLAY=:0` evaluation: PASS. The user reported that it
"looks fine."

## 2026-07-28: synchronous atomic opponent rotation

Implemented the initial conservative generation barrier:

- native self-play rejects multiple processes/GPUs, multiple frozen banks,
  `base.async != 0`, and nonzero `selfplay.eval_games` before native runtime
  creation;
- rotation runs only after a completed synchronous rollout and learner update;
- display/log throttling cannot skip rotation;
- every fight using the rotating bank is truncated and reset;
- the new frozen checkpoint is loaded and all frozen recurrent state is zeroed;
- affected primary recurrent rows are zeroed;
- fresh observations, rewards, terminals, and masks are published before the
  next inference;
- cumulative swap truncations and per-bank generation are logged;
- the old sticky `boundary_reached`/`pending_path` mechanism was removed.

Forced-rotation training command:

```bash
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
timeout 180s ./puffer train dogfight \
  base.run_id=phase5_atomic_swap_20260728_a \
  base.seed=42 \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.async=0 \
  base.cudagraphs=-1 \
  base.checkpoint_interval=1 \
  vec.total_agents=1024 \
  vec.num_buffers=2 \
  vec.num_threads=8 \
  vec.num_frozen_banks=1 \
  vec.frozen_bank_pct=0.5 \
  vec.frozen_bank_hidden_size=64 \
  vec.frozen_bank_num_layers=3 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  train.total_timesteps=262144 \
  selfplay.enabled=1 \
  selfplay.eval_games=0 \
  selfplay.opp_timeout_steps=1 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1
```

Result: PASS. Four completed training rollouts caused four ordered rotations.
Each rotation truncated all 256 affected fights, for exactly 1,024 swap
truncations. Async prefetch was rejected by configuration, so no rollout could
cross a generation boundary.

Final affected suite:

```text
44 passed, 3 skipped
```

Artifact:

```text
checkpoints/dogfight/phase5_atomic_swap_20260728_a/0000000000262144.bin
SHA-256 1398106a6ae90c63787c478f42e1ea54271337357ae887d4ab785fca58195c40
```

Independent fixed-stage-0 evaluation: `perf=0.8937`, `score=0.771`.

Human-visible `DISPLAY=:0` evaluation: PASS. The user reported that it
"looked fine."

## Phase 5 release decision

PASS for the constrained native self-play implementation:

- one GPU/process;
- one frozen bank;
- synchronous rollouts;
- final pool evaluation disabled;
- CPU environments only.

Phase 6 may now establish the first trusted one-bank self-play baseline. These
constraints must remain until later phases add and test broader execution
modes.
