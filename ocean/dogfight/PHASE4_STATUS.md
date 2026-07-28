# Dogfight Phase 4 Status

Date: 2026-07-28

Branch: `dogfight5c`

Base: Joseph/PufferAI `origin/5c` at
`ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`

## Current verdict

Phase 4 is complete for its stage-0 release gate. Its structural, regression,
determinism, independent-eval, and human-visible flight gates pass.

The direct two-agent environment now has:

- two directly controlled native agent slots;
- per-aircraft perspective observations;
- independent action routing;
- symmetric, versioned dense rewards;
- identical trigger and cooldown handling;
- two-phase hit/crash/supersonic/timeout resolution;
- deterministic logical-to-physical role randomization;
- logical `slot_0_score`, `slot_1_score`, and `draw_rate` logs;
- clean-fight and shot accounting.

The accepted artifact is a bounded two-agent warm start from the proven Phase 3
curriculum checkpoint. Random initialization was tested and rejected because
balanced slots mostly timed out; balanced slot scores alone are not evidence of
successful combat learning.

## TDD evidence

The focused two-agent regression covers:

- fresh observations for both slots after reset;
- swapping physical aircraft swaps observations;
- each logical action buffer controls its assigned aircraft;
- both terminal pulses and reset observations are published;
- mutual hits resolve as an order-invariant draw;
- mutual crashes resolve identically for both aircraft;
- both aircraft receive identical fire cooldown rules;
- swapping aircraft and actions swaps dense rewards;
- forced logical role assignments route observations and actions correctly;
- logical win/draw metrics follow the role assignment;
- equal seeds produce equal role-assignment sequences while exercising both
  assignments;
- successful fire is reflected in `shots_fired` and `accuracy`.

The red tests captured the pre-fix failures explicitly:

```text
simultaneous hit was order-biased: -1.000 1.000
simultaneous crash was order-biased: 0.250 -1.000
cooldown mismatch: 10 9
role-swap reward mismatch
physical-player shot accounting failed
```

Focused command that passed:

```bash
source /home/claude/dogfight5/.venv/bin/activate
pytest -q ocean/dogfight/tests/test_c_regressions.py -k two_agent_adapter
```

Result:

```text
1 passed, 21 deselected
```

Full standing suite:

```bash
source /home/claude/dogfight5/.venv/bin/activate
pytest -q ocean/dogfight/tests
```

Result:

```text
31 passed, 3 skipped
```

The skipped tests are pre-existing, explicitly deferred contracts. They are not
new Phase 4 failures.

## Build environment and command

This exact environment was required in this checkout:

```bash
source /home/claude/dogfight5/.venv/bin/activate
export PATH=/home/claude/dogfight5/.venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CCACHE_DIR=/tmp/ccache
export NCCL_DIR=/home/claude/dogfight5/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
mkdir -p /tmp/dogfight5c-nccl
ln -sf "$NCCL_DIR/libnccl.so.2" /tmp/dogfight5c-nccl/libnccl.so
export LIBRARY_PATH=/tmp/dogfight5c-nccl
export LD_LIBRARY_PATH=/tmp/dogfight5c-nccl:$NCCL_DIR
```

Working native build:

```bash
./build.sh dogfight
```

Result: PASS. The known `dogfight_render.h` narrowing warnings remain. Omitting
the NCCL shim caused the linker to fail with `cannot find -lnccl`.

## Native adapter smoke

This bounded command verified that the native trainer accepts two direct slots:

```bash
timeout 30s ./puffer train dogfight \
  base.run_id=phase4_green_two_agent_adapter \
  base.async=0 \
  base.cudagraphs=-1 \
  vec.total_agents=64 \
  vec.num_buffers=1 \
  vec.num_threads=1 \
  policy.hidden_size=16 \
  policy.num_layers=1 \
  train.total_timesteps=1024 \
  train.horizon=16 \
  train.minibatch_size=64 \
  selfplay.enabled=0 \
  env.num_agents=2
```

Result: PASS. The former `env->num_agents == 1` assertion is gone.

## Rejected random-initialization diagnostic

These diagnostic runs used the stock current-5c policy configuration but
incorrectly discarded the Phase 3 curriculum policy. No Dogfight-specific
encoder, PPO variant, or frozen-policy bank was introduced.

Run A:

```bash
timeout 180s ./puffer train dogfight \
  base.run_id=phase4_behavior_20260728_a \
  base.seed=42 \
  base.async=0 \
  base.cudagraphs=-1 \
  base.checkpoint_interval=25 \
  vec.total_agents=1024 \
  vec.num_buffers=2 \
  vec.num_threads=8 \
  train.total_timesteps=5000000 \
  selfplay.enabled=0 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1
```

Run B used the identical command with:

```text
base.run_id=phase4_behavior_20260728_b
```

All corresponding A/B checkpoints were byte-identical:

| Step | SHA-256 |
|---:|---|
| 1,638,400 | `0b4592479e69ec6972529796e932eb9bc2240f84572854dbcf20ceacf8320931` |
| 3,276,800 | `63bb213ff01155487350c218c9e6b07fd70c2e334cc464c71b85f6277416dcec` |
| 4,915,200 | `06c3909731278fdaa1cb3404d8efd7a547b27f6b1959ac3846c172cff4b5b620` |
| 4,980,736 | `d2fc06a12d070cd4100c24deff65d6ef9a83155d8f70690a4d550fd27b018233` |

The repeated final training window included:

```text
perf=0.516
slot_0_score=0.495
slot_1_score=0.505
draw_rate=0.264
player_ground=0.190
clean_fights=0.809
shots_fired=15.978
```

These stochastic-rollout metrics show balanced logical slots and more resolved
episodes than the 262,144-step canary. They do not override independent eval.

## Rejected random-start numerical evaluation

Working command:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase4_behavior_20260728_a/0000000004980736.bin \
  base.num_games=4096 \
  policy.hidden_size=128 \
  policy.num_layers=1 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Result:

```text
perf=0.1908
score=-0.305
```

The native eval summary exposes the legacy physical-player `perf` and `score`,
not the logical slot metrics. This result rejects the random-start checkpoint.

## Accepted Phase 3 to Phase 4 warm start

Before any two-agent updates, the proven Phase 3 checkpoint was evaluated
directly through the Phase 4 environment:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
  base.num_games=4096 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Result:

```text
perf=0.8967
score=0.775
```

This proves the Phase 4 mechanics preserve the learned Phase 3 policy. The
accepted bounded training gate then used the native warm-start handoff:

```bash
timeout 180s ./puffer train dogfight \
  base.run_id=phase4_warmstart_20260728_a \
  base.seed=42 \
  base.load_model_path=checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
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

Run B used the identical command with:

```text
base.run_id=phase4_warmstart_20260728_b
```

All four corresponding A/B checkpoints were byte-identical. The accepted final
checkpoint is:

```text
checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin
SHA-256 927ce244bac762b83644ec2b93dd5ce49734ec254c2f5595afb927dda485f032
```

Independent accepted-artifact evaluation:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  base.num_games=4096 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Result:

```text
perf=0.8980
score=0.778
```

## Human-visible evaluation

Working command:

```bash
DISPLAY=:0 timeout 20s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase4_warmstart_20260728_a/0000000000262144.bin \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.num_agents=2 \
  env.reward_version=1 \
  env.role_randomization=1 \
  env.fixed_stage=0
```

Mechanical result: PASS. Raylib opened a 1280x720 window, loaded
`ocean/dogfight/p40.glb`, and ran until the expected timeout exit code `124`.

Human result: PASS for the accepted artifact. The user reported that it
"looked good" and "flew pretty smooth." It looked like a low curriculum stage
because the gate deliberately fixed `env.fixed_stage=0`.

The accepted render reported approximately:

```text
slot_0_score=0.502
slot_1_score=0.498
draw_rate=0.105
perf=0.894
player_ground=0.001
```

The earlier random-start render remains a useful negative control. The user
correctly reported that it was "mostly timeout, so not winning."

## Next work

1. Preserve the accepted Phase 3 warm-start handoff. Do not restart the
   two-agent phase from random weights.
2. Preserve the direct symmetric mechanics and their standing tests.
3. Treat the random-start timeout result as a negative control, not the release
   artifact.
4. Repeat the complete build, test, deterministic train A/B, independent eval,
   and `DISPLAY=:0` render loop after any behavior-affecting change.
5. Keep later-stage visual quality as a standing gate when Phase 5 expands
   opponent diversity; this Phase 4 render intentionally isolated stage 0.
6. Do not customize PufferLib PPO or the policy encoder without concrete
   environment-side evidence.

## Command provenance

These commands were executed successfully in this checkout on 2026-07-28. They
are dated evidence, not gospel. Future agents must start with these forms,
bound expensive commands, confirm current 5c argument/config behavior, and
document any required changes. In particular, native checkpoint loading uses
`base.load_model_path=...`; it does not use the older Python CLI form.
