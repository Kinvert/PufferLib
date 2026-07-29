# Native Self-Play Evidence: 2026-07-28

## Status

PufferLib 5c native FIFO self-play works with Dogfight without changes to
PufferLib core. Starting from the Phase 6 native checkpoint, the existing
Dogfight spawn curriculum advanced through stage 11 while native self-play
rotated frozen opponents.

This is dated experimental evidence, not a promise that these command-line
options will remain valid after later PufferLib changes. Future agents should
start with these commands, check the current CLI/config schema, and adjust only
what is necessary.

- Branch: `dogfight5c`
- Source commit used: `a8fc98fb`
- Policy topology: hidden size 64, 3 layers
- Native pool: FIFO, maximum 8 opponents
- Opponent rotation cadence: 2,097,152 steps
- PufferLib core files changed for this experiment: none
- Tracked source files changed for this experiment: none

## Required Runtime Environment

The local binary needs the virtual environment and NCCL library path:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
```

## Training Evidence

### Original 90 Percent Gate

The first native run used the existing `mastery_threshold=0.90`. It mastered
stage 0 after about 10.9M steps, then remained at stage 0 through about 65M
steps. The decisive kill ratio stabilized near 78 to 83 percent, so the
90 percent gate was not appropriate for this native self-play topology.

- Run ID: `native_official_150m_20260728_a`
- Rotations observed before stopping: 31
- Last checkpoint:
  `checkpoints/dogfight/native_official_150m_20260728_a/0000000065011712.bin`
- Stop reason: informative plateau, not a crash

### Foundation Run at 75 Percent

This continuation used the native system unchanged and only overrode the
Dogfight mastery gate.

```bash
./puffer train dogfight \
  selfplay.mode=native \
  base.run_id=native_official_curriculum75_100m_20260728_a \
  base.load_model_path=checkpoints/dogfight/native_official_150m_20260728_a/0000000065011712.bin \
  train.total_timesteps=100000000 \
  base.checkpoint_interval=32 \
  selfplay.opp_timeout_steps=2097152 \
  selfplay.max_size=8 \
  selfplay.eval_games=0 \
  env.mastery_threshold=0.75
```

Observed curriculum progress:

| Event | Approximate run step |
| --- | ---: |
| Mastered stage 1 | 5.6M |
| Mastered stage 2 | 64.5M |
| Mastered stage 3 | 67.2M |
| Mastered stage 4 | 69.4M |
| Mastered stage 5 | 72.1M |

The run completed 100M steps and 47 FIFO rotations without NaN or Inf output.
It then held at stage 5, where the decisive kill ratio was approximately
60 percent.

- Checkpoint:
  `checkpoints/dogfight/native_official_curriculum75_100m_20260728_a/0000000099942400.bin`

### Advanced Run at 60 Percent

This continuation resumed from the stage-5 policy and tested whether native
self-play plus Dogfight's existing spawn curriculum could raise difficulty
automatically.

```bash
./puffer train dogfight \
  selfplay.mode=native \
  base.run_id=native_official_curriculum60_stage5_150m_20260728_a \
  base.load_model_path=checkpoints/dogfight/native_official_curriculum75_100m_20260728_a/0000000099942400.bin \
  train.total_timesteps=150000000 \
  base.checkpoint_interval=32 \
  selfplay.opp_timeout_steps=2097152 \
  selfplay.max_size=8 \
  selfplay.eval_games=0 \
  env.curriculum_target=5.9 \
  env.mastery_threshold=0.60
```

Observed curriculum progress:

| Event | Approximate run step |
| --- | ---: |
| Mastered stage 6 | 5.6M |
| Mastered stage 7 | 56.9M |
| Mastered stage 8 | 59.2M |
| Mastered stage 9 | 62.0M |
| Mastered stage 10 | 64.5M |
| Mastered stage 11 | 67.0M |

The run reached 149.9M steps, completed 71 FIFO rotations, and produced no
NaN or Inf output. It correctly stopped advancing at stage 11 because stage 11
was not mastered. Reported throughput at the end was about 521K SPS.

The process did not exit after reaching its timestep limit and repeatedly
printed the final dashboard. It was stopped with Ctrl-C after confirming that
the final checkpoint existed. Treat this as a PufferLib finalization issue to
investigate, not a Dogfight training failure.

- Final checkpoint:
  `checkpoints/dogfight/native_official_curriculum60_stage5_150m_20260728_a/0000000149946368.bin`
- Full local log:
  `/tmp/dogfight_native_official_curriculum60_stage5_150m_20260728_a.log`

## Both-Seat Match Evidence

Matches used the 100M stage-5 checkpoint as the common opponent. Each policy
was evaluated in both seats. PufferLib may finish the current vector batch, so
the structured evidence can contain slightly more than the requested 2,048
games.

The required fixed-stage options are:

```bash
env.curriculum_enabled=1 env.fixed_stage=STAGE
```

Do not use `env.curriculum_enabled=0` for a fixed-stage comparison. That
disables the fixed-stage selection and can make nominally different stage
matches identical.

Native match template:

```bash
./puffer match dogfight \
  base.seed=SEED \
  base.load_model_path=PRIMARY_CHECKPOINT \
  base.load_enemy_model_path=ENEMY_CHECKPOINT \
  base.num_games=2048 \
  env.num_agents=2 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  vec.num_frozen_banks=1 \
  vec.frozen_bank_hidden_size=64 \
  vec.frozen_bank_num_layers=3 \
  env.curriculum_enabled=1 \
  env.fixed_stage=STAGE
```

For the second seat, swap `PRIMARY_CHECKPOINT` and `ENEMY_CHECKPOINT`. Interpret
the candidate as slot 0 in the first run and slot 1 in the swapped run.

Aggregated results:

| Candidate | Stage | Candidate wins | Opponent wins | Draws | Candidate decisive share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 65.0M boundary | 10 | 2,523 | 1,580 | 2 | 61.5% |
| 65.0M boundary | 11 | 993 | 2,583 | 522 | 27.8% |
| 67.1M boundary | 10 | 2,535 | 1,566 | 3 | 61.8% |
| 67.1M boundary | 11 | 1,124 | 2,380 | 595 | 32.1% |
| 149.9M final | 10 | 2,200 | 1,895 | 4 | 53.7% |
| 149.9M final | 11 | 1,362 | 2,027 | 714 | 40.2% |

Interpretation:

- Native continuation added substantial stage-10 capability.
- Continued stage-11 training improved stage-11 performance from 32.1 to
  40.2 percent decisive share.
- Some stage-10 specialization was lost, but the final checkpoint still beat
  the earlier policy at stage 10.
- The stage-11 gate is doing useful work by preventing premature advancement.
- The final checkpoint is the best advanced-stage candidate from this run, but
  it is not yet a release checkpoint.

The full structured match output is in:

```text
/tmp/dogfight_native_checkpoint_screen_20260728.log
/tmp/dogfight_native_intermediate_screen_20260728.log
```

## Human-Visible Evaluation

This command successfully opened the 1280x720 C renderer on display 0 at fixed
stage 11:

```bash
DISPLAY=:0 timeout 25s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/native_official_curriculum60_stage5_150m_20260728_a/0000000149946368.bin \
  env.curriculum_enabled=1 \
  env.fixed_stage=11
```

Human observation: the policy was flying acceptably, but not greatly. It was
stable enough to continue development and clearly represented learned flight.

The render reported:

- `avg_stage=11.000`
- approximately equal aggregate seat scores
- `draw_rate` approximately 0.31
- `base_stage_kills` approximately 0.39
- nonzero ground losses on both sides

## What This Proves

PufferLib's official native FIFO system does not need Elo-aware matchmaking for
Dogfight to raise difficulty. Native opponent rotation supplies changing
policies, while Dogfight's existing curriculum supplies progressively harder
spawn geometry. Together they produced automatic progression from stage 5
through stage 11 without PufferLib core modifications.

Native FIFO still uses fixed-cadence snapshots and pool sampling; it does not
perform Dogfight-specific promotion or rating-aware curriculum control. Keep
the persistent Dogfight coordinator available for future promotion gates and
reproducible match evidence.

## Recommended Next Experiment

Do not lower the mastery gate merely to force stage advancement. Resume from
the final checkpoint at stage 11 and improve that frontier while periodically
checking stages 6, 10, and 11 in both seats. Preserve the 67.1M checkpoint as a
stage-10 regression anchor.

Any source change must return to the required loop:

```text
failing test -> implementation -> affected tests -> build -> deterministic
training check -> quantitative eval -> DISPLAY=:0 human flight eval
```
