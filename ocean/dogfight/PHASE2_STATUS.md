# Dogfight 5c Phase 2 Status

Date: 2026-07-28

Status: PASS

Phase 2 restores Dogfight3 flight-test coverage, trainer global-step delivery,
mastery-gated curriculum progression, deterministic training evidence, fixed-stage
headless evaluation, and visible human evaluation through stage 2.

These commands were verified on branch `dogfight5c` against the current 5c
code. Future agents should try them first, but must treat them as dated evidence,
not gospel. If 5c changes, confirm the current CLI and adapt the smallest
necessary part.

## Dependency Shell

The donor virtual environment supplies Python and NCCL. CUDA 12.8 builds the
native trainer.

```bash
export PATH=/home/claude/dogfight5/.venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CCACHE_DIR=/tmp/ccache
export NCCL_DIR=/home/claude/dogfight5/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
mkdir -p /tmp/dogfight5c-nccl
ln -sf "$NCCL_DIR/libnccl.so.2" /tmp/dogfight5c-nccl/libnccl.so
export LIBRARY_PATH=/tmp/dogfight5c-nccl
export LD_LIBRARY_PATH=/tmp/dogfight5c-nccl:$NCCL_DIR
```

## TDD Evidence

The controller test was written first and failed because
`PufCurriculumState`, `puf_curriculum_init`, and `puf_set_global_step` did not
exist.

The native cadence test was written after the first training probe and failed
because curriculum updates occurred after the wall-clock dashboard throttle.
The hook now polls every rollout epoch.

The first long deterministic replay produced byte-identical weights and
identical mastery steps, but its diagnostic windows differed with dashboard
timing. That red determinism result led to an explicit dashboard-counter-clear
contract. Two fresh post-fix runs now produce exact curriculum traces and
byte-identical checkpoints.

## Build and Test Gate

```bash
./build.sh dogfight --local
ASAN_OPTIONS=detect_leaks=0 ./dogfight

PYTHONDONTWRITEBYTECODE=1 \
  /home/claude/dogfight5/.venv/bin/python \
  -m pytest -p no:cacheprovider ocean/dogfight/tests -q

./build.sh dogfight --float
./build.sh dogfight
```

Verified results:

- Local standalone build: PASS
- Standalone smoke: `dogfight phase1 smoke: ok`
- Pytest: `18 passed, 4 skipped`
- Float native build: PASS
- Default native build: PASS

The six structured flight suites and their donor provenance are recorded in
`PHASE2_FLIGHT_TESTS.md`.

## Working Stage-Climb Training Command

This is the supported subset of the prior Dogfight5 clean-climber profile.
The profile keeps Dogfight3 environment/reward behavior while using
native-5c training hyperparameters that actually climb.

```bash
./puffer train dogfight \
  base.run_id=phase2_climber_b \
  vec.total_agents=4096 \
  vec.num_buffers=4 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  train.total_timesteps=33554432 \
  train.gpus=1 \
  train.learning_rate=0.0025 \
  train.minibatch_size=4096 \
  train.gamma=0.996 \
  train.gae_lambda=0.999 \
  train.ent_coef=0.02 \
  train.clip_coef=0.06 \
  train.vf_coef=4.6 \
  train.vf_clip_coef=1.5 \
  train.max_grad_norm=3.4 \
  train.momentum=0.9896 \
  train.prio_alpha=0.4 \
  train.prio_beta0=0.82 \
  train.replay_ratio=0.95 \
  train.vtrace_rho_clip=0.1 \
  train.vtrace_c_clip=2.5 \
  env.domain_randomization=0.05 \
  env.vertical_spawn_prob=0.02 \
  env.warmup_steps=1000000 \
  env.eval_interval=60000 \
  env.min_eval_episodes=50 \
  env.max_stage=2
```

Verified curriculum milestones:

- Initial target `0.9` failed its first gate and correctly retreated to stage 0.
- Stage 0 mastered at `24,903,680` steps.
- Stage 1 mastered at `25,165,824` steps.
- Stage 2 mastered at `25,952,256` steps.
- Run finished at target `2.0`.
- Final stage-2 mastery window was approximately 77% kills.

Checkpoint:

```text
checkpoints/dogfight/phase2_climber_b/0000000033554432.bin
```

The config file retains the conservative donor-style baseline. The full
override profile above is required to reproduce this short native stage climb.

## Determinism Gate

The two 33,554,432-step runs produced byte-identical final checkpoints:

```text
3a16815298fa1c2685376a91699965be4f8d6a10ed23bc70aafc5358ec4372f4
```

The replay used `base.run_id=phase2_climber_b_replay` with every other argument
unchanged.

After the dashboard-counter fix, two fresh 4,194,304-step runs produced exact
curriculum traces and byte-identical checkpoints:

```text
bcd81b295bc1462be9045b10dfb90f99dedd3f984ccd3b25ab23838b6f2633da
```

The short run IDs were `phase2_cadence_a` and `phase2_cadence_b`.

## Headless Fixed-Stage Evaluation

Checkpoint topology arguments are mandatory. Loading this checkpoint with the
default 128-wide, one-layer policy is invalid.

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase2_climber_b/0000000033554432.bin \
  base.num_games=1024 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.fixed_stage=0
```

Repeat with `env.fixed_stage=1` and `env.fixed_stage=2`.

Verified results:

- Fixed stage 0: perf `1.0000`, score `1.000`
- Fixed stage 1: perf `0.7096`, score `0.564`
- Fixed stage 2: perf `1.0000`, score `1.000`

The vector evaluator may overshoot `base.num_games`; this run completed at
least 1,024 episodes for each fixed stage.

## Visible Human Evaluation

```bash
DISPLAY=:0 timeout 20s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase2_climber_b/0000000033554432.bin \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.fixed_stage=2
```

Expected exit code is `124` because `timeout` ends the otherwise continuous
render loop.

Mechanical result: PASS. Raylib opened a 1280x720 window, loaded
`ocean/dogfight/p40.glb`, and rendered for the full timeout.

Human result: PASS. The plane did not fly perfectly, but it clearly learned,
visibly won about 75% of the early-curriculum fights, and did not merely produce
a black window or static scene.

## Remaining Risks

- Flight quality is learned but not yet polished.
- Fixed stage 1 is materially weaker than stages 0 and 2.
- The renderer prints excessive dashboard output.
- The visible run used Mesa llvmpipe, so rendering is CPU-backed on this host.
- Advanced curriculum stages, self-play, and longer-horizon flight quality are
  later-phase work.
