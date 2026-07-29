# Dogfight Fixed-Stage Evaluation Matrix

## Purpose

Native self-play training metrics are not the acceptance oracle. A candidate can
score well against its training distribution while failing basic numbered
curriculum stages or learning a persistent lateral control bias.

Promotion is based on fixed stages 0 through 10:

- `128` episodes per seed and mirror cell.
- Fresh seeds `42`, `314159`, and `271828`.
- Exact lateral mirrors `0` and `1`.
- Every cell for a stage must reach `perf >= 0.90`.
- Evaluation stops after the first complete stage matrix that fails.
- Candidates rank first by highest contiguous mastered stage, then worst-cell
  performance, mean performance, and control-bias diagnostics.

The final goal is highest contiguous stage `10`. A short screening run may use
fewer seeds or stages, but it is not final evidence.

## Exact paired geometry

`env.eval_lateral_mirror=1` reflects a normally generated fixed spawn without
consuming RNG:

- Position, velocity, and previous velocity map from `{x,y,z}` to `{x,-y,z}`.
- Quaternion maps from `{w,x,y,z}` to `{w,-x,y,-z}`.
- Angular velocity maps from `{x,y,z}` to `{-x,y,-z}`.
- Left/right and hard-left/hard-right autopilot modes swap.

The flag applies only when numbered fixed evaluation is active for stages
0 through 10. It does not modify training defaults, randomized eval spawns, or
PufferLib core.

Identical stage and seed values with mirror `0` and `1` form one exact pair:

```bash
bash ocean/dogfight/eval_checkpoint.sh \
  headless "$CHECKPOINT" "$STAGE" 128 "$SEED" "$MIRROR"
```

`env.role_randomization` is not a geometry mirror. `env.eval_spawn_mode` only
applies to randomized eval and therefore cannot mirror numbered stages.

## Full matrix command

```bash
python ocean/dogfight/eval_stage_matrix.py "$CHECKPOINT" \
  --stages 0..10 \
  --seeds 42,314159,271828 \
  --mirrors 0,1 \
  --episodes 128 \
  --threshold 0.90 \
  --json-output fixed-eval.json
```

Exit `0` with valid JSON means all requested stages mastered. Exit `2` with
valid JSON is a normal candidate rejection. Missing or malformed JSON, another
exit code, a timeout, or a checkpoint mismatch is an infrastructure failure.

The matrix runs each cell serially. The C evaluator uses CUDA, so do not run it
concurrently with native training. Sweep orchestration drains fixed evaluations
only after the native sweep process exits.

## Current rejection reference

Checkpoint:

```text
ocean/dogfight/sweeps/df42-native-sweep-20260729-094258/checkpoints/dogfight/sweep_1785343379066_0000/0000000671088640.bin
```

Stage `0`, seed `42`, `128` episodes:

| Mirror | Perf | Score | Timeout rate | Signed aileron bias |
|---:|---:|---:|---:|---:|
| 0 | 0.359375 | 0.039062 | 0.640625 | 62.691963 |
| 1 | 0.468750 | 0.203125 | 0.531250 | 54.503548 |

Both cells fail, so the highest contiguous mastered stage is `-1`. The native
training metric for this run was about `0.632`, demonstrating why fixed
evaluation must be recorded separately.

## Deterministic end-to-end canary

The no-name command below completed native training, df42 history upload,
post-sweep fixed screening, and fixed-summary upload:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --max-runs 1 \
  --timesteps 8388608
```

Verified experiment:

```text
ocean/dogfight/sweeps/df42-native-sweep-20260729-102730
```

The manifest ended with native exit `0`, `uploaded_runs=1`,
`fixed_eval_summary.completed=1`, `rejected=1`, `failures=0`, and
`status=completed`. The stage-0 screen evaluated six cells and reported:

- Highest contiguous stage: `-1`.
- Minimum perf: `0.046875`.
- Mean perf: `0.06630516666666667`.
- Maximum absolute signed aileron bias: `42.275864`.

Two independent canaries using the same seed and configuration produced the
same final checkpoint SHA-256 and identical fixed metrics:

```text
fbf5dc6cfd8477382596c315797c7848465c4da3ccc018468802d74164d0b868
```

This is the current deterministic smoke-test reference, not a learned-policy
quality result.

Interrupted post-sweep screening resumes without training:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --screen-existing EXPERIMENT_DIR
```

## TDD gate

```bash
/home/claude/PufferLib/.venv/bin/python -m pytest -q ocean/dogfight/tests
```

The exact-mirror C regression checks stages 0 through 10, RNG identity,
reflected plane state, autopilot direction, and the observation sign contract.
The existing lateral-symmetry regression covers mirrored controls and one-step
flight physics.

Build commands:

```bash
source .venv/bin/activate
CUDA_HOME=/usr/local/cuda ./build.sh dogfight
CUDA_HOME=/usr/local/cuda bash ocean/dogfight/build_eval.sh
```

Use clone-local `.venv` on the 5090. The `/home/claude/PufferLib/.venv` path is
only a compatibility fallback on g240.
