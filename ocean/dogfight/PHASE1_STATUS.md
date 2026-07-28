# Dogfight 5c Phase 1 Status

Date: 2026-07-28

Phase 1 establishes a one-agent Dogfight3-compatible environment on PufferLib
5c. The CPU contracts, native builds, bounded training, deterministic replay,
and headless checkpoint evaluation pass.

## Gate status

- Build: PASS for `--local`, `--float`, and default BF16/native.
- Test: PASS, `10 passed, 4 skipped`.
- Sanitizers: PASS for the finite standalone smoke. Leak detection is disabled
  only because LeakSanitizer cannot run under the host ptrace wrapper.
- Train: PASS for a four-epoch, 262,144-step native CUDA canary.
- Determinism: PASS. Two independent fixed-seed runs matched on all extracted
  final gameplay metrics.
- Headless checkpoint eval: PASS for 1,024 requested games.
- Visible `DISPLAY=:0` eval mechanics: PASS. Native 5c now exposes `render`,
  compiles the real `c_render` under NVCC, loads the aircraft model, and remained
  alive until the bounded 20-second timeout.
- Human flight-quality assessment: FAIL. The 262,144-step stage-0 checkpoint
  flies poorly and visibly behaves like an early-curriculum policy. Renderer
  availability must never be confused with acceptable aircraft behavior.

## What Phase 1 added

- Dogfight3 flight model, spawning, observations, scripted opponent, and
  renderer source.
- A current 5c `Env`/`Agent` adapter with one trainable policy agent and an
  internal scripted opponent.
- Opponent-aware 26-float observations and five continuous controls.
- Canonical action clamping before physics.
- An immutable configured episode cap restored at reset.
- Per-environment RNG binding for native vector workers. Direct donor fixtures
  retain the pinned Dogfight3 process-global RNG behavior.
- A finite standalone smoke executable and C fixture/provenance tests.
- A generic 5c eval fix: eval horizon now uses `ADV_VEC_WIDTH` instead of the
  invalid hardcoded value `1`.
- A native `render` mode, NVCC-safe Raylib compatibility declarations, the
  pinned Dogfight3 `p40.glb`, a resizable 1280x720 window, and a complete
  26-value observation overlay.

## Deferred tests

The four skips are explicit rather than hidden:

- Multi-episode logging/reset fields: Phase 2.
- Duplicated `State` roundtrip: must be rewritten without importing that
  rejected Dogfight5 design.
- Curriculum max-step variants: Phase 2.
- Later curriculum geometry: Phase 2.

## Commands observed working in this checkout

These are observations from this branch and machine, not permanent API
guarantees. Future agents must re-check 5c CLI/config changes rather than taking
them as gospel.

Dependency setup used here:

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

Build:

```bash
./build.sh dogfight --local
./build.sh dogfight --float
./build.sh dogfight
```

CPU test gate:

```bash
ASAN_OPTIONS=detect_leaks=0 ./dogfight
PYTHONDONTWRITEBYTECODE=1 /home/claude/dogfight5/.venv/bin/python \
  -m pytest -p no:cacheprovider \
  ocean/dogfight/tests/test_fixture_provenance.py \
  ocean/dogfight/tests/test_c_regressions.py -q
```

Native 5c training canary:

```bash
./puffer train dogfight \
  base.run_id=phase1_determinism_a \
  train.total_timesteps=262144 \
  train.gpus=1
```

Native 5c headless checkpoint eval:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase1_determinism_a/0000000000262144.bin \
  base.num_games=1024
```

Native 5c human-visible checkpoint eval:

```bash
DISPLAY=:0 ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase1_render_gate/0000000000262144.bin
```

Confirmed bounded render smoke:

```bash
DISPLAY=:0 timeout 20s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase1_render_gate/0000000000262144.bin
```

Omit `timeout 20s` when the human needs an open-ended inspection session.

Command provenance: native argument syntax and modes were confirmed directly
from this checkout's `src/pufferl.cu` and executed successfully. The older
`python -m pufferlib.pufferl ...` command from Dogfight4/5 donors is not valid
with the donor virtual environment here: its compiled `pufferlib._C` predates
5c's `load_config` ABI.

The native sweep form is:

```bash
./puffer sweep dogfight \
  sweep.max_runs=1 \
  sweep.gpus=1 \
  train.gpus=1
```

Do not use a 262,144-step override with the inherited default sweep space. The
global sweep requires at least 30 million timesteps, and current Dogfight
defaults also exceed inherited ranges for `gae_lambda` and `max_grad_norm`.
Define deliberate Dogfight-specific sweep bounds before treating sweep as a
green gate.

## Determinism baseline

Both fixed-seed 262,144-step runs produced:

```text
perf=0.064
score=-0.405
episode_return=-2.126
episode_length=283.004
shots_fired=24.268
accuracy=0.219
avg_abs_bias=36.106
avg_control_rate=3.012
base_stage_kills=0.064
```

Timing, SPS, GPU utilization, and dashboard refresh cadence are not
determinism criteria.

## Required loop from here

Use build, test, train, then eval for every meaningful port chunk. If a change
intentionally breaks the numerical baseline, land the new expected baseline
with the change; subsequent identical fixed-seed runs must match it. Flight
tests remain mandatory before trusting reward or curriculum results.
