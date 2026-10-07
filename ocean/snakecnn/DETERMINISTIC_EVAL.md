# Exact assigned-episode Snake evaluation

October 6, 2026 launcher follow-up. The October 5 four-family native builds
remain unchanged. Independent CPU **game/receipt** tests and fourteen
native-host/synthetic-audit/launcher tests pass. These execute no neural policy.
GPU inference, recurrent reset, checkpoint
reload, repeatability and learning remain pending. Both GPU holds and
encoder-5's separate 5090 gates remain.

## Episode and scoring contract

`snake-local-episodic-v1` and matching `snakebench` share the one-agent world,
RNG, actions, rewards and declared finite game horizon. This is not stock
Snake. Each explicit episode ID maps to world and native Philox action seeds
using the versioned Connect4 identity mapping. Drawing selection doesn't
consume game RNG. Encoder/core size doesn't select starts.

Evaluation ends on first death or game horizon. Death at the horizon remains
death; rewarded food at the horizon remains food. Capture ending counters
before automatic reset loses live values, then stop that slot without
stepping a replacement. Wave padding never contributes outcomes. Every ID
must end exactly once; a process failure or missing ID invalidates the mean.
There is no extra evaluator cap that removes survivors or labels unfinished
episodes losses. The horizon is a frozen game boundary, not proof of unlimited
survival; its learning suitability still needs calibration.

The primary estimand is mean **ending snake length over all assigned
episodes**. Also retain food count, clipped length/120 perf, decisions, return
and death/horizon counts. Live length saturates at ring capacity minus one;
later food still counts and earns reward. Food count and length are distinct;
neither is wins. Audit recomputes length/perf and float32 running return.
Repeated food-reward additions round differently from `foods * reward_food`.

## Native and external evidence

`BINARY eval_exact_info` and `BINARY eval_exact_manifest FULL.ini` dispatch
before CUDA discovery. They initialize only native game/raster, never a policy.
Manifests contain IDs, both seeds, drawing, post-reset RNG and explicit
byte-order world/image hashes. Bare state INIs need no appearance keys.
Invalid settings/IDs fail before the CSV header.

Native `eval_exact` is implemented but **must not run under current holds**.
It keeps the full INI's policy/core and explicit `base.load_model_path`, checks
parameter bytes/finiteness before inference, sets terminal=1 on assigned starts
for production recurrent reset and uses a non-default CUDA actor stream.
Its actual GPU checkpoint/reset/graph behavior remains unvalidated. The
supervised `run` launcher is now implemented and host-tested; use that wrapper
for scheduled acceptance rather than bypassing provenance/failure checks.

Python is external suite/audit glue. Metadata freezes world, seeds, IDs, quota,
slots, drawing, source/effective INIs and build/file hashes. Rehashed but
inconsistent settings fail too. Receipt audits require every assigned ID once,
expected starts/seeds/slots, legal counts/endings and matching finite metrics.
CSV audit alone cannot certify a GPU produced it or certify neural math; its
output explicitly reports `policy_runtime_certified=false`.

## GPU-free preparation

Compile all four normal targets from [README.md](README.md) in a fresh build
directory, then use only native host modes and CPU environment fixtures:

```bash
source ocean/connect4cnn/runtime_env.sh
bash ocean/snakecnn/tests/run_exact_episode.sh build/snakecnn/FRESH_EPISODE_CHECK_ID
.venv/bin/python ocean/snakecnn/tests/test_exact_eval.py \
  --binary-dir build/snakecnn/FRESH_NATIVE_BUILD_ID
.venv/bin/python ocean/snakecnn/tests/verify_host_suites.py \
  --binary-dir build/snakecnn/FRESH_NATIVE_BUILD_ID \
  --out build/snakecnn/FRESH_HOST_SUITE_CHECK_ID
```

The verifier retains 1,000 starts per drawing by default: all six default
pixel manifests match IMPALA/Impoola/large-core manifests, state worlds match,
and the final ID replayed alone starts identically. Image hashes differ where
drawings/state differ. This proves host starts, not GPU batching repeatability.

Prepare a frozen suite using a **full resolved INI**:

```bash
.venv/bin/python research/prepare_pixel_robustness.py --environments snakecnn \
  --out build/pixel-robustness/FRESH_SNAKE_ID
.venv/bin/python ocean/snakecnn/exact_eval.py suite \
  --binary build/snakecnn/FRESH_NATIVE_BUILD_ID/default \
  --config build/pixel-robustness/FRESH_SNAKE_ID/snakecnn/r0/flex_quality-s53111/config/default.ini \
  --seed 56111 --episodes 1000 --slots 64 --representation 0 \
  --out build/snakecnn/FRESH_SUITE_ID
```

Each drawing has its own suite. State uses matching state binary/full INI with
the same world, seed and IDs. Output directories refuse overwrite. Choosing
easier drawings or horizons after scores arrive is not robustness.

## Prepare an existing checkpoint without GPU execution

Supply its actual matching full training INI/binary and Snake checkpoint;
the paths below deliberately require replacement. Other games' action-head
weights aren't interchangeable. `--prepare-only` checks finite float32 bytes,
preserves all policy/core keys, fixes the suite's evaluation appearance and
compares actual native starts against its manifest. It saves original/effective
INIs, copied suite and launcher-tool hashes, with `prepared-not-executed` status.
It never queries a GPU or starts the policy. Preparation **cannot certify
parameter compatibility, neural math or encoder-5 acceptance**.

```bash
.venv/bin/python ocean/snakecnn/exact_eval.py run --prepare-only \
  --binary build/snakecnn/FRESH_NATIVE_BUILD_ID/default \
  --config /absolute/path/to/full/resolved-snake-training.ini \
  --checkpoint /absolute/path/to/snake-checkpoint.bin --family flex \
  --suite build/snakecnn/FRESH_SUITE_ID/suite.json \
  --out build/snakecnn/FRESH_CHECKPOINT_PREPARATION_ID --timeout 600
```

When Kinvert explicitly schedules GPU acceptance, use the same command without
`--prepare-only`, a fresh output ID, and an agreed timeout. The launcher reserves
the shared measurement lock, refuses busy/failed/empty hardware queries, bounds
the native process group and retains monotonic evaluation-process timing. It
requires complete ID/count/end-reason/length/food/parameter summary agreement
and unchanged original/copied/effective/tooling inputs before successful output.
Failures retain partial rows/logs/status/timing without a successful report.
The process timeout is a failure bound, not a replacement game horizon; 600
seconds is a provisional wrapper default, not measured runtime calibration.

Tests include 24 pixel family/core preparations and matching state H256/L2,
forbidden preparation GPU queries, malformed weights, reservations, busy/failed
queries, timeout preservation, completion/parameter/count mismatch, missing/
duplicate summaries, changed copies and absent/backward clocks. Temporary
finite dummy weights and simulated hardware/process/episode outcomes are
**not valid trained policies or GPU execution evidence**. Reports retain
explicit pending runtime qualification even after a completed synthetic fixture.

## Remaining sequence

1. When explicitly scheduled, pass native train/reload and exact inference
   acceptance for quality/Nature/IMPALA/Impoola/state using actual matched
   checkpoints. Don't substitute other games' weights.
2. Test 65 IDs/64 slots, repeated/eager/graph runs, final ID replay, recurrent
   resets across waves, altered core sizes, death/horizon/growth saturation and
   every drawing. Keep batch size fixed for bitwise-repeat claims; investigate
   changed-batch numerical effects separately. Preserve failures/assertions.
3. Calibrate common learning/horizon recipes without held-out score selection,
   then retain paired-seed full curves across tasks/drawings. Plumbing success
   is not learning, Pareto improvement or superiority evidence.

[Retained exact preparation](../../research/results/snakecnn/exact-preparation-20261005/README.md).
[Separate launcher evidence](../../research/results/snakecnn/exact-launcher-20261006/README.md).
[Full robustness plan](../../research/MULTI_ENV_ROBUSTNESS.md).
