# Vanilla Self-Play Phase 0 Evidence

Date: 2026-07-28

## Decision

Phase 0 is complete. The clean-core Dogfight baseline builds, passes its full
test suite, trains deterministically, masters fixed stage 0, evaluates through
the Dogfight-local headless and visible wrappers, and repeats within the
declared `5%` elapsed-time tolerance.

This is a one-agent policy against the internal scripted opponent. It is the
pre-change behavioral and performance baseline, not a native two-agent
self-play result.

## Source State

- Branch: `dogfight5c-vanilla-selfplay`
- Baseline commit: `fe1c1332c0198fd239d3c4b39690411ee8d7348a`
- Pinned Joseph 5c source:
  `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`
- `git diff --quiet <pinned-5c> -- src` reported a clean PufferLib core.
- No implementation file was modified during Phase 0.
- Root launchers `dogfight` and `puffer` are generated build artifacts and must
  not be committed.

## Runtime Environment

The commands below were observed working in this checkout. They are evidence
for this machine and revision, not guaranteed future PufferLib CLI syntax.

```bash
source /home/claude/PufferLib/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
```

GPU training and evaluation must run in a context with access to the real CUDA
device and pinned host memory. The first sandboxed training attempt aborted in
`src/pufferl.cu:xpin` because `cudaHostAlloc` failed. Running the same command
outside the restricted sandbox succeeded. This was an execution-context
failure, not a Dogfight simulation failure.

## Build and Test Gate

Observed working command:

```bash
./build.sh dogfight
bash ocean/dogfight/build_eval.sh
python -m pytest ocean/dogfight/tests -q
```

Observed result:

```text
46 passed, 3 skipped in 23.16s
```

Both native binaries built successfully. The compiler reported existing
narrowing-conversion warnings in `dogfight_render.h`; these did not fail the
build.

The suite includes RK4 and flight physics, energy behavior, recovery, static and
dynamic observations, action bounds, reset and terminal references, curriculum
geometry and mixing, RNG isolation, two-agent adapter behavior, training
profile, checkpoint manifests, and matched evaluation-wrapper tests.

## Short Determinism Canary

Observed working commands:

```bash
bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase0_s42_20260728 1048576 0

bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase0_s42_repeat_20260728 1048576 0
```

Both checkpoints had this SHA-256:

```text
6d18e8b2793e18d80accc797c69fd78e023fa7a6dad0ea86379b62bd65c042e6
```

The one-million-step policy was intentionally immature:

- Training final evaluation: approximately `7.6%`
- Independent headless evaluation: `5/64`, or `7.81%`
- Visible evaluation: `0/8`
- Human observation: it flew badly, with a possible tendency to roll right, but
  the observation was uncertain because the viewer was distracted.

This run proves exact fixed-seed checkpoint reproduction. It is not a
flight-quality acceptance checkpoint.

## Intermediate Learning Check

Observed working command:

```bash
bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase0_stage0_8m_s42_20260728 8388608 0
```

The final internal stage-0 result was approximately `33.5%`. It was not used for
visible acceptance because it was still clearly immature.

## Full Fixed-Stage Baseline

Observed working commands:

```bash
bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase0_stage0_32m_s42_20260728 33554432 0

bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase0_stage0_32m_s42_repeat_20260728 33554432 0
```

Authoritative checkpoint:

```text
checkpoints/dogfight/vanilla_phase0_stage0_32m_s42_20260728/0000000033554432.bin
```

Both full-run checkpoints had this SHA-256:

```text
fd0ed432f36422f543b8e4c9e8d02b82ff2d32967178e59c9954501c21236676
```

Repeated results:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Agent steps | 33,554,432 | 33,554,432 |
| Uptime | 36.419s | 38.146s |
| Whole-run agent SPS | 921,344 | 879,632 |
| Final internal performance | 95.7% | 95.7% |
| Final internal score | 0.936 | 0.936 |
| Player-ground rate | 0.0% | 0.0% |

The elapsed-time spread was `4.74%`, within the Phase 0 `5%` repeatability
tolerance. The paired whole-run baseline is approximately `0.90M agent SPS`.
Short dashboard samples varied more widely and should not be used as the
authoritative baseline.

Because this mode has one trainable agent against an internal scripted
opponent, its agent SPS is not directly comparable to a native two-agent match.
For the future two-agent mode, match steps/s will be approximately half of
reported agent SPS.

## Independent Evaluation

Observed working headless command:

```bash
bash ocean/dogfight/eval_checkpoint.sh headless \
  checkpoints/dogfight/vanilla_phase0_stage0_32m_s42_20260728/0000000033554432.bin \
  0 64 42
```

Observed result:

```text
56/64 wins
performance = 0.875
score = 0.812
```

Observed working visible command:

```bash
DISPLAY=:0 bash ocean/dogfight/eval_checkpoint.sh visible \
  checkpoints/dogfight/vanilla_phase0_stage0_32m_s42_20260728/0000000033554432.bin \
  0 8 42
```

Observed result:

```text
8/8 wins
```

Human observation: the evaluation was very fast and the policy appeared to win
frequently. The earlier possible right-roll tendency was not confirmed or
cleared with a careful sustained-flight observation, so the existing control
polarity, level-flight, recovery, and visible-flight gates remain mandatory.

## Phase 1 Entry Conditions

- Preserve the authoritative checkpoint and hash above.
- Preserve `46 passed, 3 skipped` as the starting test result.
- Preserve approximately `0.90M agent SPS` as the one-agent whole-run baseline.
- Make no PufferLib core changes.
- Add two-agent contract tests before changing vanilla self-play behavior.
- Repeat build, full tests, deterministic training, headless evaluation, and
  `DISPLAY=:0` evaluation before Phase 1 is accepted.
