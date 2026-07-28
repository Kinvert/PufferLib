# Dogfight 5c Phase 3 Status

Date: 2026-07-28

Pinned PufferLib 5c base:
`ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`

Phase 3 implementation and runtime evidence are complete. The checkpoint
manifest publisher is implemented and tested. Publishing the accepted
checkpoint sidecars is intentionally the final post-commit action because
`dogfight_behavior_version` must name the full commit containing the Phase 3
behavior.

The release artifact selected by the final build-test-train-eval loop is:

```text
checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin
SHA256 525452dacc75e20ca22911a5c684d3b4428b234d2c537b32a8b92a72ae39ce2d
```

## Scope and decisions

Phase 3 delivered:

- current-5c native curriculum baseline evidence;
- deterministic fixed-stage segmented training;
- an optional deterministic single-stage rehearsal mix;
- geometry and behavior coverage for all 21 curriculum stages;
- generic native training warm-start support;
- exact checkpoint byte-count validation;
- a runtime proof that warm-start weights become self-play pool entry zero;
- atomic manifest-last checkpoint publication tooling;
- explicit learning-rate decisions for every accepted segment.

The proposed custom Dogfight encoder A/B remains deferred. The stock current-5c
encoder with `policy.hidden_size=64` and `policy.num_layers=3` mastered stages
0 through 11, so policy topology is not the demonstrated early-learning
constraint. Reopen the encoder work only after a reproducible environment-
fidelity-controlled plateau.

The older PufferLib 5 state-curriculum system is not a Phase 3 deliverable.
Current 5c recurrent state is already active through
`base.reset_every_horizon=0`. The removed generic state-buffer API and the
older duplicated Dogfight `State` mirror were not ported.

## TDD and regression gate

The following commands passed on 2026-07-28:

```bash
export PATH=/home/claude/dogfight5/.venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CCACHE_DIR=/tmp/ccache
export NCCL_DIR=/home/claude/dogfight5/.venv/lib/python3.12/site-packages/nvidia/nccl/lib
mkdir -p /tmp/dogfight5c-nccl
ln -sf "$NCCL_DIR/libnccl.so.2" /tmp/dogfight5c-nccl/libnccl.so
export LIBRARY_PATH=/tmp/dogfight5c-nccl
export LD_LIBRARY_PATH=/tmp/dogfight5c-nccl:$NCCL_DIR

./build.sh dogfight --local
./build.sh dogfight --float
./build.sh dogfight

PYTHONDONTWRITEBYTECODE=1 \
  /home/claude/dogfight5/.venv/bin/python \
  -m pytest -p no:cacheprovider ocean/dogfight/tests -q
```

Result:

```text
30 passed, 3 skipped in 26.52s
```

All three builds passed. Existing compiler warnings remain non-fatal:

- two unused-variable warnings in the local build;
- six integer-to-float narrowing warnings in the renderer;
- the existing `nvcc -arch=native` fallback warning.

The Phase 3 tests cover:

- all 21 curriculum-stage table and geometry invariants;
- stage 8's max-step boundary;
- stage 17 hard-turn/weave behavior;
- stage 18 crossing behavior;
- stage 19 evasive behavior;
- stage 20 AutoAce behavior and a finite scripted trace;
- deterministic fixed/rehearsal stage assignment;
- training warm-start ordering and exact checkpoint sizes;
- manifest schema, topology, hashes, lineage, and atomic publication.

## Generic warm-start and self-play pool proof

The native trainer now resolves and loads `base.load_model_path` immediately
after policy creation and before `Selfplay` constructs or saves its step-zero
pool entry. Async actor weights are synchronized before rollout.

Checkpoint loads require the exact expected byte count. For the accepted
26-observation, 5-action, hidden-64, three-layer topology:

```text
weights: 38,917
bytes:   155,668
```

A 155,669-byte trailing-data file and a 155,664-byte truncated file both fail
with an expected-versus-actual checkpoint size error.

A zero-learning-rate Dogfight warm-start run reproduced the supplied baseline
checkpoint exactly:

```text
input SHA256:  1677049ecd7d0846f248cdea98860b2e4700f3e2e6fa05a5b4264ffe975d6fa3
output SHA256: 1677049ecd7d0846f248cdea98860b2e4700f3e2e6fa05a5b4264ffe975d6fa3
```

Dogfight intentionally exposes one agent until Phase 4, so the complete
self-play pool runtime invariant was exercised with current-5c Robocode's
two-agent adapter. This tests the same generic native trainer path.

Working proof command:

```bash
./build.sh robocode

./puffer train robocode \
  base.run_id=phase3_warmstart_fixture \
  base.async=0 \
  base.cudagraphs=-1 \
  vec.total_agents=64 \
  vec.num_buffers=1 \
  vec.num_threads=1 \
  vec.num_frozen_banks=0 \
  vec.frozen_bank_pct=0 \
  policy.hidden_size=16 \
  policy.num_layers=1 \
  train.total_timesteps=1024 \
  train.horizon=16 \
  train.minibatch_size=64 \
  train.gpus=1 \
  selfplay.enabled=0 \
  selfplay.eval_games=0

./puffer train robocode \
  base.run_id=phase3_warmstart_pool \
  base.load_model_path=checkpoints/robocode/phase3_warmstart_fixture/0000000000001024.bin \
  base.async=0 \
  base.cudagraphs=-1 \
  vec.total_agents=64 \
  vec.num_buffers=1 \
  vec.num_threads=1 \
  vec.num_frozen_banks=1 \
  vec.frozen_bank_pct=0.5 \
  vec.frozen_bank_hidden_size=16 \
  vec.frozen_bank_num_layers=1 \
  policy.hidden_size=16 \
  policy.num_layers=1 \
  train.total_timesteps=1024 \
  train.horizon=16 \
  train.minibatch_size=64 \
  train.gpus=1 \
  selfplay.enabled=1 \
  selfplay.eval_games=0 \
  selfplay.opp_timeout_steps=100000000

sha256sum \
  checkpoints/robocode/phase3_warmstart_fixture/0000000000001024.bin \
  checkpoints/robocode/phase3_warmstart_pool/0000000000000000.bin

cmp \
  checkpoints/robocode/phase3_warmstart_fixture/0000000000001024.bin \
  checkpoints/robocode/phase3_warmstart_pool/0000000000000000.bin
```

Result:

```text
input SHA256: ada41aae35528f5a3cf33c076c0c0d2b7b9e781a2baabd801a4d23208d0d489c
pool0 SHA256: ada41aae35528f5a3cf33c076c0c0d2b7b9e781a2baabd801a4d23208d0d489c
byte compare: MATCH
```

## Baseline curriculum result

Baseline artifact:

```text
checkpoints/dogfight/phase3_encoder_baseline/0000000057671680.bin
SHA256 1677049ecd7d0846f248cdea98860b2e4700f3e2e6fa05a5b4264ffe975d6fa3
```

Resolved configuration:
`logs/dogfight/phase3_encoder_baseline.ini`

Important effective settings:

```text
total agents:       4096
buffers:            4
threads:            8
horizon:            64
minibatch:          4096
hidden size:        64
recurrent layers:   3
total steps:        57,671,680
learning rate:      0.0025
LR schedule:        linear anneal to zero
max stage:          11
async:              disabled
recurrent carry:    enabled
```

Mastery milestones:

| Stage | Global agent steps |
|---:|---:|
| 0 | 29,884,416 |
| 1 | 30,146,560 |
| 2 | 30,932,992 |
| 3 | 31,195,136 |
| 4 | 31,457,280 |
| 5 | 31,719,424 |
| 6 | 32,243,712 |
| 7 | 33,030,144 |
| 8 | 34,340,864 |
| 9 | 35,913,728 |
| 10 | 45,613,056 |
| 11 | 49,545,216 |

This satisfies Phase 3's pinned stage-10/11 learning gate with the stock
current-5c encoder.

## Accepted segmented lineage

All accepted segments used the same seed, topology, 4096 agents, four buffers,
64-step horizon, 4096 minibatch size, and recurrent carry. Each warm start
restarted optimizer state and therefore is a warm start, not an exact trainer
resume.

| Artifact | Parent | Segment steps | Cumulative steps | Stage protocol | LR decision | SHA256 |
|---|---|---:|---:|---|---|---|
| `phase3_encoder_baseline/0000000057671680.bin` | none | 57,671,680 | 57,671,680 | native climb through 11 | 0.0025, linearly annealed | `1677049ecd7d0846f248cdea98860b2e4700f3e2e6fa05a5b4264ffe975d6fa3` |
| `phase3_stage13_segment_a/0000000004194304.bin` | baseline | 4,194,304 | 61,865,984 | fixed 13 | fixed 0.00025 | `ef6f9381107d02738f98e67e8b28dc9ceae88a58ae88f3203d8bac30da182654` |
| `phase3_stage13_segment_c/0000000008388608.bin` | stage13 A | 8,388,608 | 70,254,592 | fixed 13 | fixed 0.0001 | `0b7a9cf9e243a012e04735d8783c6c9338c9cf8a3093136cfa7227235aa89b54` |
| `phase3_release_gate_20260728_a/0000000016777216.bin` | stage13 C | 16,777,216 | 87,031,808 | stage 20 plus stage 11 at stride 8 | fixed 0.00005 | `525452dacc75e20ca22911a5c684d3b4428b234d2c537b32a8b92a72ae39ce2d` |

Deterministic exact repeats:

| Run A | Run B | Result |
|---|---|---|
| `phase3_stage13_segment_a` | `phase3_stage13_segment_b` | byte-identical |
| `phase3_stage13_segment_c` | `phase3_stage13_segment_d` | byte-identical |
| `phase3_stage20_mix11_a` | `phase3_stage20_mix11_b` | byte-identical |
| `phase3_release_gate_20260728_a` | `phase3_release_gate_20260728_b` | byte-identical |

The earlier `phase3_stage20_mix11_a` pair had SHA256
`ebc7bd2109c9bbaaece1b2d8cf77f1689f870c17daf9371752332b9e244b8169`.
After the final native rebuild, two fresh runs instead produced the identical
SHA256 `525452dacc75e20ca22911a5c684d3b4428b234d2c537b32a8b92a72ae39ce2d`.
Their resolved configurations match except for run ID and equivalent textual
formatting of zero. This is recorded as build-boundary drift. Current execution
is deterministic, and the fresh evaluation remained behaviorally comparable,
so the fresh pair is the release evidence.

The final release training log reported approximately:

```text
avg_stage=19.045
avg_control_rate=2.850
player_ground=0.014
```

The 19.045 average confirms that the stage-20/stage-11 deterministic mix was
actually exercised.

## Rejected experiments

These artifacts remain useful negative evidence but must not be used as the
accepted lineage.

| Artifact | Experiment | Reason rejected | SHA256 |
|---|---|---|---|
| `phase3_stage20_segment_a/0000000008388608.bin` | stage13 C to fixed stage 20, 8,388,608 steps, fixed LR 0.00005 | stage 11 regressed to performance 0.733 and score 0.599 | `2b689581b3193482d94626faaf357beff52aab0fdd3cca8ddd3c23bdf363c0c6` |
| `phase3_stage11_rehearsal_a/0000000002097152.bin` | rejected stage20 artifact to fixed stage 11, 2,097,152 steps, fixed LR 0.00002 | stage 11 remained at performance 0.717 and score 0.575 | `c71dcc1a782249da54235486158b5666e39f796d60fff9af75ab652e38c16e4b` |

This is why the accepted stage-20 run used simultaneous deterministic
rehearsal rather than a later repair segment.

## Independent evaluation matrix

Command form that worked:

```bash
./puffer eval dogfight \
  base.load_model_path=checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
  base.num_games=1024 \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.fixed_stage=STAGE
```

Results:

| Stage | Performance | Score |
|---:|---:|---:|
| 0 | 1.0000 | 1.000 |
| 1 | 0.6646 | 0.497 |
| 2 | 0.8815 | 0.822 |
| 3 | 1.0000 | 1.000 |
| 4 | 1.0000 | 1.000 |
| 5 | 1.0000 | 1.000 |
| 6 | 1.0000 | 1.000 |
| 7 | 0.5712 | 0.357 |
| 8 | 1.0000 | 1.000 |
| 9 | 1.0000 | 1.000 |
| 10 | 1.0000 | 1.000 |
| 11 | 1.0000 | 1.000 |
| 12 | 0.9971 | 0.998 |
| 13 | 0.9475 | 0.945 |
| 14 | 0.9698 | 0.970 |
| 15 | 0.9766 | 0.969 |
| 16 | 0.9708 | 0.961 |
| 17 | 1.0000 | 1.000 |
| 18 | 1.0000 | 1.000 |
| 19 | 1.0000 | 1.000 |
| 20 | 0.9134 | 0.827 |

The accepted advanced policy is not an all-stage-mastered artifact. It retains
the Phase 3 stage-10/11 gate and strong advanced-stage performance, but it
forgets substantial behavior at stages 1, 2, and 7. That is recorded curriculum
quality debt, not a reason to extend Phase 3 beyond its written gate.

## Human-visible flight gate

Working command:

```bash
DISPLAY=:0 timeout 20s ./puffer render dogfight \
  base.load_model_path=checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
  policy.hidden_size=64 \
  policy.num_layers=3 \
  env.fixed_stage=20
```

The release artifact's numerical stage-20 evaluation reported:

```text
performance=0.9134
score=0.827
```

Human assessment:

- the policy is combat-capable and winning;
- it works but does not fly great;
- it remains twitchy and is not yet flying especially well;
- flight stability is quality debt, not the current Phase 3 blocker.

Flight behavior remains a permanent gate. Phase 4 and later changes must keep
running physics tests, training determinism checks, independent evaluation, and
human-visible `DISPLAY=:0` renders.

## Manifest-last publication

`ocean/dogfight/checkpoint_manifest.py` publishes
`<checkpoint>.manifest.json` only after validating the complete payload. It
uses a temporary file, `fsync`, and atomic replacement so the sidecar is the
last artifact published.

The tool validates:

- schema version;
- environment observation/action topology;
- policy hidden size and recurrent layer count;
- exact weight count and byte count;
- payload SHA256;
- cumulative global agent steps;
- parent lineage;
- full commit-shaped behavior version;
- explicit learning-rate schedule decision.

Do not replace `PHASE3_COMMIT` below with a pre-Phase-3 commit. After the Phase
3 behavior is committed, publish and validate the lineage in order:

```bash
PHASE3_COMMIT=FULL_PHASE3_COMMIT_SHA
PUFFERLIB_COMMIT=ebf5ed03cc3524076b6c1a4033bd69cec0b3db22

python ocean/dogfight/checkpoint_manifest.py publish \
  checkpoints/dogfight/phase3_encoder_baseline/0000000057671680.bin \
  --dogfight-behavior-version "$PHASE3_COMMIT" \
  --pufferlib-commit "$PUFFERLIB_COMMIT" \
  --global-agent-steps 57671680 \
  --curriculum-stage 11 \
  --hidden-size 64 \
  --num-layers 3 \
  --lr-schedule-decision "0.0025 linearly annealed to zero over 57671680 steps" \
  --from-scratch

python ocean/dogfight/checkpoint_manifest.py publish \
  checkpoints/dogfight/phase3_stage13_segment_a/0000000004194304.bin \
  --dogfight-behavior-version "$PHASE3_COMMIT" \
  --pufferlib-commit "$PUFFERLIB_COMMIT" \
  --global-agent-steps 61865984 \
  --curriculum-stage 13 \
  --hidden-size 64 \
  --num-layers 3 \
  --lr-schedule-decision "fixed 0.00025 for 4194304 warm-start steps" \
  --parent checkpoints/dogfight/phase3_encoder_baseline/0000000057671680.bin.manifest.json

python ocean/dogfight/checkpoint_manifest.py publish \
  checkpoints/dogfight/phase3_stage13_segment_c/0000000008388608.bin \
  --dogfight-behavior-version "$PHASE3_COMMIT" \
  --pufferlib-commit "$PUFFERLIB_COMMIT" \
  --global-agent-steps 70254592 \
  --curriculum-stage 13 \
  --hidden-size 64 \
  --num-layers 3 \
  --lr-schedule-decision "fixed 0.0001 for 8388608 warm-start steps" \
  --parent checkpoints/dogfight/phase3_stage13_segment_a/0000000004194304.bin.manifest.json

python ocean/dogfight/checkpoint_manifest.py publish \
  checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin \
  --dogfight-behavior-version "$PHASE3_COMMIT" \
  --pufferlib-commit "$PUFFERLIB_COMMIT" \
  --global-agent-steps 87031808 \
  --curriculum-stage 20 \
  --hidden-size 64 \
  --num-layers 3 \
  --lr-schedule-decision "fixed 0.00005 for 16777216 warm-start steps; stage 11 rehearsal stride 8" \
  --parent checkpoints/dogfight/phase3_stage13_segment_c/0000000008388608.bin.manifest.json

python ocean/dogfight/checkpoint_manifest.py validate \
  checkpoints/dogfight/phase3_encoder_baseline/0000000057671680.bin
python ocean/dogfight/checkpoint_manifest.py validate \
  checkpoints/dogfight/phase3_stage13_segment_a/0000000004194304.bin
python ocean/dogfight/checkpoint_manifest.py validate \
  checkpoints/dogfight/phase3_stage13_segment_c/0000000008388608.bin
python ocean/dogfight/checkpoint_manifest.py validate \
  checkpoints/dogfight/phase3_release_gate_20260728_a/0000000016777216.bin
```

These commands are dated evidence from the pinned 2026-07-28 checkout, not
timeless CLI documentation. Future agents must inspect current help and
resolved `.ini` logs, try the recorded command, and adjust only for verified
upstream CLI changes.

## Phase 3 exit gate

| Gate | Result |
|---|---|
| Reproduce pinned stage-10/11 learning | PASS |
| Supplied weights load before self-play initialization | PASS |
| Initial self-play pool entry hash matches supplied weights | PASS |
| Incompatible and trailing checkpoint bytes fail loudly | PASS |
| Every continuation has an explicit LR decision | PASS |
| Fixed/segmented curriculum control exists | PASS |
| Stages 17-20 geometry/progress behavior is tested | PASS |
| Local, float, and native builds pass | PASS |
| Full Dogfight regression suite passes | PASS |
| Human-visible advanced-stage flight gate | CONDITIONAL PASS: winning but twitchy |
| Accepted manifests published against Phase 3 commit | POST-COMMIT ACTION |

No Phase 4 two-agent or Phase 5 self-play environment behavior was pulled
forward into this phase.
