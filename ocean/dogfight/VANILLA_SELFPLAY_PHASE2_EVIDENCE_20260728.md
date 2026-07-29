# Vanilla Self-Play Phase 2 Evidence - 2026-07-28

## Scope

Phase 2 is the smallest official PufferLib 5c self-play canary:

- two native trainable Dogfight agents per match
- PufferLib's built-in frozen-policy bank and historical-opponent pool
- no custom `selfplay_coordinator`
- no PufferLib core changes
- fixed stage 0
- 4,096 agent rows
- 8,388,608 agent steps
- seed 42

This is a topology and throughput canary. It is not a claim that 8M steps
produce a useful fighter policy.

## TDD Gate

Focused profile tests:

```bash
source /home/claude/PufferLib/.venv/bin/activate
pytest -q ocean/dogfight/tests \
    -k 'vanilla_selfplay_profile or two_agent_adapter'
```

Result:

```text
3 passed, 21 deselected
```

Full Dogfight suite:

```bash
source /home/claude/PufferLib/.venv/bin/activate
pytest -q ocean/dogfight/tests
```

Result:

```text
48 passed, 3 skipped
```

Builds:

```bash
./build.sh dogfight
bash ocean/dogfight/build_eval.sh
```

Both completed successfully before the canary.

## Training Commands

Dry run:

```bash
bash ocean/dogfight/train_vanilla_selfplay.sh \
    --dry-run vanilla_phase2_native_8m_s42_20260728 8388608 4096
```

GPU runs require PufferLib's NCCL directory:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

bash ocean/dogfight/train_vanilla_selfplay.sh \
    vanilla_phase2_native_8m_s42_20260728 8388608 4096

bash ocean/dogfight/train_vanilla_selfplay.sh \
    vanilla_phase2_native_8m_s42_repeat_20260728 8388608 4096
```

The first attempted repeat omitted the NCCL directory and failed before
training with `libnccl.so.2` not found. It changed no source and produced no
training result.

## Native Pool Evidence

Each successful run produced policy checkpoints at:

```text
0
2,097,152
4,194,304
6,291,456
8,388,608 agent steps
```

That is the configured initial generation plus four successive generations
made available on the same cadence as the official historical-opponent pool.
The Dogfight two-agent contract raises `boundary_reached` only for tagged
historical matches, matching Robocode's contract.

There was no crash, deadlock, buffer fault, or incomplete run.

## Throughput

Run 1:

```text
8,388,608 agent steps
9.314 seconds
about 900,645 end-to-end agent SPS
about 450,322 match steps/second
1.9 GiB VRAM
```

Run 2:

```text
8,388,608 agent steps
8.830 seconds
about 950,012 end-to-end agent SPS
about 475,006 match steps/second
1.9 GiB VRAM
```

The wall-time spread was about 5.3%. Dashboard SPS is an instantaneous
measurement and must not replace end-to-end steps divided by wall time.

## Seat Symmetry

Both runs finished with effectively symmetric seats.

Run 1:

```text
slot_0_score 0.499
slot_1_score 0.501
draw_rate 0.746
slot_0_gun_kills 0.126
slot_1_gun_kills 0.128
```

Run 2:

```text
slot_0_score 0.499
slot_1_score 0.501
draw_rate 0.746
slot_0_gun_kills 0.126
slot_1_gun_kills 0.128
```

The matching aggregate results and near-50/50 seats are evidence against a
slot or physical-plane advantage. They are not evidence of combat mastery.

## Determinism Boundary

The initial and 2,097,152-step checkpoints were byte-identical across the two
runs:

```text
initial:
cd2e3ad836c505c6d5aeb9a8631913f4fd10bb8be65a469ee0e970010891d010

2,097,152:
bfe64a12255bb2933e85523391fe4568d26b76b90a33e0fe9ed288f9ccd5f85f
```

The runs diverged at 4,194,304 steps, after historical-opponent rotation
began. Final hashes were:

```text
run 1:
75cbdba7c61e6bdd550ddeed862c1e894eb24e89f4936d943a16b364a6f102ad

run 2:
4b702b8397a67b88035fdcf84c8a657eb0f81d34c0c0d4474af4c0f0564d1287
```

Therefore:

- deterministic initialization and current-current training are preserved
- official historical-pool routing is not byte-deterministic in this
  multithreaded profile
- the repeated aggregate behavior remained nearly identical
- future work must retain exact deterministic non-pool guard runs and use
  bounded metric variance, rather than identical hashes, for native pool runs

Do not describe the full native self-play run as byte-deterministic.

## Automated Evaluation

The primary final checkpoint was evaluated against the fixed scripted stage-0
opponent:

```bash
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

bash ocean/dogfight/eval_checkpoint.sh headless \
    checkpoints/dogfight/vanilla_phase2_native_8m_s42_20260728/0000000008388608.bin \
    0 16 42
```

Result:

```text
2/16 wins
perf 0.125
score -0.406
```

This is substantially below the strong one-agent Phase 0 baseline and is not
an acceptable learned policy.

## Human-in-the-Loop Evaluation

Command:

```bash
DISPLAY=:0 \
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-} \
bash ocean/dogfight/eval_checkpoint.sh visible \
    checkpoints/dogfight/vanilla_phase2_native_8m_s42_20260728/0000000008388608.bin \
    0 8 42
```

Automated result:

```text
0/8 wins
```

Human verdict:

```text
always rolling right, flies like shit
```

This is a hard flight-quality failure. Do not promote, warm-start from, or use
this checkpoint as a skill baseline.

## Phase 2 Verdict

Structural gate: **PASS**

- official PufferLib native self-play launches and completes
- historical generations are produced
- tagged historical matches use the required boundary contract
- seats are symmetric
- throughput is roughly 0.90M-0.95M end-to-end agent SPS
- PufferLib core remains untouched

Policy-quality gate: **FAIL**

- only 12.5% headless wins against scripted stage 0
- 0/8 visible wins
- persistent right-roll behavior

The same build produced a strong, visually acceptable one-agent Phase 1
policy, so the evidence points to an undertrained or collapsed self-play
policy rather than an RK4 or evaluator regression. Phase 3 must improve and
measure the training configuration before any Dogfight hot-path rewrite.
