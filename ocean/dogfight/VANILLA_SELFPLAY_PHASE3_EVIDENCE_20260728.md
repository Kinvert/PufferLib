# Vanilla Self-Play Phase 3 Evidence - 2026-07-28

## Purpose

Phase 3 scales the official PufferLib native self-play configuration before
considering Dogfight hot-path rewrites. All runs used:

- unchanged RK4 physics
- fixed stage 0
- two native policy rows per match
- four vector buffers
- eight vector threads
- seed 42
- PufferLib's official historical-policy pool
- no PufferLib core changes

The 8,388,608-step canaries varied only `vec.total_agents`.

## Scaling Matrix

| Agent rows | Trainer uptime | Trainer-average SPS | Process wall time | Process-wall SPS | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 12.077 s | about 0.695M | 17.93 s | about 0.468M | 795,732 KiB |
| 4,096 | 8.830-9.314 s | about 0.90M-0.95M | about 12.87 s observed | about 0.65M | not separately measured |
| 8,192 | 5.711 s | about 1.469M | 8.43 s | about 0.995M | 800,340 KiB |
| 16,384 | 4.434 s | about 1.892M | 6.10 s | about 1.375M | 808,936 KiB |

The dashboard's terminal instantaneous SPS at 16,384 rows was 2.1M.
Trainer-average SPS is total steps divided by dashboard uptime. Process-wall
SPS includes Python, CUDA, model, and environment startup and is the honest
command-to-command throughput.

## Scaling Verdict

16,384 agent rows was the best tested point:

- it is also Robocode's official scale
- it more than doubled process-wall throughput relative to the 4,096-row
  canary
- memory growth was negligible
- slot symmetry remained approximately 50/50
- no crash or buffer failure occurred

Do not quote the dashboard's 2.1M instantaneous SPS as end-to-end throughput.

## 128M From-Scratch Run

Command:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

bash ocean/dogfight/train_vanilla_selfplay.sh \
    vanilla_phase3_16k_128m_s42_20260728 \
    134217728 \
    16384
```

Result:

```text
process wall time: 71.62 seconds
trainer uptime: 69.288 seconds
process-wall SPS: about 1.874M
trainer-average SPS: about 1.937M
terminal instantaneous SPS: 2.3M
peak RSS: 809,140 KiB
```

Final self-play metrics:

```text
perf 0.515
score 0.273
slot_0_score 0.503
slot_1_score 0.497
draw_rate 0.485
slot_0_gun_kills 0.261
player_ground 0.000
opponent_ground 0.000
```

Compared with the 8M canary, decisive fights rose from about 25% to 51.5% and
draws fell from about 74.6% to 48.5%. Seats remained symmetric.

## Fixed-Opponent Evaluation

Checkpoint:

```text
checkpoints/dogfight/vanilla_phase3_16k_128m_s42_20260728/0000000134217728.bin
```

Headless command:

```bash
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-} \
bash ocean/dogfight/eval_checkpoint.sh headless \
    checkpoints/dogfight/vanilla_phase3_16k_128m_s42_20260728/0000000134217728.bin \
    0 16 42
```

Result:

```text
5/16 wins
perf 0.3125
score -0.156
```

This improved on the 8M self-play canary's 2/16 result but remained far below
the known-good one-agent stage-0 baseline.

## Human Flight Evaluation

The first visible evaluation produced 4/8 wins, but the user did not see it.
It was rerun for 16 episodes:

```bash
DISPLAY=:0 \
LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-} \
bash ocean/dogfight/eval_checkpoint.sh visible \
    checkpoints/dogfight/vanilla_phase3_16k_128m_s42_20260728/0000000134217728.bin \
    0 16 42
```

Automated result:

```text
5/16 wins
perf 0.3125
player_ground 0.250
```

Human verdict:

```text
always rolls right; flight remains bad
```

This is a hard flight-quality failure.

## Phase 3 Verdict

Configuration scaling: **PASS**

- 16,384 rows provides about 1.87M process-wall SPS over 128M steps
- terminal instantaneous throughput exceeds the plan's 2M floor
- no Dogfight or PufferLib core optimization was needed to reach that floor

From-scratch policy quality: **FAIL**

- 128M steps did not remove persistent right roll
- scripted stage-0 mastery remained only 31.25%
- self-play's improved internal decisiveness did not translate into robust
  fixed-opponent flight

Do not spend more compute on the identical from-scratch fixed-stage profile.
Stock pinned 5c does not consume `base.load_model_path` when starting native
training, so configuration-only warm-start must not be claimed or attempted.
The next clean-core experiment is Dogfight-local progressive asymmetric spawn
difficulty within one uninterrupted official native self-play run. Stronger
historical-opponent exposure can follow if the spawn curriculum alone is
insufficient.
