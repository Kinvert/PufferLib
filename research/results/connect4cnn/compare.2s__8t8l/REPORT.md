# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. The input representation and encoder differ; parameters/FLOPs are not matched. State uses a modified configuration.
Requested decisions: 13,279,232; expected completed decisions: 13,279,232.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 13,279,232 | 77.10% | 0.5420 | 1096 | 138,528 | 162.552 | 81,692 | 81,900 |
| nature_cnn | 74 | 13,279,232 | 88.33% | 0.7686 | 1037 | 138,528 | 164.117 | 80,913 | 81,076 |
| nature_cnn | 75 | 13,279,232 | 70.68% | 0.4457 | 1095 | 138,528 | 166.168 | 79,915 | 80,068 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| nature_cnn | 73 | 3,319,808 | 13.72% | 43.069 |
| nature_cnn | 73 | 6,639,616 | 75.34% | 83.769 |
| nature_cnn | 73 | 9,959,424 | 76.94% | 123.085 |
| nature_cnn | 73 | 13,279,232 | 77.10% | 162.458 |
| nature_cnn | 74 | 3,319,808 | 4.44% | 42.931 |
| nature_cnn | 74 | 6,639,616 | 69.31% | 84.079 |
| nature_cnn | 74 | 9,959,424 | 88.61% | 124.099 |
| nature_cnn | 74 | 13,279,232 | 88.33% | 164.020 |
| nature_cnn | 75 | 3,319,808 | 15.32% | 43.052 |
| nature_cnn | 75 | 6,639,616 | 48.79% | 84.721 |
| nature_cnn | 75 | 9,959,424 | 64.81% | 125.541 |
| nature_cnn | 75 | 13,279,232 | 70.68% | 166.077 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Trials alternate policy order by seed to reduce systematic order effects. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
