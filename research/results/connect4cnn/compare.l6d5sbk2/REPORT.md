# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. The input representation and encoder differ; parameters/FLOPs are not matched. State uses a modified configuration.
Requested decisions: 13,279,232; expected completed decisions: 13,279,232.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 13,279,232 | 98.99% | 0.9798 | 1090 | 270,496 | 1227.228 | 10,821 | 10,824 |
| impoola_cnn | 73 | 13,279,232 | 61.75% | 0.2350 | 1098 | 151,712 | 1224.202 | 10,847 | 10,851 |
| impoola_cnn | 74 | 13,279,232 | 73.59% | 0.4718 | 1064 | 151,712 | 1228.583 | 10,809 | 10,812 |
| impala_cnn | 74 | 13,279,232 | 99.01% | 0.9801 | 1207 | 270,496 | 1232.505 | 10,774 | 10,778 |
| impala_cnn | 75 | 13,279,232 | 99.57% | 0.9914 | 1169 | 270,496 | 1229.103 | 10,804 | 10,808 |
| impoola_cnn | 75 | 13,279,232 | 73.14% | 0.4628 | 1076 | 151,712 | 1224.713 | 10,843 | 10,846 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| impala_cnn | 73 | 3,319,808 | 82.99% | 307.928 |
| impala_cnn | 73 | 6,639,616 | 97.67% | 613.967 |
| impala_cnn | 73 | 9,959,424 | 98.63% | 920.561 |
| impala_cnn | 73 | 13,279,232 | 98.99% | 1227.112 |
| impoola_cnn | 73 | 3,319,808 | 40.29% | 308.862 |
| impoola_cnn | 73 | 6,639,616 | 58.78% | 614.105 |
| impoola_cnn | 73 | 9,959,424 | 62.17% | 919.055 |
| impoola_cnn | 73 | 13,279,232 | 61.75% | 1224.054 |
| impoola_cnn | 74 | 3,319,808 | 70.26% | 308.804 |
| impoola_cnn | 74 | 6,639,616 | 72.46% | 615.544 |
| impoola_cnn | 74 | 9,959,424 | 72.62% | 921.959 |
| impoola_cnn | 74 | 13,279,232 | 73.59% | 1228.447 |
| impala_cnn | 74 | 3,319,808 | 86.94% | 309.059 |
| impala_cnn | 74 | 6,639,616 | 98.50% | 616.882 |
| impala_cnn | 74 | 9,959,424 | 98.84% | 924.558 |
| impala_cnn | 74 | 13,279,232 | 99.01% | 1232.393 |
| impala_cnn | 75 | 3,319,808 | 84.69% | 308.717 |
| impala_cnn | 75 | 6,639,616 | 99.40% | 615.124 |
| impala_cnn | 75 | 9,959,424 | 99.49% | 922.007 |
| impala_cnn | 75 | 13,279,232 | 99.57% | 1228.983 |
| impoola_cnn | 75 | 3,319,808 | 71.62% | 307.808 |
| impoola_cnn | 75 | 6,639,616 | 74.07% | 613.647 |
| impoola_cnn | 75 | 9,959,424 | 72.85% | 919.282 |
| impoola_cnn | 75 | 13,279,232 | 73.14% | 1224.581 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- When several policies are selected, trials alternate policy order by seed. Separate invocations are not interleaved. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
