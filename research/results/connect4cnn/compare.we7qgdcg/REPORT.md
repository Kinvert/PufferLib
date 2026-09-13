# Connect4: state versus pixels

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials.
The input representation and encoder differ; parameters/FLOPs are not matched. This is a common-recipe comparison.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 73 | 13,279,232 | 21.88% | -0.5426 | 1056 | 55,552 | 121.865 | 108,967 | 109,330 |
| tiny_cnn | 73 | 13,279,232 | 65.91% | 0.3286 | 1053 | 151,680 | 135.649 | 97,894 | 98,136 |
| tiny_cnn | 74 | 13,279,232 | 66.89% | 0.3541 | 1045 | 151,680 | 135.361 | 98,102 | 98,318 |
| state | 74 | 13,279,232 | 18.56% | -0.6057 | 1083 | 55,552 | 121.714 | 109,102 | 109,354 |
| state | 75 | 13,279,232 | 15.09% | -0.6762 | 1047 | 55,552 | 122.474 | 108,425 | 108,718 |
| tiny_cnn | 75 | 13,279,232 | 59.32% | 0.2082 | 1148 | 151,680 | 136.614 | 97,203 | 97,448 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| state | 73 | 3,319,808 | 0.35% | 32.914 |
| state | 73 | 6,639,616 | 2.98% | 63.550 |
| state | 73 | 9,959,424 | 17.17% | 92.983 |
| state | 73 | 13,279,232 | 21.88% | 121.787 |
| tiny_cnn | 73 | 3,319,808 | 13.47% | 36.685 |
| tiny_cnn | 73 | 6,639,616 | 56.62% | 69.794 |
| tiny_cnn | 73 | 9,959,424 | 64.52% | 102.666 |
| tiny_cnn | 73 | 13,279,232 | 65.91% | 135.550 |
| tiny_cnn | 74 | 3,319,808 | 8.29% | 37.061 |
| tiny_cnn | 74 | 6,639,616 | 51.41% | 70.021 |
| tiny_cnn | 74 | 9,959,424 | 65.56% | 102.549 |
| tiny_cnn | 74 | 13,279,232 | 66.89% | 135.294 |
| state | 74 | 3,319,808 | 0.44% | 32.583 |
| state | 74 | 6,639,616 | 1.64% | 63.423 |
| state | 74 | 9,959,424 | 13.35% | 92.555 |
| state | 74 | 13,279,232 | 18.56% | 121.643 |
| state | 75 | 3,319,808 | 0.44% | 32.677 |
| state | 75 | 6,639,616 | 3.47% | 63.638 |
| state | 75 | 9,959,424 | 13.45% | 93.130 |
| state | 75 | 13,279,232 | 15.09% | 122.382 |
| tiny_cnn | 75 | 3,319,808 | 5.58% | 37.498 |
| tiny_cnn | 75 | 6,639,616 | 43.69% | 71.175 |
| tiny_cnn | 75 | 9,959,424 | 57.58% | 103.919 |
| tiny_cnn | 75 | 13,279,232 | 59.32% | 136.487 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Trials alternate policy order by seed to reduce systematic order effects. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
