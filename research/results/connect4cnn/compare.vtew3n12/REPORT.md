# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. Encoders differ; parameters/FLOPs are not matched. State, if included, uses a modified configuration.
Requested decisions: 13,312,000; expected completed decisions: 13,312,000.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 173 | 13,312,000 | 71.78% | 0.4357 | 1042 | 160,736 | 77.819 | 171,063 | 171,897 |
| nature_cnn | 173 | 13,312,000 | 66.11% | 0.3296 | 1071 | 138,528 | 83.790 | 158,873 | 159,390 |
| impala_cnn | 173 | 13,312,000 | 96.49% | 0.9298 | 1025 | 270,496 | 461.455 | 28,848 | 28,866 |
| impoola_cnn | 173 | 13,312,000 | 82.77% | 0.6554 | 1126 | 151,712 | 476.149 | 27,958 | 27,974 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| flex_quality | 173 | 1,024,000 | 0.08% | 6.631 |
| flex_quality | 173 | 2,048,000 | 0.37% | 12.929 |
| flex_quality | 173 | 3,072,000 | 12.89% | 19.177 |
| flex_quality | 173 | 4,096,000 | 29.70% | 25.292 |
| flex_quality | 173 | 5,120,000 | 47.85% | 31.312 |
| flex_quality | 173 | 6,144,000 | 64.15% | 37.161 |
| flex_quality | 173 | 7,168,000 | 68.36% | 42.924 |
| flex_quality | 173 | 8,192,000 | 69.89% | 48.657 |
| flex_quality | 173 | 9,216,000 | 72.14% | 53.960 |
| flex_quality | 173 | 10,240,000 | 71.58% | 59.465 |
| flex_quality | 173 | 11,264,000 | 71.74% | 65.560 |
| flex_quality | 173 | 12,288,000 | 71.84% | 71.613 |
| flex_quality | 173 | 13,312,000 | 71.78% | 77.686 |
| nature_cnn | 173 | 1,024,000 | 0.00% | 7.121 |
| nature_cnn | 173 | 2,048,000 | 0.19% | 13.999 |
| nature_cnn | 173 | 3,072,000 | 9.97% | 20.879 |
| nature_cnn | 173 | 4,096,000 | 20.59% | 27.644 |
| nature_cnn | 173 | 5,120,000 | 25.66% | 34.358 |
| nature_cnn | 173 | 6,144,000 | 35.49% | 40.544 |
| nature_cnn | 173 | 7,168,000 | 44.82% | 47.062 |
| nature_cnn | 173 | 8,192,000 | 52.12% | 53.564 |
| nature_cnn | 173 | 9,216,000 | 59.78% | 59.918 |
| nature_cnn | 173 | 10,240,000 | 64.19% | 65.819 |
| nature_cnn | 173 | 11,264,000 | 65.17% | 71.894 |
| nature_cnn | 173 | 12,288,000 | 65.70% | 77.933 |
| nature_cnn | 173 | 13,312,000 | 66.11% | 83.688 |
| impala_cnn | 173 | 1,024,000 | 0.36% | 38.300 |
| impala_cnn | 173 | 2,048,000 | 45.84% | 76.648 |
| impala_cnn | 173 | 3,072,000 | 70.41% | 114.600 |
| impala_cnn | 173 | 4,096,000 | 77.74% | 149.678 |
| impala_cnn | 173 | 5,120,000 | 81.49% | 184.145 |
| impala_cnn | 173 | 6,144,000 | 92.61% | 218.682 |
| impala_cnn | 173 | 7,168,000 | 94.13% | 253.288 |
| impala_cnn | 173 | 8,192,000 | 95.15% | 287.935 |
| impala_cnn | 173 | 9,216,000 | 95.52% | 322.583 |
| impala_cnn | 173 | 10,240,000 | 96.88% | 357.207 |
| impala_cnn | 173 | 11,264,000 | 96.10% | 391.919 |
| impala_cnn | 173 | 12,288,000 | 96.25% | 426.624 |
| impala_cnn | 173 | 13,312,000 | 96.49% | 461.328 |
| impoola_cnn | 173 | 1,024,000 | 0.00% | 34.991 |
| impoola_cnn | 173 | 2,048,000 | 12.48% | 69.653 |
| impoola_cnn | 173 | 3,072,000 | 79.46% | 104.057 |
| impoola_cnn | 173 | 4,096,000 | 83.76% | 138.215 |
| impoola_cnn | 173 | 5,120,000 | 83.35% | 174.193 |
| impoola_cnn | 173 | 6,144,000 | 83.00% | 212.079 |
| impoola_cnn | 173 | 7,168,000 | 82.42% | 249.743 |
| impoola_cnn | 173 | 8,192,000 | 82.79% | 287.570 |
| impoola_cnn | 173 | 9,216,000 | 82.58% | 325.349 |
| impoola_cnn | 173 | 10,240,000 | 82.84% | 363.019 |
| impoola_cnn | 173 | 11,264,000 | 82.86% | 400.620 |
| impoola_cnn | 173 | 12,288,000 | 82.77% | 438.275 |
| impoola_cnn | 173 | 13,312,000 | 82.77% | 476.031 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Confirmation rotates model order by seed according to protocol.json; other comparisons alternate forward/reverse order. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
