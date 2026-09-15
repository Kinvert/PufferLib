# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. Encoders differ; parameters/FLOPs are not matched. State, if included, uses a modified configuration.
Requested decisions: 65,536; expected completed decisions: 65,536.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 9173 | 65,536 | 0.00% | -0.9599 | 299 | 160,736 | 0.815 | 80,425 | 144,612 |
| nature_cnn | 9173 | 65,536 | 0.00% | -0.9549 | 288 | 138,528 | 0.765 | 85,691 | 140,784 |
| impala_cnn | 9173 | 65,536 | 0.00% | -0.9615 | 286 | 270,496 | 2.818 | 23,256 | 26,027 |
| impoola_cnn | 9173 | 65,536 | 0.00% | -0.9593 | 295 | 151,712 | 2.818 | 23,258 | 25,937 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| flex_quality | 9173 | 16,384 | 0.00% | 0.402 |
| flex_quality | 9173 | 32,768 | 0.00% | 0.503 |
| flex_quality | 9173 | 49,152 | 0.00% | 0.605 |
| flex_quality | 9173 | 65,536 | 0.00% | 0.707 |
| nature_cnn | 9173 | 16,384 | 0.00% | 0.321 |
| nature_cnn | 9173 | 32,768 | 0.00% | 0.426 |
| nature_cnn | 9173 | 49,152 | 0.00% | 0.532 |
| nature_cnn | 9173 | 65,536 | 0.00% | 0.636 |
| impala_cnn | 9173 | 16,384 | 0.00% | 0.847 |
| impala_cnn | 9173 | 32,768 | 0.00% | 1.465 |
| impala_cnn | 9173 | 49,152 | 0.00% | 2.082 |
| impala_cnn | 9173 | 65,536 | 0.00% | 2.696 |
| impoola_cnn | 9173 | 16,384 | 0.00% | 0.842 |
| impoola_cnn | 9173 | 32,768 | 0.00% | 1.464 |
| impoola_cnn | 9173 | 49,152 | 0.00% | 2.084 |
| impoola_cnn | 9173 | 65,536 | 0.00% | 2.702 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Confirmation rotates model order by seed according to protocol.json; other comparisons alternate forward/reverse order. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
