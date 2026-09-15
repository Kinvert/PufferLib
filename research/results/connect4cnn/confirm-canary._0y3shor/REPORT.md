# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. Encoders differ; parameters/FLOPs are not matched. State, if included, uses a modified configuration.
Requested decisions: 65,536; expected completed decisions: 65,536.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 9173 | 65,536 | 0.00% | -0.9599 | 299 | 160,736 | 1.267 | 51,720 | 73,646 |
| nature_cnn | 9173 | 65,536 | 0.00% | -0.9549 | 288 | 138,528 | 1.267 | 51,736 | 69,956 |
| flex_fast | 9173 | 65,536 | 0.00% | -0.9571 | 303 | 261,856 | 1.267 | 51,729 | 70,717 |
| impala_cnn | 9173 | 65,536 | 0.00% | -0.9615 | 286 | 270,496 | 6.680 | 9,811 | 10,488 |
| flex_small | 9173 | 65,536 | 0.00% | -0.9539 | 304 | 109,768 | 1.267 | 51,735 | 72,901 |
| impoola_cnn | 9173 | 65,536 | 0.00% | -0.9593 | 295 | 151,712 | 6.630 | 9,885 | 10,517 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| flex_quality | 9173 | 16,384 | 0.00% | 0.584 |
| flex_quality | 9173 | 32,768 | 0.00% | 0.776 |
| flex_quality | 9173 | 49,152 | 0.00% | 0.968 |
| flex_quality | 9173 | 65,536 | 0.00% | 1.160 |
| nature_cnn | 9173 | 16,384 | 0.00% | 0.588 |
| nature_cnn | 9173 | 32,768 | 0.00% | 0.792 |
| nature_cnn | 9173 | 49,152 | 0.00% | 0.988 |
| nature_cnn | 9173 | 65,536 | 0.00% | 1.188 |
| flex_fast | 9173 | 16,384 | 0.00% | 0.598 |
| flex_fast | 9173 | 32,768 | 0.00% | 0.798 |
| flex_fast | 9173 | 49,152 | 0.00% | 0.994 |
| flex_fast | 9173 | 65,536 | 0.00% | 1.194 |
| impala_cnn | 9173 | 16,384 | 0.00% | 1.981 |
| impala_cnn | 9173 | 32,768 | 0.00% | 3.510 |
| impala_cnn | 9173 | 49,152 | 0.00% | 5.042 |
| impala_cnn | 9173 | 65,536 | 0.00% | 6.566 |
| flex_small | 9173 | 16,384 | 0.00% | 0.595 |
| flex_small | 9173 | 32,768 | 0.00% | 0.783 |
| flex_small | 9173 | 49,152 | 0.00% | 0.975 |
| flex_small | 9173 | 65,536 | 0.00% | 1.171 |
| impoola_cnn | 9173 | 16,384 | 0.00% | 1.942 |
| impoola_cnn | 9173 | 32,768 | 0.00% | 3.466 |
| impoola_cnn | 9173 | 49,152 | 0.00% | 4.990 |
| impoola_cnn | 9173 | 65,536 | 0.00% | 6.522 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Confirmation rotates model order by seed according to protocol.json; other comparisons alternate forward/reverse order. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
