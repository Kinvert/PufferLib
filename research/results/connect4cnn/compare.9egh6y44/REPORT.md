# Connect4 training results

Stock Connect4 training configuration, float32, one GPU, serial seeds. Evaluation uses the common 64-agent setup. This is a separately configured baseline, not a matched-hyperparameter comparison.
Requested decisions: 13,272,299; expected completed decisions: 13,238,272.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_state | 73 | 13,238,272 | 98.51% | 0.9703 | 1076 | 209,408 | 41.913 | 315,852 | 320,983 |
| stock_state | 74 | 13,238,272 | 99.06% | 0.9820 | 1164 | 209,408 | 40.634 | 325,793 | 330,723 |
| stock_state | 75 | 13,238,272 | 99.04% | 0.9809 | 1151 | 209,408 | 47.695 | 277,564 | 281,410 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| stock_state | 73 | 3,407,872 | 87.98% | 11.464 |
| stock_state | 73 | 6,815,744 | 96.14% | 21.548 |
| stock_state | 73 | 10,223,616 | 98.42% | 32.296 |
| stock_state | 73 | 13,238,272 | 98.51% | 41.780 |
| stock_state | 74 | 3,407,872 | 76.88% | 10.726 |
| stock_state | 74 | 6,815,744 | 96.76% | 20.890 |
| stock_state | 74 | 10,223,616 | 98.88% | 31.570 |
| stock_state | 74 | 13,238,272 | 99.06% | 40.506 |
| stock_state | 75 | 3,407,872 | 80.33% | 19.678 |
| stock_state | 75 | 6,815,744 | 98.86% | 29.059 |
| stock_state | 75 | 10,223,616 | 99.22% | 38.903 |
| stock_state | 75 | 13,238,272 | 99.04% | 47.531 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Trials alternate policy order by seed to reduce systematic order effects. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
