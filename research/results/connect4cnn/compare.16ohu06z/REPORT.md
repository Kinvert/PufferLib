# Connect4: state versus pixels

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials.
The input representation and encoder differ; parameters/FLOPs are not matched. This is a common-recipe comparison.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| state | 73 | 65,536 | 0.00% | -0.9466 | 281 | 55,552 | 1.217 |
| tiny_cnn | 73 | 65,536 | 0.00% | -0.9441 | 286 | 151,680 | 1.167 |
| tiny_cnn | 74 | 65,536 | 0.00% | -0.9401 | 284 | 151,680 | 1.116 |
| state | 74 | 65,536 | 0.00% | -0.9454 | 293 | 55,552 | 1.017 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| state | 73 | 32,768 | 0.00% | 0.793 |
| state | 73 | 65,536 | 0.00% | 1.121 |
| tiny_cnn | 73 | 32,768 | 0.00% | 0.709 |
| tiny_cnn | 73 | 65,536 | 0.00% | 1.081 |
| tiny_cnn | 74 | 32,768 | 0.00% | 0.674 |
| tiny_cnn | 74 | 65,536 | 0.00% | 1.042 |
| state | 74 | 32,768 | 0.00% | 0.627 |
| state | 74 | 65,536 | 0.00% | 0.951 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Trials alternate policy order by seed to reduce systematic order effects. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
