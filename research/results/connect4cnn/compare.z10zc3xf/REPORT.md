# Connect4 training results

Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. The input representation and encoder differ; parameters/FLOPs are not matched. State uses a modified configuration.
Requested decisions: 65,536; expected completed decisions: 65,536.

## Final results by training seed

| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 65,536 | 0.35% | -0.9366 | 284 | 138,528 | 1.317 | 49,762 | 67,315 |

## Checkpoint evaluation curves

| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |
|---|---:|---:|---:|---:|
| nature_cnn | 73 | 32,768 | 0.35% | 0.805 |
| nature_cnn | 73 | 65,536 | 0.35% | 1.229 |

## Reading these results

- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.
- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.
- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.
- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.
- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.
- Trials alternate policy order by seed to reduce systematic order effects. Short smoke timings are not steady-state speed benchmarks.
- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.
- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions.
