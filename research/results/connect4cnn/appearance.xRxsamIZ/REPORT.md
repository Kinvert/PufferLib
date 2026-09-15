# Native mixed-appearance canary

16 short training runs and 16 checkpoint reload evaluations. Two checkpoints per run were finite and had expected parameter counts. Paired checkpoints and evaluation records matched byte-for-byte across one/two CPU workers.

Appearance mode 1, seed 12345, 64 slots; assignment CSVs are retained. Training seed 56173, evaluation action seed 66173. Game RNG remains the native per-slot stream. All four encoders use the same per-game learner recipe and 16,384 decisions. This is a reproducibility check, not learning or speed evidence.

Evaluation is pooled v1, may overshoot 64 games and is bounded by a 30-second process timeout. This does not fix sparse Pong match completion or implement paper-grade exact episode quotas. A timeout makes the canary fail; it is never a zero score.

| Game | Model | Workers | Train s | Process SPS | Eval s | Eval perf |
|---|---|---:|---:|---:|---:|---:|
| connect4cnn | flex_quality | 1 | 1.40 | 11,703 | 0.44 | 0.00% |
| connect4cnn | flex_quality | 2 | 0.58 | 28,248 | 0.41 | 0.00% |
| connect4cnn | nature_cnn | 1 | 0.71 | 23,076 | 0.47 | 0.00% |
| connect4cnn | nature_cnn | 2 | 0.62 | 26,426 | 0.41 | 0.00% |
| connect4cnn | impala_cnn | 1 | 2.08 | 7,877 | 0.55 | 0.00% |
| connect4cnn | impala_cnn | 2 | 1.97 | 8,317 | 0.50 | 0.00% |
| connect4cnn | impoola_cnn | 1 | 2.09 | 7,839 | 0.53 | 0.00% |
| connect4cnn | impoola_cnn | 2 | 1.98 | 8,275 | 0.51 | 0.00% |
| pongcnn | flex_quality | 1 | 0.46 | 35,617 | 0.46 | 0.00% |
| pongcnn | flex_quality | 2 | 0.47 | 34,860 | 0.45 | 0.00% |
| pongcnn | nature_cnn | 1 | 0.49 | 33,437 | 0.48 | 0.00% |
| pongcnn | nature_cnn | 2 | 0.49 | 33,437 | 0.46 | 0.00% |
| pongcnn | impala_cnn | 1 | 1.86 | 8,809 | 0.94 | 0.00% |
| pongcnn | impala_cnn | 2 | 1.88 | 8,715 | 0.89 | 0.00% |
| pongcnn | impoola_cnn | 1 | 1.88 | 8,715 | 0.95 | 0.07% |
| pongcnn | impoola_cnn | 2 | 1.88 | 8,715 | 0.92 | 0.07% |

Process SPS includes startup/writes and uses /usr/bin/time's rounded timer; native timing/SPS remain in each metrics directory. Source snapshots/checks, resolved native configs, commands, binary/checkpoint hashes and raw failures are retained. Large artifacts remain local.
