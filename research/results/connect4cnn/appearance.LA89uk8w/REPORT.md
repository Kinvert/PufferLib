# Native mixed-appearance canary

16 short training runs and 16 checkpoint reload evaluations. Two checkpoints per run were finite and had expected parameter counts. Paired checkpoints and evaluation records matched byte-for-byte across one/two CPU workers.

Appearance mode 1, seed 12345, 64 slots; assignment CSVs are retained. Training seed 56173, evaluation action seed 66173. Game RNG remains the native per-slot stream. All four encoders use the same per-game learner recipe and 16,384 decisions. This is a reproducibility check, not learning or speed evidence.

Evaluation is pooled v1, may overshoot 64 games and is bounded by a 30-second process timeout. This does not fix sparse Pong match completion or implement paper-grade exact episode quotas. A timeout makes the canary fail; it is never a zero score.

| Game | Model | Workers | Train s | Process SPS | Eval s | Eval perf |
|---|---|---:|---:|---:|---:|---:|
| connect4cnn | flex_quality | 1 | 0.58 | 28,248 | 0.33 | 0.00% |
| connect4cnn | flex_quality | 2 | 0.43 | 38,102 | 0.33 | 0.00% |
| connect4cnn | nature_cnn | 1 | 0.50 | 32,768 | 0.35 | 0.00% |
| connect4cnn | nature_cnn | 2 | 0.42 | 39,010 | 0.33 | 0.00% |
| connect4cnn | impala_cnn | 1 | 1.00 | 16,384 | 0.39 | 0.00% |
| connect4cnn | impala_cnn | 2 | 0.92 | 17,809 | 0.33 | 0.00% |
| connect4cnn | impoola_cnn | 1 | 0.99 | 16,549 | 0.35 | 0.00% |
| connect4cnn | impoola_cnn | 2 | 0.94 | 17,430 | 0.34 | 0.00% |
| pongcnn | flex_quality | 1 | 0.34 | 48,188 | 0.36 | 0.00% |
| pongcnn | flex_quality | 2 | 0.34 | 48,188 | 0.35 | 0.00% |
| pongcnn | nature_cnn | 1 | 0.35 | 46,811 | 0.34 | 0.00% |
| pongcnn | nature_cnn | 2 | 0.35 | 46,811 | 0.35 | 0.00% |
| pongcnn | impala_cnn | 1 | 0.88 | 18,618 | 0.47 | 0.00% |
| pongcnn | impala_cnn | 2 | 0.89 | 18,409 | 0.47 | 0.00% |
| pongcnn | impoola_cnn | 1 | 0.83 | 19,740 | 0.46 | 0.07% |
| pongcnn | impoola_cnn | 2 | 0.82 | 19,980 | 0.44 | 0.07% |

Process SPS includes startup/writes and uses /usr/bin/time's rounded timer; native timing/SPS remain in each metrics directory. Source snapshots/checks, resolved native configs, commands, binary/checkpoint hashes and raw failures are retained. Large artifacts remain local.
