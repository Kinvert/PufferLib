# Evaluation telemetry regression — G240, September 15

Native float32 Pong evaluator built with existing CUDA 12.8/NCCL and `sm_120`. No new training, kernel math, environment rules or system changes. Source patch and baseline/patched binary hashes are retained here; original executables/checkpoints stay in local `build/`. The Bash script records exact commands, checkpoint identity and isolated configs; it is a local regression recipe, not a portable benchmark launcher.

Two existing nonzero-score checkpoints evaluated for 256 requested matches with seed 29173 produced **byte-identical final CUDA_EVAL records** before/after the telemetry patch. Both saved checkpoint hashes remained unchanged. A seven-second, deliberately unreachable-target probe exited 124 with a flushed progress line and no success record. It is a diagnostic timeout, not another failed benchmark evaluation.

| Checkpoint | Old process seconds | Patched process seconds | Result |
|---|---:|---:|---|
| Trial 5, 2,562,048 decisions | 10.71 | 11.68 | score 18.648438, perf .903211, 256 games, 160224 parameters, identical |
| Trial 7, 2,332,672 decisions | 5.03 | 3.00 | score 20.941406, perf .997337, 256 games, 160224 parameters, identical |

These single-pair timings varied and do **not** establish a throughput improvement. The initial patch failed at the final dashboard because it omitted mandatory utilization fields; `initial-failure.txt` preserves the failure. The corrected patch refreshes telemetry at periodic and final standalone dashboard prints. Training-owned dashboard resource fields stay frozen. Quiet evaluation avoids hardware telemetry in this loop.

Three CPU runtime tests passed (PATH tool preference/idle, busy rejection, query-failure rejection), four Pong config tests and three existing hardware tests passed, and Bash syntax checks passed. The GPU regression covers standalone headless Pong evaluation with two checkpoints; it does not constitute all-mode validation or a complete baseline performance audit. See [next 5090 audit instructions](../../../PONG_EVALUATION_AUDIT_HANDOFF.md).
