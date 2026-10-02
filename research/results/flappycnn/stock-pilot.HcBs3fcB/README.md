# Stock-learner pilot: blocked before training

September 26, 2026. Native float32 compilation and source hashes passed; stock learner/vector/original environment settings were checked against `config/default.ini` plus `config/flappy.ini`. The quality encoder and H128/L1 core are the network exception. [Resolved configs](config/) and [pilot specification](../../../FLAPPY_STOCK_PILOT.md).

The idle-GPU preflight returned failure. Direct NVML inspection reports `GPU access blocked by the operating system`; this managed session prohibits escalation (`approval_policy=never`). Status: `blocked_gpu`. No model execution, training, evaluation, checkpoint or performance measurement. The compute-process query may fail silently; the separate full NVML query is preserved in `gpu-access.txt`.

Local directory: `build/flappycnn/stock-pilot.HcBs3fcB`. Source hashes, base revision/integration diff, binary hash, build command/log and failed preflight are retained. Full source snapshot and binary remain in ignored local build output. Source is a dirty worktree, not reproduced by the base revision alone. No training command was executed; its arguments and time limits are in [stock_pilot.sh](../../../../ocean/flappycnn/stock_pilot.sh).

Run that launcher from a device-accessible terminal when the 5060 is idle. It creates a fresh attempt, preserves this blocked attempt and saves native metrics, wall times, checkpoints and separately seeded final evaluation. Successful process exit still requires receipt/finite-weight/metric audit before conclusions.
