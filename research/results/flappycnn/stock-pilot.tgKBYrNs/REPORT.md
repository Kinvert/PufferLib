# Original stock Flappy versus quality pixel CNN on RTX 5060

September 26, 2026. Original state-based Flappy completed one native float32 training/reload/evaluation run. Both rows use 19,922,944 actual decisions (stock 20M requested), train seed 73, stock learner/vectorization/game settings and final action-seed 10073 evaluation. The pixel row is the already completed pilot; it was not retrained.

| Model | Core | Parameters | Process train seconds | Process SPS | Native avg SPS | Evaluation pipes/episode | Clipped perf | Completed eval episodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original stock state Flappy | H64/L2 | 25,152 | 9.73 | 2,047,579 | 2,264,975 | 48.533836 | 0.947180 | 266 |
| Our selected quality pixel CNN | H128/L1 | 160,096 | 56.12 | 355,006 | 361,619 | 52.252918 | 0.975875 | 257 |

Stock state training is **5.7677× faster by whole-process SPS** (6.2634× by saved native average SPS). The pixel pilot's final evaluation score is 3.7191 pipes higher, with clipped-perf difference +0.028695. These are descriptive single-seed outcomes, not established score superiority or an isolated measurement of encoder overhead.

`perf` is mean min(pipes passed/20, 1), not wins or completion fraction. State exposes bird velocity and offscreen pipe coordinates; pixels show visible geometry at reduced precision. Network/core sizes differ as requested: untouched stock state H64/L2 versus selected pixel quality H128/L1. Equal learner settings therefore do not isolate input modality or encoder cost. This is whole native **training throughput**, not a pure environment-simulation SPS measurement.

## Recipe and evaluation

Original `ocean/flappy/flappy.h`, original six state observations and unchanged copied `config/flappy.ini`. The state binary is built for `flappy`, without any custom pixel encoder. Random initialization. Stock learner: LR 0.01 with annealing, gamma 0.995, GAE 0.92, entropy 0.003, value coefficient 1.5, replay 1, gradient norm 1.5, 2,048 agents, four buffers/eight CPU workers, horizon 64, minibatch 16,384, async=1 and CUDA graphs. Float32 on one 5060 is shared with the pixel run. No tuning, adaptive search or checkpoint selection by evaluation score.

Checkpoints every 1,048,576 decisions, plus the final checkpoint (19 total). Operational changes to stock are artifact/run paths, cadence and disabling embedded training evaluation; final evaluation is separately timed. Both final evaluations request 256 completed episodes with 64 slots. Pooled-v1 accounting returns 266 for state and 257 for pixels; outcomes have unequal episode counts and native environment RNG remains slot based. No exact-quota or independently seeded environment panel is claimed.

State process evaluation time: 12.36 seconds. Saved native uptime: 8.796100 seconds; last saved native SPS: 2,434,353; last VRAM sample: 1.853 GiB. Last training-window score/perf: 49.944057 / 0.964336; these are separate from final evaluation. Native history uses stock downsample=5; raw dashboards and checkpoint sequence are retained. Process SPS includes startup/checkpoint writes and excludes compilation/evaluation; native average SPS divides actual steps by saved adjusted uptime.

## Checks and provenance

- Original stock state config copy is byte-identical to the captured source `config/flappy.ini`.
- Executed learner/vector/selfplay keys match stock source configs and match the pixel run. Original game keys match stock; executed policy matches the prepared stock H64/L2 config. Train seed, async, CUDA graphs, checkpoint cadence and disabled embedded evaluation agree.
- Actual steps match 152 full rollouts. All 19 expected checkpoints exist, have exactly 25,152 float32 parameters, are finite and match recorded SHA256s. All native saved metrics are finite.
- Final evaluation completion has finite score/perf, at least 256 episodes and the expected parameter count. Reloaded the final checkpoint, not a selected earlier snapshot.
- Normal CUDA build and source hashes passed. GPU preflight showed desktop Xwayland only, no competing compute process. No continuous contention or exclusive-host monitoring is claimed.

Final state checkpoint SHA256: `c5184a9d23b04f535f134bc49eaff40c3769afa8c2d56d22b1e8eea907a7c0d5`.

State campaign: `build/flappycnn/stock-pilot.tgKBYrNs`, tmux launch `flappy-state-stock-20260926-2138`, status `completed_pending_audit` from native runner; receipt/config/checkpoint audit above was performed after successful exit. [Pixel pilot evidence](../stock-pilot.DN7C2JF6/REPORT.md). Raw configs/metrics/train and eval logs/commands/source and checkpoint hashes are preserved alongside this report. Full source snapshots, binaries and weights remain in local ignored build output. Dirty worktree provenance requires captured hashes rather than the base commit alone. No repeatability test or multi-seed inference was performed.
