# FlappyCNN stock-learner pilot on RTX 5060

September 26, 2026. One fixed model, one training seed, no adaptive search or baseline comparison. Native float32 GPU training and final-checkpoint reload/evaluation completed without a timeout.

| Measurement | Result |
|---|---:|
| Requested decisions | 20,000,000 |
| Actual decisions | 19,922,944 |
| Whole-process training seconds | 56.12 |
| Whole-process SPS | 355,006 |
| Saved native uptime seconds | 55.093792 |
| Native average SPS (decisions / saved uptime) | 361,619 |
| Last saved native SPS | 360,625 |
| Evaluation mean pipes passed | 52.252918 |
| Evaluation mean clipped pipes/20 | 0.975875 |
| Completed evaluation episodes | 257 (256 requested) |
| Whole-process evaluation seconds | 14.06 |
| Parameters | 160,096 |
| Saved VRAM used, last sample | 4.976 GiB |
| Checkpoints | 19, all hashes verified and all weights finite |

Process SPS includes startup and checkpoint writes. Native SPS uses saved adjusted timing; the raw terminal's final dashboard rate is a different sample. Native histories are downsampled into five bins by stock `sweep.downsample=5`; the final saved uptime and decision count are exact final entries. Preserve the full raw dashboard log and all intermediate checkpoints, including any declines. Final training-window score/perf were 39.266666/0.95, based on only 15 completed episodes in that window; they are not the final separate evaluation result.

## Model and recipe

Existing Connect4-selected `flex_quality`: encoder 4, one 16-channel 7×7 convolution with stride 4, flattened spatial readout, projection 64 and H128/L1 recurrent core. Input is one 36×44 grayscale frame. Pixel appearance 0 stays fixed. Random initialization, training/action seed 73, stock async=1 and CUDA graphs. No pretrained weights or state-observation input.

All stock Flappy learner/vectorization values and original environment values match the **executed** native INI: 2,048 agents, four buffers/eight CPU environment workers, horizon 64, minibatch 16,384, LR 0.01 with stock annealing, gamma 0.995, GAE 0.92, entropy 0.003, value coefficient 1.5, replay 1 and gradient norm 1.5. H128/L1 replaces stock Flappy H64/L2 to retain our selected full network. Artifact paths, checkpoint cadence and separate evaluation are operational changes. Training's default embedded 10,000-episode evaluation was disabled.

Final evaluation uses action seed 10073, 64 native slots and pooled-v1 accounting. `perf` is mean clipped pipes/20, **not a win rate or fraction of episodes completing the cap**. The 4,096-decision native cap and its terminal/crash-reward semantics are preserved. Native environment RNG is slot based, so changing the action seed does not create an independent environment-seed test panel. This pilot demonstrates learnability; it does not establish deterministic replication, baseline superiority, a Pareto improvement or SOTA. Other encoders and appearance panels were not trained here. Encoder-5 math and legacy numerical regression gates remain separate.

## Evidence audit

Executed learner/vector/selfplay and original environment keys match stock source configs; executed encoder/core keys match prepared configs. Final agent steps agree with 152 full rollouts. All expected checkpoint names at 1,048,576-decision cadence are present, all 19 recorded SHA256s match their bytes, all arrays have exactly 160,096 float32 parameters and are finite. All saved metric arrays are finite. The final CUDA_EVAL completion has finite score/perf, the expected parameter count and at least the requested episode count.

Final checkpoint SHA256: `76d475af63169d83dd49da42bea35d36eee67d564cd19bdf250d1a0f721a5efe`.

GPU: RTX 5060, 8GB; preflight showed no competing compute process and desktop Xwayland only. Source/build/config commands and hashes identify a dirty worktree on the recorded base revision; HEAD alone does not reproduce the measurement. Small native metrics, raw train/evaluation logs and receipts are archived here; full source snapshot, binaries and checkpoints remain in `build/flappycnn/stock-pilot.DN7C2JF6`. Earlier blocked attempts are preserved. No continuous exclusive-host monitoring or cross-hardware claim is made.
