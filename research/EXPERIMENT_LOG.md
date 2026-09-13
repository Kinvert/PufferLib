# CNN experiment history

Persistent results across code and configuration changes. The comparison runner appends a dated entry after each comparison; keep earlier results, including failures. Each entry links the exact protocol, source/binary hashes, recipe, hardware record, raw logs, and checkpoints. Add interpretation below an entry rather than rewriting its measured numbers.

## Metric definitions

- **Perf / win rate:** independent checkpoint evaluation against the unchanged scripted Connect4 opponent. The board remains 7 columns × 6 rows.
- **Process SPS:** training agent decisions divided by measured training-process wall time, including startup and checkpoint writes. Compilation and later evaluation are excluded.
- **Native average SPS:** training agent decisions divided by the trainer's final logged uptime. Its timing boundary differs from process wall time.
- **Native last SPS:** final logged SPS sample/bin; not an overall average or guaranteed steady-state rate.
- **VRAM last GB:** last native logged GPU-memory reading, not a measured peak or exclusively encoder memory.
- Retain training budget, seeds, float precision, parameters, core/learner settings, device, and concurrent-load context. Compare speed alongside learning; a faster policy with lower win rate is not automatically better.

## Current measured baselines (2026-09-12 local time)

**IMPALA/Impoola extension in progress:** `build/connect4cnn/compare.l6d5sbk2` runs the same 13,279,232-decision common recipe, seeds 73/74/75, four checkpoint evaluations, and 1,024 requested evaluation games. Its report records completed trials as they finish; the full comparison is pending and is not included in the completed three-seed means below. Both variants use original-SAME pooling and the same 16/32/32 backbone; GAP is the readout change. See the environment README for reference adaptations and numerical validation. Archive the full results after completion.

Arithmetic means across training seeds 73/74/75, evaluated with separate paired seeds 10073/10074/10075. All runs use float32 on G240's RTX 5060, and the same game/opponent and 64-agent evaluation setup. Stock, Nature, and the earlier state/tiny pair ran in separate invocations, not one interleaved speed experiment.

| Policy | Training recipe | Actual decisions per seed | Parameters | Mean wins | Mean process SPS | Mean train wall s |
|---|---|---:|---:|---:|---:|---:|
| Stock state | Stock training config | 13,238,272 | 209,408 | 98.87% | 306,403 | 43.414 |
| State | Modified common recipe | 13,279,232 | 55,552 | 18.51% | 108,831 | 122.017 |
| Tiny CNN | Modified common recipe | 13,279,232 | 151,680 | 64.04% | 97,733 | 135.875 |
| Adapted Nature CNN | Modified common recipe | 13,279,232 | 138,528 | 78.71% | 80,840 | 164.279 |

Reports: [stock state](results/connect4cnn/compare.9egh6y44/REPORT.md), [state/tiny CNN](results/connect4cnn/compare.we7qgdcg/REPORT.md), [Nature](results/connect4cnn/compare.2s__8t8l/REPORT.md). Stock wins here, with a different learner/network/batching configuration. Nature outperformed tiny CNN in each paired seed under the common recipe, while processing about 17.3% fewer decisions per second. Nature is the adapted three-convolution stack, not a complete DQN reproduction. No pixel policy has yet been tested with the stock training recipe.

## Policy identities and hyperparameters for compare.we7qgdcg

**The 18.51% state result is PufferLib's original Connect4 environment and default network architecture with a modified configuration. It is not a measurement of untouched stock PufferLib tuning.** Both tested policies use the same modified learner/core recipe. Stock training settings were subsequently measured separately in `compare.9egh6y44`: **98.87% mean wins**, using float32 and common evaluation. See the dated results below.

| Property | `state` — modified-config baseline | `tiny_cnn` — custom pixel encoder |
|---|---|---|
| Input | Original 42 board values | 1×36×44 grayscale; each board cell is 6×6 pixels, with side padding |
| Encoder | Default linear projection, 42 → 128 | Conv4×4, stride 4, 1 → 8 channels → ReLU → flatten 792 → linear 128; no biases |
| Recurrent core | Existing PufferLib MinGRU, hidden 128, one layer | Same |
| Trainer and action/value heads | Existing PufferLib implementation | Same |
| Total parameters | 55,552 | 151,680 |
| Mean final evaluation win rate | 18.51% | 64.04% |
| Mean process SPS | 108,831 | 97,733 |

Tiny CNN is the custom encoder implemented in [connect4cnn.cu](../ocean/connect4cnn/connect4cnn.cu), not Nature, IMPALA, or Impoola. Both environments preserve the same 7-column × 6-row game, seven actions, scripted opponent, rewards, and termination rules.

The following values describe this completed experiment, not every future invocation. Stock values come from the run's saved upstream [Connect4 config](results/connect4cnn/compare.we7qgdcg/source/config/connect4.ini) plus [default config](results/connect4cnn/compare.we7qgdcg/source/config/default.ini), at revision `89414204ce85`. Tested values come from the [saved common recipe](results/connect4cnn/compare.we7qgdcg/recipe.ini), command overrides, and resolved run INIs.

| Setting | Tested state | Tested tiny CNN | Stock configuration |
|---|---:|---:|---:|
| Training decisions per seed | 13,279,232 | 13,279,232 | 13,272,299 |
| Core hidden size | 128 | 128 | 256 |
| Core layers | 1 | 1 | 1 |
| Initial learning rate | 0.001 | 0.001 | 0.00847027 |
| Replay ratio | 1 | 1 | 3.16619 |
| Parallel agents | 64 | 64 | 4,096 |
| Vector buffers | 1 | 1 | 8 |
| Environment threads | 2 | 2 | 2 |
| Minibatch size | 2,048 | 2,048 | 8,192 |
| Rollout horizon | 32 | 32 | 32 |
| Async execution | 0 | 0 | 1 |
| Gamma | 0.8 | 0.8 | 0.8 |
| GAE lambda | 0.962627 | 0.962627 | 0.962627 |
| PPO clip coefficient | 0.511829 | 0.511829 | 0.511829 |
| Value loss coefficient | 5 | 5 | 5 |
| Value clip coefficient | 1.99178 | 1.99178 | 1.99178 |
| Maximum gradient norm | 0.552251 | 0.552251 | 0.552251 |
| Entropy coefficient | 0.0000324222 | 0.0000324222 | 0.0000324222 |
| Optimizer momentum | 0.878636 | 0.878636 | 0.878636 |

Both tests also use learning-rate annealing to zero, entropy annealing disabled, CUDA graphs enabled, one GPU, and float32 builds. `min_ent_coef_ratio` is overridden from 0.1 to 0 but is inactive with entropy annealing disabled. Checkpoint interval changes from 500 updates to 1,621, yielding four checkpoints at 3,319,808-decision intervals. Output directories/run IDs are isolated per trial. Training sets `eval_episodes=0`; separate evaluation processes request 1,024 games per checkpoint instead of the stock evaluation default of 10,000, with actual batched counts retained in the report.

Both policies use training seeds 73/74/75 and corresponding separate evaluation seeds 10073/10074/10075. All six resolved training configurations were checked to agree after excluding environment names, seeds, and artifact paths. Equal seed labels do not imply identical weight initializations or trajectories across different architectures.

**What the result establishes:** the custom CNN achieved higher evaluation win rates under this shared modified recipe. Direct state provides the game information, but the network still must learn useful features. The CNN adds nonlinear feature extraction and more parameters; either may contribute, and this experiment does not isolate their effects. It does not establish that pixels are intrinsically better, that the CNN beats stock-tuned PufferLib, or that either policy has converged. The subsequent stock-config run achieves much higher win rates than either earlier policy.

## Earlier workflow checks (2026-09-12)

- Tiny CNN native checks and repeated 65,536-step training produced identical finite checkpoints. See [environment verification](../ocean/connect4cnn/README.md).
- [Matched smoke comparison](results/connect4cnn/compare.16ohu06z/REPORT.md): state and tiny CNN, seeds 73/74, 65,536 steps each, two checkpoint evaluations each. All final win rates were zero. This was infrastructure validation; these short process times are not a meaningful speed ranking. SPS was not yet a dedicated report column; original native timing samples remain in each run's resolved INI.
- Earlier `compare.otwm9q9m` was a reporting diagnostic: its selected native score metric was win rate. Its report is marked superseded; do not read that score column as episode return.

## Planned first learning-budget comparison

Keep the same common recipe and both architectures; increase training to **13,279,232 decisions per run**, the first multiple of rollout-batch × four checkpoints above the stock config's 13,272,299 decisions. Use three training seeds (73/74/75), separate paired evaluation seeds (10073/10074/10075), and 1,024 requested evaluation games per checkpoint. This changes only the budget from the smoke comparison. The shared recipe still differs from the stock Connect4 tuning; learning failure in both variants would require recipe investigation before judging CNN quality.

## 2026-09-13T04:51:13+00:00 — compare.we7qgdcg

Change/purpose: First full-budget comparison; unchanged common recipe and architectures, increased from 65,536-step smoke budget

Revision `89414204ce85`; recipe SHA256 `6cdd8042d7c848923d119aa65062adfe4c52a8cb5bf85ce4ab8e503b5a6b34cf`. [Report](results/connect4cnn/compare.we7qgdcg/REPORT.md) · [CSV](results/connect4cnn/compare.we7qgdcg/results.csv) · [Source/build hashes](results/connect4cnn/compare.we7qgdcg/protocol.json) · [GPU](results/connect4cnn/compare.we7qgdcg/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 73 | 13,279,232 | 21.88% | -0.5426 | 55,552 | 121.865 | 108,967 | 109,330 | 116,265 | 1.316 |
| tiny_cnn | 73 | 13,279,232 | 65.91% | 0.3286 | 151,680 | 135.649 | 97,894 | 98,136 | 99,146 | 1.365 |
| tiny_cnn | 74 | 13,279,232 | 66.89% | 0.3541 | 151,680 | 135.361 | 98,102 | 98,318 | 102,663 | 1.365 |
| state | 74 | 13,279,232 | 18.56% | -0.6057 | 55,552 | 121.714 | 109,102 | 109,354 | 113,224 | 1.316 |
| state | 75 | 13,279,232 | 15.09% | -0.6762 | 55,552 | 122.474 | 108,425 | 108,718 | 114,764 | 1.316 |
| tiny_cnn | 75 | 13,279,232 | 59.32% | 0.2082 | 151,680 | 136.614 | 97,203 | 97,448 | 101,845 | 1.365 |

Matched learner/core recipe; state and CNN parameter counts differ. See the report for checkpoint curves and actual evaluation counts.

Interpretation (2026-09-12 local time): all six training jobs and 24 checkpoint evaluations passed. Hardware was G240's RTX 5060 (8 GB), float32, with serial trials; the GPU was idle before launch. Arithmetic means across the three training seeds: state **18.51% wins / 108,831 process SPS / 122.017 s**, tiny CNN **64.04% wins / 97,733 process SPS / 135.875 s**. The CNN achieved higher win rates in each paired seed, with about 10.2% lower throughput and 11.4% more training wall time. This establishes useful learning at this budget; it does not establish convergence or isolate architecture from parameter count. The common recipe is not the untouched stock tuning. Nature, IMPALA, and Impoola still need implementation and measurement.

## 2026-09-13T05:09:12+00:00 — compare.9egh6y44

Change/purpose: Stock Connect4 training configuration, float32; same held-out evaluation protocol as the modified-recipe baselines

Revision `89414204ce85`; recipe SHA256 `536a53ff54dd958d85e9384cfe6ecbd7031aefc5f2d8d9244d48f756e0965af6`. [Report](results/connect4cnn/compare.9egh6y44/REPORT.md) · [CSV](results/connect4cnn/compare.9egh6y44/results.csv) · [Source/build hashes](results/connect4cnn/compare.9egh6y44/protocol.json) · [GPU](results/connect4cnn/compare.9egh6y44/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_state | 73 | 13,238,272 | 98.51% | 0.9703 | 209,408 | 41.913 | 315,852 | 320,983 | 551,820 | 2.429 |
| stock_state | 74 | 13,238,272 | 99.06% | 0.9820 | 209,408 | 40.634 | 325,793 | 330,723 | 599,676 | 2.429 |
| stock_state | 75 | 13,238,272 | 99.04% | 0.9809 | 209,408 | 47.695 | 277,564 | 281,410 | 627,027 | 2.429 |

Stock training configuration in float32; common 64-agent evaluation. Training hypers differ from the earlier state/tiny-CNN comparison.
See the report for checkpoint curves and actual evaluation counts.

Interpretation: all three runs and 12 evaluations passed. Mean wins **98.87%**, mean process SPS **306,403**, mean training wall **43.414 seconds**. Stock hidden size 256 gives 209,408 parameters. All stock train/vec/policy/env/selfplay settings and async mode were checked against the resolved INIs. Training budget 13,272,299 truncates to 13,238,272 actual decisions in 101 rollout batches; checkpoint controls yield evaluations at 26/52/78/101 batches. Separate evaluation uses the earlier 64-agent, one-buffer, two-thread synchronous setup and seeds 10073/10074/10075. This is stock **training configuration in float32**, not a default-precision claim. The result shows that the smaller shared recipe substantially weakened the state baseline; the tiny CNN did not outperform stock-config PufferLib. Parameter count, optimization settings, and vectorization all differ.

## 2026-09-13T05:21:22+00:00 — compare.z10zc3xf

Change/purpose: Nature encoder smoke: validated forward/backward; first native training and checkpoint evaluation

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.z10zc3xf/REPORT.md) · [CSV](results/connect4cnn/compare.z10zc3xf/results.csv) · [Source/build hashes](results/connect4cnn/compare.z10zc3xf/protocol.json) · [GPU](results/connect4cnn/compare.z10zc3xf/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 65,536 | 0.35% | -0.9366 | 138,528 | 1.317 | 49,762 | 67,315 | 76,667 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T05:22:02+00:00 — compare.ltxw6alr

Change/purpose: Nature same-seed repeat to verify byte-identical training checkpoints

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.ltxw6alr/REPORT.md) · [CSV](results/connect4cnn/compare.ltxw6alr/results.csv) · [Source/build hashes](results/connect4cnn/compare.ltxw6alr/protocol.json) · [GPU](results/connect4cnn/compare.ltxw6alr/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 65,536 | 0.35% | -0.9366 | 138,528 | 1.317 | 49,759 | 68,001 | 77,442 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T05:35:01+00:00 — compare.2s__8t8l

Change/purpose: First adapted Nature CNN at the same 13.28M-decision common recipe as state and tiny CNN; stock training configuration recorded separately

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.2s__8t8l/REPORT.md) · [CSV](results/connect4cnn/compare.2s__8t8l/results.csv) · [Source/build hashes](results/connect4cnn/compare.2s__8t8l/protocol.json) · [GPU](results/connect4cnn/compare.2s__8t8l/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 13,279,232 | 77.10% | 0.5420 | 138,528 | 162.552 | 81,692 | 81,900 | 82,193 | 1.552 |
| nature_cnn | 74 | 13,279,232 | 88.33% | 0.7686 | 138,528 | 164.117 | 80,913 | 81,076 | 83,865 | 1.552 |
| nature_cnn | 75 | 13,279,232 | 70.68% | 0.4457 | 138,528 | 166.168 | 79,915 | 80,068 | 81,722 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T06:11:19+00:00 — compare.ij5_xpo7

Change/purpose: IMPALA/Impoola training smoke after numerical validation; original SAME pooling, base widths, common recipe

Revision `43850f172637`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.ij5_xpo7/REPORT.md) · [CSV](results/connect4cnn/compare.ij5_xpo7/results.csv) · [Source/build hashes](results/connect4cnn/compare.ij5_xpo7/protocol.json) · [GPU](results/connect4cnn/compare.ij5_xpo7/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 65,536 | 0.00% | -0.9377 | 270,496 | 6.679 | 9,812 | 10,513 | 10,744 | 3.503 |
| impoola_cnn | 73 | 65,536 | 0.00% | -0.9491 | 151,712 | 6.633 | 9,880 | 10,531 | 10,682 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T06:12:52+00:00 — compare.sqngvlom

Change/purpose: IMPALA/Impoola identical-seed smoke repeat for training determinism

Revision `43850f172637`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.sqngvlom/REPORT.md) · [CSV](results/connect4cnn/compare.sqngvlom/results.csv) · [Source/build hashes](results/connect4cnn/compare.sqngvlom/protocol.json) · [GPU](results/connect4cnn/compare.sqngvlom/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 65,536 | 0.00% | -0.9377 | 270,496 | 6.630 | 9,885 | 10,510 | 10,741 | 3.503 |
| impoola_cnn | 73 | 65,536 | 0.00% | -0.9491 | 151,712 | 6.580 | 9,960 | 10,605 | 10,758 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.
