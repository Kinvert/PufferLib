# Connect4CNN: frozen five-seed confirmation

**October 5 caveat:** this campaign used the inherited incorrect draw rule.
Preserve these measurements as legacy-task evidence; ranking impact is unknown.
[Correction and new benchmark boundary](CONNECT4_DRAW_CORRECTION.md).

**Measured source:** `b2fa7787a754d36362374ac271ea6c7b23beeb25`, reconstructed and verified against all 38 captured file hashes. [Exact source state, later feature revision, and recovery instructions](BENCHMARK_STATE.md).

Completed September 14, 2026: `confirm.ol9tcj5k`, all **30 training jobs and 390 held-out checkpoint evaluations**, no native failures or W&B upload failures. This is the confirmation requested in paper-plan item 1, not another architecture search. All models used the original square representation, the same learner/core settings, five fresh paired seeds and the same 13 checkpoints through 13,312,000 decisions. [Frozen protocol](CONFIRMATION_PROTOCOL.md).

**Observed mean time/score tradeoff:** our fixed models form the low-cost frontier; IMPALA forms the high-performance frontier. Nature and Impoola have no points on the combined frontier of five-seed checkpoint means. This is a descriptive mean frontier, not a statistically established dominance result. The final quality-model advantage over Nature is promising but its paired uncertainty interval narrowly includes zero.

[Interactive full curves](results/connect4cnn/confirm.ol9tcj5k/analysis/frontier.html) · [Complete numerical report and frontier](results/connect4cnn/confirm.ol9tcj5k/analysis/REPORT.md) · [All 390 observations](results/connect4cnn/confirm.ol9tcj5k/analysis/observations.csv) · [All time/step, per-model/combined, mean/per-seed frontiers](results/connect4cnn/confirm.ol9tcj5k/analysis/frontiers.csv).

## Final checkpoint comparison

Five-seed arithmetic means. Wall time here is the entire training process; checkpoint-frontier times below use launch-to-checkpoint timestamps and differ slightly. Native SPS uses the trainer's adjusted timer. Intervals resample training seeds, not evaluation games.

| Fixed model | Mean held-out wins | Pointwise 95% interval | Min–max across seeds | Mean train wall | Native SPS |
|---|---:|---:|---:|---:|---:|
| Ours quality | 79.66% | 71.51–87.82% | 66.73–90.82% | 144.47 s | 92,382 |
| Ours small | 78.01% | 66.12–88.03% | 58.45–89.25% | 143.28 s | 93,134 |
| Ours fast | 71.66% | 63.67–78.99% | 58.20–81.80% | 147.12 s | 90,687 |
| Nature | 73.35% | 66.24–80.63% | 60.89–85.96% | 158.71 s | 84,066 |
| IMPALA | 97.45% | 96.69–98.39% | 96.30–99.18% | 1,231.81 s | 10,811 |
| Impoola | 80.09% | 74.85–88.31% | 73.05–95.82% | 1,229.68 s | 10,830 |

The names quality/fast/small preserve their development-selection roles. In this confirmation, the model named fast was slower and scored lower on average than quality/small. All three are reported; no new model or seed is substituted after inspecting outcomes.

## The whole Pareto front

All 13 checkpoints per model and all five seeds are included in the artifact. The chart switches between mean curves and each paired seed, between wall time and decisions, and between the first 180 seconds and the full range. It shows chronological learning curves (including declines), uncertainty bars, and the observed nondominated points. The full numeric frontier includes early near-zero-performance points; they are not silently discarded.

Selected checkpoints illustrating the **combined mean wall-time frontier**:

| Model | Agent decisions | Mean checkpoint wall | Mean held-out wins |
|---|---:|---:|---:|
| Ours small | 5.120M | 58.25 s | 55.55% |
| Ours quality | 6.144M | 69.03 s | 68.87% |
| Ours quality | 8.192M | 90.55 s | 77.27% |
| Ours quality | 13.312M | 144.35 s | 79.66% |
| IMPALA | 4.096M | 379.75 s | 86.65% |
| IMPALA | 5.120M | 474.19 s | 93.44% |
| IMPALA | 6.144M | 568.69 s | 95.92% |
| IMPALA | 11.264M | 1,042.42 s | 97.56% |

The frontier ends before IMPALA's final checkpoint because its mean declines slightly thereafter; its final mean is 97.45%. That earlier checkpoint was identified from the observed curve, not preselected as an optimal stopping budget. All checkpoints follow the full-run learning-rate schedule, so this does not establish that a separately trained shorter-budget run would achieve the same result.

Nature's mean curve peaks at 73.60% around 134.80 s, after ours has already reached higher mean performance at lower cost. Impoola eventually reaches 80.09%, but an IMPALA checkpoint reaches 86.65% at approximately 380 s. This explains why neither reference appears on the combined mean wall-time frontier; their complete individual curves/frontiers remain visible.

The **decision-count frontier** tells a different story: it consists entirely of IMPALA checkpoints in the mean comparison. IMPALA is substantially more sample-efficient in this experiment, while our models take less wall time to reach moderate performance. There is no evidence that ours beats IMPALA across the entire tradeoff.

## Seed robustness and uncertainty

At the final matched step budget, quality beats Nature in four of five paired seeds. Its mean advantage is **+6.31 percentage points**, with a paired percentile-bootstrap interval of **−0.07 to +13.00 points**. Small's advantage is +4.66 points with a much wider interval, −11.47 to +20.78. Speed is consistently higher here (quality has about 9.9% higher mean native SPS), but the score advantage is not firmly resolved with five seeds.

The original discovery winner's roughly 92% held-out score did **not** reproduce as a typical result: quality averages 79.66%, with final scores spanning 66.73–90.82%. Across all measured checkpoints, quality reached 90% in 1/5 seeds; small, fast and Nature in 0/5; IMPALA in 5/5; Impoola in 1/5. These threshold counts are exploratory descriptive summaries, not predeclared hypothesis tests. Non-achievement is retained in [thresholds.csv](results/connect4cnn/confirm.ol9tcj5k/analysis/thresholds.csv).

Intervals enumerate all 3,125 ordered bootstrap resamples of five seeds; differences use the same seed resample on both models. They are pointwise percentile intervals, with no multiple-comparison adjustment or simultaneous-band guarantee. They do not quantify uncertainty in the frontier itself or establish universal dominance. Five seeds still give limited coverage of rare training failures and broad variability. Reporting uncertainty rather than only selected point estimates follows the evaluation concern raised by [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264); this analysis uses its own explicit small-sample bootstrap, not the rliable package.

## Audit and practical limits

- Verified all 390 checkpoint hashes and float32 parameter counts/finiteness, all 38 archived source hashes, each compiled binary hash, the complete variant × seed × checkpoint matrix, paired evaluation seeds, monotonic checkpoint times, and matched effective non-encoder configs.
- Total training decisions: 399,360,000. Summed training-process time: 15,275.30 s (4 h 14 m 35 s). Evaluation-process time: 258.83 s. Builds and between-job W&B uploads add further campaign overhead. All 30 uploads were recorded successful in `kinvert-k/cnn2`.
- No competing compute process appeared in the pre-job GPU snapshots. Those snapshots are not continuous resource monitoring. Brief CPU-only development/tests occurred during the campaign; no strict exclusive-host claim is made. The comparison uses this RTX 5060 and these native float32 implementations.
- Model adaptation, initialization, common recipe and fixed recurrent core are those in the frozen protocol. Equal learner settings are not equal per-family hyperparameter tuning budgets, parameters or FLOPs. These correctness-oriented CUDA convolution implementations have not undergone a comprehensive backend optimization comparison.
- Original square rendering only. No new Pong or representation experiment is mixed into these results. This is one development game, not evidence of architecture transfer across games.
- Small receipt files are archived, and all captured source/config files are recoverable from measured-source commit `b2fa7787a754d36362374ac271ea6c7b23beeb25`. Full binaries and checkpoint arrays remain in the original local build directory. The original launch base had uncommitted work; see the [provenance note](BENCHMARK_STATE.md) for the retrospective reconstruction and its limits.

Analysis code: [analyze_confirmation.py](analyze_confirmation.py), [chart template](confirmation_frontier.html). Reproduce from the original local campaign with `.venv/bin/python research/analyze_confirmation.py build/connect4cnn/confirm.ol9tcj5k`. The generated JavaScript passed a DOM smoke check across 48 view combinations and empty-model selection; actual browser appearance was not automatically tested.

The next claim to test is whether the low-cost advantage survives matched appearance tests and a second game with frozen models. More architecture search should not treat these confirmation scores as a new untouched test set. Finish the remaining fairness, generalization and practical-delivery actions in [PAPER_PLAN.md](PAPER_PLAN.md).
