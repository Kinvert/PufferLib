# Frozen confirmation: full numerical results

All 30 jobs and 390 checkpoints audited. Five fresh seeds per fixed architecture; same 13 checkpoints and learner settings.
[Interactive chart](frontier.html) · [Every observation](observations.csv) · [All per-seed/family/time/step frontiers](frontiers.csv)

## Final checkpoint means

| Model | Wins | Pointwise 95% interval | Seed range | Whole-process seconds | Native SPS |
|---|---:|---:|---:|---:|---:|
| Ours quality | 79.66% | 71.51%–87.82% | 66.73%–90.82% | 144.47 | 92,382 |
| Nature | 73.35% | 66.24%–80.63% | 60.89%–85.96% | 158.71 | 84,066 |
| Ours fast | 71.66% | 63.67%–78.99% | 58.20%–81.80% | 147.12 | 90,687 |
| IMPALA | 97.45% | 96.69%–98.39% | 96.30%–99.18% | 1231.81 | 10,811 |
| Ours small | 78.01% | 66.12%–88.03% | 58.45%–89.25% | 143.28 | 93,134 |
| Impoola | 80.09% | 74.85%–88.31% | 73.05%–95.82% | 1229.68 | 10,830 |

## Entire combined frontier of checkpoint means

Includes near-zero early points. Cost is mean launch-to-checkpoint wall time, not whole-process time.

| Model | Decisions | Wall seconds | Wins | Pointwise 95% interval |
|---|---:|---:|---:|---:|
| Ours quality | 1,024,000 | 12.31 | 0.05% | 0.02%–0.08% |
| Ours fast | 1,024,000 | 12.43 | 0.07% | 0.02%–0.12% |
| Ours quality | 2,048,000 | 24.21 | 0.18% | 0.11%–0.26% |
| Ours small | 2,048,000 | 24.29 | 0.20% | 0.11%–0.30% |
| Ours fast | 2,048,000 | 24.44 | 0.37% | 0.26%–0.49% |
| Ours quality | 3,072,000 | 35.83 | 7.96% | 5.21%–10.57% |
| Ours quality | 4,096,000 | 47.14 | 29.59% | 23.29%–35.89% |
| Ours quality | 5,120,000 | 58.18 | 52.83% | 40.74%–64.91% |
| Ours small | 5,120,000 | 58.25 | 55.55% | 45.76%–61.54% |
| Ours small | 6,144,000 | 68.89 | 67.07% | 56.48%–75.16% |
| Ours quality | 6,144,000 | 69.03 | 68.87% | 59.26%–78.76% |
| Ours small | 7,168,000 | 79.48 | 73.65% | 59.94%–84.34% |
| Ours quality | 7,168,000 | 79.79 | 73.88% | 66.00%–82.10% |
| Ours small | 8,192,000 | 90.12 | 75.72% | 61.81%–86.41% |
| Ours quality | 8,192,000 | 90.55 | 77.27% | 71.42%–84.22% |
| Ours quality | 9,216,000 | 101.32 | 78.17% | 71.03%–84.96% |
| Ours quality | 10,240,000 | 112.07 | 78.83% | 71.33%–86.03% |
| Ours quality | 11,264,000 | 122.83 | 79.32% | 71.32%–87.29% |
| Ours quality | 12,288,000 | 133.58 | 79.64% | 71.56%–87.71% |
| Ours quality | 13,312,000 | 144.35 | 79.66% | 71.51%–87.82% |
| IMPALA | 4,096,000 | 379.75 | 86.65% | 82.55%–91.57% |
| IMPALA | 5,120,000 | 474.19 | 93.44% | 91.18%–95.79% |
| IMPALA | 6,144,000 | 568.69 | 95.92% | 94.68%–97.17% |
| IMPALA | 7,168,000 | 663.35 | 96.76% | 95.56%–98.11% |
| IMPALA | 8,192,000 | 757.96 | 96.96% | 96.13%–98.21% |
| IMPALA | 9,216,000 | 852.57 | 97.34% | 96.61%–98.36% |
| IMPALA | 10,240,000 | 947.65 | 97.34% | 96.70%–98.27% |
| IMPALA | 11,264,000 | 1042.42 | 97.56% | 96.81%–98.53% |

## Final paired differences versus Nature

Same training/evaluation seed pairs at the full step budget. Units are percentage points.

| Model | Mean difference | Pointwise 95% interval | Higher-scoring seeds |
|---|---:|---:|---:|
| Ours quality | +6.31 | -0.07 to +13.00 | 4/5 |
| Ours fast | -1.69 | -14.87 to +5.46 | 4/5 |
| Ours small | +4.66 | -11.47 to +20.78 | 3/5 |
| IMPALA | +24.10 | +17.20 to +30.68 | 5/5 |
| Impoola | +6.74 | +0.95 to +12.57 | 4/5 |

Intervals enumerate all 3,125 ordered bootstrap resamples of the five seeds, pairing seed indices for differences. They are pointwise, not simultaneous or multiplicity-adjusted, and five seeds provide limited statistical resolution.
Exploratory first-observed threshold crossings and failures to reach them are retained in thresholds.csv. They are not independently trained shorter-budget runs or guarantees of sustained performance.
No GPU compute processes appeared in pre-job snapshots, but there was no continuous exclusive-resource audit. Brief CPU-only development work occurred during training. Native source snapshots/binaries/checkpoints remain in the original local campaign directory.
