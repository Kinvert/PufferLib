# CNN2: full observed Pareto-front comparison

Analyzed September 14, 2026. Campaign `sweep._u86vi03` completed 128 trials / 40 active architectures in **14,950.940 seconds (4h 9m)**, consuming 1,355,720,704 agent decisions. No worker failures. All 128 final checkpoint arrays are finite; the saved binary matches its recorded hash. All non-budget learner settings, recurrent core, vectorization, environment, and selfplay settings match the archived reference runs.

**The observed time frontier has room for both our CNN and IMPALA. Our configurations improve the middle of the score/time tradeoff; IMPALA retains the highest scores. Nature remains useful at lower time budgets. Impoola contributes no points to the combined frontier in the matched-seed comparison.** These are measured points, not fully optimized frontiers for each reference architecture.

- [Interactive full frontier](results/connect4cnn/sweep._u86vi03/analysis/frontier.html): all points, per-family frontiers, combined frontier, time/steps axes, a short-time zoom, seed-73/three-seed-reference views, and optional winner checkpoints. Default view uses all Flex final checkpoints.
- [Every observation](results/connect4cnn/sweep._u86vi03/analysis/observations.csv), [all frontier memberships](results/connect4cnn/sweep._u86vi03/analysis/frontiers.csv), [time-budget table](results/connect4cnn/sweep._u86vi03/analysis/budget_table.csv).
- [Original sweep report](results/connect4cnn/sweep._u86vi03/REPORT.md), [audit](results/connect4cnn/sweep._u86vi03/analysis/summary.json), [evaluation receipts](results/connect4cnn/sweep._u86vi03/analysis/evaluations.csv).

## What was compared

Every one of the **128 final Flex checkpoints** was evaluated with seed 10073 and 1,024 requested games, against the unchanged scripted Connect4 opponent. Actual batched episode counts are recorded. An additional 12 earlier checkpoints of the training-selected winner were evaluated, giving 140 Flex observations. These are compared with all **36 archived reference checkpoints**: four checkpoints × three training seeds × Nature/IMPALA/Impoola. No training was run for this analysis.

The primary comparison pairs training seed 73 / evaluation seed 10073. The alternate reference view averages three training seeds 73/74/75 and their paired evaluation seeds. Flex still has only one training seed; averaging reference seeds does not make the search winner a three-seed result. The best training-score trial remained the best final Flex checkpoint after evaluating all 128.

An observed point is Pareto-dominated if another point achieves at least its win rate at no greater cost, with at least one strict improvement. All raw points remain visible and archived, including near-zero scores. Frontier membership uses unrounded stored scores and costs; tiny differences at near-zero performance have little practical significance.

## Complete combined wall-time frontier: Flex final checkpoints versus reference checkpoints

Training seed 73 for every row. This table includes all combined-frontier points, not just a selected winner. It excludes earlier checkpoints of the Flex winner; those are a separate view below. All 128 final Flex points and all 12 seed-73 reference checkpoints enter the comparison.

| Point | Actual steps | Approx. wall seconds | Held-out wins |
|---|---:|---:|---:|
| Flex wild-maple-6 | 2,193,408 | 26.48 | 0.00% |
| Flex lucky-robin-7 | 2,332,672 | 26.57 | 0.17% |
| Flex wild-maple-2 | 2,414,592 | 27.09 | 0.17% |
| Flex wild-falcon-50 | 2,791,424 | 34.61 | 0.25% |
| Nature, checkpoint 1 | 3,319,808 | 43.07 | 13.72% |
| Flex happy-bear-34 | 7,239,680 | 76.81 | 15.75% |
| Nature, checkpoint 2 | 6,639,616 | 83.77 | 75.34% |
| Flex cosmic-wolf-116 | 9,484,288 | 103.73 | 81.55% |
| Flex gentle-tree-110 | 10,854,400 | 115.67 | 82.94% |
| Flex silver-robin-80 | 11,126,784 | 123.09 | 85.63% |
| Flex silver-owl-109 | 12,013,568 | 127.17 | 87.80% |
| Flex calm-otter-91 | 12,730,368 | 137.18 | 88.21% |
| Flex brave-comet-51 | 12,662,784 | 137.57 | 92.00% |
| IMPALA, checkpoint 2 | 6,639,616 | 613.97 | 97.67% |
| IMPALA, checkpoint 3 | 9,959,424 | 920.56 | 98.63% |
| IMPALA, checkpoint 4 | 13,279,232 | 1,227.11 | 98.99% |

Nature's later checkpoints, IMPALA's first checkpoint, and all four Impoola checkpoints are dominated by other observed points in this view. This does not establish that the underlying architecture could never win after tuning.

## Earlier checkpoints improve our observed frontier further

The training-selected `brave-comet-51` already achieves the following held-out results during its 12.66M-step run:

| Steps | Approx. wall seconds | Held-out wins |
|---|---:|---:|
| 3,072,000 | 35.72 | 7.77% |
| 4,096,000 | 47.14 | 34.51% |
| 5,120,000 | 58.21 | 70.81% |
| 6,144,000 | 69.03 | 84.82% |
| 7,168,000 | 79.80 | 88.24% |
| 8,192,000 | 90.56 | 90.53% |
| 9,216,000 | 101.31 | 91.12% |
| 10,240,000 | 112.11 | 92.05% |
| 12,662,784, final | 137.57 | 92.00% |

Thus the observed first checkpoint above 90% is **8.192M steps / 90.56 seconds**. The exact crossing was not measured. The 10.24M checkpoint dominates its own final checkpoint in the measured time/score space; the tiny score difference should not be interpreted as meaningful regression. Learning-rate annealing was configured for the full run, so independently training with an 8.192M total budget is not equivalent to loading this checkpoint.

With this optional curve included, our intermediate checkpoints dominate most final Flex points and Nature's later points; IMPALA retains the 97.67–98.99% region. Evaluation density is unequal: the references have four checkpoints each and only our selected winner has a full curve. These observations justify testing an earlier stop, not claiming a proven globally optimal stopping time. The chart keeps this view separate rather than silently mixing it into the final-trial frontier.

## Throughput and final quality context

These SPS figures all use **native adjusted uptime**, not whole-process SPS. Reference values are three-seed arithmetic means; Flex values are individual selected trials.

| Model | Parameters | Final steps | Native average SPS | Native seconds | Final held-out wins |
|---|---:|---:|---:|---:|---:|
| Flex brave-comet-51 | 160,736 | 12,662,784 | 92,180 | 137.37 | 92.00% |
| Flex cosmic-wolf-116 | 261,856 | 9,484,288 | 91,600 | 103.54 | 81.55% |
| Flex clever-otter-60 | 109,768 | 12,431,360 | 91,259 | 136.22 | 86.29% |
| Nature, three-seed mean | 138,528 | 13,279,232 | 81,015 | 163.93 | 78.71% |
| IMPALA, three-seed mean | 270,496 | 13,279,232 | 10,803 | 1,229.21 | 99.19% |
| Impoola, three-seed mean | 151,712 | 13,279,232 | 10,837 | 1,225.40 | 69.49% |

The selected Flex winner has **13.8% higher native SPS than Nature**, and roughly **8.5×** the native SPS of our current IMPALA/Impoola implementations. It has higher measured final quality than Nature/Impoola but lower than IMPALA. These are preliminary observed tradeoffs, not seed-confirmed architecture superiority. The earlier reports' Nature 80,840 SPS and IMPALA 10,800 SPS were whole-process figures; the small difference here is a timing-boundary change, not newly measured reference throughput.

The winning architecture is a single SAME 7×7/s4 convolution with 16 channels, ReLU, flatten, projection 64, then a 128-wide output feeding the unchanged hidden-128/one-layer recurrent core. No pooling or residual branch. Despite fewer convolutional layers it has **16.0% more total parameters than Nature**, largely due to the flattened projection. The smaller finalist uses 8 channels, 8×8/s4, projection 64 and has 20.8% fewer parameters than Nature. Kernel profiling has not been performed; these results do not isolate implementation improvements from architecture cost.

## What the search learned, and what it did not test fairly

- **114/128** runs reached at least 6.6M decisions; the broader budget range gave useful learning signals. 100 trials exceeded 50% training wins, 41 exceeded 80%, 22 exceeded 85%, and one exceeded 90%.
- **120/128** trials used one stage. All eight deeper trials received only **2.19–2.79M decisions**. Their near-zero scores cannot establish that deeper networks, pooling, or skips are inferior. Depth and sufficient training budget were confounded by the adaptive search.
- The most common configuration (16 channels, 8×8/s4, projection 64) received 37 runs. The 7×7 and 6×6 counterparts received 18 and 17. The sweep concentrated on a narrow region after its initial exploration.
- Four groups repeated the exact active configuration, requested budget, and training seed: sizes 6, 3, 14, and 2. Within every group, checkpoints were byte-identical. That is **21 repeat runs beyond the first**, or 16.4% of the campaign, with no new architecture/budget/seed coverage.
- Identical checkpoint groups had varying reported training scores (one group 78.69–81.59%). Native training metrics summarize episodes between wall-clock-driven logging events, so equal final weights can receive different last-window scores. Their held-out evaluation uses a fixed protocol. Do not interpret those repeated training-score fluctuations as independent learning gains or failures of checkpoint determinism.

## Limits and next experiment

All policies use the same 7×6 Connect4 game and 1×36×44 pixel observations, float32, RTX 5060, common learner settings, and fixed recurrent core. Core/learner/vec/env equality was checked from all resolved trial INIs against all nine reference runs. Budgets, encoder, checkpoint frequency (500 versus 1,621 updates), history downsampling, source revision, and concurrent W&B sidecar differ. Source/config/binary provenance is archived. Point wall costs are approximate launch-to-checkpoint times; full campaign time includes search overhead, while point costs exclude compilation and subsequent evaluation. No modern framework or CUDA-stack changes were made.

The baseline families were not independently architecture/hyperparameter-swept under the same 128-trial search budget. This compares their observed checkpoint curves with a selected single-seed search, not a complete or fair-search-budget claim about the best attainable Nature/IMPALA/Impoola frontier. All 128 Flex final scores now form a used analysis set; further tuning needs fresh confirmation.

Next, confirm several tradeoff points with new training seeds and the same evaluation/timing protocol, including the earlier stopping point. Deduplicate active configuration/budget/seed combinations, and guarantee some longer-budget coverage for deeper candidates. Sweep the reference families' budgets with comparable search resources before claiming their optimized frontiers are beaten. The current result supports pursuing a faster middle-to-high-score region while retaining IMPALA as the high-score reference.

Reproduce locally without training: source `ocean/connect4cnn/runtime_env.sh`, run `.venv/bin/python research/analyze_cnn2.py build/connect4cnn/sweep._u86vi03 --all-final` (resumes evaluation receipts), then `.venv/bin/python research/cnn2_frontier.py build/connect4cnn/sweep._u86vi03`. GPU evaluation requires execution outside the sandbox. Full checkpoints remain in the ignored build directory; the archive preserves small evidence and chart data.
