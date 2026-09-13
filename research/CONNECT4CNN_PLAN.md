# Connect4CNN: first end-to-end CNN experiment

Started 2026-09-12. This is a staged implementation plan, not a benchmark result.

## First decision: enlarge the image, keep the game

Copy the native Connect4 environment into `ocean/connect4cnn/connect4cnn.h`, with its own `config/connect4cnn.ini`. Preserve the standard six-row, seven-column game, seven actions, scripted opponent, rewards, and terminal/reset behavior. Keep upstream Connect4 untouched as a differential-test reference.

Use a single-channel 36-high × 44-wide image: every board cell is a solid 6×6 pixel block, with one blank pixel column on each side. Float intensities are empty=0, player=1, opponent=0.5. Store contiguous row-major pixels, top row first; the policy-facing shape is `[N,1,36,44]`. These are normalized synthetic grayscale pixels, not a screenshot of the decorated human viewer. No frame stacking or privileged state is added. The native interface carries a flat array of 1,584 floats; the future CNN adapter must explicitly reshape it.

**Correction to the initial 36×42 suggestion:** fitting the convolution kernels does not guarantee full image coverage. Nature's final 1×1 feature at that size sees only the leftmost 36 columns and drops the entire seventh board column. Width 44 yields two overlapping final receptive fields spanning the full image. The two padding columns preserve all cells without changing cell size or game rules.

Larger cells add no new game information. Their purpose is to accommodate the reference convolution stacks while keeping the task small. The game is fully visible; recurrence is not needed to recover hidden board cells. Fix the opponent implementation and seed protocol in comparisons.

Making every cell one pixel on a much larger game board would be a different experiment: it changes the action count, horizon, strategy, and opponent cost. The existing bitboard and opening book also assume 6×7 and cannot simply be resized to 36×42 cells. Defer that experiment until the basic workflow works.

## Architecture shapes and limitations

| Encoder | Spatial progression for 36×44 |
|---|---|
| Nature, valid convolutions 8/s4, 4/s2, 3/s1 | 8×10 → 3×4 → 1×2 |
| IMPALA, three SAME stride-two pooling stages | 18×22 → 9×11 → 5×6 |
| Impoola | Same IMPALA stack, then global average pooling |

Keep reference kernel sizes; adapt the input channel count and final projection dimensions explicitly. All encoders receive the same image and use the same feature dimension and head/core configuration within a comparison. This minimal input is a plumbing target, not a neutral representation for ranking all CNNs: Nature's stride alignment may favor certain board positions. A later common larger-resolution ablation can assess that sensitivity.

## Milestones and acceptance criteria

1. **Pixel environment (current work):** copy Connect4, implement observations, fix reset and human-viewer assumptions about the old vector, and add the environment config. Validate every pixel against known boards, buffer bounds, reset/terminal behavior, and seeded trajectory parity with upstream Connect4. This does not yet implement a CNN learner.
2. **One tiny CNN learns:** add a bounded encoder interface and reference implementation; validate forward/backward calculations, then run a short training smoke test. Save resolved config, code identity, observation contract, seeds, logs, and checkpoints. Confirm checkpoint reload and independent evaluation. Establish learning relative to a fixed random-policy baseline before expanding.
3. **Two-encoder workflow:** add faithful adapted Nature through the same interface. One command schedules both architectures and produces return-versus-agent-decisions and return-versus-elapsed-time plots plus a machine-readable result table. Include failures and runs that do not reach the target. Freeze evaluation seeds separately from training seeds; opponent tie-breaking is seeded stochasticity.
4. **Complete the reference set:** add IMPALA and Impoola with exact documented padding/readout choices and numerical checks. Start with a common learner recipe, seed lists, budget, and tuning allowance. The tiny custom CNN can lose; a trustworthy comparison is the milestone.
5. **Broaden the task panel:** add one slightly harder visual task, then one harder task, after checking baseline learnability and integration cost. Keep Connect4CNN as a regression test. These custom tasks do not inherit published Procgen/ALE scores.
6. **Improve architectures and implementations:** methodically change one factor at a time. Compare architecture changes on one backend, then port/profile measured bottlenecks. Record combined whole-agent improvements separately from implementation-only speedups.

Measure observation generation, encoder and learner time, full training time, memory, evaluation return/win rate, invalid-action rate, and episode length. Report same-seed repeatability separately from uncertainty across different seeds. Select meaningful training budgets and target returns from pilot data, not invented numbers.

## Scope and constraints

- The copied config initially retains the vector Connect4 training recipe for provenance. It is not tuned for images. The native build now selects the tiny custom CNN; use the documented bounded smoke runner before broader comparisons.
- No system CUDA, drivers, global packages, or other virtual environments may change. Any later Torch dependency belongs only in this project's existing `uv venv`.
- Use PufferLib's existing `build.sh`, shared root-level Raylib dependency, and CPU runner. No separate per-environment Raylib copy, header shim, or screenshot path. Pixel generation writes directly to the native observation buffer. Headless environment tests use the shared library with a local C compiler; no GPU or display is needed. Generated test files belong under `build/connect4cnn/`.
- Work stays local. Keep changes reviewable and preserve the original conversation transcripts.

## Milestone 1 completed, 2026-09-12

Implemented [Connect4CNN](../ocean/connect4cnn/README.md) and its config. Standard CPU build and a 10-episode headless smoke run passed. The test harness checks all cells/owners, reset and terminal paths, observation bounds, 4,096 matched transitions against upstream Connect4, and repeatability, with ASan/UBSan. The same shared Raylib 5.5 package used by normal PufferLib builds is reused at the checkout root; no per-environment rendering dependency or system change was introduced. CNN integration and training remain milestone 2.

## Milestone 2 training path verified, 2026-09-12

Added a tiny native `Conv4×4/s4,8 channels → ReLU → Linear(hidden)` encoder, using existing PufferLib allocations, GEMMs, MinGRU, optimizer, and checkpoint/evaluation paths. Registered it with five added lines in `src/ocean.cu`. Float32 GPU forward/gradient reference checks and eager/graph repeatability passed on G240's RTX 5060. Two 65,536-step training runs produced identical finite checkpoints; checkpoint reload/evaluation passed. The GPU is accessible through WSL outside the sandbox; no Docker or system changes are required.

The policy is still weak (zero wins in the first short checkpoint evaluation). The training plumbing portion of milestone 2 is complete; meaningful learning against a fixed baseline, broader performance measurement, and BF16 validation remain. See the environment README for exact commands and artifact locations.

## Matched state/pixel runner verified, 2026-09-12

Added `ocean/connect4cnn/compare.sh`, `compare.py`, and one shared `compare.ini`. Original state-based Connect4 is a permanent comparison baseline. The runner builds separate binaries from current source, checks effective settings agree, trains serially with paired seed lists and alternating order, evaluates fixed saved checkpoints on separate seeds, and writes Markdown/CSV with parameters, timing, actual episode counts, and preserved artifacts. Four runs (two policies × two seeds) and eight checkpoint evaluations passed at 65,536 steps per run. Both final win rates were zero at that smoke budget. There were no additional `src` changes for this runner.

## Longer learning comparison completed, 2026-09-12

At 13,279,232 decisions per run and three training seeds, tiny CNN evaluation win rate averaged 64.04%, versus 18.51% for state, with process throughput of 97,733 versus 108,831 SPS. All six runs and 24 fixed-checkpoint evaluations passed. This supersedes the smoke-only zero-win observation above and demonstrates useful learning, without establishing convergence. The dedicated fixed random-policy baseline and BF16 validation remain outstanding. Full seed results and timing definitions are in [the persistent experiment history](EXPERIMENT_LOG.md); future comparisons append there automatically. Proceed toward faithful Nature integration using this measured workflow.

## Stock baseline and Nature validation, 2026-09-12

Stock training settings in float32 achieved **98.87% mean wins and 306,403 process SPS** over three seeds, at 13,238,272 actual decisions each. Evaluation controls match the earlier comparisons. This shows the modified common recipe substantially weakened the state baseline; tiny CNN has not beaten stock-config PufferLib. Keep `stock_state` separate from `state` in all tables.

Adapted Nature is now implemented and selectable with `--variants nature_cnn`, using the same pixel environment/core/learner as tiny CNN. Float32 numerical and repeatability checks passed, including byte-identical checkpoints in two same-seed smoke runs. The Nature convolution stack and bias/ReLU choices are preserved; input dimensions and projection width are explicitly adapted. The full-budget three-seed evaluation completed with **78.71% mean wins and 80,840 process SPS**, all 12 evaluations passed. See the environment README for commands and architecture details, and the experiment history for the four-policy table. Milestones requiring plots, a dedicated random baseline, BF16, IMPALA, and Impoola remain outstanding.

## Reference set measured, 2026-09-13

Adapted IMPALA and Impoola now share the validated original-SAME backbone and differ by flatten versus GAP readout. Numerical checks and repeated training smoke tests passed. All six full-budget runs and 24 held-out evaluations completed: IMPALA averaged **99.19% wins / 10,800 SPS**, Impoola **69.49% / 10,833 SPS**. The reference set is measured under the common recipe; these are initial native implementations, not optimized speed limits. Plots, a dedicated random baseline, and BF16 validation remain outstanding. The next proposed work is [INI-controlled CNN architecture search with native PROTEIN](CNN_SWEEP_INTEGRATION.md); that integration has only been researched, not implemented.
