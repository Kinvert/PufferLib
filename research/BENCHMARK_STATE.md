# Source state for the five-seed CNN results

October 5 caveat: the measured source below contains an inherited draw-rule bug,
now corrected in both state and pixel games. These results remain **legacy-rule
evidence**; they cannot be pooled with corrected-rule results. See the
[defect, regression checks and version boundary](CONNECT4_DRAW_CORRECTION.md).
The effect on model rankings has not been measured.

Recorded September 14, 2026 for campaign **`confirm.ol9tcj5k`**. See [results and full Pareto curves](CONFIRMATION_RESULTS.md) and the [frozen protocol](CONFIRMATION_PROTOCOL.md).

## Git revisions

| Revision | What it identifies |
|---|---|
| `b2fa7787a754d36362374ac271ea6c7b23beeb25` | Reconstructed measured source: **all 38 captured source/config files match their recorded SHA-256 hashes**. Use this revision when referring to the implementation that produced these results. |
| `9ed6bc2a192e87c9b006667f2b06392d049a9ab9` | Original Git HEAD at launch, with uncommitted work. This base alone does not identify the measured implementation. |
| `5352d24ec1b740ed05ebe11e01126c8e3886cfb0` | Later workspace state: archived results and analysis, PongCNN, and Connect4 representation presets. Those later features were not part of the confirmation. |

The measured-source commit was created **after** the experiment by overlaying the captured files on the original base. It did not exist at launch. It matches the complete captured manifest, not a claimed snapshot of every uncaptured file in the original dirty workspace. The [original protocol receipt](results/connect4cnn/confirm.ol9tcj5k/protocol.json) retains its original `revision` and `source_sha256` fields unchanged.

This note is committed separately after the two revisions above, so their hashes can be recorded without a self-referential commit hash.

## State that produced the performance

- Native C/CUDA PufferLib training on G240's **RTX 5060, float32**.
- Original **7-column × 6-row Connect4**, same scripted opponent, grayscale **1×36×44** input and solid 6×6 cell rendering. This corresponds to the later `representation=0`; the selector did not yet exist.
- Six fixed models, each trained from scratch for **13,312,000 agent decisions**, five paired training seeds **173–177**, and 13 equally spaced checkpoints. Evaluation seeds **20173–20177**, 1,024 requested games per checkpoint; actual batched counts are recorded.
- Common hidden-128, one-layer recurrent core, 64 agents, rollout horizon 32, minibatch 2,048, replay ratio 1, learning rate .001 annealed across the full budget, synchronous rollout and CUDA graphs. Exact effective configs are archived per job, for example [quality/seed 173](results/connect4cnn/confirm.ol9tcj5k/flex_quality-s173/config/connect4cnn.ini).
- **30 completed jobs, 390 held-out evaluations**, no native or upload failures. Checkpoint hashes, finite parameter arrays, compiled binary hashes and matched non-encoder settings were audited.

Final-checkpoint five-seed means; wall time is the entire training process and SPS is the native trainer metric:

| Model | Held-out wins | Train wall | Native SPS |
|---|---:|---:|---:|
| Ours quality | 79.66% | 144.47 s | 92,382 |
| Ours small | 78.01% | 143.28 s | 93,134 |
| Ours fast | 71.66% | 147.12 s | 90,687 |
| Nature | 73.35% | 158.71 s | 84,066 |
| IMPALA | 97.45% | 1,231.81 s | 10,811 |
| Impoola | 80.09% | 1,229.68 s | 10,830 |

Across all checkpoint means, ours forms the low-cost time frontier and IMPALA the high-score frontier. IMPALA owns the mean decision-count frontier. Quality's paired final score advantage over Nature is +6.31 percentage points, with a pointwise 95% bootstrap interval of **−0.07 to +13.00**: the score advantage remains uncertain. These are one game's results with these native implementations, not an established general-purpose CNN win.

## Later work and recovery

The later feature revision adds ten Connect4 appearance presets and PongCNN. Representation CPU checks and a three-trial native GPU canary passed; no long representation sweep was run. PongCNN passed CPU parity/pixel/sanitizer checks, but native GPU build/train/reload remains pending. Neither experiment contributes to the table above.

To inspect the measured source without replacing current work, create a separate worktree when needed:

```bash
git worktree add --detach ../cnn-confirmation b2fa7787a754d36362374ac271ea6c7b23beeb25
```

Use the recorded per-job configs and commands to reproduce a particular run; the revision's default config alone does not specify all six models and seeds. Consult `AGENTS.md` and `ocean/connect4cnn/runtime_env.sh` for existing build dependencies. Rebuilding requires the appropriate local dependencies; matching captured source does not promise bitwise-identical builds or equal speed on another GPU.

The later feature/results revision contains the [raw receipts](results/connect4cnn/confirm.ol9tcj5k/), [analysis code](analyze_confirmation.py), [full report](results/connect4cnn/confirm.ol9tcj5k/analysis/REPORT.md), and [interactive frontier](results/connect4cnn/confirm.ol9tcj5k/analysis/frontier.html). Full checkpoint arrays, compiled binaries, and the original source-copy directory remain local under ignored `build/connect4cnn/confirm.ol9tcj5k/`; they are not included in Git. The captured source files themselves are recoverable from the measured-source commit.

On this machine, regenerate the audited analysis from the current checkout with:

```bash
.venv/bin/python research/analyze_confirmation.py build/connect4cnn/confirm.ol9tcj5k
```

Preserve the raw evidence. The archived protocol SHA-256 is `3b80841dde59b719e465015b0369b7cbc06a4362de6706181f3c4ff615effa79`; results CSV SHA-256 is `a7a56eef74bab7be9314c502e180470559834de3091783bfdee69d3f614ce054`.
