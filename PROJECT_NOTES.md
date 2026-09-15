# CNN project notes

## 5090 hardware-comparison handoff

Prepared `cnn-research` for the Kinvert-owned `Kinvert/PufferLib` fork only; official `PufferAI/PufferLib` pushing remains disabled. [Hardware comparison instructions](research/HARDWARE_COMPARISON.md) provide fresh-checkout/local-venv setup using the existing CUDA/NCCL toolchain, a four-model canary, and fixed full-budget comparison. Portable `nvidia-smi` lookup, optional competing-process rejection and host/compiler receipts were added to the reporting runner; native model/training code is unchanged. Twenty-one CPU configuration/sidecar/hardware tests passed. No additional GPU job was launched on G240, and the separate 5090 host has not yet been tested. A same-revision G240 run after Pong completes is needed for a controlled cross-host timing comparison.

## Additional potential deliverable — CNN constructor and search tool

Kinvert proposed delivering a native PufferLib CNN constructor that starts with small, fast candidates and explores more expensive features to find useful per-environment Pareto frontiers. [Brief plan](research/CNN_CONSTRUCTOR_PLAN.md) records the workflow, staged implementation, exportable INIs, search-cost accounting and required comparisons. This complements the fixed-CNN research; it does not assume official acceptance, global optimality or a completed SOTA result.

## 2026-09-14 — Pong campaign recovery

`hypers.jwyApeXS` stopped after 24 ours and 24 Nature training trials because one Nature evaluation timed out at 180 seconds. Original failure retained; no selective extended retry. Recovery tmux `pong-cnn3-resume-20260914`, log `build/pongcnn/cnn3-hypers-resume.log`, uploads Nature's completed results and continues only unstarted IMPALA/Impoola using original binaries/configs/caps. Completed training is not repeated. [Protocol](research/PONG_HYPER_SWEEP.md) records the revised failure-reporting behavior and provenance. Do not restart the original whole campaign or actively poll the recovery.

## 2026-09-14 — Pong frozen-model training search launched in cnn3

User authorized Pong experiments while locking our CNN architecture and searching training hypers. [Protocol](research/PONG_HYPER_SWEEP.md): Connect4-selected quality shape and H128/L1 core fixed, with fixed Nature/IMPALA/Impoola references receiving the same nine training/budget dimensions. Native C/CUDA PROTEIN and training, Bash runner, external reporting/W&B; no core/kernel/system/dependency changes. Five-model train/reload canary and eight-trial online sweep canary passed; 16 CPU safeguards/regressions passed. A 13.312M-decision quality pilot scored zero at 61.82 seconds, retained as negative evidence.

Launched detached campaign `build/pongcnn/hypers.jwyApeXS`, tmux `pong-cnn3-hypers-20260914`, log `build/pongcnn/cnn3-hypers-launch.log`, W&B `kinvert-k/cnn3`. Limits are 24 completed trials and 3,600 native sweep-process seconds per family, four families serially. Separate evaluation of every completed final checkpoint and W&B upload follow each family, so runs appear online in batches. Startup session verified; full results pending. Do not launch a duplicate or actively monitor. This is development search, not fresh-seed confirmation or a Nature/SOTA claim.

## 2026-09-14 — Thesis-defense standard for the Nature advantage

Kinvert explicitly requires substantially more certainty that our advantage over Nature is real, with claims in the arXiv paper verified as rigorously as a thesis defense. [The paper plan](research/PAPER_PLAN.md#required-evidence-standard-defend-the-claim-like-a-thesis) now requires predeclared claims and statistical design, adequate independent replication, frontier-appropriate uncertainty, fair and efficient baselines, checks of alternative explanations, and independently reproducible evidence. The current five-seed score interval includes zero; the completed confirmation is not proof of superiority. Preserve negative and inconclusive evidence, and restrict claims to the tasks and budget region actually supported.

## 2026-09-14 — Benchmark source and later work committed separately

Measured-source commit `b2fa7787a754d36362374ac271ea6c7b23beeb25` matches all 38 captured file hashes for `confirm.ol9tcj5k`; it retrospectively reconstructs the captured files from the dirty launch state based on `9ed6bc2a`. Later features and archived results are in `5352d24ec1b740ed05ebe11e01126c8e3886cfb0`. [Benchmark state and recovery instructions](research/BENCHMARK_STATE.md) records the distinction, measured settings and performance. No push or new training was performed for these commits.

## 2026-09-14 — Five-seed confirmation complete and audited

`confirm.ol9tcj5k` completed all 30 jobs, 390 evaluations and W&B uploads without failures. [Full analysis and interactive curves](research/CONFIRMATION_RESULTS.md). Means at 13.312M: ours quality 79.66%/144.47 s, small 78.01%/143.28 s, fast 71.66%/147.12 s; Nature 73.35%/158.71 s; IMPALA 97.45%/1,231.81 s; Impoola 80.09%/1,229.68 s. All 390 checkpoint hashes/counts/finiteness and source/config/binary receipts were audited. The combined mean wall-time frontier contains ours at lower costs and IMPALA at higher scores; Nature/Impoola contribute no mean points. IMPALA owns the mean decision-count frontier. Quality's paired final advantage over Nature is +6.31 pp with pointwise 95% bootstrap interval −0.07 to +13.00 pp: promising but unresolved. The development winner's 92% score is not typical across fresh seeds. No new training was launched for this analysis. Paper-plan item 1 is complete; broader fairness/generalization/delivery remain open.

## 2026-09-14 — Sweepable Connect4 representations

Added native integer `env.representation` IDs 0–9, default 0 preserving the old observation. Presets cover square sizes/gaps, circles, X/O, and smaller centered boards (including one pixel per cell), all in the same 36×44 tensor with unchanged rules and encoder kernels. [Exact presets and sweep usage](ocean/connect4cnn/REPRESENTATIONS.md). CPU fixtures and 4,096 original-game parity transitions per preset, repeats, invalid-ID rejection and ASan/UBSan passed. Native canary `sweep.wstneiqe` completed three 32,768-decision trials, representations 0/2/4, same 160,736-parameter architecture, all weights finite. Twelve tooling tests and two confirmation tests passed. No long sweep or online W&B run was launched. Appearance is recorded separately from architecture; robust performance requires coverage across representations rather than letting the optimizer pick only the easiest rendering.

## 2026-09-14 — PongCNN development task

Added native `ocean/pongcnn/pongcnn.h`, copied from original Pong with unchanged rules/opponent/rewards. Synthetic float32 1×36×44 images include paddles, ball and two score bars; direct C buffer writes, one observation per decision, no renderer dependency in training. Two native factory conditions now also accept `PUFFER_PONGCNN`, reusing all existing CNN kernels. Original Pong is untouched. CPU tests passed 49,152 state/RNG/reward/reset transitions against the original, repeatability and pixel fixtures under ASan/UBSan. Five isolated common-config canary inputs were prepared and checked; no GPU build or training was run during Connect4 confirmation. Details, pending native validation, information differences and commands: [PongCNN README](ocean/pongcnn/README.md). This is a development task; no cnn3 sweep was started.

## 2026-09-14 — Frozen confirmation queued

User authorized paper-plan item 1. The fixed panel, fresh seeds and matched measurement schedule are in [CONFIRMATION_PROTOCOL.md](research/CONFIRMATION_PROTOCOL.md); all six paper actions are explicitly listed in [PAPER_PLAN.md](research/PAPER_PLAN.md). Corrected six-model canary `confirm-canary._0y3shor` passed 24 evaluations and six online W&B uploads. Both canary attempts are archived, including the first missing-INI-key failure. No native training/kernel source changes were needed for this confirmation tooling.

Full confirmation is queued in tmux `cnn2-confirm-20260914`, launcher log `build/connect4cnn/confirmation-launch-20260914.txt`. At startup verification it was waiting for GoldenEye evaluation PID 937816 to release the GPU; the automatic wait has a one-hour cap. Once free, it runs 30 serial jobs (six fixed models × five seeds), 13.312M decisions each, 13 checkpoints each, with online `kinvert-k/cnn2` logging between jobs. Estimated training time is roughly four hours after starting. The campaign's generated `confirm.*` path will appear in the launcher log. Do not actively monitor or launch a duplicate; check on user request. Completion and valid uncontended timing remain to be established from receipts.

## 2026-09-14 — Paper with a practical software contribution

Kinvert wants to treat the project as a paper backed by real use, clear evidence, and apples-to-apples comparisons. Current Connect4-only support is an acknowledged weakness; the next research goal is architecture generalization across environments. Architecture invention remains paused while measurement and verification improve. [PAPER_PLAN.md](research/PAPER_PLAN.md) records the proposed claim/evidence matrix, matched versus equally tuned comparisons, fresh seeds and held-out game/level splits, complete uncertain Pareto frontiers, native pixel-input portability, and runnable PufferLib delivery criteria. Architecture transfer means freezing the design before retraining it from scratch on unseen tasks; policy transfer with the same weights is a separate claim. The plan is not a completed study or authorization to launch an unspecified large experiment.

## 2026-09-11 — Project intent and workspace

### User context

- This g240 computer has many PufferLib clones, mostly for environment work. Several contributions were merged, cherry-picked, or independently implemented upstream.
- Affine Lock was developed as a potential benchmark: serious work intended to help the field, attract attention, and support finding a job.
- This CNN project is also intended as serious work, with a possibility of producing a state-of-the-art result and something useful to PufferLib.
- The user wants help developing and executing the project while learning the details.

### Goals expressed by the user

- Build a CNN suitable for the PufferLib ethos, described by the user as: "Determinism with orders of magnitude higher speed than all the other researchers running python garbage code."
- Treat determinism and speed as central goals. Orders-of-magnitude speedups and state-of-the-art performance are ambitions to investigate and measure, not established results.
- Produce useful work with credible evidence that can also demonstrate the user's abilities to potential employers.
- Bring PufferLib into this workspace, preferably with its repository files directly under `/home/claude/cnn`, rather than in a nested PufferLib directory.
- Start maintaining Markdown notes before proceeding with setup or experiments.

### Existing source material

- `JOSEPH_CONVERSATION.md`: verbatim account of the conversation with Joseph Suarez.
- `CLAUDE_RESEARCH.md`: verbatim pasted prior assistant discussion, including research suggestions and proposed questions.
- The pasted research is source material for investigation; its citations, factual claims, and novelty claims have not been verified here.

### Research question carried forward

Joseph's question concerns the compute-efficient convolutional encoder for pixel-based RL and the allocation of computation between that encoder and a recurrent core. Nature DQN and IMPALA-style encoders are the initial reference points. An encoder/core allocation study and a faster encoder implementation are related possibilities; the final contribution has not yet been chosen.

### Workspace findings and proposed setup

- Initialized Git directly in `/home/claude/cnn` and checked out official PufferLib's `5.0` branch onto local branch `cnn-research`.
- Starting commit: `bea26c718b61cf2377f8e2ff247fcf687478bcdf` (2026-09-11, "Minor fixes"), the latest official `5.0` commit fetched during setup. The official default branch was still `4.0`, so `5.0` was selected explicitly.
- `upstream`: `https://github.com/PufferAI/PufferLib.git`, with its push URL set to `DISABLED` to block accidental pushes through that remote.
- `origin`: `https://github.com/Kinvert/PufferLib.git`. The user is Kinvert and owns this fork. The fork was verified to exist; it did not advertise a `5.0` branch during setup.
- `cnn-research` tracks `upstream/5.0` for updates; the default push remote is `origin`.
- Existing Markdown files had no upstream path collisions and were preserved during checkout. This checkout is independent of the other local PufferLib clones.
- No code has been pushed, installed, built, or trained as part of repository setup.

### Publication preference

- The user explicitly requires work to be worthwhile and **clean** before contributing it to official PufferLib.
- Keep development local for now. Future authorized publication should use the Kinvert fork; prepare clean, reviewable contributions before proposing anything upstream.

### Decisions to resolve as the project develops

- Starting revision is pinned above; decide later whether any existing local work should be brought into this project.
- Which hardware and execution path should the implementation target?
- What does determinism require: repeated runs on one machine, reproducible training trajectories, or agreement across devices?
- Which result is the primary target: encoder latency, full training throughput, time to a target return, or return under a fixed compute budget?
- Which pixel tasks and memory requirements should the benchmarks cover?
- What baselines, training budgets, seeds, and measurement procedures will make the results convincing?

### Working principles proposed by the assistant

- Establish measured baselines before claiming improvements or novelty.
- Measure end-to-end training alongside encoder performance: fewer FLOPs alone do not establish a wall-clock speedup.
- Evaluate learning quality alongside speed so an implementation improvement does not conceal a worse policy.
- Keep provenance, configurations, results, and decisions in the project as it develops.

## 2026-09-11 — Research library and project design

- User authorized research, local paper-to-Markdown conversion, and local retrieval over papers/code for future agents. Sub-agents were authorized for this research phase.
- Created `research/` with primary-paper PDFs/HTML, Markdown conversions, acquisition manifest, and source/version/hash receipts. The initial collection contains 25 papers, including Nature DQN, IMPALA, Impoola, BBF, scaling, recurrence, efficiency, statistics, and visual-memory benchmarks.
- Added reports on exact CNN architectures, CUDA implementation and swappable experiment scaffolding, and benchmark/sweep methodology. `research/README.md` is the entry point.
- Created `.venv` using `uv venv --python /usr/bin/python3.12`; actual interpreter is Python 3.12.3. Locked CPU research dependencies include LanceDB 0.38.0 and FastEmbed 0.8.0. Binary-wheel installation, package compatibility, CPU embeddings, and initial retrieval checks succeeded.
- Search combines keyword and semantic retrieval, returns source paths/line ranges, checks file hashes, and supports incremental refresh. The model is `snowflake/snowflake-arctic-embed-xs`; downloaded papers and generated caches stay out of Git by local ignore rules.
- User's system constraint: CUDA-related reading/inspection is allowed; do not change the system CUDA stack, drivers, cuDNN, system libraries, global settings, or other environments. Basic Torch work may be confined to `/home/claude/cnn/.venv`. Current research tooling does not require or install Torch.
- Important corrections: BBF supports successful encoder scaling; Dreamer scales CNN and recurrent components together; this checkout's Breakout uses state features rather than pixels. The earlier pasted response remains unchanged and should not be treated as verified literature.
- Proposed first implementation target: faithful Nature/IMPALA references and native comparison harness, explicit encoder/feature/core dimensions, original pixel-environment protocol, then staged controlled sweeps. No CNN/trainer implementation or learning experiment was performed during research.

## 2026-09-12 — Upstream 5.0 refresh

- Checked official `upstream/5.0`; it had eight new commits. Fast-forwarded `cnn-research` from `bea26c718b61cf2377f8e2ff247fcf687478bcdf` to `89414204ce8509c3fecede41f5dc432e33e35908`.
- Changes include Breakout CPU and multi-agent GPU fixes, OSRS asset compression, and cleanup. Upstream removed the tracked `puffer` executable and moved `scripts/bench_craftax_sps.c` into `tests/`.
- Local research files were preserved. No push, build, training, dependency installation, or system CUDA change was performed. Research code assessments describe the original revision unless subsequently rechecked; consult current source before implementation.

## 2026-09-12 — Connect4CNN environment milestone

- User authorized writing the staged workflow plan and copying Connect4 into a pixel-observation environment. See `research/CONNECT4CNN_PLAN.md` and `ocean/connect4cnn/README.md`.
- Preserved the original 6×7 game; cells become 6×6 grayscale blocks. Added one blank column on each side for 36×44 observations: unpadded 36×42 would let Nature's striding discard the seventh board column.
- User requested the normal PufferLib dependency/build approach. Reused an existing Raylib 5.5 package from the main clone by copying it into this checkout's usual shared root-level dependency directory. The other clone and system packages were not modified. Pixel observations are direct C writes, independent of Raylib rendering.
- Standard CPU build and 10-episode headless smoke run passed. ASan/UBSan pixel/reset/terminal tests and a 4,096-step differential trace against original Connect4 passed; the pixel trace also reproduced exactly.
- No CNN encoder or learning run yet. The copied training config is a starting recipe and still uses the default encoder until milestone 2 adds the CNN path.

## 2026-09-12 — Tiny CNN native training path

- Added the first custom CNN in `ocean/connect4cnn/connect4cnn.cu`; only five integration lines were added to `src/ocean.cu`. Conv4×4/stride4, eight channels, ReLU, and a linear projection feed the existing core/decoder. No trainer/optimizer/build-system changes.
- Corrected GPU discovery: G240 is WSL. `/usr/lib/wsl/lib/nvidia-smi`, executed outside the sandbox, reports the RTX 5060. Followed the Admiral/F-Zero process-local NCCL setup. Added explicit instructions to `AGENTS.md`; no Docker, driver changes, or system package changes.
- GPU numerical tests passed in float32, including forward/weight-gradient reference agreement, rollout/train parity, blank inputs, and eager/graph repeated execution. The NumPy reference passed finite differences and all-cell image-coverage checks.
- Two native 65,536-step smoke runs and GPU checkpoint evaluation passed. Both final checkpoints are identical and finite; both CNN parameter matrices updated. Artifacts: `build/connect4cnn/smoke.a0cFzy` and `build/connect4cnn/smoke.EMKokT`.
- This establishes training plumbing, not useful gameplay: the first short evaluation won zero of 286 completed games. Next work is learnability/baseline measurement and architecture comparisons, with BF16 validation before making default-precision performance claims.

## 2026-09-12 — State versus pixel comparison workflow

- User requested a clean common workflow comparing native state-based Connect4 with pixel CNNs to measure representation/encoder overhead.
- Added a shared comparison recipe and one runner (`ocean/connect4cnn/compare.sh`). It builds separate state/tiny-CNN binaries, runs matched settings/seeds, checks effective INIs, evaluates fixed checkpoints, and emits Markdown/CSV with preserved source/config/build/run artifacts and failures.
- Verified two seeds × two policies at 65,536 training steps, two checkpoints per run: all four training jobs and eight evaluations passed. Artifact directory: `build/connect4cnn/compare.16ohu06z`. Both final win rates were zero; this is workflow validation, not a performance ranking. The earlier `compare.otwm9q9m` diagnostic selected `perf` as the native score metric; the final runner fixes/validates `sweep.metric=score` to report return and win rate separately.
- Original Connect4 and `build.sh` remain unchanged. The comparison runner required no additional `src` edits. Nature/IMPALA/Impoola are not implemented yet.

## 2026-09-12 — Persistent performance history and longer learning comparison

- User requested tracking SPS and other performance measurements in Markdown across changes. Added `research/EXPERIMENT_LOG.md`; the comparison runner automatically appends dated results with a `--note`, source/config/build provenance, steps, seeds, parameters, win rate, score, wall time, process/native SPS, and last logged VRAM. Added future-agent instructions in `AGENTS.md`.
- Completed matched state/tiny-CNN training at 13,279,232 decisions per seed, seeds 73/74/75, four fixed checkpoint evaluations with separate paired evaluation seeds. All six training jobs and 24 evaluations passed; verified timing arithmetic, finite metrics, and history append. Artifacts: `build/connect4cnn/compare.we7qgdcg`.
- Three-seed means: state 18.51% evaluation wins and 108,831 process SPS; tiny CNN 64.04% wins and 97,733 process SPS. The CNN learns useful gameplay at this budget while processing about 10.2% fewer steps per second. Parameter counts differ (55,552 versus 151,680); these are common-recipe pipeline results, not an isolated perception penalty or convergence claim.
- G240 RTX 5060, float32, serial runs. No architecture/learner changes for this experiment, and no system CUDA or dependency changes. The longer budget replaces the smoke-only evidence for learnability; faithful Nature/IMPALA/Impoola comparisons remain next implementation work.

## 2026-09-12 — Baseline identity and hyperparameter clarification

- User requested explicit documentation of what the two results represent. `state` uses vanilla PufferLib's environment/default architecture/trainer with a modified configuration; its 18.51% result must not be labeled untouched stock performance. `tiny_cnn` is our custom single-convolution encoder, not Nature/IMPALA/Impoola.
- Added policy descriptions and a side-by-side table of tested state, tested CNN, and stock configuration in [the experiment log](research/EXPERIMENT_LOG.md#policy-identities-and-hyperparameters-for-comparewe7qgdcg), linked from the environment README. Both tests share the same hypers and 13,279,232 decisions per seed. Documented stock differences in core width, learning rate, replay, vectorization, minibatch, async mode, and run/evaluation controls.
- The CNN's additional parameters and nonlinear feature extraction are possible explanations for better learning, not established causes. A stock-config baseline remains unmeasured. This clarification changes documentation only and preserves the recorded results.

## 2026-09-12 — Stock baseline and adapted Nature encoder

- User authorized the next steps: measure the stock-config baseline, then validate/integrate Nature using the existing workflow. Added `--stock` and `--variants` to the runner without changing the stock config, trainer, or tiny encoder.
- Stock run `build/connect4cnn/compare.9egh6y44`: all three seeds and 12 checkpoint evaluations passed. Mean wins 98.87%, mean process SPS 306,403, mean wall time 43.414 seconds, 209,408 parameters. Stock's requested 13,272,299 decisions become 13,238,272 actual decisions (101 complete batches); SPS uses the actual count. Training settings remain stock; float32 and separate common evaluation controls are explicit adaptations.
- This resolves the earlier unmeasured stock baseline. The shared small-batch/low-learning-rate configuration weakened the state result substantially. Tiny CNN's 64.04% wins do not beat stock-config PufferLib. Preserve that distinction when comparing results.
- Added `ocean/connect4cnn/nature.cu`: Nature Conv8/s4,32 → Conv4/s2,64 → Conv3/s1,64 → linear, with bias/ReLU throughout, adapted to grayscale 36×44 and hidden 128. Existing MinGRU/heads/learner retained. Encoder 88,352 parameters; full policy 138,528. Nine total integration lines now differ from upstream in `src/ocean.cu`; the tiny encoder remains unchanged.
- Float32 forward and all parameter gradients match the independent NumPy reference. Finite differences, all-cell coverage, inactive ReLUs, rollout/train parity, and eager/graph repeatability passed. Nature smoke runs `compare.z10zc3xf` and `compare.ltxw6alr` produced identical checkpoints; all weights were finite and every encoder weight/bias array updated. Final SHA256 `1bbecd40be2327277c1852ce273720e09f026aab072b53a87f02e0d81cbfcb1b`.
- Completed the three-seed Nature run at the common 13,279,232-decision budget in `build/connect4cnn/compare.2s__8t8l`: mean wins 78.71%, mean process SPS 80,840, mean wall 164.279 seconds. All three jobs and 12 evaluations passed; checkpoints are finite, SPS arithmetic is valid, and resolved settings match the earlier common-recipe comparison. Results appended automatically; the experiment history now includes a four-policy summary. Nature beats tiny CNN under this recipe while running slower; neither beats the separately configured stock baseline. No system CUDA or package changes.

## 2026-09-12 — Local commit preparation

- User authorized local commits. Preserved small evidence from all seven comparison runs and both original tiny-CNN smoke runs under `research/results/connect4cnn/`, including the superseded diagnostic. Updated report links to committed copies. Full binaries, checkpoints, downloaded papers, and search caches remain local and ignored.
- Split the milestone into research/documentation and implementation/benchmark commits. Existing Git author configuration is retained. Publication to either remote is outside this local commit step.

## 2026-09-12 — IMPALA and Impoola implementation

- User authorized completing the reference set. Added a shared 15-convolution, 16/32/32-channel IMPALA backbone with original-SAME pooling and two preactivation residual blocks per stage. Impoola differs only by global average pooling after the final ReLU. Input and projection are adapted to the same grayscale board and hidden-128 core.
- Added independent NumPy/CUDA numerical tests. All 16 weight/bias pairs passed finite differences, all board cells affect the output, pooling border/tie cases passed, and native forward/all gradients matched for B=1/3/32 and H=16/32/128. Rollout/train and repeated eager/graph execution matched exactly.
- `compare.ij5_xpo7` and `compare.sqngvlom` passed two-encoder training/checkpoint smoke tests. Each variant reproduced its checkpoints byte-for-byte; every encoder parameter array updated and all values were finite. Zero wins at the short budget; approximately 9,800–10,000 SPS and 3.5 GB VRAM. These are initial implementation timings, not architecture limits.
- Started the full 13,279,232-decision, three-seed-per-encoder common-recipe comparison in `build/connect4cnn/compare.l6d5sbk2`. Per-training/evaluation-process timeout is 2,400 seconds. Completed September 13: IMPALA 99.19% mean held-out wins / 10,800 SPS; Impoola 69.49% / 10,833 SPS. All six jobs and 24 evaluations passed; small evidence is archived under `research/results/connect4cnn/compare.l6d5sbk2/`. Stock-derived CNN training remains a later experiment.
- Existing tiny/Nature encoders, game, core, training config, and system CUDA stack remain unchanged. The source integration adds only a conditional include; all new encoder code stays under `ocean/connect4cnn/`.

## 2026-09-13 — First configurable CNN canary sweeps

- Added optional policy-configuration handoff to the existing encoder factory. Numeric ID 0 preserves the compiled default/reference; ID 1 selects the experimental CNN. All CNN construction, forward/backward, allocation, and training remain native C/CUDA. Experimental channels 8/16/32, residual blocks 0/1/2, and flatten/GAP yield 18 allowed shapes.
- The native PROTEIN scheduler searches channels/blocks/pooling plus training timesteps. Temporary Python glue prepares an isolated config directory, invokes the native binary, and reports results; it never rewrites the checkout defaults. Kinvert explicitly requires replacing that glue with native tooling before final delivery. The external W&B sidecar follows the F-Zero/Admiral saved-result pattern.
- Two 12-trial canaries completed (`sweep.r3qepq1g`, finalized tooling `sweep.7s4ovb72`), each covering 8 shapes and reaching a GP-guided suggestion. Final sweep process wall was 19.409 s, excluding build and subsequent SDK sync. All checkpoints were finite; all expected parameter counts and fixed learner/core settings were checked. First 11 trial checkpoints matched byte-for-byte across repeats; adaptive timing-based suggestions are not claimed invariant.
- All 18 shapes passed native float32 numerical/repeatability tests; CPU configuration/sidecar tests passed. Default tiny-encoder checkpoints matched the old binary exactly. Vanilla state Connect4 trained and reloaded successfully; the 684,224-parameter CNN variant reloaded/evaluated successfully. These are plumbing checks, not new learning baselines.
- Both canaries produced 12 offline W&B runs. Repeating the sidecar did not create duplicate offline runs. W&B 0.21.4 and compatible protobuf 6.33.6 are pinned in the project-only venv lock; LanceDB/FastEmbed versions are unchanged. No Torch or system CUDA modifications. Online logging remains untested.
- Small evidence is archived under `research/results/connect4cnn/sweep.*/`. Binaries, full source snapshots, checkpoints, and W&B binary artifacts remain local under `build/connect4cnn/`. The first canary preceded final metadata/validation refinements; the second is the reproducible tooling reference.

## 2026-09-13 — Flexible architecture controls

- Encoder ID 4 adds 18 architecture knobs: depth, independent stage widths/kernels/strides/pools/skips, readout and projection. Any legal subset can now be swept, including fixed-budget searches. Full recipe: `ocean/connect4cnn/sweep_flex.ini`; usage and semantics: `research/FLEX_CNN_SWEEP.md`. Existing IDs/checkpoints retain their meanings.
- Numerical checks passed for 78 targeted flexible configurations, Nature regression, and all 54 old compact cases. Ten tooling tests passed. Full and one-knob canaries completed (12 and four trials); finite checkpoints and parameter counts verified. Maximum-workspace layout fit the common 2,048-decision batch, produced identical repeated checkpoints, and reloaded successfully. Evidence is archived in `sweep.jfu9hfqr` and `sweep.jd06qqgp`.
- Only bounded validation was run. The default 128-trial recipe has not been launched, and no new learning-quality advantage is claimed.

## 2026-09-13 — Compact search and Nature control (results)

- Completed: 12 Nature trials in 191.242 s and 24 compact trials in 919.912 s. All 36 final checkpoints finite; fixed learner/core/env settings matched. At the same 3,317,760 steps, compact C8/depth1/stride4/projection32 reached 87,241 SPS versus Nature 76,836 (+13.54%). Best compact final training score was 22.76% at 6,639,616 steps, 72.66 s, 91,379 SPS. Nature's adaptive sweep never exceeded 3,317,760 steps, so learning/time superiority is unestablished. Fixed longer-budget controls and held-out multi-seed checks are next. Reports are archived under `research/results/connect4cnn/sweep.4h5iffsm` and `sweep.od0_6e75`.

- Implemented a shared-kernel compact family: depth 1/2/3, channels 8/16/32, first stride 2/4, projection 32/64/128, while keeping the recurrent core fixed at 128. Numeric encoder 2 selects exact Nature; 3 selects compact. Existing IDs 0/1 remain available. Native source changes are limited to Connect4CNN selection in `src/ocean.cu`; the implementation stays in `ocean/connect4cnn/nature.cu`.
- All 54 compact configurations and Nature regressions passed. Nature-equivalent compact outputs/gradients and old/new Nature training checkpoints matched exactly. Both 12-trial canaries completed and compact checkpoint reload passed; evidence is archived. Eight CPU tooling tests passed.
- Launched the serial 12-Nature/24-compact campaign under `build/connect4cnn/fast-search.o61nFcx4`, tmux `cnn-small-20260913`, with online `kinvert-k/puffer-cnn` logging. Same learner/core/budget range, 25 history points. Read `research/COMPACT_CNN_SWEEP.md` and the experiment log for commands, limits, validation, and future confirmation requirements. Leave training unattended.

## 2026-09-13 — Learning-scale discovery launch

- Completed: all 24 trials, eight shapes, 13 model-guided proposals, no native failures, 3 h 13 m 31 s sweep wall. All final checkpoints finite and all 24 W&B runs finished with corrected names/native metrics. Fastest observed 100% final training point: `lucky-maple-21`, channels 32 / blocks 1 / flatten, 5,785,600 decisions, 641.40 s native cost, 9,020 average SPS. Held-out and multi-seed confirmation remain pending. Evidence is archived under `research/results/connect4cnn/sweep.co1g6diu/`; see the experiment log for interpretation and the expected original-sidecar replacement error.

- Launched `build/connect4cnn/sweep.co1g6diu` in tmux session `cnn-discovery-20260913`: 24 native PROTEIN trials, about 3.3M–13.3M decisions, existing 18-shape family, fixed learner/core, checkpoint interval 1,000, 12-hour whole-sweep deadline. See `research/EXPERIMENT_LOG.md` for launch provenance and pending evaluation. Do not actively monitor it.
- Verified online W&B with all 12 finalized canary trials, then a two-trial concurrent-sidecar canary. Online campaigns now upload completed trials automatically to https://wandb.ai/kinvert-k/puffer-cnn; discovery group is `sweep.co1g6diu`. These appear as grouped runs because PROTEIN owns the sweep. All five CPU tooling tests passed. Native source and system CUDA are unchanged.
