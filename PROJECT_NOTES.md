# CNN project notes

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
