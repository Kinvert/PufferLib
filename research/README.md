# CNN research workspace

- [Native encoder profiler preparation](NATIVE_ENCODER_PROFILER.md): production
  registration audits and isolated future CUDA timing/workspace probes;
  compile/host checks and supervised preparation pass; GPU execution remains pending.

This is the local research library and retrieval tooling for Kinvert's PufferLib CNN project. The training checkout starts at official PufferLib `5.0`, commit `bea26c718b61cf2377f8e2ff247fcf687478bcdf`. Research began September 11, 2026 (America/Los_Angeles); download receipts use UTC.

On September 12, the checkout was fast-forwarded eight commits to `89414204ce8509c3fecede41f5dc432e33e35908` on upstream `5.0`. Existing code assessments retain their original review date; check current source before implementation. Upstream removed the tracked `puffer` executable; this refresh did not rebuild it.

## Start here

- [October 7 5090 feedback handoff](../NEXT_5090_FEEDBACK.md): two commands build,
  prepare, run the tiny native cross-game PROTEIN canary and audit/archive it.
  Kinvert authorized research delivery to his fork. GPU-free local launcher
  preparation passes; actual remote feedback execution remains pending. This
  changes the earlier delivery hold, not the numerical or publication gates.

- [Completed short-budget RTX 5060 mini benchmark](MINI_SHORT_BUDGET_5060.md): quality versus Nature, two paired seeds, six games/all 41 drawings, checkpoints 262,144/524,288 decisions. All 24 jobs/328 evaluations/48 repeat-eager checks pass audit; [curves and SPS table](results/mini-short-5060-20261006/README.md). Ours processes decisions 9–22% faster in these small runs, but learning is weak and frontier marks stay suppressed. Three native uptime inconsistencies are retained; monotonic receipts define the timing comparison.

- [Learner-batch numerical follow-up failed](GPU_ENCODER_LEARNER_BATCH.md): quality H256/B2048 stops after 33 first-worker passes. A separate CUDA diagnostic supports a near-zero ReLU rounding explanation; Nature/second worker remain unexecuted. Original fixed tolerances/failure are retained. The next protocol separates independent layer-forward accuracy from backward correctness on computed float32 branches; it is planned, not implemented. No full-learner or speed acceptance.

- [Actual independent CUDA encoder math smoke](GPU_ENCODER_NUMERICAL_SMOKE.md): frozen quality/Nature at H128/B1/3/64 pass all output/parameter-gradient checks against isolated float64 CUDA, exact graph/process/device-parameter checks and 28 selected GPU finite differences on RTX 5060. Two workers/144 harness calls take 1.78s; no CPU CNN reference, Torch install or production change. B2048/full-policy/general architectures remain separate.

- [Actual candidate evaluator comparisons on RTX 5060](CANDIDATE_EVALUATOR_ACCEPTANCE.md): offline audited-case export feeds saved ours/Nature checkpoints into existing native evaluators. Ten drawing-zero cases across five newer games pass all 40 graph/repeat/eager/independent-tail processes and 520 assigned episodes at 16 slots. No training or model-math/learning/frontier claim; other drawings/batches remain separate.

- [Completed quality versus Nature RTX 5060 smoke](NATURE_MULTIGAME_5060_SMOKE.md): matched six-game native workflow passes 12 training jobs, all 82 drawing evaluations and 24 repeat/eager comparisons. Full-policy layout/size gate is exercised; 93.24s campaign, 9.68s training process total. One seed/65,536 decisions per game is plumbing evidence, with no quality/frontier selection.

- [Candidate panel native policy layouts](CANDIDATE_BASELINE_COMPARISON.md): opt-in metadata tools now register each full policy before training, with owned source/action binding, repeated pre-query registration and checkpoint-size enforcement. Actual 60-job/five-model/two-seed preparation passes, zero execution; [52 focused checks and retained evidence](results/candidate-policy-layout-20261006/README.md) distinguish scalar layouts from math/memory/learning acceptance.

- [Discovery on drawing mixtures](DISCOVERY_PIXEL_MIXTURES.md): explicit fixed appearance seed for six-game native discovery preparation; Flappy's seven-ID catalog now works. Eighteen CNN-only dimensions/per-game learners stay fixed. Six actual preparations/384 native scalar rows and 26 host/configuration checks pass; no policy executes.

- [Arbitrary-candidate full curves](CANDIDATE_FRONTIER_VIEW.md): offline candidate-panel audit plus a self-contained viewer for every declared CNN/reference, including missing models. [Actual smoke view](results/candidate-viewer-20261006/smoke-view/curves.html) retains all 82 existing observations with smoke marks suppressed; [five-model allocation view](results/candidate-viewer-20261006/allocation-view/curves.html) displays all 820 missing cells. No new policy ran; source/data/display checks pass without a quality claim.

- [Checkpoint layout before GPU acceptance](CHECKPOINT_LAYOUT_GATE.md): opt-in shared evaluator packet-v2 freezes native shape/stat tools and checks configurable CNN/Nature/IMPALA/Impoola weight counts before a GPU adapter. Existing Flappy quality checkpoint prepares successfully without model execution. Wrong-length/core/nonfinite artifacts retain failures; bytes/layout do not certify model math or runtime binary provenance. Individual game launchers and training panels retain separate gates.

- [Matched arbitrary CNNs and fixed references](CANDIDATE_BASELINE_COMPARISON.md): opt-in panel-v2 adds Nature, IMPALA and Impoola with identical per-game learners, seeds, budgets, mixtures and exact suites. Eighteen normal targets compile; actual preparation passes 60 jobs/82 suites/310 native start-and-pixel comparisons, with 820 planned evaluations and zero observations. Full curves retain missing cells, declines and complete paired means. [Retained artifact verification](results/candidate-baselines-preparation-20261006/README.md) passes without GPU/model execution; qualification/calibration and scheduling remain pending.

- [Native cross-game feedback prototype](CROSS_GAME_PROTEIN_FEEDBACK.md): temporary macro-gated core bridge closes the adaptive loop using native PROTEIN and six-game exact scalar feedback. Compiled/prepared and host-tested; GPU loop unexecuted. Local research only: no push, no upstream PR claim, no long campaign.
- [One CNN across games/drawings](CROSS_GAME_CNN_SWEEP.md): the generic panel expands independent native proposals into all six environments and 41 deterministic drawing targets. Architecture is common across games; learners are fixed per game. Earlier short 5060 allocations are complete; full 5090 work requires separate scheduling.

- [Native checkpoint size/layout preflight](POLICY_CHECKPOINT_PREFLIGHT.md): actual full-policy registration, ordered encoder/head/MinGRU shapes and stat-only checkpoint checks before GPU scheduling. Three targets and seven host/scalar checks pass; this does not certify checkpoint contents, identity or model math.
- [Offline multi-game curves](PIXEL_FRONTIER_REPORTING.md): self-contained HTML with all four CNNs, per-condition/seed controls, complete cost range, explicit missing/failure coverage and separate Pong censoring bounds. Current actual collection remains empty; no new learning or frontier claim. Six environments are enough for now.
- [Paired dense-copy numerical acceptance](DENSE_ALIAS_ACCEPTANCE.md): baseline/candidate fixed-graph/all-gradient/parameter-preservation and exact process/eager/graph checks, including quality H64. Two libraries compile, 39 host/synthetic tests and actual preparation pass; GPU execution and broader policy/timing/learning gates remain pending.
- [Isolated dense-input copy candidate](DENSE_PATCH_ALIAS_CANDIDATE.md): profiler/test-only implementation for quality and Nature; paired compilation/host counts pass, owned allocations and production trainer remain unchanged. GPU math, timing and whole-policy qualification are pending.
- [Deterministic mixed-drawing training preparation](MIXED_PIXEL_TRAINING.md): matched four-CNN mixtures using native persistent slot assignments, retained counts and per-drawing checkpoint tests. Six-game/two-seed host preparation and 60 scalar/configuration checks pass; no policy, training or GPU executes.
- [Same-policy drawing transfer](PIXEL_APPEARANCE_TRANSFER.md): bind each trained checkpoint to every drawing without extra training. Six-game host preparation passes 24 planned jobs/38 suites/152 targets and 50 host/configuration checks; GPU transfer/learning and final-test exclusion remain unverified.
- [Per-game fixed-CNN budgets](TASK_BUDGETS.md): explicit shared-within-game decisions/cadence, native scalar receipts, preserved v1/v2 compatibility and v3 evaluation binding. Six-game host preparation passes; its large resource example is uncalibrated and not authorized for execution.
- [Learner memory preflight](LEARNER_MEMORY_PREFLIGHT.md): actual large-batch native registrations matched against independent C counts. Stock IMPALA/Impoola buffers exceed 5060 capacity; an equally applied smaller-batch candidate is prepared. Full-policy fit/learning remain unverified; no GPU allocation/execution.
- [Shared fixed-model learner recipes](SHARED_LEARNER_RECIPES.md): explicit numeric overlays, native scalar preflight, unchanged legacy panels and full build/config closure. Two Flappy candidates prepare 96 jobs across all drawings/three seeds, with paired native host starts; zero trained. GPU/memory/evaluation/calibration gates remain.
- [Learning-recipe geometry and calibration](LEARNING_RECIPE_CALIBRATION.md): native scalar INI arithmetic, effective updates/coverage/budget rounding, current-stock versus common-recipe differences and a staged fair multi-game learning plan. Host/UBSan audits pass; no policy, GPU or learning qualification.
- [Paired multi-game evaluation bindings](PIXEL_EVALUATION_BINDINGS.md): captured-recipe audits and fixed development suites for all six tasks/38 drawings/152 fixed-model jobs. Actual host starts/configuration checks pass; no policies or GPUs execute, learning/runtime/inference gates remain open.
- [Shared GEMM workspace acceptance preparation](SHARED_GEMM_ACCEPTANCE.md): isolated native forward/dW/dX, actual event fork/join and independent GPU dyadic references; compile/host checks and 96-case preparation pass. Exclusive supervisor/offline audits pass host/synthetic checks; GPU runtime remains pending and production helpers are unchanged.
- [Shared evaluator acceptance](EVAL_ACCEPTANCE.md): existing-checkpoint host preparation/inspection for the five pending games; scheduled repeat/eager/same-batch-tail receipt comparisons. No policy execution, new score or GPU authorization from host tests; Connect4 and encoder-5 gates remain separate.
- [MazeCNN navigation preparation](../ocean/mazecnn/README.md): direct pixels of the original local crop, six drawings and preserved native level/reset rules. [Dedicated exact evaluation](../ocean/mazecnn/DETERMINISTIC_EVAL.md) now compiles with identity-based levels, all-assigned success auditing and supervised checkpoint preparation. Game/counter/sanitizer and host checks pass; GPU/learning and verified held-out exclusion remain pending.
- [Pong texture robustness](PONG_TEXTURE_ROBUSTNESS.md): two reversible native texture presets, preserved legacy mixed assignments and archived environment/build/host-start checks. No policy ran; GPU holds remain.
- [Native baseline backend/count audit](BASELINE_BACKEND_AUDIT.md): shared cuBLAS implementations, architecture adaptations, reproducible C shape arithmetic and a concrete stream/workspace reset issue requiring GPU validation. Counts aren't performance measurements; no GPU execution or production change.
- [Deterministic Snake evaluation groundwork](../ocean/snakecnn/DETERMINISTIC_EVAL.md): exact first-death/game-horizon adapter and all-assigned length/food/return auditing. Four builds and host/game/audit checks pass; supervised checkpoint preparation/audits pass, GPU acceptance pending.
- [Deterministic Flappy evaluation groundwork](../ocean/flappycnn/DETERMINISTIC_EVAL.md): native adapter compiles; supervised existing-checkpoint preparation and source/manifest configuration closure pass 21 host/audit tests. Original cap/rewards and explicit legacy-suite limitations are preserved; GPU quota/reset/reload/repeatability remain pending.
- [Deterministic Breakout evaluation groundwork](../ocean/breakoutcnn/DETERMINISTIC_EVAL.md): first-terminal frame-skip capture, administrative frame caps and exact assigned IDs. Four native targets and host/simulation/audit checks pass; supervised launcher adds GPU-free preparation and retained failures. GPU acceptance remains pending. No game/training changes or learning claims.
- [Pong exact-match evaluation](../ocean/pongcnn/DETERMINISTIC_EVAL.md): native adapter compiles; whole-match counters/point receipts and censoring bounds pass environment/host audits. Supervised checkpoint preparation/audits pass; GPU acceptance pending. [Inspected scoring design](PONG_EXACT_EVALUATION_DESIGN.md).

- [Fixed CNN multi-environment/appearance robustness](MULTI_ENV_ROBUSTNESS.md): six native pixel tasks, 41 drawings, frozen quality/Nature/IMPALA/Impoola preparation, direct-buffer Breakout/Maze and separately versioned SnakeCNN/SnakeBench. GPU/exact-evaluation gates remain; no new GPU campaign launched.
- [Flappy geometric robustness](FLAPPY_GEOMETRIC_ROBUSTNESS.md): rounded bird/outlined pipes, legacy-compatible mixing, preserved captured catalogs, pixel-stream identity and native/host preparation receipts; no learned robustness result.
- [Multi-game frontier reporting](PIXEL_FRONTIER_REPORTING.md): offline native-clock/raw-episode audits, separate game/drawing metrics, complete paired seeds, honest missing/failure cells and censoring bounds; no policy or certified dominance.
- [SnakeCNN local episodic protocol](../ocean/snakecnn/README.md): one-agent game/local crop, six pixel presets and matched state control. Independent environment/reference/raster/sanitizer checks and four native builds pass; original Snake unchanged. No policy execution or comparative result.

- [Exact full-frontier follow-up and paired development replication](CONNECT4_DEVELOPMENT_REPLICATION.md): all 52 historical checkpoints scored on a fixed suite, GPU checkpoint timing/failure recovery canary passed, and five fresh paired ours/Nature seeds launched with all 51 checkpoints retained.

- [Dedicated deterministic Connect4 evaluation](../ocean/connect4cnn/DETERMINISTIC_EVAL.md): freeze an N-episode suite and evaluate any matching native checkpoint/config; exact per-game allocation and seeded start receipts, with GPU acceptance across eight policy configurations.

- [RTX 5060 hardware panel](HARDWARE_5060_RESULTS.md): separately scheduled four-model pilot, estimated duration, hardware-specific result directories and whole-frontier plots; single seed and pooled-evaluation limitations are explicit.

- [October 5 Connect4 draw-rule correction](CONNECT4_DRAW_CORRECTION.md): independent fixtures exposed an inherited state/pixel bug; corrected rules and historical results must remain separate. No GPU execution. [Current 5090 preparation handoff](../NEXT_5090_TASK.md).

- [Exact evaluation and full-frontier measurement groundwork](CLAIM_PIPELINE.md): opt-in native episode accounting and completed-checkpoint timing, matched frozen preparation, artifact audits, whole curves and explicit inference gates. No new GPU run; initial synthetic uncertainty coverage is inadequate for a publication claim.

- [Current potential priorities](../potential-todos.md) and [encoder-5 verification contract](FLEX2_VERIFICATION.md): independent numerical reference, boundary probes and evidence requirements; compiled locally, GPU execution pending on the 5090.

- [Small native pixel environment candidates](NATIVE_PIXEL_ENV_CANDIDATES.md): source-based ranking and the selected direct-buffer FlappyCNN task; [implementation and canary](../ocean/flappycnn/README.md).
- [FlappyCNN stock-learner pilots](FLAPPY_STOCK_PILOT.md): fixed quality pixels and original stock state, stock Flappy learner/environment settings and 20M requested decisions each on the 5060; completed/audited outside the sandbox, with [state/pixel comparison receipts](results/flappycnn/stock-pilot.tgKBYrNs/REPORT.md).

- [Pixel-space game-frame pretraining design](PIXEL_PRETRAINING_DATASET_PLAN.md): exact frame-aligned boxes/masks, native annotation path, data splits and scratch controls. Research only; no dataset generated.
- [Expanded native CNN grammar](FLEX2_CNN_SWEEP.md): encoder 5, dilation, four stages, adaptive readout, fixed/learned activations and duplicate avoidance. Build passes; GPU validation pending.
- [September 26 literature refresh](ENCODER_RESEARCH_20260926.md): Hadamax, Aftab, learned rational activations, OCAtari and temporal pretraining, with scope and next experiments.

- [Current 5090 encoder-5 handoff](../NEXT_5090_FLEX2_SWEEP.md): GPU math/reload gates and two bounded architecture-search panels. [Earlier appearance validation](../NEXT_5090_VALIDATION.md) is historical context; completed campaigns must not be relaunched.

- [Complete 5090 session handoff](../START_HERE_5090.md): context and executable checklist from fresh clone through tests, training, tables and evidence transfer.

- [5090 fixed-model hardware comparison](HARDWARE_COMPARISON.md): Kinvert-fork checkout, existing-toolchain setup, canary/full commands and same-revision timing requirements.

- [Potential CNN constructor/search deliverable](CNN_CONSTRUCTOR_PLAN.md): native small-first architecture search, per-environment frontiers, exported training configurations and an evidence plan.

- [Pong fixed-architecture hyperparameter search](PONG_HYPER_SWEEP.md): locked transferred model and reference architectures, bounded native PROTEIN training search, cnn3 logging and negative budget-calibration evidence.

- [Benchmark source revisions and recovery](BENCHMARK_STATE.md): exact captured implementation for the five-seed results, distinguished from later Pong and representation work.

- [Completed five-seed confirmation](CONFIRMATION_RESULTS.md): all 390 checkpoint evaluations, whole time/step frontiers, uncertainty, and audit receipts for ours versus Nature/IMPALA/Impoola.

- [Connect4 representation presets](../ocean/connect4cnn/REPRESENTATIONS.md): integer-sweepable squares, circles, gaps, X/O and compact boards, with validation and robustness interpretation.

- [PongCNN implementation and validation](../ocean/pongcnn/README.md): second native development task, direct pixels, original-Pong parity checks, shared encoders and prepared matched GPU canary.

- [Frozen confirmation protocol](CONFIRMATION_PROTOCOL.md): six existing encoders × five fresh seeds, equal 13-checkpoint curves, and separate held-out/W&B metrics.
- [Paper and practical-delivery plan](PAPER_PLAN.md): claims, apples-to-apples comparisons, architecture transfer to unseen environments, and native PufferLib acceptance criteria.
- [CNN2 complete frontier comparison](CNN2_RESULTS.md): all 128 final checkpoints evaluated against Nature/IMPALA/Impoola, interactive time/step frontiers, and search coverage limitations.
- [Flexible small CNN controls](FLEX_CNN_SWEEP.md): per-stage kernels, widths, strides, skips, pooling, and arbitrary fixed/swept subsets in INI.
- [Compact CNN search](COMPACT_CNN_SWEEP.md): cheaper strided architectures, an exact Nature control, shared kernels, and the first matched sweep recipes.
- [Experiment log](EXPERIMENT_LOG.md): persistent learning/speed results across code and configuration changes, including SPS definitions and linked artifacts.
- [Connect4CNN implementation plan](CONNECT4CNN_PLAN.md): the first pixel environment and staged end-to-end workflow.
- [CNN sweep integration](CNN_SWEEP_INTEGRATION.md): current INI-to-network construction path, minimal proposed core changes, and native PROTEIN constraints for architecture search.
- [Reading list](READING_LIST.md): what to read and why, with local conversions and primary sources.
- [CNN architectures](CNN_ARCHITECTURES.md): Nature, IMPALA, pooling variants, compute accounting, and candidate designs.
- [CUDA and scaffolding](CUDA_AND_SCAFFOLDING.md): actual PufferLib integration points, reference implementations, interchangeable candidates, and a route to native kernels.
- [Benchmarks and sweeps](BENCHMARKS_AND_SWEEPS.md): October 2 next-benchmark priorities and RGB/native integration gates, suite comparison, all 16 Procgen games, and controlled architecture experiments.
- [Research assessment](ASSESSMENT.md): corrected assumptions, decisions, and a practical starting direction.
- [Search log](SEARCH_LOG.md): scope of the literature search and remaining gaps.

The original `../JOSEPH_CONVERSATION.md` and `../CLAUDE_RESEARCH.md` remain verbatim source material. Generated paper text and assistant-authored research are separate files.

## Recreate the environment

Use `uv venv` explicitly. Python 3.12.3 and the locked dependencies were installed and checked here. This environment is for research tools; it does not install or rebuild the native trainer.

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --only-binary :all: research/requirements.lock
uv pip check --python .venv/bin/python
.venv/bin/python research/search.py doctor
```

The lock includes LanceDB 0.38.0, FastEmbed 0.8.0, ONNX Runtime 1.30.0, and PyMuPDF4LLM 1.28.2. It also includes W&B 0.21.4 for the external native-training sidecar (added September 13; protobuf is pinned to compatible 6.33.6). Do not install `fastembed-gpu`/`onnxruntime-gpu` alongside the CPU packages. The current setup uses only `CPUExecutionProvider` and has no Torch dependency. Do not modify the system CUDA stack, drivers, cuDNN, system libraries, global settings, or other environments. CUDA inspection is allowed; any future basic Torch work must be confined to this directory's `.venv`.

## Collect and convert papers

```bash
.venv/bin/python research/collect_papers.py
# Or collect/reconvert selected entries:
.venv/bin/python research/collect_papers.py impala impoola
```

`papers.json` is the acquisition list. `sources/<slug>/` contains the downloaded PDF, available HTML, and a metadata receipt. The receipt records the paper version, source URLs, retrieval time, converter versions, PDF hash, page count, and conversion word count. `papers/<slug>.md` contains the converted source text, not a summary. `conversion_status.json` records successes and failures.

The manifest pins the acquired arXiv versions. The collector also resolves a version on first acquisition if a newly added entry uses an unversioned ID. Subsequent runs reuse the receipt and downloaded bytes; they do not silently move the paper to a new version. To compare another version, add a distinct manifest entry with a new slug and explicit versioned arXiv ID after checking the source.

HTML is preferred because equations can retain TeX and tables can retain structure. If HTML is unavailable or its text count is below 75% of PDF text, conversion falls back to the full PDF. This caught substantial missing DreamerV3 HTML content. PDF conversions contain page markers. The word-count threshold is a rough completeness check, not proof of correct extraction. Always consult PDFs for equations, architectural tables, plotted results, and exact quotations. PDF figures are not fully reproduced; HTML figure links may need network access.

Downloaded papers, model caches, and the database are ignored by Git. Keep authored findings, the manifest, dependency lock, and tools reviewable. Nothing has been published.

## Search locally

```bash
# Initial build downloads the small embedding model; later builds reuse vectors.
.venv/bin/python research/search.py build
.venv/bin/python research/search.py status

.venv/bin/python research/search.py query "global average pooling" --kind paper
.venv/bin/python research/search.py query "recurrent state reset on terminal" --kind code
.venv/bin/python research/search.py query "create_custom_encoder" --mode fts --kind code
.venv/bin/python research/search.py query "computation budget encoder recurrent" --mode vector --kind paper
.venv/bin/python research/search.py query "im2col gradient" --kind code --json
```

Run `.venv/bin/python -u research/verify_library.py` to check all PDF/Markdown hashes and exercise keyword, vector, and hybrid retrieval with real source-line verification. If the managed read-only sandbox stalls a search subprocess, run this local verification with the same cache-access permissions used for building/querying the index.

Default hybrid search combines full-text and vector retrieval with reciprocal-rank fusion. `fts` needs no embedding inference; `vector` finds conceptual similarity when wording differs. Queries use cached model files and require no embedding API or uploaded corpus. The local model is `snowflake/snowflake-arctic-embed-xs`, a compact general text retrieval model, not a specialized code understanding model. Keyword search remains valuable for exact identifiers.

The index includes repository source/config/scripts and Markdown, plus downloaded paper Markdown. It excludes `vendor/`, `resources/`, binary files, model artifacts, raw PDFs/HTML, and individual files over 2 MB. Skipped large/non-UTF8 files are recorded in the index state. Other PufferLib clones are not indexed.

Chunks preserve source paths, line ranges, content hashes, paper URLs, and available headings. Chunk windows have overlap and are not a parsed call graph. Semantic encoding strips Markdown link targets and uses up to 256 model tokens per chunk; full-text search retains the full chunk. Search hits are leads: read the current source file before drawing conclusions.

`status` reports new/changed/deleted files. Queries also check file hashes and omit stale hits. Rebuild after editing code, adding papers, or changing reports. Unchanged chunk embeddings are reused, new embeddings are checkpointed by batch, and a file lock prevents concurrent builders. Two alternating tables keep the last complete index available if a rebuild is interrupted. `--rebuild` deliberately regenerates all vectors; ordinarily use plain `build`.

For exact code work, start with `rg`:

```bash
rg -n 'encoder_forward|create_custom_encoder|mingru' src ocean
rg -n -i 'pooling|compute.optimal|width' research/papers
```

LanceDB helps discover related passages; direct file inspection establishes what the implementation and papers actually say.

## Primary tooling documentation

- [LanceDB hybrid search](https://docs.lancedb.com/search/hybrid-search)
- [LanceDB full-text search](https://docs.lancedb.com/search/full-text-search)
- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- [FastEmbed CPU/GPU package distinction](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/)
- [PyMuPDF4LLM conversion](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
