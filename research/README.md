# CNN research workspace

This is the local research library and retrieval tooling for Kinvert's PufferLib CNN project. The training checkout starts at official PufferLib `5.0`, commit `bea26c718b61cf2377f8e2ff247fcf687478bcdf`. Research began September 11, 2026 (America/Los_Angeles); download receipts use UTC.

On September 12, the checkout was fast-forwarded eight commits to `89414204ce8509c3fecede41f5dc432e33e35908` on upstream `5.0`. Existing code assessments retain their original review date; check current source before implementation. Upstream removed the tracked `puffer` executable; this refresh did not rebuild it.

## Start here

- [Current 5090 validation task](../NEXT_5090_VALIDATION.md): deterministic mixed appearances, bounded native canary and remaining exact-evaluation gates; historical campaigns must not be relaunched.

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
- [Benchmarks and sweeps](BENCHMARKS_AND_SWEEPS.md): suite comparison, all 16 Procgen games, first targets to beat, and controlled architecture experiments.
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
