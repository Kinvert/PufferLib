# FlappyCNN environment/configuration/build validation

September 26, 2026, G240. No GPU neural-model execution, training, evaluation or architecture search. CPU work here is environment simulation, raster fixtures, configuration reporting and CUDA compilation. No pretraining dataset was generated.

## Results

- [Environment checks](environment-tests.log): 49,152 original/pixel same-seed transitions, event fixtures, clipping/draw-order/overwrite checks, four fixed appearances, three mixed appearance seeds, repeatability and ASan/UBSan pass. [Trace hashes](trace.sha256) identify local full traces; original and pixel game-state/reward/terminal traces agree.
- [Configuration checks](config-tests.log): 15 shared sweep/sidecar tests pass, including Flappy physics/path selection and appearance bounds.
- [Matched preparation](canary-prepare.log): all five prepared models share game, learner, core and budget settings. Complete resolved configs are in [prepared-configs](prepared-configs/).
- [Build-only canary](build.log): state/Flex/Nature/IMPALA/Impoola targets compile; [live source check](source-check.txt) passes. Nature and Flex use the same native binary with different numeric encoder selection. [Binary hashes](binaries.sha256), build logs and exact commands are retained; binaries stay in ignored local build output.
- Old and expanded family search preparation pass: [Flex protocol](flex-protocol.json), [encoder-5 protocol](flex2-protocol.json). No optimizer/model process was executed. New encoder-5 math remains GPU-unvalidated.
- [GPU query](gpu-access.txt): WSL NVML reports `GPU access blocked by the operating system`. No fallback CPU model validation or system modification was attempted.

Build output: `build/flappycnn/canary.fn3qdTXH`, status `built`, not a completed train/reload canary. Preparation: `canary.0WU1xaPC`, `sweep.mm3bs4nv` and `sweep.v3rr50rb`. Sources were a dirty worktree on the [recorded base revision](revision.txt); HEAD alone does not reproduce them. [Full source hashes](source.sha256), [environment/config hashes](environment-source.sha256), and [integration diff](integration.patch) identify the observed tree. Research-note changes after the snapshots are outside the compiled source receipt. Original Flappy blob: `d5b194c7fc9ccde430689b1562c975203cc58f0d`.

An earlier build attempt, `canary.WxHbxOCH`, compiled the targets but failed its source check during concurrent reporting edits. Its failed status, source hashes, check output and launcher log are preserved in [earlier-source-check-failure](earlier-source-check-failure/). It was not treated as accepted build evidence. The subsequent successful attempt used the settled source tree and passed the complete check.

No FlappyCNN learning curve, checkpoint, parameter count, VRAM, SPS or timing result exists yet. Short canary budgets are plumbing; pooled-v1 evaluation is not exact episode allocation. Follow the [environment README](../../../../ocean/flappycnn/README.md) for a bounded GPU canary when access is available and exclusive. Final paper claims require learning calibration and the remaining evaluation/frontier gates.
