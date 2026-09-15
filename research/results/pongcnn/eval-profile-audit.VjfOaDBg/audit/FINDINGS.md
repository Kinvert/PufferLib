# Pong evaluation/backend audit findings

The bounded GPU audit completed in **771.01 seconds (12 min 51 s)**, including
builds and validation. All 57 planned process attempts have exit receipts: nine
validation attempts, 32 panel evaluations, eight short unprofiled training jobs
and eight instrumented repeats. The GPU was idle after completion. No new
full quality experiment was run. Diagnostic caps were 30 seconds for the fixed
old/new checkpoint panel, 60 seconds for short unprofiled training, and 90 seconds
for one profiled repeat per cell. The global GPU wall cap includes validation.

## Measured diagnosis

All eight original timeout checkpoints hit the new 30-second cap in **both**
arms. At the patched arm's last five-second progress sample (about 25 seconds),
14.64–18.22 million decisions had advanced, at 586,167–733,242 decisions/second
between progress samples, but only 0–13 matches had completed against a target
of 512. One checkpoint completed no matches in the entire observed progress
window. The remaining seven completed some matches. After the startup sample,
GPU activity was approximately 71–77% and process lifetime CPU usage approximately
296–299% (roughly three logical CPUs). These are active simulations with very
sparse match completion, not hard execution stalls or near-zero throughput.

The original 300-second logs independently show rounded final counts of
199.6–222.3 million decisions for these eight checkpoints. This supports long
or potentially nonterminating trajectories as the main reason a match-count
target can run so long. The present aggregate logs cannot distinguish a very
long finite match from an indefinitely surviving rally or a game-dynamics edge
case in an individual environment. No trajectory-level defect was established.
Those cases remain unresolved; no outcome was invented.

All eight successful controls completed in both arms, with identical final
score/perf/game/parameter records. Their old/new process-wall ratios ranged
from 0.951 to 1.005, median 0.998 (values above one favor the patch). This single
rotated-order diagnostic does **not** establish a speedup. Resource-query removal
is correct housekeeping but did not solve these timeouts; isolated NVML/query
CPU time was not traced. The new 30-second attempts do not resolve, replace or
extend any original 300-second outcome. `REPORT.md` tabulates all sixteen pairs;
`analysis.json` retains every classification, progress sequence and activity sample.

## Training/backend measurements

All eight unprofiled and eight instrumented 131,072-decision jobs completed with
expected parameter counts, finite checkpoints and finite native metrics. Exact
resolved recipes matched the original saved resolved configurations apart from
the declared seed, shortened total budget, checkpoint interval, output paths and
profiling flag. Each instrumented/unprofiled checkpoint pair was bit-identical.
Seed 52001 and the shorter annealing schedule make these performance diagnostics,
not new learning comparisons.

| Model | A unprofiled s | A instrumented process s | B unprofiled s | B instrumented process s |
|---|---:|---:|---:|---:|
| Ours quality | 0.608 | 2.723 | 0.750 | 2.951 |
| Nature | 0.643 | 2.691 | 0.786 | 3.208 |
| IMPALA | 4.438 | 7.667 | 10.802 | 14.979 |
| Impoola | 4.415 | 7.708 | 10.697 | 14.923 |

The instrumented columns include Nsight startup and export. They are not
throughput claims. Recipe A executes 64 effective replay updates; B executes 192.
The native float replay ratio is truncated to one versus three updates per
rollout. The added replay workload must not be attributed to architecture.

`PROFILE.md` separates coarse inference, environment, copies and training
intervals, kernel categories, graph-capture API time and activity after final
graph instantiation. Native final-interval train-model time per rollout is about
49 ms for A IMPALA versus 1.1 ms for A ours, and 149 ms versus 3.4 ms for B.
Environment time stays roughly 0.3–0.4 ms per rollout. Logged memory samples are
about 3.1–3.2 GiB for ours/Nature and 5.1–5.2 GiB for IMPALA/Impoola, not peak-memory
measurements.

Across IMPALA/Impoola traces, shared matrix multiplication accounts for 61–64%
of summed kernel durations, patch materialization 19–22%, bias gradients 5–6%,
input-gradient gathering 4–5%, and pooling about 2%. A large-K SGEMM kernel family
alone takes 1.34–1.42 seconds in A and 4.20–4.25 seconds in B. It is a concrete
hot kernel family, but these shared cuBLAS kernel names do not prove how much
belongs to convolution weight gradients versus other matrix-multiply callers.
No complete per-layer forward/input-gradient/weight-gradient decomposition is
claimed. Named optimizer kernels exclude the optimizer's shared GEMM work.

The first `fused_rollout` host NVTX range is approximately 44–45 ms, versus about
17–90 microseconds for later host ranges. First train host ranges are about
5–25 ms, versus 0.12–2.54 ms later. These host enqueue ranges include first-use
work and are not asynchronous GPU execution durations. Graph-capture API calls
sum to about 4.7–20.5 ms; most kernel time remains after final graph instantiation.
The measured reference-backend cost is therefore not explained by graph capture
alone. Logging cost is not independently isolated from host/process overhead.

## Prioritized follow-up candidates — no kernels rewritten

1. **Fix evaluation accounting before stronger quality claims.** Add per-episode
   identity, whole-match/rally decision counters and slot contribution records;
   validate the separately versioned allocation/truncation policy below. The
   sparse match completion and current completion-conditioned sample are measured
   or source-established problems. Keep this logging patch, without advertising
   a speedup.
2. **Attribute and optimize the large matrix multiplies and patch path.** Add
   layer/direction NVTX labels in a future diagnostic build, then evaluate
   deterministic float32 GEMM algorithms and tiled/implicit patch processing for
   the measured shapes. Materializing patches costs 19–22% in the residual
   backends; the shared GEMM category is larger still. Avoid claiming a GEMM
   algorithm improvement until its actual caller, workspace, numerical behavior
   and unprofiled time have been measured. Do not change precision or use TF32
   to obtain a nominally same-math result.
3. **Improve bias-gradient parallelism if worthwhile.** `imp_bias_grad` launches
   one 256-thread block per output channel and iterates over rows. Its measured
   0.21–0.22 seconds in A and 0.65–0.66 seconds in B justify testing a staged,
   deterministic partial reduction. Preserve fixed reduction ordering and avoid
   nondeterministic floating-point atomics; budget any additional scratch space.
4. **Investigate copy and launch overhead for small models.** Ours/Nature have
   about 1.1–1.25 ms of native copy time per 2,048-decision rollout, comparable
   to their 1.3–1.5 ms inference time. Existing graph tracing still records
   53,504–95,424 kernel executions in these short jobs. Consider consolidated
   staging/copies and compatible pointwise fusion, keeping observations, horizon,
   ordering and RNG consumption unchanged. Kernel execution counts are not
   equivalent to that many host launch calls because CUDA graphs are enabled.

For every future optimization, run existing independent float32 forward/all-
parameter-gradient tests at their established tolerances, pool edge/tie cases,
eager/graph checks, finite-weight audits, same-seed repeatability and unchanged
parameter/config checks. Require bitwise repeatability within each new
implementation. If a changed reduction algorithm changes roundoff relative to
the old build, report it rather than promise cross-build bitwise identity.
Retain an unprofiled paired benchmark with cold-start and steady-state boundaries,
memory use and exact hashes; do not turn profiler durations into publication
throughput. Any learning confirmation is a separate predeclared experiment.

Coverage: all eight Nsight Systems CUDA-node/NVTX profiles succeeded without
reported profiler warnings. CPU sampling, hardware counters and per-layer GEMM
labels were unavailable in this chosen trace protocol; no tools or permissions
were installed or changed. See `PROFILE.md` for the precise limits.

## Original experiment and timing

The preserved experiment is `../confirm-5090.full.JXYKB06r/`, measured source
`1d9e9e0dca88706c7a8495644c8dc584a4446f11`. Its full launcher took 11,581.80 seconds
(3 h 13 min 2 s). Forty training processes used 4,923.91 seconds; 320 evaluation
attempts used 6,611.45 seconds. Eight evaluation timeouts consumed 2,400.75 seconds
(36.31% of evaluation time). The other 312 evaluations used 4,210.70 seconds.
Thus evaluation exceeded training in aggregate because the evaluation budget
included eight checkpoints per training run and substantial timeout cost.
Removing timeout time arithmetically does not remove those missing outcomes
from the scientific analysis.

Evaluation has a completed-match target rather than a decision budget. Pong has
no match or rally decision cap. Match duration depends on the saved policy and
trajectories. The dashboard's episode length is only the tick count since the last
point reset and cannot be used as full-match length.

## Correctness and source scope

All four validation pairs completed under their 30-second caps with identical
final score, point fraction, game count and parameter records. Checkpoint hashes
were unchanged. Requests for 32 matches returned 39/35/62/64 matches respectively
for ours/Nature/IMPALA/Impoola, demonstrating the existing rollout-boundary
overshoot. Both periodic and final verbose dashboards ran successfully. The
seven-second unreachable-target attempt emitted progress and no `CUDA_EVAL`
success record. Exact outputs and times are in `validation.json`.

Only `src/pufferl.cu` differs among the 32 original native source files checked.
Its patch moves resource queries from every evaluation rollout to the two actual
standalone dashboard sites and adds flushed progress. Native math, reductions,
RNG, Pong rules, observations and CNN architectures are unchanged. Existing 5090
float32 forward/gradient and repeatability audits therefore remain applicable;
their paths and source evidence are retained. Nine existing configuration tests
and four focused CPU query-failure/evidence tests passed.

The final evidence audit verified **1,927 original file identities** unchanged,
including all original binaries/checkpoints, configs and retained results.

## Sampling semantics and a future policy

Standalone evaluation initializes the vector once and does not periodically reset
all environments. Each environment starts a new match after reaching `max_score`.
The estimator pools completed matches and stops after the first rollout whose
cumulative count reaches the target. Fast-finishing slots can contribute many
times while other slots remain in long matches. This mechanism is established
by source inspection; its empirical effect cannot be quantified from aggregate
logs without per-episode identities and allocation records.

`PONG_EVALUATION_POLICY_PROPOSAL.md` describes a separate v2 with 512 assigned
episode IDs, eight per slot, explicit decision caps and truncation bounds. It was
not substituted into this experiment. No partial or diagnostic score replaces an
original missing 300-second outcome.

## What the original full checkpoint frontier supports

The original result stays **40/40 training runs and 312/320 successful evaluations,
with eight timeouts**. There are five paired training seeds and two shared recipes.
The complete observed data consist of all 312 evaluated checkpoint observations;
the frontier over all 320 planned outcomes remains partly unidentified.

- Recipe A: ours and Nature contribute low-time mean-frontier points. Impoola
  reaches the 95% point-fraction target in all five seeds and finishes at 99.91%
  mean [99.85%, 99.95% pointwise paired-seed bootstrap interval]. IMPALA finishes
  at 79.64% with wide seed uncertainty [39.66%, 99.96%].
- Recipe B: Nature supplies nearly all observed mean frontier points; ours has
  the cheapest initial time point. Ours finishes at 29.90% [0.36%, 69.55%], IMPALA
  at 59.37% [19.68%, 99.07%], and Impoola at 0% in all five seeds. Recipe dependence
  is substantial.
- Final five-seed means are missing for A ours, A Nature and B Nature. Their
  observed-seed identification bounds are respectively 68.42–88.42%, 69.46–89.46%
  and 79.66–99.66%; these are not confidence intervals.
- Mean-frontier membership is computed only where all five evaluations exist;
  missing later observations can alter it. All per-seed curves remain visible.
  The frozen 10,000-draw whole-paired-seed bootstrap produces pointwise intervals,
  not simultaneous confidence for superiority over an entire frontier. A
  degenerate five-seed bootstrap interval does not prove population certainty.
- These measurements do not establish a quality advantage over Nature or SOTA.
  Time comparisons refer to these specific native implementations and recipes.
  Backend inefficiency must not be interpreted as intrinsic architectural cost.

See the original `review/REPORT.md`, `curves.html`, `mean-curves.csv`,
`seed-frontiers.csv`, and `target-attainment.csv` for every checkpoint, seed,
crossing and decline. This audit does not rerank the original quality frontier
using shortened performance jobs.

## Reproduction

Read `plan.json`, `frozen-panel.sha256`, `revision.txt`, `working.patch`,
`source.sha256`, `binaries.sha256`, and each attempt's `inputs.sha256` first.
Every exact process invocation is in its `command.sh`; reuse the recorded working
directory and `source build/connect4cnn/runtime-5090.sh`. Original and patched
binaries and all checkpoints stay local. Replaying an invocation constitutes a
new attempt and must use a fresh output directory rather than overwrite evidence.

The original binaries are used directly for the old arm; no separately instrumented
old scheduling build was substituted. Original dashboard decisions are rounded;
old completed-match counts are unavailable before the final success record.
Patched progress elapsed excludes construction/loading, whereas process wall
includes those costs. Low-frequency activity is in each `activity.log`; short jobs
can complete between five-second samples.
