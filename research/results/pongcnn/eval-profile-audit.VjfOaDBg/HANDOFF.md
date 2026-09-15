# RTX 5090 Pong audit — handoff to the requesting agent

## Status and decisions

The requested bounded evaluation/backend audit is complete. GPU work, including
builds and correctness validation, took **771.01 seconds (12 min 51 s)**. All 57
planned attempts have receipts. No full 40-run experiment, architecture search,
W&B work, system changes or kernel optimization was performed.

**Recommended next step:** review the evidence and agree on a separately versioned
evaluation policy before making stronger quality claims. Then attribute the hot
matrix multiplications to layers and test same-math backend optimizations. These
are proposed follow-ups, not campaigns already authorized or launched by this
handoff. Do not repeat the full experiment merely to fill missing evaluations.

## Conclusions to carry forward

1. Original training took 4,923.91 s; evaluation took 6,611.45 s. The eight original
   300-second timeouts alone cost 2,400.75 s (36.31% of evaluation time).
2. All eight timeout checkpoints again timed out under both arms' new 30-second
   diagnostic caps. Patched progress showed 14.64–18.22M decisions but only 0–13
   completed matches at approximately 25 seconds. Throughput remained
   586k–733k decisions/s. Sparse match completion is established; whether any
   individual trajectory is indefinitely nonterminating remains unresolved.
3. All four 32-match correctness pairs and all eight 512-match control pairs
   matched exactly in final score/perf/games/parameters. Checkpoint hashes were
   unchanged. Both standalone dashboard sites passed. Intentional timeout
   progress was flushed without a success record. Removing per-rollout resource
   queries produced no consistent measured speedup.
4. Eight unprofiled and eight Nsight Systems training diagnostics passed with
   expected parameters and finite metrics/checkpoints. Each profiled/unprofiled
   pair produced identical checkpoint bytes. Original resolved recipes were
   checked; recipe A executes 64 replay updates and B 192 in these short jobs.
5. The residual backends spend much of summed kernel time in shared GEMMs
   (61–64%), patch materialization (19–22%), and bias gradients (5–6%). Shared
   cuBLAS names do not fully distinguish convolution directions, core/projection
   and optimizer callers. All eight profiler summaries were independently checked.
6. The original result remains **40 training runs, 312 successful evaluations,
   eight timeouts**. All 1,927 audited original file identities are unchanged.
   Diagnostic partial matches do not replace missing original outcomes.
7. Original frontier: A favors ours/Nature at low time and Impoola at high score;
   B favors Nature across nearly all observed mean-frontier points. Missing
   evaluations and five-seed uncertainty prevent a quality-superiority claim over
   Nature. Backend inefficiency is not intrinsic architectural inferiority.

## Source and evidence identity

- Original experiment: `build/pongcnn/confirm-5090.full.JXYKB06r/`.
- Original source: `1d9e9e0dca88706c7a8495644c8dc584a4446f11`.
- Audit: `build/pongcnn/eval-profile-audit.VjfOaDBg/`.
- Audit native changes: only evaluation diagnostic scheduling/progress in
  `src/pufferl.cu`; both standalone dashboard refreshes are guarded by `!board`.
  Pong sweep/recovery launchers also received GPU-discovery/query-failure fixes.
- Exact patch SHA256:
  `aa1192166ab5eeeb7d115f29f80ca35a0b9142a62c23fcd78053ac2fed732790`.
- Original audit evidence-manifest SHA256:
  `630422da42b09571416cc4c57f381ee35cc043cd153227d2c75375e8c8384a59`.

The transfer package's `audit/` contains reports, raw process/activity logs,
configs, commands, source/patch receipts, analysis JSON, native timing histories,
and eight official Nsight kernel-summary CSVs. `original-context/` contains the
original protocol, planned jobs/results and frontier summaries. The full binary
checkpoints and Nsight trace databases remain on the 5090; their hashes and
derived summaries are included. The archived full audit manifest describes
those retained files too; use the package's `PACKAGE.sha256` to verify the files
actually included in this smaller package.

## Read/reproduce

The remainder of the consolidated `HANDOFF.md` embeds the complete findings,
paired evaluation/process tables and native/profiler timing tables. The separate
evaluation-policy proposal is under `audit/PONG_EVALUATION_POLICY_PROPOSAL.md`.
Each attempted native command and working directory is recorded in `command.sh`.

On the 5090, reuse `source build/connect4cnn/runtime-5090.sh`. CPU reporting can be
regenerated with:

```bash
.venv/bin/python ocean/pongcnn/eval_profile_audit.py build/pongcnn/eval-profile-audit.VjfOaDBg report
.venv/bin/python research/analyze_pong_audit.py build/pongcnn/eval-profile-audit.VjfOaDBg
```

The deep verification reads original checkpoints and full local profiler SQLite
traces, so it needs the retained 5090 files. No SCP, push or further GPU runs have
been initiated as part of this handoff preparation.
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
# Pong evaluation/backend audit

Original: `/home/keith/Git/ml/cnn-5090/build/pongcnn/confirm-5090.full.JXYKB06r`.

The original 40 training runs and 312 successful / 8 timed-out evaluations remain unchanged.

## Evaluation panel

| Checkpoint | Original | Old exit / seconds | New exit / seconds | New last progress steps / matches |
|---|---|---|---|---|
| 00-a-flex_quality-s31002-2621440 | timeout | 124 / 30.096502 | 124 / 30.096114 | 18190336 / 13 |
| 01-a-flex_quality-s31002-3145728 | timeout | 124 / 30.103523 | 124 / 30.098822 | 17121280 / 0 |
| 02-a-flex_quality-s31002-3670016 | timeout | 124 / 30.088995 | 124 / 30.109231 | 18219008 / 3 |
| 03-a-flex_quality-s31002-4194304 | timeout | 124 / 30.101502 | 124 / 30.095702 | 17326080 / 1 |
| 04-a-nature_cnn-s31005-3145728 | timeout | 124 / 30.099728 | 124 / 30.103729 | 15132672 / 3 |
| 05-a-nature_cnn-s31005-3670016 | timeout | 124 / 30.088661 | 124 / 30.105026 | 16050176 / 1 |
| 06-a-nature_cnn-s31005-4194304 | timeout | 124 / 30.098098 | 124 / 30.10065 | 14639104 / 1 |
| 07-b-nature_cnn-s31004-4194304 | timeout | 124 / 30.102451 | 124 / 30.100683 | 16093184 / 2 |
| 08-a-flex_quality-s31001-524288 | ok | 0 / 4.826712 | 0 / 4.899081 | — / — |
| 09-a-nature_cnn-s31001-524288 | ok | 0 / 2.119793 | 0 / 2.153207 | — / — |
| 10-a-impala_cnn-s31001-524288 | ok | 0 / 17.374578 | 0 / 17.385927 | 1996800 / 448 |
| 11-a-impoola_cnn-s31001-524288 | ok | 0 / 6.301666 | 0 / 6.280568 | 661504 / 420 |
| 12-b-flex_quality-s31001-524288 | ok | 0 / 5.617216 | 0 / 5.587326 | 3276800 / 480 |
| 13-b-nature_cnn-s31001-524288 | ok | 0 / 2.58833 | 0 / 2.581528 | — / — |
| 14-b-impala_cnn-s31001-524288 | ok | 0 / 16.155126 | 0 / 16.208337 | 1980416 / 475 |
| 15-b-impoola_cnn-s31001-524288 | ok | 0 / 1.61783 | 0 / 1.701924 | — / — |

## Short performance diagnostics

131,072 decisions, seed 52001; shortened annealing schedule. Instrumented durations include profiler overhead and are not publication throughput.

| Cell | Mode | Valid | Process seconds | Native seconds | Updates |
|---|---|---|---:|---:|---:|
| a-flex_quality | unprofiled | True | 0.607579 | 0.31502723693847656 | 64 |
| a-flex_quality | instrumented | True | 2.723064 | 0.34293198585510254 | 64 |
| a-nature_cnn | unprofiled | True | 0.642851 | 0.33883237838745117 | 64 |
| a-nature_cnn | instrumented | True | 2.69134 | 0.3636205196380615 | 64 |
| a-impala_cnn | unprofiled | True | 4.4383 | 4.154934644699097 | 64 |
| a-impala_cnn | instrumented | True | 7.666875 | 4.316418647766113 | 64 |
| a-impoola_cnn | unprofiled | True | 4.414726 | 4.129648923873901 | 64 |
| a-impoola_cnn | instrumented | True | 7.708112 | 4.262478351593018 | 64 |
| b-flex_quality | unprofiled | True | 0.750086 | 0.464357852935791 | 192 |
| b-flex_quality | instrumented | True | 2.95063 | 0.49948740005493164 | 192 |
| b-nature_cnn | unprofiled | True | 0.785922 | 0.5022721290588379 | 192 |
| b-nature_cnn | instrumented | True | 3.208242 | 0.535348653793335 | 192 |
| b-impala_cnn | unprofiled | True | 10.801918 | 10.502366542816162 | 192 |
| b-impala_cnn | instrumented | True | 14.97858 | 10.813774108886719 | 192 |
| b-impoola_cnn | unprofiled | True | 10.696782 | 10.40699553489685 | 192 |
| b-impoola_cnn | instrumented | True | 14.92263 | 10.780262231826782 | 192 |

Raw configs, commands, low-frequency activity, progress and profiler output accompany each attempt. Full measurements are in `measurements.json`. Interpretation and profiler coverage are recorded separately in `FINDINGS.md`.
# Native timing and profiler coverage

Native values below are the last logging interval divided by its recorded rollout count. One rollout is 2,048 decisions. These are coarse local intervals, not sums of the downsampled history. Training model includes forward/backward, loss, updates and optimizer work.

| Cell | Rollouts in interval | Rollout ms | Inference ms | Environment ms | Copy ms | Train model ms | Train misc ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| a-flex_quality | 63 | 2.9577 | 1.2830 | 0.3533 | 1.1111 | 1.1152 | 0.0150 |
| a-nature_cnn | 63 | 3.2684 | 1.5428 | 0.3358 | 1.1994 | 1.1702 | 0.0151 |
| a-impala_cnn | 3 | 15.3960 | 12.4259 | 0.3857 | 2.3617 | 48.9844 | 0.0198 |
| a-impoola_cnn | 3 | 14.3032 | 12.3116 | 0.3422 | 1.4708 | 47.4997 | 0.0174 |
| b-flex_quality | 63 | 3.0060 | 1.2843 | 0.3365 | 1.1429 | 3.4236 | 0.0152 |
| b-nature_cnn | 63 | 3.3926 | 1.5443 | 0.3378 | 1.2482 | 3.6250 | 0.0153 |
| b-impala_cnn | 3 | 15.0848 | 12.3801 | 0.3394 | 2.1864 | 148.6936 | 0.0202 |
| b-impoola_cnn | 3 | 14.7241 | 12.3120 | 0.3224 | 1.8777 | 146.6671 | 0.0179 |

## CUDA trace

Times sum traced kernel durations across streams; they are not elapsed wall time or GPU utilization. Matrix-multiply kernels are shared across convolution forward/input-gradient/weight-gradient, projection/core and optimizer, so those callers remain unresolved. Named kernels and all CUDA API summaries are preserved in analysis.json.

| Cell | Kernel executions | Kernel seconds | Shared GEMM % | Patch % | Bias gradient % | Input gather % | Pool % | Capture API ms | After-capture kernel seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a-flex_quality | 53504 | 0.1494 | 65.15 | 10.48 | 6.06 | 0.94 | 0.00 | 4.788 | 0.1482 |
| a-nature_cnn | 65088 | 0.1692 | 58.99 | 14.57 | 5.37 | 3.08 | 0.00 | 4.702 | 0.1678 |
| a-impala_cnn | 191872 | 3.9751 | 62.61 | 21.77 | 5.41 | 4.30 | 1.81 | 10.905 | 3.9630 |
| a-impoola_cnn | 191808 | 3.9312 | 61.36 | 22.35 | 5.69 | 4.57 | 1.83 | 12.205 | 3.9192 |
| b-flex_quality | 77056 | 0.2985 | 68.39 | 8.33 | 9.10 | 1.42 | 0.00 | 5.991 | 0.2973 |
| b-nature_cnn | 95424 | 0.3225 | 59.74 | 12.79 | 8.47 | 4.84 | 0.00 | 6.650 | 0.3211 |
| b-impala_cnn | 283264 | 10.4289 | 64.23 | 19.45 | 6.20 | 5.11 | 1.86 | 18.950 | 10.4168 |
| b-impoola_cnn | 283072 | 10.4020 | 63.58 | 19.92 | 6.36 | 5.14 | 1.87 | 20.533 | 10.3900 |

Capture API time sums cudaStreamBeginCapture/EndCapture, cudaGraphInstantiate and cudaGraphDestroy CPU calls; it does not measure the complete first-use capture region. After-capture excludes kernels starting before the final graph-instantiation return. First and later NVTX host-range durations are retained separately in analysis.json and do not equal asynchronous GPU execution time.

Nsight Systems 2025.5.2 captured CUDA node activity and NVTX under the existing base.profile CUDA profiler hooks. CPU sampling and context-switch tracing were disabled. No Nsight Compute hardware-counter replay was run. No occupancy, cache-bandwidth, per-layer GEMM attribution or isolated resource-query/logging CPU time is claimed. Five-second GPU/CPU samples may miss brief jobs and memory peaks. Instrumented full process time includes trace export overhead. Unprofiled controls remain separate.
