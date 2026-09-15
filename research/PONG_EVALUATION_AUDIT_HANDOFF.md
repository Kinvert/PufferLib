# Next RTX 5090 task: evaluation diagnosis and reference profiling

**Historical task, completed.** The received archive is analyzed in [PONG_EVALUATION_AUDIT_RESULTS.md](PONG_EVALUATION_AUDIT_RESULTS.md). Follow [current 5090 instructions](../NEXT_5090_VALIDATION.md), not another run of this diagnostic campaign.

This is an implementation/audit task after the 40-run Pong replication. **Do not rerun the full training experiment or start an architecture search.** No transfer from G240 is needed; all required checkpoints and original evidence are already on the 5090.

## Existing evidence

Work in `~/Git/ml/cnn-5090`. Read `AGENTS.md` and preserve current changes. Source for the completed replication: `1d9e9e0dca88706c7a8495644c8dc584a4446f11`; original run `build/pongcnn/confirm-5090.full.JXYKB06r/`. Read its frozen protocol, jobs, reviewed report and raw logs. Forty training runs completed; eight of 320 evaluations timed out. Do not replace or regenerate its original results with follow-up measurements.

Source `build/connect4cnn/runtime-5090.sh`; reuse the existing local venv, CUDA/NCCL and builds where applicable. No Docker, system/driver changes, Torch installation or changes to other environments. Use direct patches, native C/CUDA training/evaluation and existing external reporting only. Do not push or publish. Preserve original binaries and record new source/config/binary hashes separately.

## Concrete local findings to inspect and reproduce

At the G240 source revision, `src/pufferl.cu::trainer_eval_log` calls `log_util` after **every rollout**. `log_util` queries NVML, CUDA memory and `/proc/self/status`. Evaluation scoring needs the environment log, not those resource queries. A locally implemented fix removes `log_util(p, out)` from `trainer_eval_log` and calls `log_util(p, show)` immediately before each actual standalone dashboard print in `eval_loop`:

1. The periodic dashboard branch, gated by `!render && verbose` and the existing 0.6-second interval.
2. The final `if (verbose)` dashboard before returning the result.

Both calls are guarded by `if (!board)` so training-owned dashboard resource fields remain frozen. **The final dashboard call is mandatory**: the first local patch missed it and failed with `missing key [?] util/gpu_percent`; that failed attempt was preserved. The fix changes telemetry scheduling, not rollout count, actions, RNG, model math, games or scores. Inspect your newer source before reproducing it; do not blindly overwrite your agent's changes.

Also add periodic progress to headless verbose evaluation, without changing the final `CUDA_EVAL` line:

```c
// Before the evaluation loop:
double started = wall_clock();
double last_progress = started;
long start_step = p->global_step;

// After the render branch and obtaining cumulative completed games n:
if (verbose && now - last_progress >= 5.0) {
    printf("CUDA_EVAL_PROGRESS steps=%ld games=%ld requested=%ld elapsed=%.3f\n",
        p->global_step - start_step, n, eval_episodes, now - started);
    fflush(stdout);
    last_progress = now;
}
```

Elapsed is evaluation-loop wall time, excluding model construction/load; record full process wall separately. The line is diagnostic, not a completed score. Incomplete runs must not emit a success record. Do not collect slow hardware telemetry on every rollout to produce progress. If diagnosing individual environments, inspect counters only while rollout workers are stopped; avoid new RNG calls or mutations.

Pong currently terminates matches at `max_score`, with no match/rally decision cap. `env->tick` resets between points, so the existing logged `episode_length` is not full match length. This supports investigating long matches, but **does not establish the cause of the eight timeouts**. Do not change Pong rules, insert artificial rewards, force episode completion or lower `max_score` to make results finish.

If your Bash launchers still hardcode `/usr/lib/wsl/lib/nvidia-smi`, use PATH-first tool discovery with that location only as fallback. A failed GPU query must abort the idle check; an empty failed query is not evidence of an idle GPU. G240 added shared functions to `ocean/connect4cnn/runtime_env.sh` and updated forward-looking Pong canary/sweep launchers; the old campaign-specific recovery script remains historical.

## Stage 1: validate correctness, then diagnose the timeouts

1. Freeze a deterministic diagnostic checkpoint list from original receipts: all eight timed-out checkpoints, plus one successful checkpoint per model/recipe (eight controls). Select the first successful checkpoint in sorted job/step order in each cell; deduplicate if necessary and record actual count. These are diagnostics, not new statistical replications.
2. Before profiling, verify original versus patched evaluation on one fast successful saved checkpoint per family. Use matching saved config, seed, evaluation agents, precision and 32 requested matches; cap each run at 30 seconds. Assert identical final native score/perf/game count/parameters for completed pairs and unchanged checkpoint hashes. Incomplete checks are unresolved, never a pass. Retain the mandatory final-dashboard check. Reuse unchanged encoder numerical/gradient audits; rerun relevant ones if native math changes.
3. Test a deliberately unreachable episode target with a seven-second cap to verify flushed progress survives termination and no success record is emitted. Label this as a synthetic timeout probe, outside benchmark results.
4. Run the fixed diagnostic panel with original versus patched evaluators using each checkpoint's **original 512-match target and evaluation config/seed**, but a new explicitly labeled **30-second diagnostic process cap**. At most 32 evaluations, up to 16 minutes at the cap. Do not overwrite original 300-second results, extend their limits, or treat new diagnostic outcomes as resolved original final means. Rotate old/new order to reduce order effects.
5. Capture progress, process wall, completed games and decisions, CPU/GPU use with low-frequency external sampling, and available profiler summaries. No active agent polling. Distinguish slow rollouts from many advancing decisions with few completed matches. For old binaries lacking progress, inspect existing dashboard steps/uptime and partial counters or use an instrumented original-scheduling build, clearly labeled and hashed.

Investigate whether completion-count evaluation over repeatedly resetting vector environments favors shorter episodes while other environments stall. Do not silently change the sampling estimator. If a deterministic per-episode decision cap or fixed-episode allocation is necessary, write a separately versioned proposal with truncation/scoring and coverage rules, and validate it on a common panel before recommending a uniformly applied new study. Preserve the original incomplete results and bounds.

## Stage 2: profile the native reference implementations

Use all four existing architectures and both exact recipes from the completed protocol. Keep model/core, environment, precision, agents/threads, horizon and minibatch fixed. Run one short **131,072-decision** job per cell, seed **52001**, at most 60 seconds per job. These eight jobs are throughput diagnostics, not learning comparisons; the shortened learning schedule is explicitly different from the replication. Validate expected parameter counts and finite checkpoints.

Measure existing native `perf/*` categories, then one instrumented repeat per cell if a supported profiler is already installed (up to 90 seconds each). Inspect local `nsys --help` for supported capture flags and the existing `base.profile`/CUDA profiler API hooks. Do not install global profiling tools or assume a permission failure justifies system changes. If detailed tools are unavailable, retain native coarse timings and clearly state the limitation.

Separate environment, transfers, rollout inference, training forward/backward, optimizer and logging overhead. Where supported, report convolution forward/input-gradient/weight-gradient, pooling, activation/residual, projection/core kernels, launch counts and memory use. Record actual replay update counts: recipe B's higher replay can make model training much more expensive. Do not compare different replay workloads as a pure architecture effect. Separate first graph capture/warmup from steady-state and full process cost. Instrumented durations are not publication throughput measurements; compare against unprofiled controls and disclose perturbation.

Do not conclude that a 10x–25x model speed gap is automatically architectural. Identify evidence for extra FLOPs, memory traffic, small-kernel launch overhead, poor GPU utilization or avoidable implementation cost. Do not rewrite the kernels in this audit. Propose any subsequent optimization with a same-math numerical/gradient/repeatability acceptance test and equal treatment of baselines.

## Boundaries, reporting and follow-up

The entire new GPU diagnostic campaign has a **45-minute wall cap**, with child-process cleanup and incomplete records preserved if reached. CPU review/implementation can continue separately. Launch detached after validation, verify startup once, and do not actively monitor. No W&B integration work or full quality sweep is needed. Reuse the same fixed panel after tooling failure only if necessary, preserving both attempts; do not retry based on scores.

Write one report with: source/patch/build identities; original and diagnostic protocols; tests; explicit timeout classifications and unresolved cases; per-model/recipe unprofiled and instrumented timings; profiler coverage and limitations; full original-frontier interpretation with uncertainty/missing values; prioritized concrete fixes. No SOTA claim or changed success counts for the original study.

Archive small receipts, raw logs, scripts, configs and source hashes under a fresh `build/pongcnn/eval-profile-audit.*` directory and preserve large binaries/checkpoints locally. Make a local clean commit of only intentional tested changes if appropriate; no push or data transfer is needed to finish. Return a concise report and exact reproduction commands. We can decide how to move evidence later.

The next engineering milestone after this audit is a clean constructor supporting normal configurable image shapes/channels and an original external benchmark integration feasibility check. Those are separate from this bounded GPU task; do not launch Procgen studies or redesign the CNN while diagnosing evaluation.
