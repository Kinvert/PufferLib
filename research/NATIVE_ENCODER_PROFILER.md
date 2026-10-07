# Native encoder profiler preparation

Current paired numerical protocol [v2](DENSE_ALIAS_ACCEPTANCE.md) also includes
the common B2048 learner batch: 72 declarations, zero executed. Its actual
host preparation/inspection and 39 tests pass; the earlier 54-case v1 archive
remains unchanged. Neither variant's numerical/timing GPU gate is accepted;
both holds and full-policy/larger-batch/general-callback gates remain.

October 6 paired-math follow-up: [the dense-alias acceptance tool](DENSE_ALIAS_ACCEPTANCE.md)
now prepares the exact quality/Nature graphs, including H64's equal-projection
branch. Two fresh test libraries compile and 39 host/synthetic checks pass;
all GPU numerical paths remain unexecuted. Supervised timing remains subsequent
work in an explicitly scheduled window, with complete source/tool receipts.
These fixed-graph checks don't replace general Flex/compact/encoder5 or
concurrent learner/reload qualification. Preserve original profile packets.

October 6 isolated follow-up: [dense patch alias candidate](DENSE_PATCH_ALIAS_CANDIDATE.md)
adds opt-in profiler/test callbacks for quality and Nature without production
changes. Paired native descriptors/count audits pass; both complete 64-case
panels are prepared. Descriptor/runtime markers must agree, preventing a
candidate run from being mislabeled as the ordinary backend. Existing baseline
packets keep their original source/tool receipts. GPU math, direct paired
equivalence, timing and both holds remain pending; no measured speedup.

October 6, 2026. **Compiled and host metadata audited; GPU timing/numerical
execution remains pending. Neither GPU hold is lifted.** No trainer or
production backend was changed.

October 6 large-batch follow-up: [learner memory preflight](LEARNER_MEMORY_PREFLIGHT.md)
extends only host `--describe` to batch 65,536; scheduled `--run` still caps
2,048 and remains on hold. Optional `audit_encoder_profile.py --batches`
checks six requested batches against C arithmetic (72 comparisons). Stock
Flappy IMPALA/Impoola training tensors alone exceed 16 GiB; unchanged stock
minibatch cannot fit the 5060. This is required-payload arithmetic, not peak
VRAM, intrinsic architecture memory or neural/runtime acceptance. Original
36-comparison archives/default commands below remain unchanged.

October 6 follow-up: the exclusive supervised launcher now passes 16 host/
synthetic tests, and actual binaries prepare the full 64-case packet without
GPU queries or policy execution. New native exports retain exact byte arrays;
its host-only `--hash` mode passes nine known-answer/block-boundary checks.
GPU runtime, numerical oracles and concurrent trainer qualification remain pending.

Separate [shared GEMM acceptance preparation](SHARED_GEMM_ACCEPTANCE.md) now
compiles actual main/dW fork/join paths with independent GPU scalar-double
references, two disjoint lanes and serial/overlap scheduling. Its host/synthetic
checks pass; its exclusive supervisor now has host/synthetic preparation,
deadline/failure and complete paired-receipt checks. GPU arithmetic/runtime
remain pending.
These checks are separate from timed encoder profiling and certify neither
whole-policy concurrency nor a speedup.

The purpose is to qualify the implementation comparison in the
[multi-task plan](MULTI_ENV_ROBUSTNESS.md): distinguish encoder compute from
environment/core/optimizer costs and investigate the shared cuBLAS workspace
issue. All six native pixel tasks use `[1,36,44]`. Synthetic encoder profiling
is not learning or evidence of generality across those games/drawings.

## Permitted preparation

[encoder_profile.cu](../ocean/connect4cnn/encoder_profile.cu) includes actual
native constructors, registration and CUDA forward/backward callbacks. Its
default binary accepts locked quality (encoder 4) or adapted Nature (2);
separate IMPALA/Impoola binaries require encoder 0. Changed quality graphs,
encoder 5 and incompatible family/config pairs are rejected. The audit covers
H64/128/256, including quality's projection-equals-hidden branch. The core isn't
constructed or executed.

`--describe FULL.ini BATCH` constructs host descriptors/registers shapes and
returns before CUDA allocation, initialization or policy execution. Actual
callback payloads and 16-byte-aligned allocator sizes are distinct. Reported
activation counts exclude input/upstream, weights/gradients, core/head/replay/
optimizer/vendor workspace. They aren't measured peak VRAM or speed.

```bash
# Compilation only; explicit architecture and fresh output.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/build_encoder_profile.sh \
  build/FRESH_PROFILE_BUILD

# Host registration only; use a freshly prepared multi-task panel.
source ocean/connect4cnn/runtime_env.sh
build/FRESH_PROFILE_BUILD/default --describe \
  build/FRESH_PANEL/connect4cnn/r0/flex_quality-s53111/config/default.ini 64

# Independent configuration-arithmetic cross-check; see backend audit for C build.
.venv/bin/python research/audit_encoder_profile.py \
  --binaries build/FRESH_PROFILE_BUILD \
  --cost-binary build/FRESH_ENCODER_COST_ID/costs \
  --configs build/FRESH_PANEL/connect4cnn/r0 \
  --out build/FRESH_PROFILE_BUILD/metadata-audit --host-hash

# Freeze the full repeated profiling packet. This is still GPU-free.
.venv/bin/python ocean/connect4cnn/encoder_profile.py prepare \
  --binaries build/FRESH_PROFILE_BUILD \
  --configs build/FRESH_PANEL/connect4cnn/r0 \
  --out build/FRESH_PROFILE_BUILD/panel
```

The panel must contain exactly one config per fixed family. The audit copies
resolved configs, varies hidden width and checks batches 1/64/2048 against
[encoder_costs.c](encoder_costs.c): 36 comparisons and seven rejected
ID/width/graph/family cases. This is configuration verification, not CPU neural
validation. Input hashes are rechecked and existing outputs refused. Native
build receipts retain compiler/commands/dirty worktree/source/binary hashes;
the revision alone does not identify a dirty measured source.

## Scheduled GPU interface — do not execute under the hold

Native command format, not authorization or a supervised campaign launcher:

```text
PROFILER --run FULL.ini BATCH rollout|learner eager|graph default|reassign WARMUP SAMPLES RECEIPT_DIR
```

Initial intended panel: four encoders × two phases × eager/graph × default/
reassigned workspace, 32 cases per repetition, H128. Two separate-process
repetitions give the default 64-case panel. Rollout B64, learner B2048; identical
paired configs, seed 173, at least five warmups and equal sample counts. Retain
every failure/slower case. The supervisor now uses the shared
`build/connect4cnn/hardware-benchmark.lock`, contention/query rejection,
hardware/compiler/runtime-environment/config/binary/source receipts, bounded
process-group timeouts and preserved failure logs. Neither GPU has run it.
Idle queries and the cooperative reservation detect observed contention, not
every brief job from software that ignores the reservation. Exclusive scheduling
still matters. **Do not launch bare native commands instead.**

After Kinvert explicitly schedules an exclusive GPU window and the relevant
runtime gates, the supervised entry point is:

```text
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python ocean/connect4cnn/encoder_profile.py run --out build/FRESH_PROFILE_BUILD/panel --timeout 120
```

This is documentation of the interface, **not permission to lift either hold**.
Preparation copies binaries, all compiled source-manifest files, resolved INIs,
tooling and compiler receipts and rechecks original hashes. Launch/load rejects
changed copied inputs, protocol or supervision code. The complete panel and
repetitions are fixed; family order rotates by repetition. No subset launch,
automatic restart or overwrite of partial execution is accepted. `FAILURE.json`
preserves failed/partial cases without `REPORT.json` aggregate success. Host
tests use explicit synthetic receipts and mocked queries/processes; only the
timeout cleanup test launches a plain host sleep, never a neural policy.

`rollout` measures encoder forward. `learner` measures forward plus parameter
gradients with fixed nonzero upstream values and no update. Inputs are synthetic
grayscale arrays, not a collected dataset; weights use native seeded
initialization. CUDA events measure serial executions with synchronization
between samples; JSON retains every duration and mean. These aren't training
SPS, policy quality or concurrent throughput. Standalone graph capture doesn't
qualify the complete production training graph.

Eager output/gradient bytes are retained before timing. Warmup and final timed
execution must match exactly, nonfinite values fail and parameters must remain
unchanged. Input/upstream/parameter/output-gradient FNV hashes are comparison
receipts, not cryptographic provenance or math oracles. Native file exports happen
outside event timing: `input.f32`, `upstream.f32`, `parameters.f32` and
`output-grad.f32` (output followed by registered parameter gradients in learner
mode). The supervisor checks expected byte counts, finiteness, native FNV/file
agreement and independently computed SHA256. It requires equal bytes across
processes/modes/workspaces within each encoder/phase, equal weights across
phases and equal input/upstream arrays across families. Every duration, native
log and process-clock receipt is retained; all prior case files are rechecked
before final success. These are byte/receipt checks, not independent model math.

`PROFILER --hash FILE` uses scalar host FNV-1a and returns before INI parsing,
encoder construction or CUDA. It is artifact inspection, not CPU neural testing.
Optional `--host-hash` adds empty/known-answer/multi-block-tail fixtures for all
three native targets to the existing 36 metadata/seven rejection audit. Data
consists of a few hash-test bytes, not a game-frame dataset.

This executable intercepts stream/GEMM API calls and checks status. `default`
preserves production workspace behavior; `reassign` reapplies that handle's
separate existing 32 MiB workspace after each stream assignment. Switching a
handle's stream is rejected. Only this translation unit changes, not production
kernels/helpers. Counters are **host API calls, not GPU kernel counts**; graph
replay doesn't repeat those calls. Encoder-only paths generally don't exercise
the trainer's concurrent main/dW streams. Record the dW counter and retain that
qualification gate when zero.

## Acceptance and interpretation

1. Host supervision/immutable-input/timeout/failure/pair audits are implemented
   and tested. Their real GPU execution and byte-export behavior remain unvalidated.
2. Execute all fixed cases in a scheduled exclusive window, then repeat
   independent processes. Preserve failures/differences; no selective retries,
   relaxed tolerances or partial-success aggregate.
3. Run independent forward/all-parameter-gradient oracles on any proposed
   workspace path. Byte repeatability doesn't prove correct math. Dedicated
   overlapping main/dW streams and graph checks must precede a production fix.
4. Qualify full-policy train/reload/seeded repeatability after backend changes,
   apply improvements fairly to every family and rebuild matched learning curves.
   Kernel breakdown and competitive implicit-convolution controls remain separate.

GPU outputs are marked `runtime_qualified:false` by construction. This tool
cannot certify baseline efficiency, training determinism or cross-game
dominance. Existing Connect4 curves and task pilots remain separate evidence.

[Retained builds, metadata configurations and checks](results/encoder-profiler-20261006/README.md).
[Supervised preparation receipts](results/encoder-profiler-supervisor-20261006/README.md).
