# Shared GEMM stream/workspace acceptance preparation

October 6, 2026. **Both GPU holds remain.** One isolated native test compiles;
ten host/synthetic tests pass and a 96-case panel is frozen. None of its GPU
arithmetic, graph capture or concurrency paths has executed. No production
helper, policy, learner, encoder, global CUDA configuration or system library
was changed. [Retained evidence](results/workspace-acceptance-20261006/README.md).

This addresses the [baseline backend fairness audit](BASELINE_BACKEND_AUDIT.md).
All fixed CNNs share `src/algo.cu` GEMM helpers; PufferLib's linear backward
queues weight gradients on `g_dw_stream` while computing input gradients on
the main stream, then joins before consuming weights. A serial encoder-only
profile doesn't qualify this path.

The [cuBLAS 12.8 stream/workspace contract](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublassetstream)
states that `cublasSetStream` resets the user workspace. The
[workspace contract](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublassetworkspace)
requires at least 256-byte alignment. These API facts motivate testing separate
workspaces assigned **after** stream selection. They do not show that current
training is slower or numerically wrong. The 5090 has a different toolkit;
record its actual runtime/cuBLAS versions rather than assuming a 12.8 binary
or cross-version byte identity.

## Native test scope

`ocean/connect4cnn/tests/workspace_acceptance.cu` includes the actual production
helpers, instruments only this translation unit's cuBLAS/event calls and tests:

- `puf_mm`: forward `Y = X W^T` on the main handle.
- `puf_mm_tn_async_after`: `dW = G^T X` on the side handle, with the actual
  native record/wait fork.
- `puf_mm_nn`: `dX = G W` on the main handle, with `puf_dw_join` before any
  result is consumed. Two independent input/output lanes stress reused events
  and multiple queued disjoint weight-gradient buffers.

`serial` joins each side-stream weight gradient before submitting its input
gradient; `overlap` joins after both lanes. The latter permits concurrency; it
does **not** measure or certify actual simultaneous kernel residency. Main and
dW handles remain bound to distinct nonblocking streams. In `reassign` mode,
each uses its own existing aligned 32 MiB workspace after each stream selection;
`default` retains production selection behavior. No shared concurrent user
buffer is introduced.

CUDA initialization occurs only after valid scheduled `--run` arguments. The
`--describe` branch returns before CUDA/context/handle allocation, initialization,
encoder construction or arithmetic. Invalid fixture/mode/output arguments also
reject before CUDA. A compiled binary's mere availability isn't GPU acceptance.

| Fixture | M | K | N | Intended coverage |
|---|---:|---:|---:|---|
| Odd | 3 | 7 | 5 | Small, unaligned dimensions and tails. |
| Projection | 64 | 128 | 128 | Shared rollout projection geometry. |
| Quality convolution | 6,336 | 49 | 16 | 64 × 9 × 11 patches, grayscale 7×7 filter. |
| Nature convolution 2 | 768 | 512 | 64 | 64 × 3 × 4 patches, 32-channel 4×4 filter. |
| IMPALA residual | 25,344 | 144 | 16 | 64 × 18 × 22 patches, 16-channel 3×3 filter. |
| Learner core | 2,048 | 128 | 128 | Shared learner linear shape. |

These are GEMM geometries, not six CNNs or six games. They do not execute
im2col/col2im, activations, recurrent gates, the optimizer or complete policies.

## Arithmetic reference and repeatability

The native GPU creates bounded dyadic input values with an index/salt integer
mix: multiples of 1/8, magnitude at most 1/2. It computes independent scalar
double-precision row-major references for forward, dW and dX **on the GPU**,
without cuBLAS or the production GEMM indexing. Every product numerator is at
most 16 and every partial sum stays below the float32 exact-integer limit for
all six shapes. This permits exact numerical comparisons, without tolerances
chosen after seeing output. Positive and negative zero compare numerically
equal to the reference; raw repeat/mode comparison retains signed-zero bytes.

After one eager initialization/warmup, all outputs are poisoned before each
submission. Every warmup/replay element must be finite and exactly match its
reference. Three eager submissions or graph replays must also reproduce the
warmup output bytes. All input/weight/upstream bytes must remain unchanged.
Graph capture includes both lanes and the actual native event fork/join.

The binary checks CUDA/cuBLAS status, math/pointer modes, workspace alignment/
separation, both handle call counts and exports `input.f32`, `reference.f32`,
`actual.f32` plus `native.json`. File order is lane 0 then lane 1; each input
lane is X/W/G, each output lane is Y/dW/dX. Float arrays are native IEEE float32
on the current x86 hosts. Host API counts include warmup plus eager submissions
or capture, not graph replay kernel counts. These are **acceptance checks, not
training SPS or timing benchmarks**.

This deliberately simple exact arithmetic doesn't certify general non-dyadic
FP32 accuracy or detect all reduced-precision arithmetic (dyadics also fit
TF32). Whole-encoder forward/all-parameter-gradient oracles remain required.
Actor threads, handle changes between streams, whole training graph interactions
and actual hardware overlap remain outside this scoped test.

## Permitted commands now

Reuse existing native dependencies and Python 3.12 `.venv`; install nothing.

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_workspace_acceptance.sh \
  build/workspace-acceptance/FRESH_NATIVE

source ocean/connect4cnn/runtime_env.sh
build/workspace-acceptance/FRESH_NATIVE/workspace_acceptance --describe

.venv/bin/python research/prepare_workspace_acceptance.py prepare \
  --binary build/workspace-acceptance/FRESH_NATIVE/workspace_acceptance \
  --out build/workspace-acceptance/FRESH_PANEL

.venv/bin/python research/prepare_workspace_acceptance.py inspect \
  --packet build/workspace-acceptance/FRESH_PANEL/packet.json
```

Preparation freezes the binary, compiler/commands/dirty-tree identity, all
source-manifest files and tooling. It rejects mismatched binary/source build
receipts and input changes. Read-only hashes of Raylib, selected CUDA headers/
libraries and the existing NCCL headers/library are checked during build and
preparation; they are not copied or installed, and future runtime identity
still requires validation. Panel order is six shapes × default/reassign ×
serial/overlap × eager/graph × two independent-process repetitions = 96.
There is no model/CPU oracle fallback or dataset generation.

For host/synthetic tests, choose the actual freshly built binary explicitly:

```bash
source ocean/connect4cnn/runtime_env.sh
WORKSPACE_TEST_BINARY=build/workspace-acceptance/FRESH_NATIVE/workspace_acceptance \
  .venv/bin/python research/tests/test_workspace_preparation.py
```

Tests exercise real native metadata/early rejection, packet integrity and
synthetic array/receipt audits. Synthetic zeros are intentionally not native
GPU outputs; they cannot establish that the GPU reference ran. Sources and
console evidence are retained; disposable test directories are not policies.

## Supervised preparation and pending runtime gate

The later October 6 `research/workspace_supervisor.py` implements the exclusive
launch path; **do not execute the low-level native command directly**. The
original preparation/evidence archive retains its earlier missing-supervisor
flag. New supervised plans wrap the unchanged native packet rather than editing
that history. GPU execution remains unqualified and **both holds remain**.
Thirteen host/control-flow tests pass, including mocked full-panel comparisons,
query/reservation failures, immutable-file changes, process receipt/deadline
errors and relocated offline auditing. Array validation remains covered by the
earlier ten preparation/export tests; full-panel control-flow tests explicitly
mock array auditing rather than allocating large fake GPU results. One plain
host subprocess verifies process-group timeout cleanup. No neural model or
GPU query runs in these tests. The initial JSON tuple/list audit regression is
preserved in the [later supervisor archive](results/workspace-supervisor-20261006/README.md).

These commands are GPU-free:

```bash
.venv/bin/python research/workspace_supervisor.py prepare \
  --packet build/workspace-acceptance/FRESH_PANEL/packet.json \
  --out build/workspace-acceptance/FRESH_SUPERVISED

.venv/bin/python research/workspace_supervisor.py inspect \
  --plan build/workspace-acceptance/FRESH_SUPERVISED/plan.json
```

Preparation freezes the complete existing native packet (including its binary)
and current supervisor/preparer/process tooling. It checks retained dependency
hashes without installing or changing libraries, rejects unexpected source
packet additions and never queries a GPU or launches arithmetic. Subsequent
tool changes require fresh supervised preparation; old source/config/compiler
and failed outputs remain intact.

**Only after Kinvert explicitly schedules an exclusive GPU window:**

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/workspace_supervisor.py run \
  --plan build/workspace-acceptance/FRESH_SUPERVISED/plan.json \
  --out build/workspace-acceptance/FRESH_EXECUTION --timeout 120
```

The timeout is per native process, not the total panel. Full execution holds
the shared `hardware-benchmark.lock` throughout; it rejects reservation
collisions, GPU query errors/busy/multiple-device responses and changed hardware
identity. It records process-local runtime flags and hardware output. Queries
are sampled between cases; software ignoring the cooperative lock can still
briefly contend, so external exclusive scheduling remains necessary.

Every case uses the existing process-group supervisor, zero native exit,
positive PID/monotonic process clock, exact command/deadline, native quotas,
counter/byte/array checks and frozen-input/tool verification. Native exports
use a fresh directory created by the binary. Timeouts/nonzero exits preserve
logs, clock metadata and any partial exports. Earlier completed file hashes
are retained in memory and rechecked before success. No failed-case restart,
subset launch, overwriting execution directory or partial-success aggregate.

The whole 96-case audit requires one stable runtime/driver/cuBLAS identity,
all six groups of 16 cases, and identical input/reference/actual SHA256 within
each fixture across schedules/modes/workspaces/process repetitions. It exports
`REPORT.json` only after every check passes. All general math/full-model/
frontier certification flags remain false. Process seconds are diagnostic
launch-to-exit measurements, not training SPS, timed encoder events or speedup.

This offline command reparses native exports/process receipts, rechecks all
retained file hashes and reconstructs the paired comparisons. Unchanged
archives audit after relocation without rewriting historical commands:

```bash
.venv/bin/python research/workspace_supervisor.py audit \
  --plan build/workspace-acceptance/FRESH_SUPERVISED/plan.json \
  --execution build/workspace-acceptance/FRESH_EXECUTION
```

Offline inspection/audit requires the complete copied native packet, including
its binary, but never executes it or queries the GPU. Small Git archives omit
that binary and explicitly retain only evidence receipts; regenerate complete
local preparations on the 5090 from committed source and that clone's builds.
Artifact consistency cannot authenticate that a GPU actually executed; explicit
synthetic control-flow fixtures can pass the audit and must retain their labels.

`prepare_workspace_acceptance.py audit-case` validates already-exported arrays
and native quotas/counters without doing a CPU matrix product. It produces
`export-consistency-passed` with all math/frontier certification flags **false**;
it cannot authenticate GPU execution or replace full-panel supervision.

Only after scheduled runtime acceptance, independent whole-encoder math and
whole-policy/reload regressions should a workspace helper change be considered.
Any later change must apply equally to ours/Nature/IMPALA/Impoola and use fresh
matched learning curves. Preserve historical measurements with their original
backend. Successful backend acceptance would establish neither a speedup nor
cross-game frontier superiority.
