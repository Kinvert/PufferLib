# Isolated dense-input copy optimization

October 6, 2026. Native CUDA candidate compiled; host-only checks pass.
**No GPU numerical/timing execution, training, pretraining or new frontier
point. Both GPU holds remain. Production encoders/learner are unchanged.**

October 6 follow-up: [paired numerical acceptance](DENSE_ALIAS_ACCEPTANCE.md)
is now implemented and prepared, including the exact quality H64 branch.
Two new test libraries compile and 39 host/synthetic checks pass. All numerical
GPU paths remain unexecuted; this is preparation, not equivalence or speed.

## Why this is relevant across games

Connect4CNN, PongCNN, FlappyCNN, BreakoutCNN, SnakeCNN and MazeCNN share
1x36x44 observations and the same fixed encoder constructors. The dense
readout of our quality encoder and adapted Nature currently runs im2col even
when that operation is an identity copy of a retained activation. Removing
that copy is a small implementation candidate for those shared graphs; it
does not itself establish learning or robustness in any game.

[dense_patch_alias.cu](../ocean/connect4cnn/dense_patch_alias.cu) is included
only by the isolated profiler/test harness under `C4_DENSE_PATCH_ALIAS`.
It creates a local patch-shaped view of the previous retained output and
uses it for forward and parameter-gradient GEMMs. Eligibility requires an
internal layer, 1x1 input/output, kernel/stride one and zero padding. The
caller observation is never aliased. Original allocation registrations and
owned pointers remain intact. Pooling, bias/ReLU, residuals, input-gradient
gathers, parameter layout/initialization and shared GEMM wrappers are retained.
The original input-gradient gather is kept even for an identity case, to
avoid silently changing its signed-zero reduction behavior.

The opt-in applies to quality **and Nature**. IMPALA/Impoola are unchanged
controls and explicitly report inactive aliasing. This cannot make a fair
architectural comparison by optimizing only ours.

| Fixed graph | Skipped copy kernels per forward | Logical copied payload skipped per sample |
|---|---:|---:|
| Quality, H64/projection64 | 1 | 6,336 bytes |
| Quality, H128 or H256/projection64 | 2 | 6,592 bytes |
| Nature, H64/128/256 | 1 | 512 bytes |
| IMPALA / Impoola | 0 | 0 bytes |

These are source/shape counts, **not measured memory traffic**. All patch
allocations remain: allocated-buffer saving is zero. GEMM MACs, registered
parameters, recurrent core and mathematical architecture remain unchanged.
Pointer alignment may affect cuBLAS algorithm/reduction choices, so equivalence
and a speedup must be tested rather than assumed.

## Preparation reproduced locally

Use fresh output paths. These commands compile or inspect host descriptors;
they do not query a GPU, allocate CUDA tensors or execute a network.

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=0 \
  bash ocean/connect4cnn/build_encoder_profile.sh build/FRESH_ALIAS_BASELINE
NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=1 \
  bash ocean/connect4cnn/build_encoder_profile.sh build/FRESH_ALIAS_CANDIDATE

NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=0 \
  bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature \
  build/FRESH_ALIAS_BASELINE_TEST.so
NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=1 \
  bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature \
  build/FRESH_ALIAS_CANDIDATE_TEST.so

.venv/bin/python research/audit_dense_patch_alias.py \
  --baseline build/FRESH_ALIAS_BASELINE --candidate build/FRESH_ALIAS_CANDIDATE \
  --configs build/FRESH_PANEL/connect4cnn/r0 --out build/FRESH_ALIAS_AUDIT
```

The configs directory must contain exactly one full resolved INI per fixed
family. Generate it through the [multi-game preparation](MULTI_ENV_ROBUSTNESS.md),
using local builds/paths. The paired audit verifies identical source snapshots,
build/binary identities, parameters/allocator payloads and copy arithmetic at
four batches (1/64/2048/16384) and three core widths (64/128/256): 48 pairs.
Quality H64 explicitly checks the single-projection constructor branch.

Both builds also pass the existing independent C registration calculator:
48 checks, ten invalid-input rejections and nine host artifact-hash fixtures
each. The first arithmetic attempts used a stale local calculator with an
older CSV schema; both failed receipts are retained. Recompiling current
`research/encoder_costs.c` fixed the artifact mismatch; no assertion was waived.

The [profiler supervisor](NATIVE_ENCODER_PROFILER.md) prepares 64 cases per
variant, with explicit candidate metadata retained in both descriptor and
future runtime JSON. Its receipt auditor rejects missing/changed variant
markers. Seven scalar/build-guard tests and 17 host/synthetic supervisor tests
pass. No model library was loaded. Build warnings/logs and all source receipts
are retained in [the evidence archive](results/dense-patch-alias-20261006/README.md).

## Gates before timing or promotion

1. In a separately scheduled exclusive GPU window, validate both freshly
   compiled libraries with the existing Nature, compact and Flex independent
   forward/all-parameter-gradient tests. Keep assertions and tolerances intact.
   Include the exact quality graph and H64 equal-projection branch explicitly;
   the current broad Flex oracle uses H128 and is not that branch's acceptance.
2. Retain identical fixture inputs/weights/upstream, all outputs/gradients and
   compare baseline/candidate directly across eager/graph, rollout/learner,
   blank/zero-boundary cases and independent processes. Existing within-variant
   repeatability alone is insufficient. The [paired numerical supervisor](DENSE_ALIAS_ACCEPTANCE.md)
   covers the H64 branch and fixed H64/128/256 graphs; host preparation passes
   but GPU execution remains pending. Do not infer this gate from metadata,
   synthetic supervision tests or a generic profiling run.
3. Only after independent math/paired acceptance, execute both complete
   supervised profiling panels. Preserve every slower case/failure and check
   identical inputs/parameters and any output/gradient differences. Measure
   event-time differences separately from whole-training time/SPS. Bare native
   `--run` commands remain prohibited; packet preparation does not lift holds.
4. If worthwhile, implement a small production change with baseline parity,
   owned-buffer/lifetime and complete learner stream/graph/train-reload tests.
   Rebuild all affected families equally, version the backend and rerun matched
   complete learning curves. Earlier checkpoints/timings keep their original
   source identity; do not relabel them as optimized results.

Neither this candidate nor its profiler serves as a pretrained backbone or
adds an architecture sweep dimension. The broad multi-game Pareto claim stays
open until matched learning and robust analysis support it.
