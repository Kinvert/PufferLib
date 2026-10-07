# Native CNN baseline fairness and compute audit

October 6 implementation follow-up: [the isolated dense-input candidate](DENSE_PATCH_ALIAS_CANDIDATE.md)
skips identity patch copies equally for quality and Nature, retaining every
owned allocation, GEMM helper, parameter and gradient gather. Three-family
paired builds/host registrations pass; direct paired GPU math and timing are
pending. Neither production code nor historical measurements changed. Logical
copy counts are not speed, traffic or memory-saving measurements.

October 6, 2026. Source inspection and configuration arithmetic only. No
policy executed and no GPU was profiled. **Backend efficiency is still an
open publication gate.** This audit narrows what the existing measurements
can support; it doesn't establish a general CNN or systems speed record.

## What actually runs

All four fixed pixel encoders use native CUDA kernels and PufferLib's shared
`puf_mm`/`puf_mm_tn`/`puf_mm_nn` wrappers. Those call cuBLAS GEMM. None executes
a Python convolution/training loop. Python campaign/evaluation scripts are
external preparation/audit glue. The trainer/core/optimizer remain native.

October 6 follow-up: [isolated stream/workspace acceptance](SHARED_GEMM_ACCEPTANCE.md)
now compiles with the actual main/dW wrappers and native event fork/join. Ten
host/synthetic tests pass; the six-shape 96-case panel is frozen. Its independent
GPU dyadic/double reference and all runtime paths are **unexecuted**. The
later exclusive launch supervisor is implemented with host/synthetic checks;
actual GPU behavior remains pending. No production helper
changed; this preparation does not establish speed or numerical failure.

The float32 build uses `CUDA_R_32F`, `CUBLAS_COMPUTE_32F`, default math mode
and GEMM default algorithm. The source doesn't select `FAST_TF32`. This is a
source-level configuration fact, not runtime verification of all process/library
settings. NVIDIA distinguishes the TF32 fast compute option from ordinary
FP32 configuration; any future precision experiment must apply the same
policy to every family and revalidate numerics/determinism.
[cuBLAS 12.8 compute modes](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublascomputetype-t).

- [Nature](../ocean/connect4cnn/nature.cu): explicit valid im2col, shared GEMM,
  fused bias/ReLU, retained forward patches for weight gradients, fixed-order
  col2im and bias reductions. Its final dense projection also follows the
  1x1 patch path.
- [Fixed quality/Flex](../ocean/connect4cnn/flex.cu): SAME im2col, the same
  GEMM and bias/ReLU helpers, retained patches and fixed-order gathers. Its
  two dense projections also launch the generic 1x1 patch path. Our encoder
  isn't benefiting from an implicit-convolution vendor backend denied Nature.
- [IMPALA/Impoola](../ocean/connect4cnn/impala.cu): SAME im2col + GEMM + bias,
  preactivation residuals and SAME max pooling. Forward activations are kept,
  but one patch buffer is reused and patches are rebuilt during backward.
  Gradients gather contributors in fixed order without atomic summation.
  These choices trade scratch retention against traffic and launches.

This supports a controlled **current native implementation** comparison.
Shared GEMM helpers don't prove each convolution strategy is optimal.
NVIDIA's implicit GEMM forms convolution patches on the fly without storing
the expanded matrices; algorithm/tile choice can matter for small shapes.
That supplies a concrete competing backend to investigate, not a promised
speedup on our 36x44 workloads.
[NVIDIA convolution guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#convolution-algorithms).

## Concrete shared workspace issue to validate

`src/algo.cu:cublas_init_one` allocates/assigns a 32 MiB user workspace per
handle. `cublasGemmExDense` then calls `cublasSetStream` before every GEMM,
without reassigning that workspace. NVIDIA documents that this call resets
the workspace selection to the library's default pool, even when assigning
the same stream. Therefore initialization alone doesn't establish continued
use of these explicit buffers. This applies to every family, not just IMPALA.
[cuBLAS stream/workspace contract](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublassetstream).

This is a source/API-contract finding, **not an observed numerical failure
or measured slowdown**. Existing successful repeatability receipts remain
valid for their tested conditions; they don't certify all stream combinations.
Before new profiling, inspect actual main/dW/actor concurrency and prepare a
bounded native test of stream-then-workspace assignment with a separate buffer
for each concurrently used handle/stream. Simply sharing one buffer across
overlapping streams isn't a safe repair. Validate graph capture, gradients,
repeatability and whole-policy behavior, then freeze that source for every
family. No production change or GPU execution was made by this audit.

## Architectures and adaptations

| Encoder | Native 1x36x44 graph | Readout to shared core |
|---|---|---|
| Fixed quality | 16 channels, 7x7/s4 SAME, ReLU; 9x11 output | Flatten 1,584 → 64 → H, ReLU after projections |
| Nature | 32×8x8/s4 → 64×4x4/s2 → 64×3x3/s1, valid, ReLU | Final 1x2x64 map → H, ReLU |
| IMPALA | Three 16/32/32 stages: 3x3 SAME conv, 3x3/s2 SAME max pool, two two-conv residual blocks | Final 5x6x32 map → H, ReLU |
| Impoola | Same 15-convolution trunk as native IMPALA | Global average over 5x6 → 32 → H, ReLU |

Original [Nature DQN](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf)
used 84x84x4 input and a 512-unit hidden FC layer before action values. This
native reference keeps its conv filters/strides but adapts grayscale size and
projects to the shared H128 core. It is not original DQN, preprocessing,
optimizer, frame stacking or algorithm. The [IMPALA paper](https://arxiv.org/html/1802.01561v3)
describes a distributed learner and several network setups; our comparison
uses its residual encoder pattern with PufferLib's learner/MinGRU and adapted
head. Calling this an IMPALA algorithm reproduction would be incorrect.
The [Impoola paper](https://arxiv.org/abs/2503.05546) is the pooling reference;
native GAP is a readout adaptation within this fixed shared-head experiment.

All six current native pixel tasks reuse the same image shape and graph.
Task-specific discrete action heads change total parameters; H128/L1
recurrence stays fixed. Frozen architecture transfer uses new random weights
per game/seed. It is distinct from weight pretraining or one policy trained
jointly across games.

## Reproducible static counts

[encoder_costs.c](encoder_costs.c) is an independent, fixed-four-graph
**shape-arithmetic** utility, not policy inference or a general constructor.
It reads no images/weights and calls no CUDA. Source geometry is mirrored
explicitly, so changes require renewed inspection. The audit cross-checks all
52 retained four-model Connect4 parameter rows, verifies their three encoder
sources and shared core source match this checkout, and checks Flappy's
independently retained 160,096-parameter quality checkpoint count. Those checks anchor
parameter arithmetic; they don't independently prove MAC/tensor counts or speed.

H128/L1, seven Connect4 actions, per input sample:

| Encoder | Encoder parameters | Whole-policy parameters | Forward GEMM MACs | Forward GEMM calls | Encoder training tensor payload/sample |
|---|---:|---:|---:|---:|---:|
| Quality | 110,560 | 160,736 | 187,184 | 3 | 46,796 B |
| Nature | 88,352 | 138,528 | 647,168 | 4 | 108,544 B |
| IMPALA | 220,320 | 270,496 | 11,493,120 | 16 | 1,110,848 B |
| Impoola | 101,536 | 151,712 | 11,374,336 | 16 | 1,107,136 B |

MAC counts include dense projections and padded convolution matrix products;
one MAC = two FLOPs. They exclude bias/activation/pooling/residual/recurrent/
head/optimizer/environment work. The forward+backward GEMM column in the CSV
counts actual matrix-product shapes, skipping raw-pixel input gradients as the
current code does. It doesn't multiply a whole training run by an assumed
constant: rollout/replay/forward/update cadence also matters.

Tensor payload is the sum of **encoder activation/scratch registrations** in
float32 (plus int32 pool winners), scaled by batch. It excludes allocator
padding, weights/parameter gradients, MinGRU, optimizer, trainer input/targets,
replay buffers and cuBLAS workspaces. It is not measured peak VRAM. At a
2,048-sample encoder learner batch, these payloads alone are approximately
91/212/2,170/2,162 MiB. They are not proof of memory-bandwidth saturation.

Quality has about 3.46x less forward GEMM work than Nature and 61.4x less
than IMPALA, despite having more parameters than Nature. IMPALA's forward
im2col writes 476,496 elements/sample and rebuilds that many during backward;
Nature writes 12,544 and retains them, quality writes 6,499 and retains them.
Impoola reduces dense-head parameters strongly but changes total encoder
GEMM work by only about 1% here; this helps explain why it isn't much faster
in existing runs. These are structural/source inferences, not profiler results.

```bash
# Configuration arithmetic only, permitted under the GPU hold.
mkdir build/FRESH_ENCODER_COST_ID
cc -std=c11 -O2 -Wall -Wextra -Werror research/encoder_costs.c \
  -o build/FRESH_ENCODER_COST_ID/costs
.venv/bin/python research/audit_encoder_costs.py \
  --binary build/FRESH_ENCODER_COST_ID/costs \
  --out build/FRESH_ENCODER_COST_ID/audit
```

## Gates before a stronger speed claim

1. Freeze all architecture adaptations/source/precision and matched learning
   settings. Add smaller Nature and residual controls with comparable selection
   allowances, rather than comparing only our tuned small model with large
   fixed references. Keep family tuning and common-recipe claims separate.
2. Under an explicitly scheduled exclusive GPU window, profile encoder rollout
   and learner forward/backward separately at actual batches (64 and 2,048 in
   the common recipe), eager and production graph modes. Use native CUDA-event
   timings with warmup, fixed input/weight/gradient receipts and fresh outputs.
   Full-policy process SPS remains a separate metric. No profiler was run here.
3. Measure patch generation/rebuild, GEMM, gather, bias, pooling, residual and
   dense-readout costs. API calls aren't actual kernel counts. Test eliminating
   redundant 1x1 patch copies and implicit-GEMM alternatives fairly across
   families, keeping algorithm/architecture changes separate. Compile within
   this checkout; don't change CUDA/drivers/global packages or use CPU models.
4. Verify every alternative's forward/all-parameter gradients, pooling/padding/
   ties, graph/eager/reload and same-seed training. Deterministic custom gathers
   alone don't guarantee full cuBLAS multi-stream behavior; NVIDIA's versioned
   reproducibility conditions include hardware and stream/workspace setup.
   [cuBLAS reproducibility](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#results-reproducibility).
5. Rebuild matched full learning curves across games/drawings/seeds with the
   selected validated backends. If a faster baseline changes the frontier,
   retain and report that result. Include baseline opportunity, selection cost,
   pretraining where relevant, uncertainty and losing tasks. Only then assess
   a stronger cross-task efficiency claim; broader SOTA needs external tasks
   and current competitive references too.

Both GPU holds and encoder-5's numerical gate remain. No system or production
kernel was changed. [Retained audit](results/backend-audit-20261006/README.md).
[Multi-task design](MULTI_ENV_ROBUSTNESS.md).

Follow-up: [native profiler preparation](NATIVE_ENCODER_PROFILER.md) compiles
actual callback construction/registration and cross-checks 36 parameter/payload
cases plus seven rejected configs. This strengthens the metadata arithmetic
check; it supplies no GPU timing/math or baseline-efficiency certification.
Supervised preparation now passes 16 host/synthetic tests and freezes the full
64-case repeated packet. The new scalar native file-hash mode passes nine host
checks. Scheduled phase/workspace runtime and independent/concurrent-trainer
qualification remain pending; neither GPU hold is lifted.
