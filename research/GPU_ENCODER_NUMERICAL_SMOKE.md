# Frozen quality/Nature encoders: independent CUDA numerical smoke

The separate [learner-batch and width follow-up](GPU_ENCODER_LEARNER_BATCH.md)
adds an explicit v2 allocation with B2048/H64/256. This original v1 packet and
its narrower evidence remain unchanged; it does not gain those cases.

October 6, 2026. Kinvert authorizes short local RTX 5060 smokes and prefers GPU
neural testing. This check adds a test-only CUDA reference, not a production
encoder change, dataset, architecture sweep, Torch installation or CUDA stack
modification. The RTX 5090 and encoder-5 restrictions remain.

## Why a new reference

The existing `dense_alias_acceptance.py` numerical worker runs native CUDA but
calls `test_nature.reference` / `test_flex.reference`, which are NumPy CPU
forward/backward implementations. Starting a CUDA context before calling them
does not move those references to the GPU. That larger paired packet remains
unexecuted; its previously retained preparation isn't GPU numerical evidence.
It is not this new smoke, and isn't launched under this allocation.

`research/native_encoder_reference.cu` implements separate float64 NHWC
convolution/ReLU/flatten/linear forward and backward kernels using direct
ordered sums. It includes no production encoder/GEMM/im2col helper and uses no
cuBLAS. Each gradient/input element has one owner: no floating atomics or
unordered reduction. The oracle builds with `-fmad=false`. A CUDA scalar loss
kernel supports double-precision parameter perturbation checks. Python/NumPy
only generate fixtures, convert/copy arrays, inspect exported values and compare
scalars/bytes; they compute no CNN reference.

The isolated native harness calls the unchanged production encoder callbacks,
including actor/train comparison, eager/captured graph and all registered
parameter gradients. No dense-patch alias optimization is enabled. The harness
uses `-O1`, whereas normal trainers have their own build receipts; passing this
test is not complete certification of every compiled trainer binary.

## Fixed allocation

- Frozen quality: C16/K7/S4/SAME, flatten, projection 64, output H128. No
  residual, pooling or GAP. The H64 single-projection branch is not covered.
- Adapted Nature: C32/K8/S4, C64/K4/S2, C64/K3/S1, valid padding, flatten,
  H128 linear output; ReLU after each layer.
- Batches 1, 3 and 64; nonblank inputs, zero observations, and zero weights plus
  zero observations. Eighteen declared cases, generated from the existing fixed
  fixture seed. No game/learning/selection scores are used to choose fixtures.
- Two independent native worker processes; each case executes eager twice and
  graph twice. **144 planned harness calls**, not kernel counts or updates.
- Every output and encoder parameter gradient must match the independent CUDA
  float64 oracle under unchanged Nature rtol/atol 3e-4/3e-5 and quality
  8e-4/6e-5. There is no tolerance option or waived case.
- Actual device parameters must remain exactly unchanged. Native eager/graph/
  repetition bytes, CUDA oracle arrays and all independent-process fixtures/
  outputs/gradients/receipts must match exactly.
- The reference checks an exact dyadic linear/ReLU/dX/loss example on CUDA.
  For both nonblank B1 models it also checks one maximum-magnitude gradient
  entry per registered tensor against two CUDA float64 forward-loss
  perturbations, epsilon 1e-6, rtol/atol 2e-5/1e-8. That is 14 probes per worker,
  not finite differences for every parameter or every topology.

## Reproduction interfaces

Use existing local Python 3.12 `.venv`, toolkit/NCCL and fresh destinations.
No compiler GPU discovery: explicit architecture and source/build receipts.

```bash
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS='--threads 1' C4_DENSE_PATCH_ALIAS=0 \
  bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature FRESH_NATIVE.so
NVCC_ARCH=sm_120 bash research/build_native_encoder_reference.sh FRESH_REFERENCE.so
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/gpu_encoder_smoke.py prepare \
  --native FRESH_NATIVE.so --reference FRESH_REFERENCE.so --out FRESH_PACKET
.venv/bin/python research/gpu_encoder_smoke.py inspect --out FRESH_PACKET
```

Preparation/inspection copy/hash libraries/source/build/tool receipts but never
load a library, discover a GPU, initialize a model or compute a reference.
The scheduled `run` takes the shared GPU reservation, rejects busy/query-error/
topology changes, validates current inputs and invokes bounded worker process
groups. Private workers require their actual supervisor parent/nonce/packet
binding before loading either library. Two 120-second worker ceilings bound
native/model work; preparation/audit overhead is separate.

```bash
# Scheduled short local GPU work only; this document isn't permission to rerun.
.venv/bin/python research/gpu_encoder_smoke.py run --out FRESH_PACKET --timeout 120
# Offline, no library/model/GPU execution:
.venv/bin/python research/gpu_encoder_smoke.py audit --out FRESH_PACKET
```

The offline audit reparses seeded inputs, registered array sizes, finite oracle/
native artifacts, parameter preservation, fixed comparisons, parameter
perturbation inputs and CUDA-exported losses, process commands/cwd/clocks,
complete case logs, full execution inventory and independent-process hashes.
It computes no forward/backward reference. Synthetic host checks aren't proof
of GPU math; retained worker/process/hardware evidence stays separate.

Local builds: `build/connect4cnn/gpu-oracle-native-20261006.so` and
`gpu-oracle-reference-20261006.so`; actual packet:
`build/connect4cnn/gpu-oracle-packet-20261006`. Regenerate local paths after
authorized committed delivery to the 5090; don't run G240 archived paths.

## Boundaries

This is encoder-only, fixed H128/batches/topologies/synthetic inputs. It excludes
the learner's B2048, H64/256, generalized Flex/compact/encoder5, IMPALA/Impoola,
policy head/MinGRU/optimizer math, concurrent streams, full-policy reload,
long learning/cap calibration, timing efficiency and frontier/SOTA superiority.
Input derivatives are checked in the oracle's tiny self-test, not compared
across the complete production encoder. Actual game frames/real checkpoints
and all architecture builder choices need separate coverage.

Oracle agreement and selected CUDA finite differences are evidence, not a
formal proof that either implementation is correct over all inputs. Preserve
negative cases and all raw arrays, not merely a PASS report. Full vendor/system/
link dependency closure remains incomplete. No production code or push here.

## Actual RTX 5060 execution

The full allocation passed: 18 cases, two fresh native processes, 144 actual
harness API calls, 28 selected CUDA finite-difference probes and two exact
dyadic CUDA self-tests. Every registered encoder parameter gradient/output
matches the float64 CUDA oracle at the original tolerances; native graph/eager/
repeat bytes, device parameter preservation and independent-process arrays
are exact. The existing actor/train comparison inside the native harness also
passed. Offline raw-array/process/perturbation/receipt audit passes.

Worker process times: 0.865085968s and 0.814963816s. First worker launch through
second completion: **1.779922976 seconds**. These include worker initialization/
fixture/reference/testing/artifact writes and are not encoder kernel timings,
training SPS or proof of a speed advantage. Builds and preparation/review are
outside that interval. Hardware receipt: RTX 5060 on G240, UUID
`GPU-56b0df25-b4c9-46eb-e498-bc5c71904097`, driver 591.86, CUDA compiler 12.8.93.

| Encoder | Output maximum absolute difference | Gradient maximum absolute difference | Largest tolerance ratio (gradient) |
|---|---:|---:|---:|
| Quality | 5.34e-8 | 1.57e-7 | 0.001121 |
| Nature | 4.19e-8 | 1.37e-7 | 0.002667 |

Tolerance ratio is `abs(actual-reference)/(atol+rtol*abs(reference))`, inspected
offline across all exported native modes/processes. It must stay at most one;
these values do not change tolerances or certify other fixtures. Seven host
file/configuration/supervision checks pass separately without loading any
library/querying a GPU/computing a CNN. Synthetic host guards aren't numerical
evidence. No failed GPU cases were dropped or retried; no tolerance changed.

[Durable source, raw-input receipts and report](results/gpu-encoder-smoke-5060-20261006/README.md)
identify actual executed bytes despite the dirty worktree. This allocation is
finished and isn't scheduled to repeat. Next numerical coverage should include
the common learner B2048 and alternate projection/core widths using a fresh
declared packet, preserving this narrower completed evidence unchanged.
