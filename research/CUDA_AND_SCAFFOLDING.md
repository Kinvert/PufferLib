# Native CNNs and interchangeable experiment scaffolding

Research/design note, 2026-09-11. Code observations refer to PufferLib commit `bea26c718b61cf2377f8e2ff247fcf687478bcdf` in this checkout. Proposed interfaces below do **not** exist yet. No trainer, CUDA installation, driver, cuDNN installation, or Torch environment was changed to produce this document.

## Recommendation

Use the existing native encoder interface to compare exact Nature and IMPALA implementations, then separate architecture changes from implementation changes. Establish correctness and actual rollout/training timings before investing in custom convolution kernels. A useful result could be a better small CNN, a faster implementation of an existing CNN, a better encoder/core allocation, or a combination. These are distinct claims and need distinct controls.

PyTorch models already execute expensive GPU convolutions in native kernels. Removing Python dispatch can help, particularly with small operations, but does not itself imply a large convolution speedup. Compare against a properly configured library implementation and graph replay before claiming that custom CUDA is superior. The existing PufferLib trainer already captures CUDA graphs, so several obvious host-overhead optimizations are already present.

## What this checkout actually provides

| Seam | Code evidence | Consequence |
| --- | --- | --- |
| Encoder function table | [`src/algo.cu:23`](../src/algo.cu#L23) | Forward, backward, parameter initialization/registration, and separate training/rollout activation registration are already separate operations. |
| Encoder → core → decoder | [`src/algo.cu:911`](../src/algo.cu#L911) | Replace the encoder without replacing PPO or the policy/value heads. |
| Default architecture construction | [`src/algo.cu:984`](../src/algo.cu#L984) | Encoder output and recurrent width are both `hidden_size`; independent feature/core dimensions require an explicit design change. |
| Native recurrent core | [`src/algo.cu:1013`](../src/algo.cu#L1013) | The native default is MinGRU, not an interchangeable LSTM/GRU family. Begin by holding core type fixed. |
| Environment-selected encoders | [`src/ocean.cu:56`](../src/ocean.cu#L56) | Selection is currently by build macros/environment. Generic pixel encoder selection needs an additional explicit configuration seam. |
| Training encoder batch | [`src/algo.cu:1460`](../src/algo.cu#L1460) | The encoder sees flattened batch × sequence positions, before the core restores sequence shape. Benchmark this shape separately from rollout inference. |
| Allocation arena | [`src/pufferl.cu:234`](../src/pufferl.cu#L234) | Tensors register shape/pointer metadata before allocation; register stable buffers before graph capture. |
| Native precision | [`src/pufferl.cu:40`](../src/pufferl.cu#L40) | Default storage is BF16, with an FP32 build option; GEMM compute is FP32. Match rounding/storage behavior when comparing references. |
| Rollout/training graph replay | [`src/pufferl.cu:889`](../src/pufferl.cu#L889), [`src/pufferl.cu:1627`](../src/pufferl.cu#L1627) | Benchmark after capture/warmup; preserve stream and buffer-lifetime contracts. |
| Muon optimizer | [`src/algo.cu:1118`](../src/algo.cu#L1118) | New convolution parameter shapes and registration order must agree with optimizer expectations. |

Local `file.md#L123` source links may not jump to lines in every Markdown viewer; the displayed path and line number identify the location regardless. Lines are pinned to the commit above.

**Do not mistake a fast Ocean environment for a pixel benchmark.** For example, [`ocean/breakout/breakout.h:180`](../ocean/breakout/breakout.h#L180) constructs observations from paddle/ball positions, velocity, scalar state, and brick states. Rendering the game does not make these observations pixels. A pixel observation path and its cost must be specified before using it as CNN evidence. This is also not automatically equivalent to ALE Breakout.

## Reuse the NMMO3 integration pattern, not its task assumptions

[`ocean/nmmo3/nmmo3.cu`](../ocean/nmmo3/nmmo3.cu) contains a working-looking native convolution integration pattern, but static inspection is not a successful test run. Its stem accepts an 11 × 15 map of categorical feature codes, expands receptive-field entries into a 59-channel multihot representation, applies two convolutions, concatenates player embeddings/scalars, and projects to core width. Its geometry, channel counts, bias policy, and preprocessing are hardcoded at the top of the file.

Useful pieces to study:

- `nmmo3_encoder_forward` at line 287: im2col → `puf_mm` → layout conversion, with a ReLU after the first convolution.
- `n3_conv_wgrad` at line 326: weight-gradient GEMM reuses the forward column buffer.
- `nmmo3_encoder_backward` at line 339: projection gradient, convolution weight gradients, intermediate input gradient, and saved-activation ReLU masking.
- `n3_c2_col2im` at line 99: each input-gradient element gathers its contributors in a fixed loop, avoiding a floating-point scatter atomic for this operation.
- Registration routines at lines 397, 407, and 440: weights, gradients, training buffers, and inference buffers have distinct lifetimes.
- Factory at line 469: populate the encoder function table while preserving its input/output dimensions.

This is **im2col plus cuBLAS**, not a general cuDNN convolution wrapper. Its first-stage categorical expansion and embedding code do not apply to RGB images. Dense 64 × 64 or 84 × 84 images also create very different temporary-memory costs from this small map. Copying the implementation and changing a few constants would miss padding, general output geometry, additional layers, residual branches, and pooling backward.

The integer-accumulated embedding gradient is a useful determinism example, not a general recipe for CNN gradients. Its fixed-point scale introduces quantization and requires an overflow bound. The comment that its quantization is below an FP32 ULP is not universally true for every magnitude.

**Existing reference harness needs repair before use.** [`tests/test_nmmo3_cuda.cu:5`](../tests/test_nmmo3_cuda.cu#L5) includes `../pufferlib/src/models.cu`, which is absent here, and line 25 calls the old two-argument `create_custom_encoder`. Current selection has one argument and build macros. [`tests/test_nmmo3_encoder.py:20`](../tests/test_nmmo3_encoder.py#L20) also carries the old include path and links cuDNN, which the current main build does not list at [`build.sh:525`](../build.sh#L525). Treat these tests as design references; do not claim that they pass on this checkout or run their build command blindly.

## Porting Nature and IMPALA

First freeze a written architecture specification: image shape and channel order, pixel scaling, frame stack, every kernel/stride/padding, biases, nonlinearities, pooling tie behavior, flatten order, projection size, and initialization. “Nature CNN” or “IMPALA CNN” alone is insufficient because downstream libraries change details.

For Nature, implement each of the three conv/ReLU stages and the feature projection. For IMPALA, implement the chosen three-stage conv/pool/residual specification and final feature readout. The latter additionally needs saved pooling decisions and residual-gradient accumulation. A global-average-pooling variant is a **different architecture**, even when the backend implementation shares most code.

Use mathematical cross-correlation conventions consistently between the reference and native code; do not accidentally flip convolution filters. Preserve frame stack/channel order through flattening. Padding errors at the image boundary can appear harmless in random average-error tests but change agent behavior.

### Buffer and gradient plan

For each convolution with batch `N`, input channels `C`, kernel `R × S`, and output spatial shape `P × Q`, explicit im2col needs `N * P * Q * C * R * S` elements. Compute bytes for **each proposed rollout and training shape** before allocation. Saving every such buffer across an IMPALA encoder can be expensive. CUTLASS describes how implicit GEMM forms patches during tiled loads instead of materializing this expanded matrix. That is a reason to measure a library/implicit-GEMM backend, not a prediction that it wins every shape. [NVIDIA CUTLASS convolution](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/implicit_gemm_convolution.html)

Suggested ownership contract:

| Buffer | Training use | Rollout use |
| --- | --- | --- |
| Parameters, including optional biases and adapter | Persistent, checkpointed | Persistent; respect actor/train copies |
| Parameter gradients | Registered in parameter-corresponding order | Absent |
| Layer inputs/outputs or recomputation checkpoints | Retain until backward consumes them | Reuse scratch after last reader |
| ReLU masks / outputs | Needed for exact derivative | No saved history needed |
| Pooling argmax | Save a specified tie decision; use it in backward | Usually output only |
| Residual activations and branch gradients | Preserve both contributions until merge | Reuse after branch merge |
| Convolution workspace | Per in-flight execution; plan before capture | Per in-flight execution |
| Adapter inputs and gradients | Required if feature dim differs from core width | Forward scratch only |

The encoder backward interface returns `void`, so pixel-input gradients need not leave the encoder. Gradients with respect to intermediate activations are still required, and an optional input-gradient debug interface is useful for parity testing. Preserve parameter/gradient alignment, including padding: the allocator aligns registrations, while flat parameter and gradient views are later constructed at [`src/pufferl.cu:2006`](../src/pufferl.cu#L2006). Odd channel counts and one-dimensional biases deserve special attention. The Muon implementation interprets a multidimensional parameter as `shape[0]` rows and all remaining elements as columns; it advances flat gradient offsets by element count at [`src/algo.cu:1158`](../src/algo.cu#L1158). Thus convolution layout determines optimizer matrix orientation, and padding between registrations needs an explicit audit. Define the matrix view and test one update rather than assuming arbitrary new shapes are safe.

Start with conservative lifetimes and straightforward backward passes. Only reuse/recompute buffers after parity succeeds. Workspaces must not alias across concurrently executing actor/train streams. Allocate resources, select algorithms, warm up, and create execution plans outside capture. CUDA has dedicated graph memory mechanisms, but ordinary synchronous allocation is not a drop-in operation inside capture. [CUDA graph constraints](https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html)

## Backends worth comparing

| Backend | Purpose | Main uncertainty |
| --- | --- | --- |
| PyTorch reference, initially FP32 | Fast architecture editing and autograd oracle | Framework/dispatch overhead must be distinguished from convolution cost |
| Native im2col + existing cuBLAS wrappers | Smallest conceptual extension of this checkout | Expanded patch buffers, layout copies, and many launches |
| Native cuDNN, using compatible existing/local libraries only | Strong optimized convolution baseline | Algorithm/workspace choices, determinism, and compatibility |
| Native CUTLASS implicit GEMM | Compile-time specialization and possible fusion | Engineering effort, hardware/version constraints, backward support |
| Handwritten specialized kernels | Target a measured bottleneck or unusual tiny shape | Correct backward, sustained throughput, maintenance, and portability |

No cuDNN/CUTLASS installation is proposed as an immediate step. Inspect what is already available and pick compatible versions before an implementation task. Do not modify system CUDA, drivers, cuDNN, system libraries, or global environment settings.

Candidate optimizations to evaluate after a correct baseline: fuse pixel cast/scale/layout conversion; retain one layout through the stack; fuse bias/activation or residual epilogues; specialize fixed image shapes; remove redundant copies; specialize small-batch rollout separately from large training batches. Evaluate channel counts in hardware-friendly increments, but also include the nominal literature widths. NVIDIA's performance guide explains why layout, matrix dimensions, and tiling affect realized throughput; fewer FLOPs alone need not be faster. [NVIDIA convolution performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html)

## Determinism: make the contract testable

Set the initial target to repeated-run bitwise reproducibility with the same GPU model, software/build versions, inputs, shape, configuration, and execution mode. Treat cross-GPU/backend FP32/BF16 parity as a numerical-tolerance question. A deterministic encoder does not prove deterministic asynchronous RL: sampling, environment resets, policy lag, reductions, optimizer state, and evaluation must also be controlled.

For custom kernels, use fixed reduction trees or fixed-order gathers for overlapping gradients. For max pooling, specify tie selection and avoid unordered floating-point accumulation in backward. For libraries, select an explicitly deterministic algorithm where supported and record the selection. NVIDIA documents nondeterministic convolution-backward and max-pooling-backward cases and does not promise cuDNN bitwise identity across GPU architectures. [cuDNN reproducibility](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/misc.html#reproducibility-determinism)

**Static audit item in the existing cuBLAS wrapper:** [`src/algo.cu:88`](../src/algo.cu#L88) creates a private workspace and sets it, but [`src/algo.cu:111`](../src/algo.cu#L111) calls `cublasSetStream` before GEMM. NVIDIA documents that this call resets the workspace to the default pool. Review whether workspace binding should follow stream selection for each handle before relying on private workspaces in a determinism argument. This is a concrete inspection finding, **not a demonstrated failure** and not a change made here. cuBLAS reproducibility also has software/hardware and concurrent-stream conditions. [cuBLAS stream/workspace API and reproducibility](https://docs.nvidia.com/cuda/cublas/index.html)

Test repeated forward output, every parameter gradient, one optimizer update, graph versus eager execution, and a short fixed-seed rollout/training trace. A repeated scalar return is too weak: different parameter states can happen to achieve the same score. Record checksums and first divergence locations. Numerical parity with another backend needs documented tolerances and separate analysis of values near ReLU zero and pooling ties.

## Proposed interchangeable scaffolding

Prefer one immutable variant specification and a small encoder factory over editing the same CNN file between runs. Keep named candidates under `experiments/cnn/variants/` and archive resolved configurations with results. Compile-time variants can use separate binaries named by variant/build hash; architecture changes must never silently reuse an incompatible binary or checkpoint.

Example **proposed** configuration vocabulary:

```ini
[encoder]
family = nature
backend = native_im2col
channels = 32,64,64
feature_dim = 256
readout = flatten_linear
bias = true

[observation]
height = 84
width = 84
channels = 4
layout = nchw
pixel_scale = 0.00392156862745098

[core]
type = mingru
hidden_size = 512
num_layers = 1

[adapter]
type = linear
```

This is a specification sketch, not a runnable existing INI. Define channels as actual integers after width multipliers are rounded, and record both requested and resolved values. Record pooling/padding and precision details in the complete version.

The conceptual contract is:

```text
pixels [N,C,H,W]
  -> encoder(architecture, channels, readout) -> features [N,F]
  -> adapter(F,H), if needed               -> core input [N,H]
  -> MinGRU(hidden=H, layers=L)            -> hidden [N,H]
  -> policy/value heads
```

Here `channels` controls convolution width, `F` controls feature/readout dimension, and `H` controls recurrent width. For a bias-free linear adapter, forward cost is `2*N*F*H` FLOPs under the convention that multiply-add counts as two. Log adapter cost separately and state which side of an encoder/core budget it belongs to; otherwise a “small encoder” can hide a large interface projection.

For the first implementation, keep the public native encoder output at `H` and implement the `F -> H` adapter inside the custom encoder. This respects the current core assumption while exposing internal feature dimension for experiments. Give the adapter its own parameter/gradient names and profiler range. A later general rectangular core input interface is possible but changes more code and the scientific comparison. If `F=H`, make identity versus learned projection an explicit variant, because removing a learned layer changes both compute and model capacity.

Keep the same specification for a PyTorch constructor and a native constructor. Map named tensors explicitly, including kernel ordering, bias arrays, and projection layout. Separate backend identity from architecture identity in result records so an exact Nature port is not confused with a modified Nature design.

## Fast comparison loop and acceptance gates

1. **Specification/shape validation:** resolve all dimensions; print parameter counts, forward FLOPs, adapter FLOPs, estimated activations, and workspace. Reject infeasible shapes before training.
2. **Reference parity:** compare layers, final output, weight gradients, optional pixel-input gradients, and one optimizer step. Include small/odd batches, image boundaries, pooling ties, zero inputs, and real stored observations. Match loss reduction and upstream gradients exactly.
3. **Microbenchmark:** warm up; time with CUDA events; distinguish rollout forward from training forward/backward/update, graph replay from eager, and preprocessing from network work. Coordinate timing with all relevant streams so asynchronous side work is included. Report latency distributions and workspace, not only peak examples/second.
4. **End-to-end smoke:** fixed small training run checks finite loss/gradients, resets, checkpoint reload, and reproducibility. No claim about learning quality from this gate.
5. **Learning comparison:** matched seeds, environment protocol, tuning budget, precision, and compute accounting. Keep architecture search separate from backend search initially. Judge useful speed by time/compute to a fixed return as well as throughput and final return.
6. **Specialize measured winners:** port/fuse bottleneck operations only for promising architectures. Retest parity after each optimization, then confirm speed under the full training workload.

Store code commit plus dirty-patch hash, variant specification, build command, binary checksum, hardware/software inventory, dataset/environment protocol, seeds, backend algorithms, timing scope, and raw measurements for each run. Separate pilot runs from final held-out comparisons. Serialize performance runs on the shared GPU unless interference is part of the experiment; independent runs can still be scheduled in a large methodical queue.

Torch, if needed later, belongs only in `/home/claude/cnn/.venv`, created with `uv venv` and Python 3.12 per `AGENTS.md`. First inspect existing local packages and installed CUDA compatibility. CPU Torch is enough for small numerical reference cases; native GPU versus CPU-reference comparison can use host transfers outside timing. A GPU Torch wheel is a separate constrained dependency decision, not permission to upgrade system CUDA. No Torch package was installed for this note.

## First bounded implementation target

A single exact Nature encoder with independent channel/feature/core dimensions, native cuBLAS-based forward/backward, a local reference, and a replayed-observation benchmark is a reviewable first deliverable. Pair it with a precise pixel-environment contract. Then add exact IMPALA and one readout variant using the same interfaces. This makes the larger architecture search possible without first building a general-purpose neural-network framework or committing to handwritten Tensor Core convolutions.
