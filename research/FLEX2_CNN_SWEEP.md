# Expanded native CNN grammar (encoder 5)

September 29, 2026. Native Connect4CNN and PongCNN FP32 builds and the GPU test-library build previously passed. GPU numerical/training validation is pending. No encoder-5 sweep has been launched and no speed or learning improvement is established. Kinvert chose the separate RTX 5090 for GPU validation and training; do not run encoder-5 GPU tests or sweeps on G240's RTX 5060. See [the 5090 handoff](../NEXT_5090_FLEX2_SWEEP.md).

Implementation: `ocean/connect4cnn/flex2.cu`, normal custom-encoder interface in `src/ocean.cu`. Encoder IDs 0–4 retain their parameter order and definitions. Nature descriptors have room for 24 operations; shared Flex patch kernels accept dilation with default 1 for older callers. Those shared edits still require legacy GPU regression checks.

## Controls

The complete grammar example is [sweep_flex2.ini](../ocean/connect4cnn/sweep_flex2.ini). The first 5090 discovery panels are [fast depth one](../ocean/connect4cnn/sweep_flex2_fast.ini) and [bounded stage two](../ocean/connect4cnn/sweep_flex2_stage2.ini). All keys are numeric. Fixed keys stay in `[policy]`; `[sweep.policy.KEY]` controls only keys actually searched. Encoder selection remains fixed at 5.

| Key | Allowed values / meaning |
|---|---|
| `cnn_depth` | 1–4 stages, fixed at 2 in the starter recipe |
| `cnn_channels_N` | 8, 16, 32, 64 |
| `cnn_kernel_N` | Stage 1: 1–8; subsequent stages: 1–5 |
| `cnn_stride_N` | Stage 1: 1, 2, 4, 8; subsequent stages: 1, 2, 4 |
| `cnn_dilation_N` | 1–4, actual dilation, including 3 |
| `cnn_activation_N` | 0 ReLU; 1 SiLU; 2 exact GELU; 3 learned PReLU; 4 learned rational |
| `cnn_residual_N` | 0–2 additional identity-residual 3x3, stride-1 convolutions after the stage convolution |
| `cnn_pool_N` | 0 none; 1 SAME 3x3/stride-2 max; 2 SAME 3x3/stride-2 average |
| `cnn_readout` | 0 flatten; 1 global average; 2 adaptive 2x2 average; 3 adaptive 4x4 average |
| `cnn_projection` | 16, 32, 64, 128 |
| `cnn_projection_activation` | Same activation IDs, applied at each projection operation |

There are 32 fixed architecture controls, with up to 31 swept when depth is fixed at 4. The full example searches 17 active controls at depth 2 and is **not** the first campaign: 12 trials across that space are too sparse, while stride-1/high-channel proposals may require very large patch buffers. The 5090 panels fix 13.312M decisions and H128/L1, then search seven depth-one controls or eight controls with a fixed stage-one stem. They restrict the first stride to at least 4 and channels to at most 32. These are intentional discovery panels, not the limit of the native grammar or a final cross-environment optimum. A stage is one convolution plus its selected residual repetitions and optional pooling. The full grammar's worst case is 19 operations, including readout and projections. Projection width is separate from the recurrent core.

Convolutions use ceil-output SAME padding based on `(kernel - 1)*dilation + 1`; odd extra padding goes to bottom/right. Adaptive bins use floor(start)/ceil(end), including overlaps on odd sizes and potentially repeated bins when the requested grid is larger than the map. Residuals use the stage's dilation/activation, add the original stage-sized input before activation, and preserve shape. There is no batch normalization, depthwise/grouped convolution, arbitrary graph wiring or cross-stage skip yet.

## Fixed and learnable activation choices

SiLU uses a stable sigmoid; GELU uses the error-function definition. PReLU has one learned slope per operation, initially 0.25, not one per channel. The rational choice is `P5(x)/(1 + abs(Q4(x)))`, with a constant numerator term and denominator terms of degree 1–4. Initialize it near identity (`a1=1`, `b2=0.001`, remaining coefficients zero). Each operation owns its coefficients; there is no cross-layer sharing. At `Q=0` use sign 0 for the absolute-value subgradient.

[Adaptive rational activations, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/78df0f831fbe5854349dbdfccde7ee5d-Abstract-Conference.html) supports studying learned activation shape in Atari RL; its joint sharing is different from this implementation. [Constrained rational activations, 2026](https://proceedings.mlr.press/v330/surdej26a.html) reports a stability/expressivity tradeoff and a constrained variant evaluated principally in continuous control. This code is an experimental unconstrained variant, not a reproduction of either complete method. High powers can overflow, and positive denominators do not guarantee stable gradients or good learning. Keep finite-weight checks, clipping and failed-run evidence; profile its expensive arithmetic/reduction separately.

Coefficient tensors reserve 12 float slots to match the existing allocator/optimizer alignment. PReLU uses 1 real coefficient; rational uses 10; unused slots remain zero. The measured allocation count therefore exceeds the number of active coefficients. Parameter and gradient tensors register in identical order. Coefficient gradients and overlapping pixel gradients use fixed-order gather/reduction, without atomic additions. This design does not establish determinism until GPU repeatability checks pass. FP32 only; BF16 deliberately rejects encoder 5 for now.

## Reduce wasted search dimensions and repeated graphs

The prior CNN2 search favored shallow configurations and included identical effective graphs; see the preserved coverage audit. Use separate **fixed-depth** discovery campaigns with equal decision budgets before assuming deeper shapes lost fairly. `--depth N` sets the depth and removes all inactive stage sweep coordinates while keeping fixed policy keys for reload. Do not add a swept depth section unless accepting conditional unused dimensions and auditing their coverage.

Native PROTEIN still selects proposals and learns from native curves. Before launching an encoder-5 worker, the native sweep fingerprints the constructed operations plus training/environment/core settings and seed. Inactive stage keys and irrelevant dilation of 1x1 operations do not create distinct graphs. Budgets and seeds do create distinct experiments. String/list settings participate in the fingerprint. A 64-bit FNV fingerprint is a campaign-local duplicate guard, not a cryptographic identity or a formal collision-free proof. Source/config SHA256 receipts remain separate.

Duplicate suggestions are resampled without training, with rejection counts in `sweep.log`. After 256 consecutive duplicate proposals the process fails explicitly, preserving evidence rather than silently training duplicates. This path requires serial trials (`parallel=1`). It keeps failed graphs in the attempted set rather than retrying them automatically. PROTEIN itself has not been redesigned to condition its GP on graph equivalence; skipped proposal search still costs host/GPU time.

## Preparation and remaining gates

Preparation does not build, train, access the GPU or upload W&B:

```bash
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_fast.ini \
  --max-runs 3 --canary --prepare-only --wandb disabled
bash ocean/connect4cnn/sweep.sh --recipe ocean/connect4cnn/sweep_flex2_stage2.ini \
  --max-runs 3 --canary --prepare-only --wandb disabled
```

Build uses the existing toolkit/dependencies, with no installs:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash build.sh connect4cnn build/connect4cnn/flex2-train --float
NVCC_ARCH=sm_120 bash ocean/connect4cnn/tests/build_encoder_test.sh test_flex2
```

The bounded GPU math command, after an idle-GPU check, is:

```bash
source ocean/connect4cnn/runtime_env.sh
smi=$(puffer_find_nvidia_smi)
puffer_require_idle_gpu "$smi" && \
OPENBLAS_NUM_THREADS=1 .venv/bin/python ocean/connect4cnn/tests/test_flex2.py \
  --library build/connect4cnn/test_flex2.so \
  --report build/connect4cnn/flex2-math.json
```

The new library and Python harness compile, but have not executed on a GPU. The harness uses native GPU forwards for finite differences, samples every parameter tensor and all coefficient slots, checks actor/train and eager/graph agreement, and records nonsmooth coordinates skipped. It is not an exhaustive independent mathematical proof. Further independent forward references, boundary fixtures and training checks are required.

Run the GPU harness and existing Nature/Flex reference regressions **on the 5090** before a bounded training canary there. The canary is at most three short discovery trials per panel; remove `--prepare-only`, set a suitable process timeout, and preserve all receipts/failures. Canary budgets demonstrate plumbing, not learning. Run train/reload repeatability checks before the longer 5090 search. The original 17-dimension/12-trial grammar example is not the launch recipe.

The GPU harness must check all five activations, dilation/padding, adaptive pooling, residual repetitions, eager/graph and actor/train agreement, and parameter/coefficient gradients. It should be supplemented with independent forward references before claims. Same-seed train/reload checkpoints and sanitizer checks remain required. No experimental checkpoint or coefficient layout has yet been accepted by those runtime gates.

For learning discovery after validation: equal 13.312M decisions, shared learner/core and initialization seeds, record actual parameter/VRAM counts and measured exclusive timings. Retain whole curves. Stage count, cheap/readout choices and activations should first be explored with reduced subsets of controls, then expanded using the coverage report. A 31-dimensional GP with a dozen trials is not adequate architecture search.

Current discovery score remains training performance with the native logging/downsampling semantics. No change here fixes the held-out evaluator or converts a final logging window into a fixed-decision scientific estimator. Final claims still require the paper's frozen evaluation/frontier protocol, fresh paired seeds, baseline backend review and untouched tasks. Pretraining is a separate later axis; see [pixel-space dataset plan](PIXEL_PRETRAINING_DATASET_PLAN.md).
