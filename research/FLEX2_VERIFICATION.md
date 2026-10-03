# Encoder-5 verification contract

October 2, 2026. Status: **implemented and compiled; GPU execution pending on
the RTX 5090**. Nothing in this document is a learning, speed or SOTA result.
This is priority 2 in [potential-todos.md](../potential-todos.md).

## What changed

The previous harness checked sampled finite differences of the native CUDA
encoder. A consistently wrong forward/backward pair could pass that test.
The harness now also checks the complete forward output and every parameter
gradient against [an independent float64 oracle](../ocean/connect4cnn/tests/flex2_reference.py).
The oracle derives shapes and parameter registration from fixture INIs, not
native layer descriptors. It uses image slices, NumPy matrix operations,
Horner polynomial evaluation and scatter-based pooling/convolution gradients.
It imports no production encoder kernels and requires no Torch.

Production construction/training remains C/CUDA. This NumPy code is only a
mathematical oracle paired with actual GPU execution. Do not run CPU model
experiments in its place. No production kernels or `src/` files changed here.

## Acceptance coverage

| Check | Coverage |
| --- | --- |
| 14 complete encoders | Depths 1–4; channels 8/16/32/64; kernels 1/2/3/4/5/7/8; stride 1/2/4/8; dilation 1–4; 0–2 residual repeats; flatten/GAP/adaptive 2x2/4x4; mixed stage activations; H32/H128; equal and unequal projection/core widths; batches 1/2 |
| Independent output and backward | Every forward value and every registered weight, bias and learned coefficient; exact registration offsets/counts; per-tensor errors and nonzero-gradient counts |
| 9 direct activation probes | ReLU, SiLU, exact GELU, PReLU with positive/negative/zero slope, rational activation with learned denominator, identically zero denominator polynomial and exact denominator roots |
| 60 direct pooling probes | Max/average/adaptive; 1x1, 2x3, 5x4, 5x7 inputs; negative values, ties and ramps; adaptive output 1x1/2x2/4x4 including outputs larger than inputs; every input gradient |
| Boundary conventions | ReLU derivative at zero is 0; PReLU derivative at zero is 1; rational abs subgradient at Q=0 is 0; max ties select first valid row-major value; average excludes padded pixels |
| Native finite differences | Sample every weight/bias tensor and every activation coefficient slot using perturbed GPU forwards; record both one-sided estimates for skipped nonsmooth coordinates; too many skips fail |
| Repeatability and actor path | One repeated eager execution and two graph executions must match output/gradient bytes; each execution compares training versus rollout outputs |
| Initialization and unused slots | Finite initialized weights; zero biases; exact initial PReLU/rational coefficients; unused coefficient gradients exactly zero |

These are representative fixtures, not exhaustive testing of every legal graph.
The complete encoder interface does not return an observation gradient; direct
pool/activation probes check their input derivatives, and internal convolution
backpropagation is exercised through earlier layers' parameter gradients.

Tolerances are fixed before GPU execution: complete-encoder `rtol=8e-4`,
`atol=3e-5`; direct operators `rtol=2e-5`, `atol=2e-6`. Reports include maximum
error relative to the tolerance and reference magnitude per tensor, so small or
zero gradients are visible. GPU finite differences retain `atol=1.5e-4` plus
`0.025 * max(abs(numerical), abs(analytic))`; they use actual float32 perturbation
sizes. No tolerance is a guarantee for unseen configurations. Investigate a
failure before changing a tolerance or dropping a fixture.

## Run only on the 5090

Follow [the active handoff](../NEXT_5090_FLEX2_SWEEP.md) to update safely and load
the existing runtime. Do not install or alter CUDA, Torch or system libraries.
Check the GPU is idle first. From the checkout, after sourcing its runtime:

```bash
set -euo pipefail
export NVCC_ARCH=sm_120
export OPENBLAS_NUM_THREADS=1
mkdir -p build/connect4cnn
verification_dir=$(mktemp -d build/connect4cnn/flex2-verification.XXXXXXXX)
git rev-parse HEAD > "$verification_dir/commit.txt"
git status --short > "$verification_dir/status.txt"
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version > "$verification_dir/compiler.txt"
nvidia-smi > "$verification_dir/gpu.txt"
bash ocean/connect4cnn/tests/build_encoder_test.sh test_flex2 2>&1 | tee "$verification_dir/build.log"
.venv/bin/python -u ocean/connect4cnn/tests/test_flex2.py \
  --library build/connect4cnn/test_flex2.so \
  --report "$verification_dir/math.json" 2>&1 | tee "$verification_dir/math.log"
```

Execute each command with failure handling: **a nonzero build/test exit stops
the gate**, even if an old library remains. Never run with `python -O` or
`PYTHONOPTIMIZE`; the harness rejects disabled assertions. It refuses to overwrite
an existing report. Expect `status: passed`, 69 operator records and 14 network
records. Keep the report and its fixture directory together. It preserves exact
INI/NPZ network fixtures, source and shared-library SHA256s, working diff, seeds,
Python/NumPy versions, sampled-gradient failures and repeatability checks.
The source hashes describe files at test time; rebuilding immediately before
testing and preserving the build log is necessary to associate the binary with
that source. There is no claim that the test-time hashes alone prove compilation
provenance. Native aborts can leave status `running`; that is **not** a pass.

Before discovery, also run the existing Nature/Flex regressions, then separate
short native training/reload and same-seed checkpoint checks. A full memory/race
sanitizer audit, BF16 support, optimizer/training determinism, checkpoint
interoperability and baseline efficiency are not established by this math suite.
BF16 remains unsupported for encoder 5. Timing these diagnostic probes is not an
encoder throughput benchmark.

## Local work completed

- Native test shared library compiled for `sm_120` with the existing CUDA 12.8
  toolchain. Compiler emitted warnings in existing `src/ini.h`/`src/pufferl.cu`;
  compilation succeeded. No GPU code was executed.
- Python syntax compilation and `git diff --check` passed.
- The strengthened numerical suite has **not** run on either GPU or CPU here.
  Its own runtime behavior must still be accepted on the 5090; preserve and
  diagnose any oracle/harness errors as well as kernel mismatches.
