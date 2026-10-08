# CUDA validation before each architecture trial

October 7, 2026. The staged discovery launcher now validates the actual proposed
encoder before training it. This is implemented tooling, not a future protocol.
Use [NEXT_5090_LEARNING_FEEDBACK.md](../NEXT_5090_LEARNING_FEEDBACK.md) for the
complete preparation/run instructions. Do not restart historical allocations.

`cnn_graph_acceptance.py` supervises two fresh native workers. The production
callback harness exports intermediate activations and device parameters through
test-only functions in `test_nature.cu`. `native_flex_reference.cu` independently
implements direct float64 CUDA convolution, dense layers, pooling, residuals,
global pooling, forward loss and derivatives. It uses no production GEMM,
im2col or neural helpers. Python generates fixtures, copies artifacts and checks
arrays; it does not calculate a CNN or train a policy on CPU.

The gate covers Nature and the proposed legacy encoder4 graph at H128. Its
scalar registration supports all 18 discovery coordinates: depth1–3, stage
channels/kernels/strides/pooling/residuals, projection and global pooling.
Each worker checks signed, smooth-positive and zero fixtures at B1 and B64,
plus signed B2048. Every case runs twice eager and twice under a CUDA graph:
14 process/case instances and 56 native calls per proposed encoder. Full traces,
all parameter gradients and device parameters are saved. Actor/trainer output,
eager/graph repeats and fresh-process arrays must match exactly.

There are two distinct numerical comparisons:

1. Each layer's forward calculation is independently recomputed on its actual
   float32 input. Backward is independently recomputed on actual float32
   activations and ReLU branches; max-pool winners are chosen independently.
2. A smooth-positive fixture compares the entire independent float64 graph,
   including all gradients and selected CUDA forward-loss finite differences
   for each convolution/dense layer.

These checks address the earlier near-zero ReLU branch mismatch without
declaring the failed unconditional H256/B2048 test passed. They do not establish
unconditional float32/float64 branch agreement on arbitrary inputs. Nature keeps
rtol3e-4/atol3e-5; new variable graphs declare rtol8e-4/atol6e-5 before execution.
Finite differences use epsilon1e-6 and rtol2e-5/atol1e-8. Never adjust tolerances
after a failure. Old initial bundles retain their recorded contract.

Preparation freezes both libraries and build/source receipts. Execution checks
current owned sources, config hashes, hardware, exclusive reservation and idle
GPU status. Every worker has a hard deadline; any discrepancy stops the trial
before training and preserves its packet. Offline `audit` rechecks raw arrays,
fixtures, hashes, exact bytes, selected derivative receipts and process clocks.
It never loads a model or queries the GPU.

```bash
# Artifact-only recheck of an existing packet; use its retained source revision.
.venv/bin/python research/cnn_graph_acceptance.py audit --out PATH_TO_ENCODER_CHECK
```

This gate qualifies the specific encoder/configuration/fixtures actually run.
It does not qualify encoder5, IMPALA/Impoola, H64/H256, larger batches, arbitrary
activations, head/core/optimizer derivatives, concurrent policies or publication
claims. Native registration/checkpoint and exact deterministic evaluator checks
are separate gates. The real launcher runs quality/Nature checks first and then
checks every proposed CNN before its six-game panel.
