# Independent CUDA encoder checks at learner batch 2048

October 6, 2026. This is a fresh opt-in numerical v2 packet following the
[completed smaller CUDA-only smoke](GPU_ENCODER_NUMERICAL_SMOKE.md). Kinvert's
short-local RTX 5060 authorization applies; no slow baseline, encoder 5,
separate 5090, training, dataset, dependency or production kernel changes.

**Actual outcome: failed, not accepted.** The first worker passed 33 cases,
then stopped at quality H256/B2048/nonblank. Its 26 gradient violations are
localized to one first-convolution channel. A separate retained GPU diagnostic
supports a float32/float64 ReLU branch difference as the explanation; it does
not change the original result. Nature and the second worker were not reached.
[Retained evidence](results/gpu-encoder-learner-5060-20261006/README.md).

## What changes

`gpu_encoder_smoke.py prepare --learner-panel` declares quality/Nature × encoder
output widths H64/128/256 × batches 1/3/64/2048 × the same three input states:
**72 cases**, each checked in two fresh processes and four eager/graph modes,
**576 planned native harness calls**. H64 covers quality's single-projection
branch. The learner-batch work uses the same float32 native library and direct
float64 CUDA reference library from the smaller accepted allocation; neither
was rebuilt or changed. It does not enable the isolated dense-copy alias.

The wrapper now passes each declared width into the actual native constructor,
cross-checks reference/native fixture parameter counts and preserves independent
CUDA oracle, exact actor/train/graph/process/parameter-byte comparisons and all
encoder output/parameter-gradient checks. Tolerances remain Nature 3e-4/3e-5 and
quality 8e-4/6e-5. No subset/tolerance override or selective retry is supported.
Each nonblank B1 width/model additionally gets selected CUDA forward-loss finite
differences per registered tensor: 40 probes per worker, 80 planned in total.
The oracle's exact dyadic ReLU/input-gradient/loss self-test stays active.

Default preparation still emits the old v1 eighteen-case allocation. Readers
do not relabel it as v2 or accept v2 cases under v1. The completed original
packet/source/archive remains untouched and its existing report still audits
with the new reader. Its pinned old current-tool hashes prevent automatic
relaunch after this wrapper change. Nine host/config/supervision tests pass,
including v1/v2 allocation and false qualification-flag checks; these execute
no neural model or GPU query and don't load either library.

## Completed local allocation and historical commands

Actual local packet: `build/connect4cnn/gpu-oracle-learner-packet-20261006`.
Build inputs: `gpu-oracle-native-20261006.so` and
`gpu-oracle-reference-20261006.so`, both under `build/connect4cnn/`, with their
unchanged `.so.build/` source/compiler/command/hash receipts. Original executed
source is dirty; those receipts identify it, not HEAD alone.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/gpu_encoder_smoke.py prepare --learner-panel \
  --native build/connect4cnn/gpu-oracle-native-20261006.so \
  --reference build/connect4cnn/gpu-oracle-reference-20261006.so \
  --out build/connect4cnn/gpu-oracle-learner-packet-20261006
.venv/bin/python research/gpu_encoder_smoke.py inspect \
  --out build/connect4cnn/gpu-oracle-learner-packet-20261006
# Historical invocation: already failed; do not relaunch this allocation.
.venv/bin/python research/gpu_encoder_smoke.py run \
  --out build/connect4cnn/gpu-oracle-learner-packet-20261006 --timeout 120
# Complete-panel audit cannot succeed on this incomplete failed allocation.
.venv/bin/python research/gpu_encoder_smoke.py audit \
  --out build/connect4cnn/gpu-oracle-learner-packet-20261006
```

The shared reservation/idle/topology checks and supervisor-bound worker nonce/
parent/packet checks remain. Workers have 120-second process-group deadlines;
keep complete arrays/logs on failure. Private workers aren't raw launch
commands. No active polling, competing work or embedding refresh during the
allocation. Foreign machines regenerate local paths after authorized committed
delivery; a pull doesn't deliver this currently uncommitted work.

## Actual failure and diagnostic

The first worker ran for 2.116921885s and returned exit 1. It completed all
quality H64/H128 cases, including B2048, and the nine H256 cases below B2048.
There are 33 successful case records, 34 started case directories, **133 actual
native harness calls** and 16 completed selected CUDA finite-difference probes.
These counts are partial first-worker observations, not two-process acceptance.
No complete success report/seal exists; 576 calls and 80 probes remain planned
full-panel counts, not executed counts.

At `flex_quality-h256-b2048-nonblank`, 26 of 118,880 parameter-gradient entries
exceed the unchanged quality tolerance. All are weights/bias of first-conv
channel 12. Maximum gradient absolute difference is 0.001859357251; the final
forward-output difference is only 6.244484430e-8. Outside the first convolution,
the gradient difference is at most 1.590222249e-6. Small forward error alone
does not establish backward correctness.

The separate `diagnose_encoder_relu.py` allocation used the same libraries,
original observations and first-convolution weights. Two identity readouts
exposed all 99 spatial positions of channel 12 over 2,048 images (202,752
activations) through the existing CUDA APIs. One branch differed: image 1723,
row 5, column 10 gives native 0 and float64 reference +8.847564459e-9. The
channel's bias-gradient delta is -0.001857998466; multiplying it by the observed
7x7 image patch explains the channel's weight-gradient differences to a maximum
residual of 1.358785075e-6. This supports a rounding-related branch explanation.
The probes change downstream weights, cover only that channel, and do not
independently reconstruct the original backward pass; they aren't acceptance.
No tolerance, fixture, seed, native implementation or failed result was changed.

Offline verification of the retained diagnostic (no model/library/GPU execution):

```bash
.venv/bin/python research/verify_encoder_relu_artifacts.py \
  --packet build/connect4cnn/gpu-oracle-learner-packet-20261006 \
  --diagnostic build/connect4cnn/gpu-oracle-learner-relu-diagnostic-20261006
```

This reparses original hashes, the diagnostic seal, identity readout parameters,
exported activation branch differences and observed image patches. Its success
means retained-artifact consistency only; the original panel remains failed.
The paths refer to the retained local allocation, not another machine's clone.

## Next numerical protocol: planned, not implemented

Keep the strict failed comparison and its raw unconditioned float64 gradients.
A separate protocol should check two distinct quantities:

1. Export native float32 layer activations and, where feasible, pre-ReLU linear
   values using test-only instrumentation. Compare every layer's forward values
   with independent direct float64 CUDA. Record all sign/mask disagreements,
   including their distance from zero; final-output agreement is insufficient.
2. Compute independent float64 CUDA backward algebra using the actual exported
   float32 forward activations and branch masks as fixed inputs. Compare every
   parameter gradient with the native backward output at predeclared tolerances.
   The reference must not reuse production GEMM/dW/dX or native gradients.

The second check tests differentiation on the computed float32 branches, not
the derivative of the unrounded float64 network. Keep both comparisons visible
with separate outcomes; a conditioned-backward pass cannot retroactively pass
this original packet. Preserve zero/near-zero conventions, H64's projection
branch, full case allocation, independent reference self-tests/finite differences,
device parameters, eager/graph/repeat/process comparisons and failure retention.
Any instrumentation/reference change needs fresh library/source receipts and
a fresh explicitly scheduled packet. No production change or new GPU run is
authorized by this plan. Full-policy and performance claims remain separate.

## Interpretation boundaries

An actual success would check these two fixed **encoders** at B2048 and alternate
output widths. It would not certify a full learner minibatch: policy head,
MinGRU/core, optimizer, rollout/learner concurrency, full trainer allocation/
peak memory and checkpoint reload remain separate. Reports therefore keep
`learner_batch_2048_qualified=false` and add the narrower
`encoder_batch_2048_numerically_checked` only after complete v2 comparisons.
Encoder output width isn't proof that the associated recurrent core was tested.

Batch above 2048, generalized Flex/compact, encoder 5, IMPALA/Impoola, real
checkpoint/frame corpora, kernel/training speed, sufficient multi-game learning,
censoring calibration, held-out selection and frontier/SOTA dominance remain
unproven. This does not fix stock Flappy IMPALA's too-large learner allocation
or silently shrink only a baseline's minibatch. Don't pool 5060 and 5090 results.
Oracle/selected finite-difference agreement is evidence rather than a universal
correctness proof; full vendor/system/link dependency closure remains incomplete.
