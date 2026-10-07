# Paired native encoder numerical acceptance

**Reference-device clarification:** the declared worker below calls NumPy
CPU CNN references after CUDA initialization; these remain CPU references.
The retained paired packet is still unexecuted and isn't launched under the
current short-local GPU allocation. The separate
[CUDA-only reference smoke](GPU_ENCODER_NUMERICAL_SMOKE.md) uses native float64
CUDA kernels for the frozen quality/Nature graphs; it doesn't silently upgrade
this larger baseline/candidate/batch panel or its historical receipts.

October 6, 2026. The [dense-copy candidate](DENSE_PATCH_ALIAS_CANDIDATE.md)
now has a paired numerical supervisor. Two fresh native libraries compile;
39 host/synthetic tests and actual preparation/inspection pass. **Neither
library has been loaded. No GPU query, numerical execution, training or
dataset/pretraining occurred. Both GPU execution holds remain.**

The initial 54-fixture v1 packet is retained unchanged. Current v2 adds the
common learner's B2048 workload, with new source/tool/protocol receipts and no
native source change or model execution. Do not relabel v1 as covering that batch.

This closes an implementation gap before comparing a faster shared backend
across the six native pixel tasks. It does not add a learning/Pareto result.
Production encoders/learner and encoder-5 restrictions remain unchanged.

## What is declared

The fixed panel contains quality and Nature × H64/128/256 × B1/3/64/2048 × three
input states (nonblank, zero observation, zero weights plus observation):
72 fixtures. Quality stays depth one, channels16/kernel7/stride4/SAME,
projection64, no pool/residual/GAP. H64 therefore explicitly covers the
single-projection branch. Nature retains its adapted three conv layers.

Four independent workers use baseline/candidate in repetition zero and
candidate/baseline in repetition one. Each worker checks every fixture in
eager twice and graph twice: **1,152 planned test-harness API calls**, not kernel
counts, learner updates, environment decisions or completed GPU checks.
Native execution retains the harness's actor/train byte comparison.

Workers run the actual CUDA encoder and every registered parameter gradient.
Independent existing float64 references are computed inside the scheduled GPU
worker after CUDA initialization, never as CPU model acceptance. Nature keeps
rtol/atol 3e-4/3e-5 and Flex keeps 8e-4/6e-5, matching existing tests. There is
no tolerance CLI or waived case. Every mode also exports parameters from the
device and must preserve them exactly; refreshing fixture weights between
calls cannot hide a mutation.

Inputs, weights, upstream, independent expected arrays and all actual outputs/
gradients/parameters are retained, including zero/signed-zero bytes. Baseline,
candidate, eager/graph and independent-process bytes must match exactly.
An alignment-dependent cuBLAS difference therefore fails this initial gate;
investigate it and preserve evidence instead of relaxing the assertion.

The independent broad Nature/compact/Flex tests remain required before
promoting a general callback change. This panel qualifies only the declared
fixed graphs/batches; it excludes encoder5, IMPALA/Impoola, learner batches above
2,048, optimizer updates, concurrent learner streams and checkpoint reload.

## GPU-free preparation

From this checkout, use fresh local paths and the normal shared toolkit/NCCL:

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=0 \
  bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature \
  build/FRESH_ALIAS_BASELINE.so
NVCC_ARCH=sm_120 C4_DENSE_PATCH_ALIAS=1 \
  bash ocean/connect4cnn/tests/build_encoder_test.sh test_nature \
  build/FRESH_ALIAS_CANDIDATE.so

.venv/bin/python research/dense_alias_acceptance.py prepare \
  --baseline build/FRESH_ALIAS_BASELINE.so \
  --candidate build/FRESH_ALIAS_CANDIDATE.so --out build/FRESH_ALIAS_MATH
.venv/bin/python research/dense_alias_acceptance.py inspect --out build/FRESH_ALIAS_MATH
```

Explicit-output library builds refuse existing library/receipt paths and save
source/recheck, revision/dirty worktree, compiler/environment/command and binary
receipts alongside the library in `.so.build/`. Legacy one-argument paths keep
their existing behavior. Preparation requires identical source/compiler/arch
snapshots and correct opt-in markers, then copies libraries and owned-source/
tool closure. Vendor/system/link dependencies are not fully copied or hashed;
compiler commands record their paths. It only hashes bytes and records declarations; no library
load, GPU discovery, model/reference computation or array dataset generation.

Actual G240 preparation is `build/connect4cnn/alias-acceptance-packet-20261006-v2`.
That path and ignored libraries are not available on the 5090. Regenerate
there after authorized committed delivery. [Current durable receipts](results/dense-alias-acceptance-20261006-v2/README.md)
exclude binaries and are not an executable packet.

## Scheduled interfaces — not current authorization

Only after Kinvert schedules an exclusive window and lifts the applicable
hold, the supervised interface is:

```text
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/dense_alias_acceptance.py run --out build/FRESH_ALIAS_MATH --timeout 600
.venv/bin/python research/dense_alias_acceptance.py audit --out build/FRESH_ALIAS_MATH
```

Do not call `_worker`, load libraries or run these GPU commands under the hold.
`run` acquires the shared reservation, rejects busy/query-error/topology changes,
records hardware/runtime environment, executes bounded process groups and
preserves terminal failures/partial arrays. Interpreter/NumPy and immutable
tools/libraries must match preparation. No subset, automatic restart or overwrite
is supported. Only a complete panel can produce a final report.

Offline `audit` never launches a binary/GPU/model. It checks the retained
execution seal, exact worker commands/cwd/terminal clocks/completion logs,
serial allocation, seeded fixture bytes, all array sizes/finiteness, oracle
differences, parameter preservation and every paired SHA. Recorded old paths
remain provenance rather than being rewritten during relocation. Pinned NumPy
is required for seeded-array reconstruction; this computes no neural reference.

Host tests use explicit mocked queries/processes and synthetic retained arrays.
Passing those tests is not proof of GPU arithmetic, real exclusivity or learning.
Scheduled numerical success would still leave full-policy concurrency/reload,
fair timing, matched multi-game learning and statistical claim gates open.
