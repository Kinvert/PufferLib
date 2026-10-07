# Native policy parameter and checkpoint-size preflight

October 6, 2026. Preparation/artifact checks only. Both GPU holds remain; no
GPU query, policy/weight initialization, neural computation, training or dataset.
Native production encoder/learner/evaluator sources are unchanged.
[Retained receipts](results/policy-metadata-20261006/README.md).

The exact evaluators check finite checkpoint bytes before GPU work, but only
compare byte count with the native policy's parameter count after execution.
This standalone native tool provides a prior **size** check using actual
`build_arch` / `weights_create` and parameter-registration callbacks. It does
not allocate CUDA buffers, initialize parameter values, step a game, load a
policy or perform CPU/GPU tensor computation. Host structs store shapes only.
The opt-in [shared acceptance packet-v2](CHECKPOINT_LAYOUT_GATE.md) now calls
native registration/stat preflight before preparing or executing its GPU modes.
Individual game launchers and the candidate training panel still retain their
existing checks; this does not silently upgrade their coverage.

## Build and use without a GPU

Use existing CUDA/NCCL dependencies, unchanged. An explicit architecture avoids
GPU discovery; the build script rejects `native`, compiler overrides and an
existing output directory. Only the recorded `--threads 1` override is allowed.

```bash
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS='--threads 1' \
  bash research/build_policy_metadata.sh build/policy-metadata/FRESH_NATIVE
source ocean/connect4cnn/runtime_env.sh

# Metadata only: complete executed/merged training INI and actual action count.
build/policy-metadata/FRESH_NATIVE/default --describe FULL.ini 7

# Stat-only comparison with an existing raw float32 checkpoint.
build/policy-metadata/FRESH_NATIVE/default --check-bytes FULL.ini 7 CHECKPOINT.bin
```

`default` supports the frozen quality graph (`encoder=4`) and Nature (`2`).
`impala` and `impoola` require `encoder=0` in their matching INIs. Other graphs,
state policies, continuous/multi-head policies and encoder 5 are excluded.
Hidden sizes are multiples of eight from 8–4096; MinGRU layers 1–16. The current
registration grid checks H64/128/256 × L1/2, **not every allowed size**.

New explicit `--describe-flex` / `--check-flex-bytes` modes also describe the
existing configurable encoder-4 grammar. The legacy commands above keep their
frozen-quality restriction; encoder 5 remains excluded. Native host counts for
the opt-in acceptance path are checked against independent scalar arithmetic,
not neural forward/gradient computation. Old source archives stay unchanged;
their source-matching verifiers require their own retained tool revision.

All six current pixel games use the same shared 1×36×44 encoder/MinGRU/linear
head implementation. This isolated target compiles that shared implementation
with Connect4's encoder macro and accepts the head's discrete action count
explicitly. It does **not** inspect another game's runtime binary or infer its
head from the INI. Independently match the count to native environment metadata
and the captured header before using it:

| Game | Discrete actions |
|---|---:|
| Connect4CNN | 7 |
| PongCNN | 3 |
| FlappyCNN | 2 |
| BreakoutCNN | 3 |
| SnakeCNN | 4 |
| MazeCNN | 5 |

Supply the executed **merged** INI, not just `config/default.ini` when settings
were separately loaded from the game INI. The real Flappy pilot's default file
alone is correctly rejected because it lacks `policy.encoder`; its saved full
metrics INI gives the expected H128/L1, 160,096 parameters and 640,384 bytes.
The old checkpoint is only statted/hashed, never loaded as a policy. This does
not upgrade its historical pooled-v1 evaluation or establish present-source
numerical acceptance. Preserve both the failed incomplete-input receipt and the
successful merged-input receipt.

## What is checked, and what is not

Metadata exports the encoder, decoder and MinGRU counts plus every ordered
parameter tensor's shape, group and aligned byte offset. Counts come from the
actual production registration function, not a Python formula or guessed file
size. The scalar C calculator independently checks the four frozen graphs.
Raw checkpoint size must equal the total float32 payload, and `--check-bytes`
rejects parameter layouts with alignment padding that would contradict the
native contiguous raw-checkpoint assumption. Registered descriptors have no
tensor data pointers; no allocator create/init callback is called.

`--check-bytes` uses `stat` only. It **does not** validate contents/finiteness,
architecture identity, world settings, learner recipe, checkpoint ancestry,
runtime binary identity, parameter values, math, reload or GPU behavior. An
equal-size wrong/corrupt checkpoint can pass; a deliberate NaN-bearing synthetic
artifact in the host tests demonstrates this limit. Keep the existing finite-byte
preflight and immutable config/source/binary/weight bindings. Learning-rate and
world changes may leave shapes unchanged. Do not treat size acceptance as
checkpoint compatibility or numerical certification.

The build captures owned sources, compiler/command/environment/revision/worktree
and binary hashes, refuses overwrite, and verifies copied/current source hashes.
Vendor, system and linked dependency contents are not fully copied/hashed: this
is not complete toolchain certification or proof of another binary's source.

## Executed checks

Three standalone native targets compile with CUDA 12.8 / explicit `sm_120`.
Inherited INI/vec allocation compiler warnings are retained. Seven host tests
pass: 144 six-game/four-model/H64/128/256/L1/2 registration cases match independent
C parameter arithmetic and ordered layout checks; H64 projection branch,
invalid settings/family/action IDs, raw-size rejection, horizon-only metadata
changes, descriptor repeats and build discovery/overwrite guards are checked.
There are 175 retained native host calls (152 successful, 23 rejected), plus
five build-guard subprocess cases. Thirty unique scalar configurations yield
120 retained independent model-count rows. All are configuration arithmetic or
artifact operations, not model tests. No GPU execution or policy score exists.

Reproduce host checks only after compiling fresh local targets:

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror research/encoder_costs.c \
  -o build/policy-metadata/FRESH_COSTS
source ocean/connect4cnn/runtime_env.sh
POLICY_METADATA_DIR=build/policy-metadata/FRESH_NATIVE \
POLICY_METADATA_COSTS=build/policy-metadata/FRESH_COSTS \
POLICY_METADATA_CHECKS=build/policy-metadata/FRESH_CHECKS \
  .venv/bin/python -m unittest research.tests.test_policy_metadata -v
```

This external Python test driver calls only native host metadata/scalar modes;
it is not a learner or architecture search. New graphs/core formats require
their own descriptor/oracle coverage. Scheduled evaluator/full-model acceptance,
matched learner/cap calibration and learning/frontier inference remain gates.
