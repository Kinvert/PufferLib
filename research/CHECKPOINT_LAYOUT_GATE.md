# Native checkpoint layout before evaluator acceptance

The candidate training panel now has a separate opt-in pre-training registration
path through `candidate_panel.py prepare --policy-metadata LOCAL_BUILD`; see
[CANDIDATE_BASELINE_COMPARISON.md](CANDIDATE_BASELINE_COMPARISON.md). It supports
both panel versions and preserves old packets without the option. Individual
game adapters still retain their separate checks. This follow-up doesn't change
the existing-checkpoint acceptance packet's scope or lift either GPU hold.

October 6, 2026. This is a host configuration/artifact gate for the shared
evaluator acceptance path, not a neural test or permission to launch a GPU.
Construction, learning and policy inference remain native C/CUDA.

`eval_acceptance.py prepare --policy-metadata LOCAL_BUILD` opts into packet-v2.
It copies the existing three-family native metadata build and owned source/build
receipts, checks those sources against this checkout, and freezes each game's
discrete-action header. Every case invokes only the native registration/stat
mode with its full training INI and original checkpoint. Wrong checkpoint size
fails before any policy launcher or GPU query; the failure and host process log
remain in the fresh preparation directory. All cases must pass before a packet
exists. Equal-size wrong weights remain possible: this gate supplements the
existing finite-byte, family/config/source/binary/checkpoint bindings.

The default target's new explicit `--describe-flex`/`--check-flex-bytes` modes
permit the existing encoder-4 grammar, including depth, kernels/channels/strides,
residuals, parameterless pooling/GAP and projection. The unchanged legacy
`--describe`/`--check-bytes` modes still restrict encoder 4 to the frozen quality
graph. Nature and the matching compiled IMPALA/Impoola targets remain supported.
Encoder 5, state, tiny/experimental/compact and other observation shapes are
excluded from the opt-in gate; do not fall back silently to packet-v1 for a
failed checkpoint. Legacy v1 acceptance packets retain their separate coverage.

Native `build_arch`/`weights_create` register encoder/head/MinGRU shapes without
creating CUDA buffers, initializing weights, loading weights into a policy or
computing neural outputs. Python validates the recorded ordered shapes/offsets,
component counts, action/core dimensions and raw float32 byte length. These are
scalar checks, not a Python learner or CPU neural oracle. The native executable
remains dynamically linked to the existing local dependencies; their full
contents are not certified. No system dependency changes are needed.

## Fresh local preparation

```bash
NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS='--threads 1' \
  bash research/build_policy_metadata.sh build/policy-metadata/FRESH_NATIVE
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/eval_acceptance.py prepare \
  --cases LOCAL_EXISTING_PIXEL_CASES.json \
  --policy-metadata build/policy-metadata/FRESH_NATIVE \
  --out build/eval-acceptance/FRESH_LAYOUT_PACKET --timeout 600
.venv/bin/python research/eval_acceptance.py inspect \
  --packet build/eval-acceptance/FRESH_LAYOUT_PACKET/packet.json
```

The case JSON remains the existing schema described in
[EVAL_ACCEPTANCE.md](EVAL_ACCEPTANCE.md). Supply actual merged training INIs and
matching existing pixel checkpoints/development suites. The header supplies the
head's discrete action count; it doesn't prove the runtime binary's source.
Keep its separate native build/source receipts. No short hand-written INI or
another game's checkpoint is a valid substitution.

`load_packet(..., current_inputs=False)` audits frozen descriptors, native
process logs/clocks, configs, tools, action headers and hashes offline, including
after relocation. It executes no native binary. Current-input inspection also
requires the original inputs and owned source to match. A later **separately
scheduled** v2 `run` repeats shape/stat registration before calling any GPU
adapter, then uses the existing reservation/timeouts and four graph/repeat/eager/
independent-tail checks. Rechecking shapes does not extend GPU authorization.

Both GPU holds remain after the completed local smoke. Mathematical gradients,
learner-batch memory, full policy/reload/concurrency, evaluator acceptance,
learner/cap calibration, learning, held-out selection and simultaneous frontier
inference remain separate gates. Passing bytes/layout cannot establish a
Pareto advantage or SOTA.
