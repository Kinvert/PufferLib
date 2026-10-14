# Fixed-image 5090 calibration delivery

October13, G240 audit worktree; parent revision15e87ceb plus the source changes
hashed in `receipts/CAMPAIGN.json`. Delivery is temporary Kinvert research, not
an upstream PR. Read the root `NEXT_5090_IMAGE_CALIBRATION.md` for complete
no-edit remote setup/prepare/run/report commands.

Actual GPU-free preparation through the published wrapper **passes**. Six fresh
float32 RGB trainers, six host metadata tools and five native/independent
CUDA-reference library pairs compile with existing CUDA12.8.93, explicit sm120
and the existing uv Python3.12 venv. Frozen registry/catalog/source inspection
and compact summary packaging also pass. No GPU query, model execution or CPU
CNN occurred in this delivery verification. These are compiled test libraries,
not newly passed numerical allocations. Fresh hardware-matched5090 checks for
all three models remain mandatory before any training.

The focused host/scalar/mock suite passes54 tests. New coverage checks matched
probe/main learners and seed roles, rollout/eight-checkpoint-aligned measured
budget admission, refusal of missing/duplicate/altered probe data, report
plan/result/hardware bindings, preserved numerical failures, restart refusal,
learning-gated reference freezing and compact archive exclusions. Mocked
supervisor success is not actual CUDA or learning evidence.

The fixed comparison has quality-reference, spatial-control and Nature. An
18-job, all41-drawing timing probe precedes budget admission. If actual costs
permit it,54 paired training jobs use seeds7/42/69,2M–13M decisions/game and
actual initial plus eight trained checkpoints. Per-game scalar learners and
new explicit16-actor/H16/minibatch16 geometry remain identical across models.
There are3,321 drawing/checkpoint evaluations plus108 repeat/eager checks.
All games/replicas/drawings must remain; an insufficient minimum-budget fit
stops instead of substituting another tiny plumbing run. The approximate12h
window includes numerical checks, probe, calibration and reporting; episode
length changes and offline array audits can exceed estimates and are recorded.

At least one of the fixed quality/Nature controls must pass the existing
learning floors on every development replica for each game. The spatial
control cannot rescue an unlearned reference pair. No PROTEIN, confirmation,
winner/Pareto/generalization/SOTA claim follows automatically.

Executed preparation commands:

```bash
IMAGE_PYTHON=/home/claude/cnn/.venv/bin/python bash research/run_image_calibration_5090.sh prepare build/image-calibration-delivery-wrapper-20261013
IMAGE_PYTHON=/home/claude/cnn/.venv/bin/python bash research/run_image_calibration_5090.sh inspect build/image-calibration-delivery-wrapper-20261013
IMAGE_PYTHON=/home/claude/cnn/.venv/bin/python bash research/run_image_calibration_5090.sh handoff build/image-calibration-delivery-wrapper-20261013
```

The earlier developer reuse invocation omitted NVCC_ARCH and failed before
compilation/execution; its unchanged failure receipt is included. A fresh reuse
preparation subsequently passed, followed by the fresh public-wrapper build
above. No failed allocation was overwritten or GPU allocation restarted.

`receipts/` retains exact campaign/config/role hashes, native/metadata/catalog
descriptors, compiler identity, successful logs, test output and the initial
failure. `SHA256SUMS` covers these durable files. The original96MiB prepared
tree and compact summary archive remain under ignored `build/`; summary paths
and archive hash are in `receipts/handoff.json`. The small committed selection
does not include binaries, arrays or a full source/toolchain closure and cannot
independently replay numerical checks. The5090 must prepare its own fresh tree;
G240 paths are provenance, not portable launch inputs.
