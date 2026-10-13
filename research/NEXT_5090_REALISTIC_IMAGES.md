# Next 5090 iteration: realistic RGB frames

The native builds, short 5060 checkpoint/evaluator smoke and Nature/C8 encoder
checks on all five RGB shapes atH128/B1/4/16 pass. The overnight
launch is **not ready**: full-image learning controls, the practical search
envelope and hardware-specific timing still need qualification. This document
currently covers reproducible build preparation, not authorization to start an
uncalibrated sweep. Preserve the earlier retained campaign checkout/history.

After the verified delivery is pushed, use a separate research checkout:

```bash
git pull --ff-only origin cnn-research

# Reuse this checkout's existing venv.
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python 'numpy==2.5.3'
fi
.venv/bin/python -c 'import sys,numpy; assert sys.version_info[:2] == (3,12)'
```

Do not recreate an existing venv. No Torch, driver, toolkit or system package
installation is needed. Keep the venv interpreter path; resolving its symlink
with `realpath` loses the venv's packages.

Use the existing 5090 process-local CUDA/NCCL helper, then build:

```bash
if [[ -f build/connect4cnn/runtime-5090.sh ]]; then
    source build/connect4cnn/runtime-5090.sh
fi
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120

IMAGE_PYTHON="$PWD/.venv/bin/python" \
    bash research/build_image_delivery.sh \
    "build/realistic-images-5090-$(date -u +%Y%m%dT%H%M%SZ)"
```

The build command creates six default float32 RGB trainers, native image-info
receipts, a training registry and per-game host metadata. It performs no GPU
query, training or evaluation. Nature uses selector 2 in those same binaries;
the custom CNN uses selector 4. IMPALA/Impoola require separate family builds
and are outside this first Nature-focused delivery.

| Game | Native/policy frame, height × width |
|---|---|
| Connect4 | 576 × 672 RGB |
| Pong | 640 × 620 RGB |
| Flappy | 640 × 420 RGB |
| Breakout | 330 × 576 RGB |
| Snake, Maze | 616 × 616 RGB |

There is no hidden resize. Snake/Maze retain their original local visibility.
Learners, budgets and checkpoint schedules are matched per game across models.
Declare Nature's common-H128 flattened readout adaptation and actual parameter
counts; this is not the exact original DQN policy or an Atari-wide result.

The remaining launch sequence will be: matched CUDA controls → fixed
quality/Nature learning controls with initial plus eight checkpoints → audited
all-game eligibility → measured complete-panel runtime pilot → bounded native
PROTEIN discovery → final replay, complete curves and evidence archive.
Search varies CNN coordinates only; `research/recipes/image_seed_roles.json`
freezes development/confirmation seeds. Confirmation never feeds PROTEIN.

The discovery budget will be approximately 12 hours, with reserves for a whole
next panel and reporting that grow with the retained history. Failures remain
failures; no partial-game averages, completed-allocation retries or best-seed
selection. Actual 5090 calibration, not tiny-image or 5060 SPS, must determine
the usable budgets and envelope. Final run/report commands will be added after
these gates pass. No 5090 source editing should be required.

Local proof: [October 13 report](results/image-curve-groundwork-5060-20261013/README.md).
Protocol: [overnight specification](IMAGE_OVERNIGHT_PROTOCOL.md).

The new `prepare_image_calibration.py prepare` freezes the two predeclared
controls, all three development replicas, separate evaluation seeds and initial
plus eight trained checkpoints. It requires explicit per-game budgets/learners
from a curve-v4 recipe and completed hardware-matched all-shape CUDA evidence for
both controls. It is preparation only; it cannot turn the 512-decision fixtures
into calibrated learning. The numerical manifests must name `quality-reference`
and `nature-cnn` respectively and retain their original raw allocations.

After separately scheduled native control training completes, this offline
command replays the artifacts, writes complete plots/SPS exports and evaluates
the learning gate:

```bash
.venv/bin/python research/prepare_image_calibration.py qualify \
    --allocation "$IMAGE_CONTROL_ALLOCATION" --out "$IMAGE_CONTROL_REVIEW"
```

Both variables must identify the newly prepared control allocation and a fresh
review directory. An unresolved game produces exit status 2 and an explicit
`learning-eligibility.json`; all controls and plots stay available. It creates
Nature-time reference anchors only when every game qualifies. The follow-on
pilot uses `report/review/analysis.json` and `nature-reference.json` from that
review. These commands do not schedule GPU work. Actual learning/envelope/timing
qualification is still required before an overnight launch command is supplied.
