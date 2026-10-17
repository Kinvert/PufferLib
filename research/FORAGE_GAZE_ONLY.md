# Forage ablation: only see where the policy looks

Kinvert requests removing the automatic player-centered pixel window.
The opt-in attention build now accepts `FORAGE_PLAYER_VIEW=0`; its default1
retains the previous behavior. This is a research build choice, not a changed
default observation or action contract for the normal environment.

With0, partial observations contain only the selected gaze's row-major3x3
cells. There are no additional player-neighbor tiles or full-map features.
The gaze can still be centered on the player by the policy, and starts there
on reset. A player avatar appears only if it lies in the selected window.
Off-board cells still use the tree tile. Movement/gaze actions remain5/20/15;
the four existing player-position/relative-gaze metadata floats remain.

Each tile stays32x32 RGB; no downsampling or pretraining. Pixel input shrinks
from52,228 to27,652 floats. The same shared tile CNN produces sixteen features
per tile; its H128 projection now has148 rather than276 inputs. Complete
policy parameters fall from103,976 to87,592. Matching visibility is available
for future state controls too: one-hot partial state becomes121 values.

## Allocation

Raw packet: `build/sprite-forage-gaze-only-5060-20261015`.
Reference: the completed17-tile pixel run in
`build/sprite-forage-pixels-matched-5060-20261015`,53.71% collection/
55.08% grape exits at295.053s and4,587,520 decisions,15,548 process SPS.
Do not rerun or replace that completed reference automatically.

The new gaze-only policy trains from scratch, seed7, same A64/H64/MB256,
four threads, sync/H128-L1, learner, rewards, drawing0 and game rules.
Separate131,072-decision calibration selects a roughly300s decision budget
from throughput alone. All eight scheduled checkpoints remain; primary
selection is nearest300 monotonic seconds, tie earlier, never best score.
If final differs, evaluate/report it as well. Evaluation assigns128 IDs each
at development suite seeds420/1337,16 slots,512-step cap, with exactly the
same worlds/policy RNG as the completed reference. Position metadata stays
matched; removing it would be a separate ablation.

This is an equal-time system comparison, not a fixed-decision or
parameter-matched experiment. Removing tiles changes visibility, projection
size, throughput, number of decisions and the decision-space LR schedule.
Any score difference cannot be attributed solely to missing information.
One training seed and reused development worlds do not establish a robust
ranking, generalization or a Pareto/SOTA result.

Before full training, gaze-only ASan/UBSan tests pass24,000 matched world
steps/all1,500 skin/gaze cases, full-frame crop parity, hidden-cell isolation,
geometry, all actions and RNG independence. All seven normal game/raster
suites also pass. Independent CUDA float64 reference checks pass all33,192
encoder gradients/outputs on smooth-positive and zero-weight fixtures at
B1/3/8/64/256, with exact eager/graph repeats and unchanged weights. These
fixtures do not qualify every ReLU branch or full learner derivatives.
Calibration policies must pass exact graph/repeat/eager/independent-tail
episode rows before the long allocation. GPU jobs remain serial, idle-checked
and under the shared reservation. No CPU neural learning or system change.

Build without GPU policy execution, using fresh outputs:

```bash
bash research/build_forage_attention.sh \
  build/sprite-forage-speed-5060-20261015/source \
  build/GAZE_BUILD_FRESH 2 1 1 1 0
```

Trailing options are gaze mode, relative geometry, exact evaluator, state
one-hot, and player view. The original six-argument commands retain their
previous17-tile contract. `build_forage_attention_checks.sh` now reads these
captured flags so its encoder fixture matches the selected view.

For the current explicitly assigned packet:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/forage_gaze_only.py prepare build/sprite-forage-gaze-only-5060-20261015
.venv/bin/python research/forage_gaze_only.py calibrate build/sprite-forage-gaze-only-5060-20261015
.venv/bin/python research/forage_gaze_only.py run build/sprite-forage-gaze-only-5060-20261015
.venv/bin/python research/audit_forage_pixels_matched_time.py build/sprite-forage-gaze-only-5060-20261015
```

Read the packet's COMPLETE marker and offline audit before reporting success.
These commands are not authorization to restart it. Earlier frozen packets,
binaries, sources, metrics and weights remain intact. Normal PufferLib core
files are unchanged; this uses a fresh copied compatible research core.
