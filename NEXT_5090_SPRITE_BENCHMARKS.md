# Next sprite benchmark handoff — preparation stage

Read `research/SPRITE_PIXEL_BENCHMARKS.md`. This iteration prepares identical
96x192 RGB player/gaze input for our tile CNN, Nature, IMPALA, Impoola and Flex.
Each agent learns its own movement and gaze. No pretrained weights.

Current status: build, raster parity and host packet preparation are available;
this is **not yet an overnight benchmark launch handoff**. New baseline GPU
math/runtime checks and the bounded execution supervisor remain required.
Do not launch unreviewed long native sweeps or reuse prior scores as evidence.

When the owner supplies the committed revision on Kinvert/PufferLib
`cnn-research`, pull it in the5090 checkout, activate the existing uv venv and
source `ocean/connect4cnn/runtime_env.sh`. Then run the build/prepare commands
from the linked document using fresh output names. No source edits or system/
CUDA/Torch changes are required. Prepare/inspect perform no GPU work.

This turn commits locally; it does not push or claim the5090 can already pull
these changes. A later delivery must name the pushed revision and supply the
tested run command, calibrated resource/time caps and evidence return path.
