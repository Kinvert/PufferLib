# Optional 5090 autoencoder pretraining

October20,2026. Build/host inspection pass locally; GPU qualification status must
come from the result report, not this document. This is a separate optional
deliverable from the earlier compression packet. Its WSL memcheck failure is
preserved. No long pretraining, corpus collection or5090job has started here.
Local qualification did not execute: another campaign occupied the 5060 for the
entire bounded wait. Build/host checks are the verified scope for this increment.
Delivery target is Kinvert's `cnn-research` branch. Never push official
PufferLib upstream. Pull the implementation before following these commands.
If pull refuses local changes or divergent history, preserve them and report;
do not reset, overwrite or edit around the failure on the 5090.

**Evaluation clarification:** the reconstruction-decrease canary below verifies
short training behavior, not useful perception. Read
`research/ENCODER_INFORMATION_EVALUATION.md` and
`research/ENCODER_EVALUATION_LITERATURE_20261020.md` before choosing a long run.
The proposed object/position probes and background-resistant metrics are not yet
implemented. Existing `qualify` does not certify them or downstream usefulness.

## 1. Pull, build and inspect — no dataset or GPU work

Use the5090research checkout and its existing working runtime. No installs of
Torch/CUDA/NCCL, Docker or system changes. Python is standard-library process/
manifest glue; convolution, reconstruction, gradients and Adam run native CUDA.

```bash
git pull --ff-only origin cnn-research
if [ -f build/connect4cnn/runtime-5090.sh ]; then source build/connect4cnn/runtime-5090.sh; fi
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash research/autoencoder_pretrain.sh prepare --out build/autoencoder-5090-01
bash research/autoencoder_pretrain.sh inspect --packet build/autoencoder-5090-01
.venv/bin/python -m unittest discover -s research/tests -p test_autoencoder_pretrain.py -v
```

Preparation captures the exact dirty source bytes, toolchain/binary hashes,
native manifests and recipes. It builds both game collectors and one common
variable-shape trainer. Current encoder:36,776parameters,128latent values;
training-only decoder:34,987parameters. Multiple raw frame shapes share weights.
Flappy640x420and Pong640x620stay full RGB, with original aspect ratios.
No semantic state, object IDs or sprite shortcuts enter the network.

## 2. Bounded qualification before any long pretraining

Wait for exclusive GPU access and the other campaign's supervisor to finish;
do not steal gaps or stop its processes. No active TUI watching.

```bash
bash research/autoencoder_pretrain.sh qualify --packet build/autoencoder-5090-01 --hardware 5090 > build/autoencoder-5090-01/qualification.log 2>&1
bash research/autoencoder_pretrain.sh report --packet build/autoencoder-5090-01
```

The900second cap includes24independent CUDA math workers (small/even,odd and
actual per-game shapes,B1/B2,two fresh replicas), all parameter/input derivatives,
64selected finite-difference probes, eager/graph repeats, MSE/tail-mask and Adam
checks. Then a tiny native-frame **test fixture** (at most8frames,about32MB)
tests the actual shard reader:8step/repeat/4step/resume-to8training, identical
parameter/moment checkpoints, loss reduction and per-game encoder imports plus
wrong-shape rejections. This fixture is not the pretraining corpus.
Finally two numerical Compute Sanitizer suites and a mixed-shape real training/
Adam/checkpoint memory check must pass. Any failure stops the stage;
failed/completed allocations cannot be retried in place. The short checks are
not RL/generalization evidence. G240uses a separate300second cap.

Require `stages/qualify.json` and REPORT status passed. A successful numerical
worker under a failed sanitizer is NOT a memory pass. WDDM/debugger/device errors
must be returned to the coding host; change no Windows/CUDA/system settings and
do not bypass the training gate. The5090agent does not need to edit code.

## 3. Only if choosing to pretrain:collect a bounded pilot

This explicit command generates a corpus. Do not run it merely because the
tools built. Default:16training+4validation trajectories/game,maximum32frames
each,256decision cap,cadence4,all seven skins cycling in episode order.
Worst-case storage about5.1GB,8GiB cap plus20GiB free reserve. Actual episodes
can end earlier; keep all short/weak trajectories. Random actions are a known
coverage limitation. No best-seed/episode replacement or seed sweep.

```bash
bash research/autoencoder_pretrain.sh collect --packet build/autoencoder-5090-01 --out build/autoencoder-data-5090-01
bash research/autoencoder_pretrain.sh dataset --dataset build/autoencoder-data-5090-01
```

Train/validation environment/action seeds are fixed,derived deterministically
from disjoint bases7/420. The manifest records episode boundaries,skins,raw
frame hashes,collection source and duplicate-image overlap across splits.
Identical pixels in separate trajectories are disclosed; new trajectory seeds
do not create unseen-game evidence. Nothing is labeled with invisible state.

## 4. Optional native training

```bash
bash research/autoencoder_pretrain.sh train --packet build/autoencoder-5090-01 --dataset build/autoencoder-data-5090-01 --out build/autoencoder-fit-5090-01 --hardware 5090 > build/autoencoder-fit-5090-01.log 2>&1
```

The default recipe is research/recipes/autoencoder.ini:batch2,1024updates,
checkpoints every128,LR0.0003,seed7,900second cap. A user-selected longer
recipe can be supplied with `--recipe FILE`; the native cap is at most12hours,
and no long job is launched automatically. No seeds are swept. The output
reports complete versus budget_exhausted honestly, and keeps all checkpoints.
Each step cycles streams deterministically,using epoch-specific fixed shuffles
within each stream. Validation masks tail padding. Raw and optional visible-edge
weighted reconstruction losses are separate. The decoder is discarded later.

To resume, use a fresh output folder and `--resume` pointing to a retained
step-N.state. The matching sidecar,dataset/source/trainer and optimizer recipe
must agree. A longer recipe may increase `steps` and `max_seconds`; it cannot
silently change LR,batch,seed,loss or checkpoint cadence. Resume restores Adam
moments,global step and the derived stream/shuffle sequence.

## 5. Export the last scheduled encoder

```bash
bash research/autoencoder_pretrain.sh export --packet build/autoencoder-5090-01 --run build/autoencoder-fit-5090-01 --stream flappycnn-640x420 --out build/backbones/ours-ae-flappy-01
bash research/autoencoder_pretrain.sh export --packet build/autoencoder-5090-01 --run build/autoencoder-fit-5090-01 --stream pongcnn-640x620 --out build/backbones/ours-ae-pong-01
```

Both exports use the same trained encoder weights,with shape-specific native
manifests. They include all encoder layers/projection and exclude the decoder
and Adam state. Use encoder8/compression_model0,hidden128 and set
`policy.compression_init` to the export directory. Ordinary encoder initialization
consumes its usual RNG before replacement; core/heads initialize normally.
Full policy reload clears compression_init. No frozen-prefix caching is implied:
that separate integration is still future work. Compare scratch versus pretrained
with matching downstream learners and count pretraining/data generation costs.

## Existing weights, without pretraining

```bash
.venv/bin/python research/backbone_assets.py fetch-taesd --out build/backbones/taesd-614f768
.venv/bin/python research/backbone_assets.py fetch-mobilenet --out build/backbones/mobilenetv3-small-050-5188129
```

This downloads9.8MBof pinned safetensors/config/model-card data,validates its SHA
and tensor spans,and executes no model or pickle. TAESD's pretrained encoder has
1,222,532parameters and a spatial H/8xW/8x4latent. It needs its own native adapter
or an offline teacher path; it cannot be loaded into our36,776parameter encoder.
See research/PRETRAINED_BACKBONES_20261020.md for ranked alternatives and links.
MobileNetV3-Small-050 is a separate 6.42 MB ImageNet feature-backbone checkpoint;
it also requires its own native adapter. Neither fetch command performs inference.

## Return evidence

Return source/binary/config hashes,hardware/driver,qualification status and every
failure,canary checkpoint repeat/resume hashes,dataset manifest/hash and exact
frame overlap,all raw/weighted loss curves,steps/process seconds/images per
second,checkpoint/encoder hashes and storage. Keep5090and5060separate.
Reconstruction success is not proof of better RL or a Pareto improvement.
