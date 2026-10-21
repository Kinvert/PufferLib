# 5090: train a spatial image compressor

Pull `Kinvert/PufferLib:cnn-research` and run the commands below without coding.
This is the new spatial autoencoder, not the earlier 128-number vector pilot.
Do not rerun the old pilot or modify the existing CUDA/NCCL stack.

## What runs

```text
pretrain: full 640x480 RGB -> encoder -> spatial features -> decoder -> RGB
                                                       object-weighted loss

later:   full RGB -> saved encoder -> spatial features -> CNN -> RL policy
```

Encoder: Conv4/stride4/8 channels, Conv3/stride2/16 channels, then
Conv4/variable stride/variable channels; SAME padding and ReLU. No pooled-vector
or dense bottleneck. Decoder: nearest expansion and three convolutions.

|Total stride|Channels|Feature grid HWC|Fewer FP32 values than input|
|---|---|---|---|
|8|8|60x80x8|24x|
|16|8|30x40x8|96x|
|16|16|30x40x16|48x|
|32|16|15x20x16|192x|

Ratios describe numbers of values, not speed or quality. Strides 8/16/32 and
channels 4/8/16/32 are configurable in `research/recipes/spatial_autoencoder.ini`.
Change it only before a fresh preparation; every selected size must qualify.

Every size gets plain and object-balanced reconstruction controls with identical
initialization/data/update targets. Object loss averages instance foreground
errors within each present class, then across classes, plus separately normalized
background error weighted at 0.1. Labels/masks are training/evaluation targets;
the encoder sees pixels only. Lookalike markings have separate reconstruction
metrics. No classifier is fitted.

## 1. Pull, host tests and prepare

From the existing 5090 checkout:

```bash
git pull --ff-only origin cnn-research
mkdir -p build
if [ -f build/connect4cnn/runtime-5090.sh ]; then source build/connect4cnn/runtime-5090.sh; fi
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python -m unittest discover -s research/tests -p 'test_*autoencoder*.py' -v
.venv/bin/python -m unittest discover -s research/tests -p test_forage_representation.py -v
NVCC_ARCH=sm_120 bash research/spatial_autoencoder.sh prepare --out build/spatial-ae-5090-01 > build/spatial-ae-5090-01.prepare.log 2>&1
```

Preparation freezes source/binary/config hashes and native shape/tensor manifests,
checks independent parameter counts, and runs native annotation/host sanitizers.
It does not query a GPU or collect a corpus. If that directory exists, use a new
suffix consistently; never erase failed evidence to reuse a name. Preserve local
changes if pull refuses; do not reset or overwrite them.

## 2. Mandatory target qualification

Wait for exclusive GPU availability; the runner reserves it and refuses foreign
work rather than stopping jobs or launching into campaign gaps.

```bash
bash research/spatial_autoencoder.sh qualify --packet build/spatial-ae-5090-01 --hardware 5090 > build/spatial-ae-5090-01.qualify.log 2>&1
```

One-hour cap, not an ETA. Checks every prepared size:

- Independent float64 CUDA forward/all parameter gradients/image gradients on
  smooth-positive 160x192/B1, 161x193/B2 and 480x640/B1 fixtures, two fresh workers
  each; selected finite differences and exact eager/graph repeats.
- Loss/Adam/ROI metric primitives and a vector-model regression.
- Actual four-update object-loss training, exact repeat and optimizer/data-order
  resume using a tiny six-frame fixture, not a full corpus.
- Encoder export and exact reloaded spatial features on every fixture image;
  image/label alignment, frame/cell quotas and incorrect-shape artifact rejection.
- Frozen and fine-tuned CNN consumers, selected consumer finite differences,
  and eager/graph repetition.
- Repeated actual frozen-policy training, fine-tuned training, and checkpoint
  reload/one-update continuation. These 8,192-decision jobs verify plumbing.
- Compute Sanitizer for math, both consumers, actual AE and actual policy
  training/reload. Unsupported WDDM/debugger/device failures fail the gate;
  do not change the system/CUDA or waive it.
- Three real native PROTEIN trials selecting numeric encoder-bank variants.
  Seed stays fixed. This is wiring, not quality-search or GP-active evidence.

Require `stages/spatial-qualify.json` with `status=passed`. Old vector qualification
does not qualify this graph. If anything fails, preserve logs and return the
failure; no in-place retry or long pilot follows.

## 3. Run the spatial reconstruction pilot

```bash
bash research/spatial_autoencoder.sh run --packet build/spatial-ae-5090-01 --hardware 5090 > build/spatial-ae-5090-01.run.log 2>&1
bash research/spatial_autoencoder.sh report --packet build/spatial-ae-5090-01 > build/spatial-ae-5090-01.audit.json
```

The runner explicitly collects one shared corpus, then trains eight autoencoders
(four sizes x two objectives) and evaluates reconstruction. No classifier training
or long RL sweep. Fixed recipe:

- 16 training worlds/four validation worlds, skins 0/1/2; eight final worlds,
  skins 0/1/2/3/4. Same world/action trace across skins, same frames for all models.
  Unseen skins 3/4 remain separate report rows.
- Fixed collection seed bases 7/420/1337; pretraining seed 7. Never sweep/select
  seeds. One independent pretraining replica, not a robust generalization study.
- At most four frames/trajectory, 400 frames/about 1.48 GB decimal, under a 2 GiB
  corpus cap and 20 GiB free reserve. Retain short/dying trajectories and report
  exact frame overlap. Random play is limited coverage, not expert gameplay.
- Each fit targets 1,024 batch-2 updates, checkpoints every 128, maximum 900s.
  Actual updates and exhausted caps remain visible. Equal requested budgets
  are not equal completed work when caps bind.
- Four-hour total pilot cap including collection/fitting/evaluation, separate
  from the one-hour qualification cap. These limits are not elapsed-time estimates.

Read:

```text
build/spatial-ae-5090-01/spatial-run/REPORT.md
build/spatial-ae-5090-01/spatial-run/reconstructions.html
build/spatial-ae-5090-01/spatial-run/progress.json
build/spatial-ae-5090-01/spatial-run/encoder-bank/bank.json
```

The viewer pairs originals/reconstructions, including fixed first-instance object
crops. Errors use unclipped predictions; display RGB is clipped. Tables show
class/skin/split object MSE, distinguishing-mark MSE and background MSE; absent
classes are N/A. Retain actual time/images per second/update counts/latent sizes.
No single background-dominated accuracy or classifier-F1 verdict.

Exports use the final scheduled checkpoint, never the best test score. If you
choose architecture/objective from these test results, these worlds become
development data; later confirmation needs new untouched worlds.

## Downstream PufferLib and size sweeps

Research encoder 8 with `compression_model=3` runs the saved compressor, a small
CNN on the feature grid, then normal H128 recurrent policy. `compression_frozen=1`
excludes prefix weights from the optimizer; `0` trains them too. Neither mode
runs the reconstruction decoder or classifier. Frozen prefixes are external
artifacts: keep the encoder bank/hash/config with policy checkpoints. A policy
checkpoint alone does not contain the frozen prefix.

Object-balanced exports form a bank. Numeric `policy.compression_variant` selects
a complete pretrained size; only that coordinate appears in the generated native
sweep config. Other learners and seed remain fixed. This searches pretrained
sizes, not arbitrary graphs or a joint PROTEIN-to-pretraining loop.

`spatial-run/policy-config/config/` uses the **8,192-decision canary learner**.
It is not a meaningful learning benchmark. Do not launch its 24-run discovery
as a quality experiment. Before a larger downstream campaign, lock a suitable
per-game learner, decisions/checkpoints, fixed training replicas and independent
deterministic evaluations. Compare simple resizing and scratch/full-pixel paths;
measure wall-clock/SPS including prefix inference and report pretraining cost
separately. This handoff does not launch that larger campaign or establish an
RL/Pareto improvement.

## Return evidence

Return source revision, qualification status, per-size object/marking/background
errors, seen/unseen skin differences, actual updates/time/images per second,
failures and viewer path. Keep corpus, weights and raster previews on the 5090.
Commit/push compact reports and source/command/hash/failure receipts to Kinvert's
fork only. No official upstream push or automatic restart of finished packets.
