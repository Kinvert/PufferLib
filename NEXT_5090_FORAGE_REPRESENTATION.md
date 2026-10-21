# 5090: does pretraining preserve Forage item identities?

October 21: these commands describe the existing **vector-autoencoder pilot**.
They do not implement the requested spatial compressor. Read
`research/SPATIAL_AUTOENCODER.md` before preparing further experiments.
The reconstruction objective and per-object/marking errors are the primary AE
measurements; the linear classifier is a supporting diagnostic. Do not rerun a
completed overnight pilot merely to follow this historical handoff.

This is a bounded **perception diagnostic**, not an RL sweep or a game-win
benchmark. Run the committed code without editing it on the 5090. Keep the
existing CUDA/NCCL stack; no Torch, Docker or system changes are required.
If a pull/build/check fails, preserve the logs and return the failure.
Local build/host checks and brief RTX 5060 primitive/network/integration smokes
pass; see `research/results/forage-representation-20261020/README.md` for scope.
Those short checks do not replace the new 5090 qualification below.

## What runs

All cases see the same full **640x480 RGB** Forage frames. No oracle crop, ROI
coordinate, object ID or reward enters the encoder. The frozen encoder emits
128 values. A separate 503,100-parameter linear diagnostic predicts 13 identities
at each of the 300 board cells. That diagnostic is discarded; it is not the RL
policy. It is deliberately held identical across encoder cases.

The four cases are constant features, random frozen features, plain-MSE
pretraining, and object-balanced pretraining. The two trained encoders have
identical initial weights, data, Adam settings and target updates. The object
loss gives equal weight to present object classes and instances, plus a separate
background term. This uses simulator-derived **training targets**, and must be
described as privileged supervision. The input is still the original image.

Reports separately show apple versus spotted apple, safe mushroom versus
toadstool, every class's recall/confusion, and reconstruction error on both whole
objects and their distinguishing marks. There is no background-dominated
"encoder accuracy" score. The pair-only score is conditional on true object
locations; use the full confusion matrix to assess detection/localization too.

## 1. Pull, build, host checks

From the 5090 research checkout:

```bash
git pull --ff-only origin cnn-research
mkdir -p build
if [ -f build/connect4cnn/runtime-5090.sh ]; then source build/connect4cnn/runtime-5090.sh; fi
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash research/forage_representation.sh prepare --out build/forage-representation-5090-01 > build/forage-representation-5090-01.prepare.log 2>&1
.venv/bin/python -m unittest discover -s research/tests -p test_forage_representation.py -v
.venv/bin/python -m unittest discover -s research/tests -p test_autoencoder_pretrain.py -v
```

Preparation captures source bytes/toolchain/binary hashes and builds native CUDA.
It runs host template checks and ASan/UBSan over actual native reset/step pixels
for all five skins. It does not query the GPU, collect a corpus or train a net.
The small Forage adapter is under `research/`; the PR environment is unchanged.
If pull refuses local changes/divergent history, do not reset or overwrite them.

## 2. Mandatory bounded GPU qualification

Wait until the GPU is exclusively available. Do not stop another campaign or
launch into its gaps. This command refuses an occupied GPU and never retries.

```bash
bash research/forage_representation.sh qualify --packet build/forage-representation-5090-01 --hardware 5090 > build/forage-representation-5090-01.qualify.log 2>&1
bash research/autoencoder_pretrain.sh report --packet build/forage-representation-5090-01 > build/forage-representation-5090-01.status.json
```

The 900-second allocation covers independent CUDA encoder/decoder derivatives,
selected finite differences, the new weighted loss and probe-head arithmetic,
and 20 identity/deletion/lookalike-corruption metric controls across five skins.
A tiny fixture of at most six actual frames checks eight-update object-loss
training, exact repeat/resume state, encoder import, and two actual four-update
recognition probes with identical feature/head/prediction bytes. No final-test
score is used to pick weights. Four memory-instrumented suites must also pass:
network math, recognition primitives, actual probe fitting and actual AE training.

Require `stages/qualify.json` and a passed report. WDDM/device/debugger failures
are failures of the memory gate; do not change Windows/CUDA or bypass them.
Return the packet/logs instead. A failed/finished stage cannot restart in place.
The earlier local compression failure remains preserved and is not waived.

## 3. Run the pilot after qualification passes

```bash
bash research/forage_representation.sh run --packet build/forage-representation-5090-01 --out build/forage-recognition-pilot-5090-01 --hardware 5090 > build/forage-recognition-pilot-5090-01.log 2>&1
```

This command explicitly collects its one shared pilot corpus, then runs the
two pretraining cases and four frozen-feature probes serially. Default recipe:
`research/recipes/forage_representation.ini`.

- Training: 16 fixed worlds, skins 0/1/2; development: four different worlds,
  the same skins; test: eight further worlds, skins 0/1/2/3/4.
- World/action seeds are deterministically derived from fixed bases 7/420/1337,
  never swept. Each world uses the same actions/world RNG across its skins.
  Appearance-paired label/terminal receipts must match.
- At most four frames per trajectory: at most 400 full frames, about 1.48 GB
  decimal including tiny labels, under a 2 GiB corpus cap and 20 GiB free reserve.
  Actual short/terminal trajectories are retained. Random actions are a limited
  collection policy; this is not a coverage guarantee for expert gameplay.
- Each pretraining case targets 1,024 batch-2 updates, with checkpoints every
  128 updates and a 900-second cap.
- Each linear probe targets 512 batch-16 updates, with a 300-second cap including
  feature extraction. Normalization/class weights use training data only.
- Overall campaign cap is one hour including collection, whose own cap is five
  minutes. Actual time is not yet measured. The cap is a limit, not an estimate.

The last scheduled checkpoint is used, never the best test/probe/MSE checkpoint.
Actual update counts and `budget_exhausted` remain visible. These are fixed-seed
diagnostics, not seven independent pretraining replicas. No whole unseen game
is evaluated. Skins 3/4 are unseen during fitting and must be reported separately
from seen-skin test worlds; do not average away a regression.

## 4. Read and return evidence

```bash
bash research/forage_representation.sh report --packet build/forage-representation-5090-01 --out build/forage-recognition-pilot-5090-01 > build/forage-recognition-pilot-5090-01.audit.json
```

Read `build/forage-recognition-pilot-5090-01/REPORT.md` and `REPORT.json`.
The report rebinds sources, exact native commands, every prediction/target row,
all checkpoint/feature/head hashes and frame quotas. No missing/failed case is
silently averaged. Raw prediction CSVs include the per-cell majority control.
Raw features and diagnostic heads are retained for analysis, not policy inputs.
The constant-feature reconstruction is from the random model; only its head
input is zeroed, so do not treat its reconstruction as a separate autoencoder.

The `encoder-exports/plain` and `encoder-exports/object-balanced` directories
contain compatible encoder-only exports. They target the experimental
encoder-8/compression-model-0, H128 full-frame architecture, not stock Nature.
Native import is checked during qualification; no downstream RL training is
part of this pilot.

Return compact reports, source/command receipts, native math/annotation logs,
data/array/checkpoint hashes and every failure. Keep the roughly 1.5 GB corpus,
weights and feature arrays local; do not commit or transfer them by default.
Do not claim an RL, Pareto, generalization or SOTA improvement from probe scores.
