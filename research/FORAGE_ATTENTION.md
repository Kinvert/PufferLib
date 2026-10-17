# Sprite Forage attention prototype

Latest pixel ablation: [gaze-only](results/sprite-forage-gaze-only-5060-20261015/README.md)
is faster but falls to4.88% collection versus53.71% with the automatic player
view. `FORAGE_PLAYER_VIEW=0` is opt-in; the default remains17 visible tiles.
See [the gaze-only protocol](FORAGE_GAZE_ONLY.md).

Latest state results: [matched action/observation ablations](results/sprite-forage-state-controls-5060-20261015/README.md).
The factored-X/Y, one-hot, relative-offset candidate reaches93.55% collection
perf /90.23% grape exits at97.3 seconds, with exact same-seed checkpoint and
episode-row reproduction. This is a single-seed development result. Current
control/build contracts and remaining gates are in
[FORAGE_STATE_ABLATIONS.md](FORAGE_STATE_ABLATIONS.md). Downsampling has not
begun; the original pixel inputs/encoder below remain unchanged.

The native implementation supports these INI switches in the **attention
research build**:

```ini
[env]
observation_mode = 1     # 0 = state; 1 = RGB pixels
partial_observation = 1  # 0 = full board; 1 = local + agent-selected gaze
```

| Partial | Mode | Observation | Policy |
|---:|---:|---|---|
|0|0|Existing141-float state layout2|Standard state encoder|
|0|1|Existing640x480 RGB float32 frame|Full-frame encoder, Flex in supplied recipe|
|1|0|17 visible cell values +4 geometry floats|Standard state encoder|
|1|1|17 raw32x32 RGB tiles +4 geometry floats|Shared tile CNN|

State_layout/copies/trees still determine full-state length. Partial state is
21 floats by default, or225 with opt-in FORAGE_STATE_ONEHOT; partial pixels
remain52,228 floats/208,912 bytes. The partial input
is tile-major HWC, not a downsampled full frame. No hidden object coordinates,
global identity grid or nearest-target helper is passed. Only selected tiles
are copied directly from the cached atlas; a full frame is never built first.
The current straightforward partial path copies all17 selected slots per
observation. It does not yet cache unchanged attention-slot images.

Local slots0–7 are row-major neighbors with the player's center omitted.
Gaze slots8–16 are a complete row-major3x3. Off-board slots use the wall/tree
tile; overlapping views can duplicate a cell. State codes are the visible
tile IDs divided by12, including the player overlay. The last four values
are playerX/playerY/gazeX/gazeY normalized by19/14/19/14. Both modes get
identical explicit geometry metadata, not hidden semantic state.

## Actions and timing

The original/default attention build has **two discrete action heads**:
movement5 and gaze300. The newer FORAGE_GAZE_MODE choices offer directional5
or separate20X/15Y heads; see the state-ablation contract above.
Gaze is an absolute row-major cell index: x=action%20, y=action/20.
At each step, apply the policy's movement and next gaze center, then return
pixels/state for the resulting game and chosen center. Waiting or a blocked
move still refreshes the selected view. Looking consumes the same game step;
there is no extra free observation action or gaze reward. Reset sets gaze
to the newly spawned player, without consuming world RNG.

Full modes in this research build retain both action heads but ignore gaze
for observation/game semantics. This differs from the original one-head
baseline's policy/loss contract. Do not relabel old baselines or load their
checkpoints into this build. The normal build, without FORAGE_ATTENTION,
retains its original one-head contract and full observations. The new INI
partial switch requires the attention build; it does not retrofit old binaries.

## Pixel encoder

The17 tiles share one small CNN: valid4x4/s2 C8,3x3/s2 C16, flatten7x7x16
to16 features. This covers every pixel of each32x32 tile. Concatenate the
ordered272 tile features and4 geometry values, then linear/ReLU to H128.
The existing H128/L1 recurrent core feeds movement and gaze outputs.
Shared convolution kernels/derivatives come from the existing native Nature
implementation; new CUDA kernels only pack pixels/features and split the
gradient. Gradients from all17 tile uses accumulate into shared weights.

The encoder has49,576 parameters, the full partial-pixel policy137,896.
There is no full-map feature extractor, transformer layer, pretrained backbone,
CPU CNN or Python learner. The first tile graph is fixed; architecture search
controls for it are future work.

## Build and use

Work in `/home/claude/cnn`. The visible main core still lacks the earlier
shared-Nature/dynamic-observation integration. This prototype therefore builds
from an explicit retained compatible core snapshot, copying the **current main
environment and research files** into a fresh build tree. Its five-line core
patch routes the partial pixel input into the new encoder. No main core file
is overlaid. This local research dependency must be packaged/integrated before
a future5090 handoff; the command is not a clean upstream delivery yet.

```bash
# Build only, no GPU policy work. The output must be fresh.
bash research/build_forage_attention.sh \
  build/sprite-forage-speed-5060-20261015/source \
  build/forage-attention-FRESH

source ocean/connect4cnn/runtime_env.sh
cd build/forage-attention-FRESH/runtime

# Examples: use fresh run IDs. Supplied defaults are only8,192 decisions.
../puffer train --env.observation_mode=0 --env.partial_observation=1 \
  --base.run_id=happy-cat-1
../puffer train --env.observation_mode=1 --env.partial_observation=1 \
  --base.run_id=mystic-tree-2
```

Use partial_observation=0 for full-visibility controls. The supplied runtime
INI merges the environment, spatial CNN full-control recipe and attention
smoke defaults; no large state-tuned batch is silently imported. Checkpoint
shapes differ between modes. The existing completed local build is
`build/sprite-forage-attention-build-20261015/puffer`.

New bounded four-mode canary, when separately requested and GPU is idle:

```bash
.venv/bin/python research/run_forage_attention_canary.py \
  build/sprite-forage-attention-build-20261015 build/forage-attention-canary-FRESH
```

The supervisor requires the completed encoder check log and matching main
header, takes the shared GPU lock, and preserves commands/configs/times,
weights, parameter counts, finiteness and aggregate repeat/eager checks.
It does not restart completed allocations. Before changing encoder code,
rebuild and rerun the matching numerical checks rather than reusing this log.

## Verified scope and remaining work

Seven ASan/UBSan game/raster suites pass, including24,000 matched steps and
all1,500 skin/gaze-center combinations, exact full-frame crops and no hidden
cell leakage. New encoder outputs/all gradients match an independent direct
float64 CUDA oracle on smooth-positive and zero-weight fixtures at B1/3/8/64;
eager/graph repeats and device parameter preservation pass. These do not
qualify arbitrary mixed ReLU branches, finite differences, full learner/
optimizer derivatives or independent fresh-process numerical repeats.

All four mode trainings and six native eval processes finish on RTX5060;
partial-pixel aggregate repeat/eager summaries match. No per-action trace
claim follows from summary equality. At8,192 decisions, partial pixels measure
about4,505 process SPS versus1,080 full pixels with the supplied full encoder.
These are short, startup-sensitive smokes with changed observability and
architecture; they do not demonstrate a learning or Pareto advantage.

Next qualify/tune state learning with the restricted view, inspect learned
gaze behavior, then run matched pixel learning and broader numerical/evaluator
checks. Add a human-viewer gaze overlay and optional attention-slot caching
after checking their benefit. Keep fixed paired seeds, other eval worlds,
all failures/declines and original baselines. No seed search.

## Approximate equal-wall-time learning comparison

`research/run_forage_attention_budget.py FRESH_OUTPUT` performs the requested
local5060 comparison. It targets11.246 seconds per training: the measured mean
of the two most recent8,388,608-decision full-state cleanup runs. This is the
recent short baseline, not the older33M-decision state experiment or the
ten-minute full-frame pixel experiment.

The allocation contains a fresh reproduction of the original one-head state
baseline, plus partial state and partial pixels using the two-head attention
build. The environment/reward settings and tuned learner scalars come from
that state baseline. Partial state retains256 agents/H64/MB2048; pixels use
8 agents/H8/MB64, four CPU environment threads become two, and H128/L1 stays
fixed. Pixel batching differs for memory/runtime practicality; this is an
equal-cost system pilot, not a controlled observation-only ablation.

One fresh speed calibration per partial mode determines a fixed whole-rollout
decision budget using measured steady SPS and estimated startup. Calibration
scores do not select budgets, architectures or seeds. Calibration compute is
retained separately from the main training cost. Actual process duration and
its deviation from11.246 seconds must be reported; approximate targeting
does not establish exactly equal wall-clock budgets. The supervisor does not
retry a policy when its score or runtime disappoints.

Final scheduled checkpoints receive32 sequential one-slot episodes at each
world offset420/1337, action seed7, with512-step evaluation caps. World RNG is
used only during reset, so the sequential generated boards do not depend on
policy actions. The evaluator retains aggregate outcomes, not independently
audited per-episode start/action receipts. Partial pixels get one additional
same-configuration aggregate repeat. All checkpoint hashes, finite weights,
requested/resolved recipes, native metrics and monotonic process receipts
are retained. The original one-head baseline must never load into the
two-head attention build. A single training seed cannot qualify a paper claim.

The completed short comparison is documented in
`research/results/sprite-forage-attention-budget-5060-20261015/README.md`.
After seeing its duration, Kinvert requested about **five minutes per partial
policy**. `research/run_forage_attention_long.py FRESH_OUTPUT` starts both
policies from scratch, keeps the short run's mode-specific learner/config,
fixes budgets using measured speed only, and saves eight evenly spaced
checkpoints. No completed short training is extended or restarted. Final
evaluation uses the same64 development episodes/policy. The full-state
97.65625% historical control is retained as a reference, not relabeled as a
five-minute training. Report actual time, decision counts and native/process
SPS; do not assume an exact300 seconds or count calibration as a free cost.

Both requested learning allocations are now complete and artifact-audited.
Read the [five-minute results](results/sprite-forage-attention-five-minute-5060-20261015/README.md).
Partial state reaches61.33% collection perf at301.8 seconds; partial pixels
reach5.47% at303.1 seconds. The pixel run's final377.1-second checkpoint
falls to3.52%; retain that decline. Its budget overrun is explicit, and the
supplemental near-five-minute checkpoints are selected by native monotonic
save time alone. Both scores are weak versus the97.66% full-state reference;
no architecture, useful-gaze or Pareto advantage is established. Sixteen
scheduled weights and all192 actual episode executions pass artifact audit.
No automatic extension or further sweep follows from these results.
