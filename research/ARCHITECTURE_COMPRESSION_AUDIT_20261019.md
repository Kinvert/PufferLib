# Architecture, compression and research-direction audit

October 19, 2026. Source inspection and primary-literature research, not a new
training result. Checkout HEAD: `9481b9f972d454bf396b2f91b5f1f5b20216b130`.
Existing uncommitted core/profiling/Pong work was preserved. No GPU work, dataset
generation, dependency installation, commit or push accompanied this audit.

## Verdict

We have a useful native experiment system and credible early efficiency leads.
We do **not** yet have evidence for a generally superior CNN, a qualified
pretrained backbone, or a novel compression algorithm. The current sprite search
explores conventional small CNNs. It cannot discover most of the representation
changes now requested, regardless of how many trials it runs.

The best next direction is **cheap visual compression that preserves control
information**, with three independently measured routes:

1. Learn or compute a compact representation from full human-visible game frames.
2. Pretrain a tiny visual encoder, then test adaptation, freezing and learner reuse.
3. Remove CUDA/memory overhead without changing the representation.

An autoencoder is one control, not the assumed winner. A small reconstruction
error can conceal losing the one dot that distinguishes a safe object from a
deadly one. Conversely, our existing tile encoder already has a very small output:
the missing improvement is avoiding expensive repeated work upstream of it.

Read the [compression/backbone design](PIXEL_COMPRESSION_AND_BACKBONE_DESIGN.md),
[annotated research reading list](PIXEL_COMPRESSION_LITERATURE_20261019.md), and
[ranked implementation/experiment plan](COMPRESSION_EXPERIMENT_PLAN.md).

## What the current code actually searches

Evidence: [sprite grammar](recipes/sprite_pixel_architecture.ini),
[Flex construction](../ocean/connect4cnn/flex.cu),
[Flex2 construction](../ocean/connect4cnn/flex2.cu),
[sprite dispatch](sprite_packed_dispatch.cu),
[native feedback bridge](../src/protein_feedback.cu).

| Capability | Current sprite overnight search | Elsewhere / missing |
|---|---|---|
| Stage count | 1–3 | Flex2 supports 1–4 |
| Stage channels | 8/16/32 | Flex2 also 64; neither admits 1/2/4 |
| Kernel | Stage 1: 1–8; later: 1–5 | Dense square kernels |
| Stride | First: 4/8; later: 1/2/4 | Flex constructor accepts first 1/2 too; INI restricts it |
| Residual work | 0/1 extra 3x3 conv per stage | Flex2 allows 0–2; not IMPALA's two-conv block |
| Pooling | None / 3x3 stride-2 max / average | Flex2 also adaptive readout |
| Readout | Flatten or global average | Flex2 adds 2x2/4x4 spatial average |
| Projection | 16/32/64/128 | Recurrent core fixed H128/L1 |
| Activation | ReLU | Flex2: SiLU, exact GELU, learned PReLU/rational |
| Dilation | 1 | Flex2: 1–4 |
| Shared tile CNN | Fixed family-6 reference | Its widths, tile features and topology are not searched |
| Downsampling before CNN | Absent | No resize/filter/latent-budget axis |
| Grouped/depthwise/partial conv | Absent | Proposed only |
| SwiGLU | Absent | Recorded in ENCODER_RESEARCH_20260926.md |
| Pretraining/freeze/latent replay | Absent from these paths | Dataset document is a design, not an implemented trainer |
| Multiscale or change-reuse graph | Absent | No general compression/cache interface |

The sprite dispatcher admits IDs 0/2/4/6/7, **not Flex2 ID5**. The native bridge
also explicitly requires encoder4/H128/L1. Changing only an INI cannot turn this
campaign into Flex2 or backbone discovery. The supervisor assumes 18 dimensions
in its history protocol. Any expansion needs versioned bridge/metadata/numerical/
checkpoint support, not just another field. Flex2's documented GPU qualification
remains pending; this audit found no new acceptance evidence to upgrade it.

**There is plenty of numerical variability but limited structural diversity.**
At depth one, there are already 2,304 nominal active combinations of width,
kernel, stride, pool, residual, projection and GAP. This counts configurations,
not unique functions. Twenty-four proposals cannot map even that small slice.
Adding every missing feature to one GP would worsen coverage.

Two concrete search problems:

- At depth one, ten of the eighteen coordinates describe absent stages. The GP
  still sees them. Use fixed-topology panels or a conditional representation.
- `sprite_overnight.run` flags duplicates only **after training**, comparing the
  full policy dictionary. Inactive-coordinate differences can hide equivalent
  graphs. The separate encoder5 native-sweep deduplication is not this path.
  Canonicalize executed operations and log raw proposals, equivalence and any
  reuse explicitly; do not silently change the meaning of historical trials.

## The search objective is not yet the whole learning frontier

Scope clarification from the implementation-handoff review: this section audits
`sprite_overnight.py`. The separate full-frame workflow already has curve-based
feedback in [image_curve_feedback.py](image_curve_feedback.py), bindings and
panel-budget tooling. Reuse that work when integrating new compression models;
do not describe endpoint-only feedback as a repository-wide limitation.

[aggregate()](sprite_overnight.py) averages **final-checkpoint** performance over
skins/replicas, then games. Cost is the sum of training-process durations, charged
once per job. This is a defensible endpoint score/cost observation, but the eight
earlier curve points do not influence proposal feedback. A candidate that learns
fast and later declines can be missed. `worst_game_utility` is recorded but is
not the scalar score fed into PROTEIN.

The [45-observation 5090 report](results/learning-continuation-5090-20261009-prefix45/README.md)
illustrates the distinction: the leading aggregate architecture cost about42%
more than quality-reference and regressed Flappy and Maze. Nine of33 additional
proposals were effective duplicates. Those results establish a new descriptive
tradeoff, not a shared architecture dominating across tasks.

For a future version, retain the full curves and use predeclared early-time
curve utility for discovery, plus per-game regression/coverage reporting. Treat
each completed **candidate**, not its eight correlated checkpoints, as an
independent optimizer observation unless the optimizer explicitly models fidelity.
Keep the final confirmatory analysis separate from adaptive search scores.

The [sprite recipe](recipes/sprite_overnight.ini) uses seven fixed replicas and
all six games on every suggestion. That is robust but expensive discovery:

| Work excluding gates | One candidate | Four controls +24 proposals |
|---|---:|---:|
| Trainings | 42 | 1,176 |
| Evaluations: 8 checkpoints ×3 skins per training | 1,008 | 28,224 |
| Assigned evaluation episodes,16 each | 16,128 | 451,584 |

All28 candidates fitting12h would average under1,543 seconds/candidate,
including its evaluation and other overhead; actual gates consume part of that
time too. Controls run first, including slow IMPALA/Impoola. The recorded canary
cost projection covers training, not a reliable total campaign forecast.
This is a cap, not evidence24 trials will finish. Repeated process startup and
model reload can dominate evaluation; measure before optimizing CNN kernels alone.

The preparation also fixes `async=0`,32 agents, horizon32 and minibatch32. These
are matched settings, but not evidence for best attainable throughput on5090.
Calibrate per-game common learner recipes once before freezing the experiment;
do not retune each architecture mid-comparison or change a running packet.

## Current visual work and misleading shortcuts

[sprite_common/sprite.h](../ocean/sprite_common/sprite.h) expands **8x8 art** to
32x32 tiles, adding textured backgrounds; Forage has separate16x16 art and an
imported-art mode. [attention.h](../ocean/sprite_common/attention.h) writes18
tiles plus4 geometry floats. Packed layout is192 high ×96 wide RGB.
Both player and gaze see pixels; movement and gaze remain learned.

The shared tile encoder uses `4x4/s2 C8 -> 3x3/s2 C16 -> 16 features/tile`,
then concatenates18 ordered embeddings and4 geometry values before projection.
That is55,296 image scalars becoming288 features:192× fewer scalars **after**
visual computation. It still copies pixels, materializes convolution patches,
and executes forward/backward during learning. The16-feature bottleneck is
already here; a pretrained/frozen/cheap stem could make it useful for speed.

The six-game renderer's block enlargement makes32→16 or32→8 a useful diagnostic
of redundant artwork, not our primary general-game solution. It does **not** prove exact equivalence of final images:
floor texture has finer structure. Forage imported art and future animation
need separate tests. Do not infer every game has8x8 intrinsic information.

Do not confuse these different claims:

- Smaller human display versus smaller policy sensor input.
- Fewer image scalars versus fewer bytes versus fewer expensive learner passes.
- Cached rendering versus cached neural features.
- General pixel encoder versus a tile-grid-specific acceleration.
- Frozen encoder architecture versus frozen encoder weights.
- Architecture transfer across games versus one shared set of pretrained weights.

## Speed opportunities without assuming the answer

The [state/CUDA audit](STATE_AND_CNN_CUDA_AUDIT.md) remains the detailed backend
reference. Reinspection confirms explicit im2col, per-GEMM stream changes,
separate activation/reduction launches and serial CNN gradients. Both state and
CNN already use native CUDA, cuBLAS, CUDA graphs and the same MinGRU/optimizer.

Priority by plausible leverage, not measured speedup:

1. **Early representation compression:** reduce rollout/patch/learner work
   together. H2D savings require compression before transfer or a different
   transport format. Sprite-atlas reduction is only a specialized diagnostic.
2. **Frozen prefix + stored features:** remove repeated encoder learning work,
   including its optimizer matrices. Actor encoding still costs time.
3. **Backend correctness and unnecessary movement:** qualify the already-noted
   upstream per-stream cuBLAS workspace change, remove dense im2col and avoid
   pixel/geometry repacking where strides can express the same layout.
4. **Implicit/direct convolution and fusion:** remove explicit patch traffic;
   compare actual batches and all backward paths. Apply improvements to Nature
   too, so an architecture win is not merely an intentionally weak baseline.
5. **Muon and overlap:** measure before batching optimizer matrices or overlapping
   CNN gradients. IMPALA shared scratch makes naive overlap unsafe.

No valid current kernel-time breakdown establishes which dominates packed sprite
training. No new SPS or speed multiplier is claimed. Before profiling, follow
`/home/claude/gpu-profiling/{README.md,COMMANDS.md,projects/cnn.md}` and verify
actual CUDA events/counters rather than accepting NVTX-only traces as GPU timing.

## Where a paper contribution could exist

Small CNNs, pooling, pretrained encoders, freezing and stored features all have
substantial prior art. Combining existing switches is useful engineering; it is
not by itself a new learning method. See the annotated literature for precedents
and negative results, including limited held-out Atari transfer.

Plausible contributions to test are:

- A reproducible **control-information versus compute frontier**: what visual
  details survive each compression budget, and when that predicts RL failures.
- A tiny shared visual encoder with low-cost adaptation that improves end-to-end
  learning time across games, counting pretraining and retaining scratch controls.
- A native deterministic system that reuses frozen visual features efficiently,
  together with an honest grid-specialized versus arbitrary-image comparison.
- A constrained architecture constructor that consistently discovers useful
  per-environment or transferable frontiers under a measured search budget.

These are proposed contributions, not novelty or SOTA findings. The six related
sprite games provide controlled diagnostics, not enough alone to claim Atari or
robotics generalization. Use existing non-sprite games as development transfer
checks; reserve genuinely untouched games/assets for eventual external claims.

## Ranked next actions

1. **Build a full-frame representation-preservation and compression panel.**
   The target is human-visible rendered games, typically around640x480 with
   their native aspect ratios. Test a general image-based stem against declared
   fixed-resize controls; sprite-atlas reductions are diagnostics, not the main
   deliverable or evidence for arbitrary games.
2. **Make discovery optimize and expose useful learning curves.** Avoid buying
   more18-dimensional endpoint search before addressing inactive coordinates,
   duplicates, weak-task regression and campaign cost.
3. **Prototype one tiny pretrained visual prefix, frozen and fine-tuned controls.**
   Compare pixel-space visual labels, temporal features and a reconstruction
   control. Measure whether freezing/reuse helps rather than assuming it does.
4. **Qualify backend improvements on matched baselines.** Repeated unprofiled
   timing and numerical checks precede claimed speedups.
5. **Expand graph families deliberately.** Full-frame bottleneck and multiscale
   stems first; qualified Flex2 activations/readouts next; grouped/depthwise,
   partial conv and SwiGLU only when they earn their runtime cost.
6. **Confirm across independent data/tasks/seeds.** Seven fixed replicas, separate
   eval worlds, fixed selections and uncertainty on curves. No best-seed envelope.

Implementation details and acceptance gates are in the linked experiment plan.
This audit changes documentation only; it does not modify the prepared5090 run.

### Scope correction from Kinvert during this audit

The main target must work with real human-visible games. Do not prioritize
recovering tiny source sprite artwork as the solution. The primary experiment
must start with real rendered image frames and learn or compute a compact
representation from those pixels. Preserve small visible cues, arbitrary object
locations and the actual scene. Existing sprite/tile results are controlled
diagnostics; atlas caching and exact cell boundaries cannot underpin a general
pixel-learning claim. This clarification governs the ranked plan above and the
backbone design, including earlier suggestions of cheap sprite-only experiments.

## Audit completion and evidence limits

| Requested question | Evidence/deliverable |
|---|---|
|Enough architectural variability?|Constructor/INI/dispatch/bridge inspection above; implemented versus reachable features separated|
|Are we innovating?|Prior-art mapping and bounded contribution proposals; no unsupported novelty claim|
|How could speed improve?|Representation, frozen-feature reuse, transport and CUDA paths; formulas separated from timings|
|How to shrink dimensionality substantially?|Full-frame spatial/token budgets and current-buffer arithmetic in the design|
|Autoencoder or pretrained backbone?|Reconstruction, visual labels, temporal objectives and distillation; scratch/frozen/adapted controls|
|Real human-visible games?|Full-frame requirement supersedes sprite-only shortcuts throughout the next plan|
|Research saved for future work?|Four linked Markdown documents plus research index, dataset and priorities pointers|

Checked local document links, table arithmetic, Markdown fences and scoped
whitespace. Source findings are inspection findings, not newly executed GPU
tests. Historical learning evidence is explicitly linked and retains its scope.
The28-candidate workload estimate is derived from the recipe, not a runtime
forecast. Literature notes distinguish direct-control evidence from robotic
imitation, classifier inference and video-language work. Neither an external
search nor a repository audit proves absence of all prior art. Actual performance
and representation quality remain experimental questions for the next phase.
