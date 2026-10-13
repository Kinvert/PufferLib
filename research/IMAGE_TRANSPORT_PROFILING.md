# Full-frame transport: next selective profiling work

October13. Investigation plan only; no transport, observation dtype, kernel,
learner or CUDA/system change is made here. This does not authorize another run.

The captured four-actor Flappy full-RGB profile assigns about63% of profiled work
to Copy and about9% to Model. It is a short plumbing workload, not a sustained
bandwidth measurement or a profile of Nature. It nevertheless makes observation
movement a better profiling lead than assuming the small CNN dominates runtime.

The current native path is explicit:

- `src/pufferl.cu:env_setup` allocates CPU observations using
  `cudaHostAllocPortable`; these buffers are already pinned.
- `cpu_upload` copies `n * OBS_SIZE * sizeof(obs_t)` CPU observation bytes per
  buffer update. The CNN environment headers define `obs_t` as float.
- The worker's Copy counter includes observation uploads and action downloads.
  It is not an isolated H2D bandwidth counter. The worker synchronizes its stream
  before CPU stepping; Env and Model counters describe different work.
- `train_epoch_gpu` transposes the full rollout observations into the learner
  layout. Rollout observation copies and this transpose are additional GPU work;
  they must not be silently attributed to the actor's Copy counter.

| Game | RGB float32 bytes per frame |
|---|---:|
| Connect4,576x672 |4,644,864|
| Pong,640x620 |4,761,600|
| Flappy,640x420 |3,225,600|
| Breakout,330x576 |2,280,960|
| Snake/Maze,616x616 |4,553,472|

These are scalar payload counts, not measured transfer rates. At16 actors and
horizon16, one Pong rollout observation tensor alone is1,218,969,600 bytes.
Preallocated/transposed copies, model activations and optimizer state add to it.

Before changing anything, profile representative matched Nature/compact runs on
the execution hardware. Separate raster generation, observation H2D, action D2H,
stream wait, rollout copy/transpose, encoder and learner work. Record actual byte
counts and CUDA-event intervals, and distinguish startup/checkpoint writes from
sustained updates. Freeze the same per-game learner, pixels and seed assignments;
use the shared reservation and refuse competing workloads. Existing512-decision
SPS cannot select the main learning budgets or establish a pipeline speedup.

Candidate changes, after that evidence:

1. Remove a demonstrated redundant copy or change a layout only if a retained
   reference path verifies exact inputs, outputs, gradients and checkpoints.
2. Investigate compact frame transport/storage with GPU conversion on consumption.
   An8-bit RGB payload is one quarter of float32 RGB, but ordinary `/255` conversion
   is not automatically bit-identical to the current semantic float palette.
   Either declare and validate a new shared pixel contract, or demonstrate exact
   reconstruction. Both Nature and our CNN must use the same transport/input path.
3. A palette-index transport could reconstruct the current synthetic renderers'
   exact float colors if every visible color is represented. Verify palette
   coverage, spatial layout and byte-identical decoded frames on every drawing.
   This is specific to these renderers, not a general camera-image representation.
4. Consider overlap/fused conversion only after proving stream dependencies,
   buffer lifetimes, graph/eager/repeat equality and correct episode resets.

Keep architecture comparisons separate from pipeline comparisons. A common
pipeline improvement might raise everyone's throughput; it does not demonstrate
that our CNN is better. Any changed path needs fresh paired learning/time evidence
before it can extend the paper's frontier claims. The immediate campaign priority
remains useful, matched full-image controls and a measured5090 runtime pilot.
