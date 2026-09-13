# Configurable native CNNs with PROTEIN

Research only, 2026-09-13. Source inspected at `2c25bf6707dbb62a3c8f8879e3f13274eed2e5c5`. This note proposes changes; none of the interfaces or INI keys proposed below has been implemented. The preceding commit preserves the validated IMPALA/Impoola implementation and smoke evidence. The long reference comparison remains separate and pending.

## Decision

Use PufferLib's existing **encoder → MinGRU → decoder** architecture. Expose CNN construction settings through the INI and let native PROTEIN search those settings plus training budget against measured performance and training time. Keep learner settings, observation representation, and recurrent size/layers fixed for the first search. Expand environments and seeds after inspecting the first frontier, as requested by Kinvert.

Prefer an `encoder` selector to a whole-agent `nn_type`. A CNN and a recurrent core occupy different slots and can be used together. Choosing an LSTM is a separate core implementation task, not a prerequisite for CNN search.

## What exists in this checkout

| Path and current location | Verified behavior | Implication |
|---|---|---|
| `src/ini.h:504`, `puf_ini_load_env` | Loads default INI, overlays environment INI, applies CLI overrides. | New environment-specific keys can be declared in its INI without changing the parser. |
| `src/ini.h:393`, `puf_ini_put` | Overrides must name existing sections/keys. | Adding a sweep range alone does not declare the actual CNN setting. |
| `src/pufferl.cu:1780`, `create_pufferl` | Reads `policy.hidden_size` and `policy.num_layers`; calls `build_arch` at line 1947 before creating weights. | This is the startup configuration handoff. |
| `src/algo.cu:23`, `Encoder` | Function table for forward/backward, initialization, weights, and rollout/train allocations; includes input/output dimensions and activation struct size. | Reuse the existing interface instead of changing PPO. |
| `src/algo.cu:911`, `arch_forward` | Encoder feeds network, which feeds decoder. Backward traverses the reverse chain. | All components already train jointly. |
| `src/algo.cu:984`, `build_arch` | Default linear encoder, optional environment override, MinGRU core, decoder. | There is no current `nn_type`, `encoder_type`, or LSTM selection here. |
| `src/ocean.cu:65`, `create_custom_encoder` | Environment-specific factories are selected by compilation macros. | Keep CNN selection in the existing custom-encoder branch. |
| `src/pufferl.cu:2565`, `run_sweep` | Discovers `[sweep.<section>.<key>]`, samples values, passes the resulting INI to a fresh native training process using the same executable. | Runtime-configurable CNNs can use the native scheduler without recompiling every suggestion. |
| `src/protein.cu:923` onward | Fits score and log-cost models and constructs Pareto candidates. | Reuse PROTEIN's objective/search rather than replacing it with an SPS scalar score. |

These are local source locations at the pinned commit, not claims about every PufferLib version. Earlier research notes were written against older revisions.

The current `policy.num_layers` counts MinGRU layers, not convolution layers. `policy.hidden_size` sets core width and also the encoder's output projection width. CNN channel/depth controls need distinct names. Core width remains fixed initially so it does not silently alter projection cost during the CNN search.

Changing core type is larger than changing an enum: `arch_reg_train` and `arch_reg_rollout` allocate `sizeof(MinGRUActivations)` directly (`src/algo.cu:933–955`), and the trainer allocates recurrent states as `[layers, agents, hidden]` (`src/pufferl.cu:1957–1989`). An LSTM's state and training implementation would need a separate design. No such change is recommended for the first sweep.

## Smallest useful implementation path

1. Thread the existing policy configuration from `create_pufferl` into `build_arch` and the custom-encoder factory. Keep default behavior when no encoder selection is supplied. Other environment factories can retain their current signatures; the dispatcher only forwards the new settings to the configurable CNN branch.
2. Give the chosen encoder access to construction settings until `create_weights` runs. A small optional borrowed `Dict*` on `Encoder` is one possible seam; the CNN then copies validated scalar settings into its own weight/shape metadata. Preserve that pointer when a factory replaces the encoder function table. Resolve all settings before tensor registration. No mutable global CNN configuration and no INI lookup during forward/backward.
3. Keep CNN settings, stage descriptions, kernels, allocations, and factory dispatch under `ocean/connect4cnn/` initially. Generalize the observation dimensions when a second pixel environment is added. Avoid a new layer framework in `src/` before the limited family proves useful.
4. Add a small generic way to choose which inherited sweep dimensions are active. This belongs in sweep orchestration, not in the PROTEIN mathematical optimizer. An optional `[sweep] parameters` list is the preferred sketch: omitted means the existing behavior; present means use only the listed declared sweep sections and reject typos.

The configuration pointer is construction-only, borrowed from a live INI. INI section arrays can move when sections are added; do not retain pointers into them as runtime network state. The CNN must copy settings into each weights object, including the async actor copy. Parameter/state metadata must be owned per architecture rather than shared between policies. This lifetime contract needs a focused test if that seam is implemented.

Expected core touch points:

| File | Proposed scope |
|---|---|
| `src/pufferl.cu` | Forward policy settings at the existing architecture construction call; filter active sweep dimensions. |
| `src/algo.cu` | Small construction interface/configuration handoff; reuse existing encoder operations and allocation order. |
| `src/ocean.cu` | Connect4CNN factory receives configuration and selects the requested encoder. |
| `src/protein.cu` | No change needed for the initial numeric search; understand the effective final-only observation behavior described below. |
| `src/ini.h` | Existing scalar/string storage and section overrides are sufficient; no parser rewrite needed. |

This is a proposed boundary, not a promised line count. Default architecture/checkpoint behavior must remain unchanged for existing environments. No CNN-specific channel fields belong in the general learner `Hypers` struct.

## First search space

Start with one bounded family: three fixed downsampling stages, channels proportional to `[1,2,2]`, zero to two residual blocks per stage, and flatten versus global average pooling. Keep the existing reference encoders unchanged as controls; implement the experimental family separately. Width/depth variations become named, resolved architectures in the results.

| Proposed key | Initial choices | Meaning |
|---|---|---|
| `policy.encoder` | Fixed `cnn` for this search | Selects the experimental encoder constructor. Named reference selection can be added alongside it. |
| `policy.cnn_channels` | 8, 16, 32, subject to memory validation | Base stage width; actual channels are `[c,2c,2c]`. |
| `policy.cnn_blocks` | 0, 1, 2 | Residual blocks in each stage; stage stem convolution/downsampling remain even at zero. |
| `policy.cnn_global_pool` | 0, 1 | Flatten or GAP readout. |
| `train.total_timesteps` | Bounded pilot range | PROTEIN's existing training-budget dimension. |

This gives 18 structural choices before budgets, a useful small beginning. Hold 3×3 kernels, pooling alignment, image size, normalization, precision, optimizer/learner settings, and core configuration fixed. Broaden stage-specific widths, kernels, strides, and block families only after this loop works.

Illustrative INI vocabulary, **not runnable in the current code**:

```ini
[policy]
encoder = cnn
hidden_size = 128
num_layers = 1
cnn_channels = 16
cnn_blocks = 1
cnn_global_pool = 0

[sweep]
parameters = policy.cnn_channels,policy.cnn_blocks,policy.cnn_global_pool,train.total_timesteps
metric = perf
metric_distribution = linear
goal = maximize
gpus = 1
downsample = 5

[sweep.policy.cnn_channels]
distribution = uniform_pow2
min = 8
max = 32
scale = auto

[sweep.policy.cnn_blocks]
distribution = int_uniform
min = 0
max = 2
scale = auto

[sweep.policy.cnn_global_pool]
distribution = int_uniform
min = 0
max = 1
scale = auto
```

The complete experiment configuration must also declare the selected training-budget range and fixed learner recipe. The first candidate comes from the base values in the INI, so those values must be valid and inside the search ranges. A later string family selector can choose Nature/IMPALA/Impoola/custom manually, but it is not automatically a categorical PROTEIN dimension.

The existing reference files all define `create_connect4_encoder` and are mutually excluded with macros. Compiling them together for runtime selection needs distinct factory names and explicit dispatch. Do not just include all three files and expect them to coexist. The experimental implementation should store a bounded, resolved stage plan, finalize its metadata arrays before registering tensor pointers, and preserve the existing output contract `[batch, hidden_size]`.

## Sweep findings that affect the plan

**Locking learner settings needs explicit filtering.** The environment INI overlays `config/default.ini`; it does not replace it. Default learner/core/vector sweep sections remain active. `run_sweep` consumes every `sweep.*` section and has no enabled/allowlist check. Setting an ordinary learning-rate value does not freeze its inherited sweep. Setting `min=max` is also unsuitable: `space_normalize` divides by the transformed range (`src/protein.cu:64`). A research-only alternative is an isolated configuration directory containing just the desired sweep sections, but a small optional native allowlist is cleaner for the intended workflow.

**PROTEIN can search numeric shapes now, but not arbitrary block strings.** The native parser accepts `uniform`, `int_uniform`, `uniform_pow2`, `log_normal`, and `logit_normal` (`src/pufferl.cu:2593–2609`); it rejects `categorical`. Binary readout and ordinal depth/width fit the initial search. Assigning unrelated families integer IDs makes them run, but also imposes artificial distances on PROTEIN's numerical model. Initially run families separately and compare their measured frontiers, or stay within the single family above. A genuine categorical/conditional search representation is a later optimizer task.

**Resolved architectures can duplicate.** Integer/power-of-two sampling rounds at unnormalization, while the scheduler retains the original normalized sample (`src/protein.cu:74`, `src/pufferl.cu:2719–2735`). Different suggestions can therefore construct the same architecture. Record resolved dimensions plus training budget and seed; do not describe every suggestion as a new network. Inactive knobs across arbitrary families would make this worse. Avoid such knobs initially. Do not discard different budgets or intentional seed repeats as duplicates.

**Multiple returned curve points overwrite one another in the current integration.** `run_sweep` observes each returned point with exactly `job->sample` (`src/pufferl.cu:2827–2830`). `protein_sweep_observe` finds matching parameter vectors and replaces their score/cost (`src/protein.cu:632–643`). Thus those calls leave the last point, not distinct training-budget observations. The worker provides `step_points`, but this loop does not use them to remap the cost dimension. This is a static code finding, not a reproduced GPU test or a fix made here. Initially retain the timesteps sweep and treat each completed trial as contributing its final performance/time pair. Keep the native `downsample=5` to preserve five-point binned curves in artifacts; do not claim the optimizer retains all five. Setting `downsample=1` is an alternative, but also reduces persisted metric curves to one binned point (`src/pufferl.cu:3373–3389`), and that point is not necessarily the final metric. Capture the printed final trial result separately. A future curve-integration fix needs its own test and allowance for budget-dependent learning-rate schedules; a checkpoint from a long schedule is not necessarily a separately trained short-budget policy.

**Budget is different from a locked learner hyperparameter.** `sweep.train.total_timesteps` is specially designated as the cost dimension (`src/pufferl.cu:2627`). Retain it so PROTEIN explores how much training different CNNs need, while learning rate, batching, update ratio, core, etc. remain fixed. Actual cost observations are measured time; step count is not being substituted for wall time. Do not replace the accepted Pareto objective with a single win-rate threshold or SPS objective.

**Native cost is adjusted training uptime.** The worker takes cost from `uptime` (`src/pufferl.cu:3192`). Timing begins after trainer construction (`2085`), and training graph capture/instantiation time is subtracted (`1622`; GPU-environment rollout capture also adjusts it at `1287`). It is not full process wall time. Use this native cost for PROTEIN as requested, and retain a separately labeled process-time measurement for practical end-to-end comparisons. Compilation, search overhead, and subsequent evaluation should also be identifiable rather than silently called training time.

**Native search performance is a training metric.** `sweep.metric=perf` selects logged `env/perf` for Connect4 (`src/pufferl.cu:2840`, `3192–3211`). Ordinary non-selfplay sweeps do not automatically score every candidate on the held-out evaluation used by `compare.py`. Use native training metrics for the initial discovery sweep, then validate promising points through the existing independent evaluator with exact saved architecture settings. Do not mix training perf and held-out wins in one frontier. The first reference comparison's tables are held-out results, so they cannot simply be appended as native search observations without reconciling the metric/protocol.

**Cost limits and early stopping are not hard timeouts.** The scheduler observes a worker after it exits. `max_suggestion_cost` filters predicted costs, and `early_stop_quantile` is not wired into live training termination here. Bound model shapes and training budgets up front, and arrange a separate process deadline for a unattended sweep if needed. Do not change those mechanisms merely to start the first small search.

## Correctness constraints for variable shapes

- Validate layer counts, output dimensions, image coverage, and estimated memory before GPU allocation. Arbitrary strides can drop a board column even when all tensor shapes are legal; Nature's initial input-size correction demonstrated this.
- Allocate stable stage/parameter metadata arrays before `alloc_register`: it retains pointers to tensor pointers and shape arrays. Resizing the metadata storage afterward would invalidate registrations.
- Preserve equal registration order/size for parameters and gradients, deterministic pooling ties, fixed reduction order, and independent rollout/train scratch. Construct once, allocate once, capture graphs, then train with a fixed architecture.
- The allocator aligns individual buffers to 16 bytes, while Muon walks flat parameter offsets by element count (`src/pufferl.cu:234–276`, `src/algo.cu:1159`). Arbitrary odd widths/bias lengths can introduce gaps that are not represented by that flat walk. Start with aligned widths and validate every registration; arbitrary dimensions would require a separate allocator/optimizer audit, not just another sweep range.
- Current checkpoint files contain raw FP32 weights without architecture metadata (`src/pufferl.cu:1722–1752`). Equal parameter counts do not prove equal architectures, and the loader does not verify an architecture fingerprint. Save the resolved specification/hash beside each checkpoint and use exact run IDs/configs for evaluation. Avoid a shared `latest` across architecture families. A sidecar check in the research evaluator can protect the first workflow without changing the core checkpoint format.
- Hold historical/selfplay opponents off initially. The existing mixed-policy configuration specifies historical hidden size/layers, not an independently selected historical CNN. Supporting unlike encoder architectures in one match is extra work.
- Test the configuration handoff through native training and evaluation, unchanged default state behavior, all allowed shape combinations, reference parity where configurations match, and same-seed smoke repeatability before searching. Do not rerun timing workloads on the occupied GPU during this research task.

## Sequence when implementation is authorized

1. Add only the configuration/factory handoff and exact architecture identity. Prove the existing default path is unchanged.
2. Add the bounded experimental family and validate the 18 structural choices. Reuse the established numerical harness and optimizer/registration checks.
3. Add active-dimension selection, print the resolved search dimensions, and assert that the intended learner/core settings remain fixed.
4. Run a bounded native PROTEIN smoke sweep with varied training budget; confirm the effective final-only observation behavior, score/cost semantics, saved curve summaries, and architecture identity for every trial.
5. Run the first discovery sweep, retain non-dominated performance/time points, and independently evaluate promising candidates and reference controls. Then extend seeds/environments and expand the search space based on evidence.

No GPU profiling, training, system/package changes, or new architecture implementation was performed for this note. The local search index refresh is deferred while the timing comparison occupies the machine; use exact file reads/`rg` for these new findings until it is refreshed.
