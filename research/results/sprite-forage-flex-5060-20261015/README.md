# Sprite Forage: our locked Flex quality CNN, RTX 5060, October 15

The custom-CNN run completes one new 32,768-decision native training and all
four assigned initial/final evaluations, with no failures or automatic retries.
Its sources include the main environment's changed-cell reset rendering.

Our locked quality graph is one padded 7x7/s4 C16 convolution, flatten/linear
64, linear 128, then the H128/L1 policy. The convolution output is 120x160x16.
Full 640x480 RGB float32 input; random initialization, no pretrained backbone.
Depth is one: layer-2/3 INI coordinates are inactive, with no residual/pooling.
The full policy has 19,721,472 parameters. The sizeable flatten projection
means this is still a large network at this input resolution.

| Measurement | Our Flex quality | Nature from immediately preceding pilot |
|---|---:|---:|
| Decisions | 32,768 | 32,768 |
| Process seconds | 89.842 | 147.219 |
| Process SPS | 364.73 | 222.58 |
| Native adjusted-average SPS | 366.85 | 223.38 |
| Native final-bin SPS | 334.78 | 224.71 |
| Parameters | 19,721,472 | 34,991,136 |
| Untrained eval perf | 2.34375% | 6.25% |
| Final eval perf | 4.6875% | 9.375% |
| Final training-bin perf | 6.25% | 0% |

Process SPS is decisions divided by monotonic training-process time, including
startup/checkpoint writes. Native adjusted times use the trainer's different
scope and CLOCK_REALTIME; both uptimes fall inside their process durations.
The observed process-rate ratio is 1.63865. This is one short paired setup,
not a qualified general speedup/frontier claim. Smaller checkpoint files also
reduce process cost; this is not an isolated convolution benchmark.

## Matched inputs

Fixed policy seed 7, training world offset 0, drawing 0, 8 agents, one buffer,
two threads, horizon 8, minibatch 8, LR .001, episode cap 512, starting game
rewards, H128/L1. Seeds are not swept. The launcher verifies matching env,
vec and train INI sections, shared default INI, game header and every native
core source file against the completed Nature packet. Only CNN coordinates
and necessary output paths change. Native core files in main were not edited;
the frozen compatible core and existing research adapter are used in build.

An explicit parameter-count assertion verifies Flex selection rather than a
silent Nature fallback. GPU model remains the local RTX 5060. The shared
benchmark lock and idle checks guard the native processes; no intentional
GPU work overlaps. Existing CUDA/NCCL only, no dependency/system/Torch change.

## Evaluation

Same initial/final assignments for each architecture: 16 sequential episodes
per world offset, one slot, fixed action/policy seed 7, drawing 0. Initial
means untrained random weights, not uniformly random actions. These are
previously used development worlds, not new confirmation seeds. The legacy
native serial aggregate evaluator is used; no new per-ID action-trace or
independent full-learner mathematical qualification follows.

| World offset | Our initial perf | Our final perf |
|---|---:|---:|
| 420 | 4.6875% | 3.125% |
| 1337 | 0% | 6.25% |
| Mean, 32 episodes | 2.34375% | 4.6875% |

World 420 declines. Both architectures perform poorly after this tiny budget;
the small mean increase does not establish reliable learning. Nature's final
perf is higher. Keep both initializations and all outcomes; no best-seed,
best-checkpoint or per-world selection is made. This is independent training
n=1 per architecture, not 32 replicas. Scores are collection-based perf,
not win rates. No convergence, robustness or Pareto superiority is claimed.

All four scheduled checkpoints (8,192/16,384/24,576/32,768 decisions) have
the expected 78,885,888 bytes, finite float32 values and distinct hashes.
The final scheduled checkpoint is evaluated. All 64 assigned initial/final
episode executions complete. Total train/evaluation campaign: 104.601s.
No training extension or retry is launched by this report.

## Evidence

Raw packet: `build/sprite-forage-flex-train-5060-20261015/`, including the owned
native sources, adapter, quality INI, binary, requested commands, runtime and
resolved configs, metrics, checkpoints, GPU queries and process receipts.
Compact results/evaluations/plan/recipe and raw SHA256 manifest are copied
here. Python supervises native CUDA processes and audits files only.
No CPU CNN, 5090 run, commit or push occurred.

Nature comparison: `research/results/sprite-forage-pixels-5060-20261015/` and
its retained raw packet. Earlier state timing remains a separately tuned,
larger-batch experiment and is not a matched speed or quality comparator here.
