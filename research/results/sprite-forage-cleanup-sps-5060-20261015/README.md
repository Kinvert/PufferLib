# Sprite Forage cleanup: quick state timing check

October15, RTX5060/G240, float32, CUDA12.8.93, driver610.88.
Four fresh serial trainings and four evaluations completed in48.831s.
The [offline receipt check](audit.json) passes. No pixel training occurred.

| Source | Whole-process SPS | Native adjusted-average SPS | Mean train seconds/run | Evaluation perf | Grape exits |
|---|---:|---:|---:|---:|---:|
| Before cleanup |742,328|782,101|11.300|97.65625%|100%|
| Current ~/cnn |745,912|770,347|11.246|97.65625%|100%|

Process throughput changed+0.48%; native adjusted throughput changed-1.50%.
This short check establishes no speedup or meaningful regression. Process SPS
uses monotonic process duration including initialization/checkpoint writes.
Native uptime/final-bin SPS are retained separately; the trainer's adjusted
clock uses CLOCK_REALTIME. Do not interchange the two timer definitions.
The raw last-bin SPS pairs are785,698/792,771 before and773,725/775,028 after.

Every scheduled checkpoint is byte-identical across all four runs:2,097,152,
4,194,304,6,291,456 and8,388,608 decisions. All sixteen checkpoint files have
the expected67,968 float32 parameters and were checked finite by the runner;
the offline check rechecks their hashes. This is strong scoped evidence that
the cleanup preserved deterministic training under these settings.

## Matched settings and limits

Both versions use8,388,608 decisions/run, fixed training seed7, H128/L1,
141 state observations,256 agents, one buffer/four threads, H64/MB2048,
two-million-decision checkpoint cadence and the same previously tuned learner.
The source recipe is the retained unsorted-local/seed7 ablation resolved INI;
only total decisions/output locations change, and the obsolete ablation option
and sweep-coordinate sections are removed from both. The shorter total budget
also shortens its LR annealing schedule equally. This is not a new sweep or
the original longer33M-decision experiment.

Order is before/after/after/before to reduce order effects. Two timing repeats
use the same seed: independent training n=1, not2 or4. Seeds are not optimized;
the seven-fixed-seed requirement still applies to the next larger scientific
comparison. No Pareto/generalization/statistical-superiority claim follows.

Each version's first final checkpoint receives32 sequential episodes each at
development world offsets420/1337, action seed7, one slot,512-step cap. These
are64 matched evaluation episodes/version, not independent training replicas
or untouched final-test data. The aggregate evaluator is sufficient for this
bounded compatibility check; it is not the newer per-ID publication protocol.

Before is the actual header saved before the metric cleanup. After is copied
from the user's real `/home/claude/cnn/ocean/sprite_forage` directory. Cleanup
removes four diagnostic metrics and the unused positive-item counter, renames
player/free-cell identifiers, adds explanatory comments and initializes only
requested sprite skins. Rules, observations and learner behavior stay matched.
The sprite-cache change has no expected state-training benefit because headless
state initialization does not generate tiles.

## Sources and receipts

Both binaries compile from the same frozen compatible research core. The main
checkout's pending core/config integration is not silently overwritten. All
new edits remain in ~/cnn; test copies live entirely under its ignored
`build/sprite-forage-cleanup-sps-20261015/` directory. Frozen before/after trees,
compiler/build logs, exact recipes/commands, GPU idle queries, process receipts,
native logs and checkpoints remain there. Source hashes are in `plan.json`;
important artifact hashes are retained in `RAW-SHA256SUMS`. The summary and
training/evaluation/audit JSONs are copied beside this report.

The shared GPU reservation was held for the serial allocation, with idle
checks before each process. Compilation finished before timed training; no
heavy CPU work or active TUI monitoring was scheduled during the runs. No
system/CUDA/dependency changes, CPU neural model,5090, commit or push occurred.
This completed allocation must not restart automatically.
