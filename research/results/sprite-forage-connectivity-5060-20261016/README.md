# Forage: removing reset-time connectivity rejection

Completed October16 on RTX5060/G240. **Unchecked placement is promising:
14.42% higher observed process SPS, with no learning degradation in this
single-seed state pilot.** Main `sprite_forage.h` is unchanged; unchecked
placement exists only in the captured ablation build.

Following this experiment, Kinvert approved removal. The main environment now
uses unchecked placement as `sprite-forage-state-v7`; the connectivity helper
and rejection loop are deleted. All seven native ASan/UBSan environment,
state/pixel, renderer, navigation, RNG, attention and imported-sprite suites
pass in `build/sprite-forage-unchecked-tests-20261016`. No additional GPU
training was run for adoption. The paragraph above describes the measurement
stage, before this approved source change.

| Map generator used for training | Mean train seconds | Process SPS | Eval on connected maps | Eval on unchecked maps |
|---|---:|---:|---:|---:|
| Current connectivity rejection | 11.748 | 714,031 | 98.83% collection | 98.83% collection |
| Unchecked placement | 10.267 | 817,022 | 99.61% collection | 99.61% collection |

Each timing mean uses two fresh trainings of8,388,608 decisions at fixed
seed7. Order: connected/unchecked/unchecked/connected. Each variant's four
scheduled checkpoints match byte-for-byte across its two repeats. These are
timing/determinism repeats, **one independent training seed** per variant.
The apparent0.78-point quality improvement is descriptive, not a robust gain.

Both use the current full-state layout2,141 observations,67,968 parameters,
H128/L1 MinGRU,256 agents, horizon64, minibatch2048, four threads, float32
and the same learner/rewards/LR schedule. This is a **state experiment**, not
a pixel/CNN benchmark. The only native behavior difference is removing
placement rejection. Each condition's first fixed final checkpoint is used
for evaluation; no best-score checkpoint selection.

Each policy receives128 exact assigned episodes per map generator:64 at
each development suite seed420/1337, cap676,16 slots. Eight native evaluations
produce512 episode CSV rows in total. Within each generator, the policies'
initial world/observation hashes and assigned world/policy RNG match. The
generators can consume different RNG draws, so identical seed numbers are
not a claim of identical maps across generators.

The connected-trained policy exits via grape in99.22% of episodes on each
generator; unchecked-trained exits in100%. Neither policy dies in these
assigned episodes. Collection perf is not a win rate. Suites420/1337 are
reused development worlds, not untouched confirmation data.

## Deterministic map diagnostic

A separate scalar native program generated4,096 consecutive resets from
initial seed7 per generator, with a second environment repeating every
reset. Board bytes, post-generation RNG, player position and state
observations match exactly across both copies for every map.

| Diagnostic | Connected | Unchecked |
|---|---:|---:|
| Maps rejected by the current connectivity predicate after generation | 0 | 46 (1.12%) |
| Maps with an unreachable apple or grape | 0 | 0 |
| Mean reset time, microseconds | 15.900 | 7.108 |

Reachability here means paths avoid trees/fatal toadstools and do not use the
terminal grape as a passage. The current connectivity predicate requires all
passable cells to connect; it can reject an isolated empty/neutral pocket
even when all objectives are reachable. Zero unreachable objectives in this
finite sample does not guarantee all future unchecked maps are solvable.

Reset timing is a single scalar diagnostic after the GPU campaign, not a
GPU training or neural benchmark, and is not a general intrinsic speedup.
The first diagnostic invocation omitted the required `seed` dictionary key
and exited before sampling; its empty failed-output files are retained.
After supplying all explicit init options, both diagnostics completed.
This diagnostic correction did not rerun or alter the completed GPU campaign.

## Receipts and interpretation

Raw packet: `build/sprite-forage-connectivity-20261016/`. Four serial
trainings plus eight evaluations finished in48.695 seconds. Process timing
uses monotonic start/end around the native process, including startup and
four checkpoint writes, excluding evaluation. Dashboard last-bin SPS is
recorded separately: connected756,491/749,594; unchecked845,092/847,958.

The packet retains both source snapshots, binaries, build logs, recipes,
resolved configs, checkpoint identities, GPU idle queries, reservation-based
serial execution, commands, process timings and per-episode CSVs. Frozen
compatible core comes from the completed state-cleanup source snapshot;
current main env/config were copied into fresh build directories. No old
allocation was restarted and no main core, system/CUDA/dependency, dataset,
commit or push changed. No CPU neural execution or pixel training occurred.
Scripts: [training/evaluation runner](../../forage_connectivity_ablation.py),
[map diagnostic](../../forage_connectivity_maps.c),
[protocol](../../FORAGE_CONNECTIVITY_ABLATION.md).

This supports removing the expensive generation constraint for the simpler
deterministic benchmark Kinvert requested. It does not establish multi-seed
quality equivalence or any CNN/Pareto claim. Removing it changes generated
worlds and their RNG streams; keep these results versioned and preserve old
connected-world checkpoints/receipts rather than relabeling them.
