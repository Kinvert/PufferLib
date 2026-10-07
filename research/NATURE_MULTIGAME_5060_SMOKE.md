# Quality versus Nature: local RTX 5060 plumbing smoke

October 6, 2026. Kinvert explicitly authorized local RTX 5060 short runs and
smokes, generally avoiding slow IMPALA/Impoola runs. This protocol uses the
existing compiled native binaries and optional full-policy registration gate;
it does not schedule a long sweep or the separate RTX 5090.

## Frozen allocation

- Two shared architectures: quality (`happy-cat-1`: encoder 4, one C16/K7/S4
  convolution, flatten, projection 64 into H128/L1), and adapted Nature
  (`nature-cnn`: selector 2, three valid convolutions into H128/L1).
- All six existing games; separately initialized weights per model/game.
  Per-game learner settings are shared across both models, not swept.
- One paired training seed 58173; each job receives 65,536 decisions and one
  final checkpoint, 64 actor slots/H32/M2048. Twelve jobs total.
- Persistent native slot mixtures, appearance seed 35173. Full catalogs:
  Connect4 10, Pong 7, Flappy 7, Breakout 5, Snake 6, Maze 6. Counts are unequal
  and preserved; no search for a favorable appearance seed.
- Evaluation seed 68173; 17 assigned episodes in 16 slots for every drawing.
  Identical world/policy RNG identities across models and drawings; partial
  final wave. Eighty-two evaluations, 1,394 assigned episodes total.
- Pong cap 512 decisions keeps censoring bounds; Breakout cap 2,048 physics
  frames keeps partial scores. Other games retain their native horizons.
- Drawing 0 gets a final-checkpoint graph repeat and eager replay per job:
  24 extra comparisons / 408 episode executions. Equality isn't independent
  numerical/gradient or full evaluator acceptance.

The 12 full-policy parameter descriptors were registered before GPU discovery.
Registration repeats before each trainer; checkpoints must match declared
sizes/counts before reads/evaluation. No encoder 5, IMPALA or Impoola policy is
trained or evaluated. The inherited registry contains their compiled builds,
which doesn't authorize their execution.

## Native execution and retention

Local packet:
`build/candidate-baselines/smoke-nature-5060-20261006/plan.json`.
Builds: `build/candidate-baselines/native-layout-20261006/registry.json`.
Metadata: `build/policy-metadata/candidate-layout-20261006`.
Preparation log: `build/candidate-baselines/smoke-nature-5060-prepare-20261006.txt`.
Runner log: `build/candidate-baselines/smoke-nature-5060-run-20261006.txt`.

```bash
source ocean/connect4cnn/runtime_env.sh
# Exact command used on the explicitly authorized idle G240/5060.
.venv/bin/python research/candidate_panel.py run --mode smoke --allow-gpu \
  --plan build/candidate-baselines/smoke-nature-5060-20261006/plan.json \
  --timeout 900 --train-timeout 60 --eval-timeout 30
```

These are reproduction receipts, not instructions to automatically repeat the
run. The supervisor uses the shared reservation/idle check, serial processes,
immutable input checks, explicit process/campaign deadlines and retained failures.
No active run watching or CPU embedding refresh during timed work. Don't
interrupt competing jobs or selectively retry a failed model/seed.

Initial authorized inspection: RTX 5060, driver 591.86, UUID
`GPU-56b0df25-b4c9-46eb-e498-bc5c71904097`, 8,151 MiB total, 496 MiB used,
0% GPU utilization and no competing compute PID. This is initial telemetry,
not continuous proof of contention-free execution or peak model memory.

The completed run passed the existing offline `candidate_panel.py audit` and
`candidate_viewer.py report`, using fresh destinations. Keep every game/drawing
separate, native monotonic checkpoint time and full-process SPS distinct from
native SPS arrays. Charge each job's training once regardless of drawing/replay;
retain failed/missing conditions and registration/query/process receipts.

## Completed RTX 5060 results

All 12 training jobs, 82 drawing evaluations, 12 repeated native layout checks
and 24 exact-byte graph-repeat/eager comparisons passed. Checkpoints had the
registered sizes/counts and finite float32 weights. All assigned episodes were
retained, including Maze timeouts; no missing evaluation cells or failed jobs.
Successful campaign window: **93.243211915 seconds**; summed training process
time: **9.679294690 seconds**. Builds/preparation/offline review are outside
that window. Hardware receipt identifies G240, RTX 5060, driver 591.86 and
CUDA compiler 12.8.93. The dirty working-tree source/build hashes, rather than
HEAD alone, identify the executed implementation.

| Game | Quality process seconds | Nature process seconds | Quality process SPS | Nature process SPS |
|---|---:|---:|---:|---:|
| Connect4CNN | 1.165676 | 1.115342 | 56,221 | 58,759 |
| PongCNN | 0.714914 | 0.764568 | 91,670 | 85,716 |
| FlappyCNN | 0.614683 | 0.714609 | 106,618 | 91,709 |
| BreakoutCNN | 0.715022 | 0.815128 | 91,656 | 80,400 |
| SnakeCNN | 0.664639 | 0.764753 | 98,604 | 85,696 |
| MazeCNN | 0.764858 | 0.865102 | 85,684 | 75,755 |

Process SPS includes startup/checkpoint writes. These subsecond-scale runs are
not steady-state throughput benchmarks. Native SPS arrays remain in the raw
metrics INIs; the offline report does not compute a native SPS average.

Drawing 0 provides a sanity check rather than a quality comparison: both
models won 0/17 Connect4 games, won no Pong points and passed no Flappy pipes.
Breakout mean scores were 0.706/0.588 (quality/Nature), Snake mean ending
length was 1.176 for both, and Maze successes were 2/17 versus 1/17. No winner
is selected from these outcomes; every other drawing stays in the report.

Durable [audited report and evidence](results/nature-multigame-5060-20261006/README.md)
and [interactive drawing-by-drawing view](results/nature-multigame-5060-20261006/view/curves.html)
keep this local smoke distinct from the 5090 campaigns. The successful
allocation is complete; do not automatically restart or extend it.

## Interpretation

This tests the new preparation/registration/train/checkpoint/all-drawing/exact
evaluation/repeat/audit path on two actual CNNs. One seed and 65,536 decisions
per game cannot establish adequate learning, robust architecture superiority
or a publication Pareto frontier. Tiny administrative caps may censor Pong or
Breakout outcomes. All drawings are exposed during training; this isn't withheld
appearance transfer. Smoke frontier marks/quality selection remain disabled.

Full independent model math, larger learner batch/memory/reload/evaluator
acceptance, adequate per-game learning/cap calibration and independent
selection-safe confirmation remain necessary. Hardware/toolchain evidence is
RTX **5060**, never relabeled or pooled with the separate **5090** results.
No system CUDA/Torch/driver/dependency changes or push are authorized here.
