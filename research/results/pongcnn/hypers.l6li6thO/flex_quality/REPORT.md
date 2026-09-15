# Pong fixed-architecture development search: flex_quality

2 completed trials; one verified architecture; 0 native worker failures. Status: completed.
PufferLib Pong, not ALE Pong. Perf is episode-averaged fraction of points won, not match win rate.
Training seed 9173; separate development evaluation seed 29173. This is hyperparameter discovery, not fresh-seed confirmation.
PROTEIN uses downsampled training curves. Native cost excludes evaluation/upload and uses adjusted uptime rounded in stdout.
Sweep wall includes search/worker overhead; see sweep-wall.txt. W&B uploads happen after the family, outside timing.
A resource cap may interrupt a trial; partial checkpoints/logs remain but are not labeled completed observations.

| Run | Decisions | Train point fraction | Native seconds | Native SPS | Eval point fraction | Eval status |
|---|---:|---:|---:|---:|---:|---|
| lucky-badger-1 | 32,768 | 0.00% | 0.28 | 117,029 |  | pending |
| bright-panda-2 | 32,768 | 0.00% | 0.28 | 117,029 |  | pending |

All per-trial native curves and effective settings: metrics/pongcnn/*.ini. Full final-point data: results.csv.
Source/build provenance: parent campaign source.sha256, revision.txt, working.patch; per-family inputs.sha256/build-command.txt.
