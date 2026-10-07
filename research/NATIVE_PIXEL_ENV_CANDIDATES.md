# Next small native pixel environments

October 6: **MazeCNN is now implemented and compiles**, preserving the original
local crop, shared level table, movement and reset/log quirks. Six direct-buffer
drawings, independent game/reference/BFS/raster/sanitizer parity and four native
targets pass. No policy ran; dedicated exact evaluation/held-out levels/GPU
acceptance remain pending. [Contract and receipts](../ocean/mazecnn/README.md).
The current panel is six tasks/38 conditions; older counts below are historical.

October 5 update: **BreakoutCNN is now implemented and compiles**, with direct
buffer pixels, five reversible appearances, original-game parity and sanitizer
checks. Breakout's exact adapter and supervised launcher now compile/pass
host preparation checks; GPU training/reload/exact acceptance remain pending.
**SnakeCNN and matched SnakeBench are implemented** under a separately named
one-agent/local-view protocol, leaving stock Snake untouched. Six direct-buffer
drawings and deterministic worlds/counters pass independent reference/sanitizer
checks. Its exact adapter compiles and native starts/audits pass; supervised
checkpoint launch now passes GPU-free preparation/audit checks (October 6);
GPU learning/exact acceptance remain pending.
[Current inventory and preparation](MULTI_ENV_ROBUSTNESS.md)
supersede implementation-status portions of the September ranking below.

September 26, 2026. Local source inspection, not measured throughput. Kinvert authorized adding another simple headless pixel task. Selection prioritizes small native simulation, direct in-memory rasterization, reusable encoders and a useful change in visual/control demands from Connect4/Pong.

| Rank | Candidate and source | Direct pixels | Value / limitation |
|---|---|---|---|
| 1 | [Flappy](../ocean/flappy/flappy.h), [config](../config/flappy.ini) | Three pipes and one bird; clear buffer and fill clipped rectangles | Two actions, gravity/flap motion, random pipe gaps and an existing finite episode cap. No simulation heap allocation; simplest useful moving-scene addition. Chosen and implemented as FlappyCNN. |
| 2 | [Breakout](../ocean/breakout/breakout.h) | Paddle, ball, up to 108 bricks under stock geometry | Richer scene occupancy/occlusion and brick removal. More collision code and reset/life semantics to audit. A strong next visual task; native Breakout is not ALE Breakout. |
| 3 | [Snake](../ocean/snake/snake.h) | Paint the existing grid/crop | Food/body/wall structure is useful, but current multi-agent arrays, one-hot/local vision and corpse semantics add integration choices. Freeze a single-agent/crop protocol before comparisons. |
| 4 | [Maze](../ocean/maze/maze.h) | Rasterize the existing local tile crop | Useful navigation/memory demand; shared levels/vector initialization and crop/boundary semantics need care. More protocol decisions than Flappy. |
| 5 | [Memory](../ocean/memory/memory.h) | Initial cue image followed by blank frames | Excellent recurrent-core regression, weak visual-encoder challenge. Do not treat success as a substantial CNN result. |
| Later | [LightsOut](../ocean/lightsout/lightsout.h) | Paint the small board | Cheap board rasterization, but its performance-driven scramble curriculum complicates interpreting a common-task frontier. Similar spatial style to Connect4. |

The ranking is an engineering judgment from code complexity and geometry, not an SPS ranking or prediction of RL learning. CPU environment stepping does not imply a CPU learner. Keep training/CNN validation on GPU, and keep timed work exclusive.

## Selected implementation

[FlappyCNN](../ocean/flappycnn/README.md) copies original Flappy and changes the observation path. It fills a float32 1x36x44 buffer directly, using the same image contract as current CNNs. The human viewer is optional. Rules, RNG, rewards, pipe spawning and cap/reset semantics are preserved; appearance selection has a separate deterministic seed and does not consume game RNG.

The original state observations carry bird velocity and future pipe data even offscreen. Pixels only show visible geometry, with reduced position precision. Record these information differences for state/pixel controls; every CNN receives the same pixel interface. No score/tick/velocity channel or annotation label is added to the CNN input.

Environment/parity/pixel/sanitizer checks pass over 49,152 transitions and explicit event fixtures. All five model targets compile, with source-hash checks passing; fifteen configuration/sidecar tests and both architecture-recipe preparations pass. [Durable validation receipts](results/flappycnn/environment-20260926/README.md) preserve the accepted checks and an earlier rejected source snapshot. The quality encoder's [5060 stock-learner pilot](results/flappycnn/stock-pilot.DN7C2JF6/REPORT.md) now completes training/reload: 19.923M decisions in 56.12 process seconds, final mean 52.25 pipes and clipped perf 0.9759. Other models, matched comparison, GPU repeatability and current-source numerical gates remain pending. No Pareto improvement has been established.

## Gates before meaningful learning comparisons

1. On an available idle GPU, pass matched state/Flex/Nature/IMPALA/Impoola train/reload canaries and verify metrics/checkpoint receipts. New encoder-5 activation/dilation math needs its separate acceptance checks first.
2. Calibrate a learning-scale pilot from the common recipe; 65K plumbing decisions do not give these agents a fair learning opportunity. Retain failed seeds and full curves, including stock state as a separately labeled tuning control if measured.
3. Predeclare shared learner settings, decision budgets/checkpoint cadence, appearance panels, training/evaluation seeds and primary metric. Report mean pipes passed and the native clipped-perf statistic distinctly.
4. Profile observation generation, environment stepping, actor and learner on exclusive hardware before claims about what dominates time. Record preprocessing/memory cost and native versus process SPS boundaries.
5. Use reliable episode allocation and simultaneous frontier analysis for claims. The existing pooled evaluator is adequate for bounded plumbing, not a substitute for the paper's open exact-evaluation gates. Flappy is now a development game; final architecture generalization needs other untouched games or an external benchmark.

Adding this game does not generate a pretraining dataset, implement an external Atari/Procgen benchmark or establish SOTA. It provides a small additional pixel workload and a future frame-aligned annotation integration candidate.
