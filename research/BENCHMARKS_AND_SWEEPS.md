# Pixel RL benchmarks and architecture sweeps

Research notes checked 2026-09-11 (America/Los_Angeles). This document proposes experiments; no training, benchmark installation, or CUDA/system change was performed. Links to local code describe this checkout and should be checked again after upstream updates.

## Next benchmark priorities (October 2, 2026)

**Recommendation: original Procgen for the next external paper milestone; native Breakout for an optional small engineering extension.** Connect4CNN, PongCNN and FlappyCNN are already development tasks. Their results do not establish transfer to untouched games or broad pixel-RL superiority. The ranking below is our research judgment, not measured integration cost or throughput. This update authorizes no GPU runs or installations.

| Priority | Benchmark | Evidence it could add | Main requirement or limitation |
|---|---|---|---|
| 1 | [Original Procgen](https://github.com/openai/procgen) | Richer visual learning and generalization to held-out procedural levels across 16 games | Support 64×64 RGB and integrate the original environment. Keep the existing proposed development panel: CoinRun, BigFish, StarPilot and Maze. A smaller initial plumbing subset does not replace that panel or establish suite-wide performance. |
| 2 | [ALE Atari](https://ale.farama.org/) | An established external pixel benchmark with diverse game dynamics | Use original ROM environments and a pinned evaluation protocol. ALE exposes a C++ interface. Native PufferLib Pong/Breakout are separate tasks and cannot inherit ALE scores. |
| 3 | [POPGym Arcade](https://arxiv.org/abs/2503.01450) | Paired fully/partially observable variants address Joseph's encoder-versus-memory question | Additional integration work; control the recurrent core and audit what changes between observability variants. |
| 4 | [Memory Gym](https://arxiv.org/abs/2309.17207) | Mortar Mayhem, Mystery Path and Searing Spotlights stress retaining visual information over time | Fix finite/endless variants, difficulty and horizons. Success can depend strongly on the recurrent model, so it is not an isolated CNN measurement. |
| 5 | [Crafter](https://github.com/danijar/crafter) | Richer 64×64 RGB survival/crafting scenes and achievement-based evaluation | Exploration and long-term credit assignment complicate identifying an encoder effect. Follow its achievement/score protocol rather than substituting raw reward. Later target. |
| 6 | [MinAtar](https://github.com/kenjyoung/MinAtar) | Compact diagnostics across five Atari-inspired games | Its 10×10 object-specific channels already identify object classes; they are not ordinary RGB frames. Unchanged VALID Nature kernels do not fit this input. Resizing or changing kernels must be explicit and weakens direct protocol comparability. |

For native development, [the source-based candidate review](NATIVE_PIXEL_ENV_CANDIDATES.md) favors **Breakout, then Snake and Maze**. Breakout can paint its ball, paddle and bricks directly into memory. Preserve original game rules, reset/life semantics and RNG, and validate the observation conversion. Its expected implementation simplicity is not a measured speed advantage. Adding more custom games alone does not replace external validation.

### Groundwork before expanding the experiments

1. **Generalize the image contract.** Current experiments use 1×36×44 grayscale. Specify channels, dimensions, layout, scale and preprocessing for 64×64 RGB; preserve existing inputs and revalidate every encoder on the new contract before comparisons.
2. **Audit original Procgen's native route.** Its C++ implementation exposes the `gym3.libenv` C interface. Verify build dependencies, buffer ownership, reset/terminal behavior, seeding and headless image generation before implementing a thin adapter. The documented Python wheels target 3.7–3.10; do not assume compatibility with our Python 3.12 research venv. No working native adapter is claimed here.
3. **Freeze development and test boundaries.** Use the four proposed development games for integration/tuning, and reserve other game identities for architecture-transfer evaluation. Held-out levels within a tuned game and entirely unseen games answer different questions. Freeze architectures before inspecting final test results.
4. **Preserve comparison fairness.** Within each game, use the same observations, learner settings, core, precision, budgets, checkpoint cadence and seed allocation across encoders. Record any tuning allowances. Retain full curves, failures, search costs and measured wall time; do not select favorable games or checkpoints after seeing results.
5. **Keep verification gates ahead of claims.** Existing encoder-5 GPU math/reload checks, baseline efficiency review and frontier-inference validation remain prerequisites. See [claim pipeline status](CLAIM_PIPELINE.md). No new benchmark resolves those open gates automatically.

This is preparation for future exclusive-GPU experiments. No environment was installed, dataset generated, or training launched for this writeup. The external suites above have not been integrated into the current matched native CNN campaign.

## What we should optimize

The practical target is **the lowest elapsed training time needed to reach a useful return, with repeatable execution and an explicitly measured resource budget**. An encoder can have fewer FLOPs and still be slower on the actual GPU. A faster encoder can also require more environment interactions to learn. Report both effects.

There are three separable claims:

| Claim | Necessary comparison |
|---|---|
| Better encoder architecture | Same environment, learner, backend, training budget, preprocessing, core family, and tuning allowance; vary encoder |
| Better encoder/core allocation | Joint encoder/core capacity grid with controlled training; compare return against FLOPs and measured time |
| Better implementation | Same architecture, weights, data, precision policy, optimizer semantics, and numerical checks; compare framework/library baseline against native implementation |

A clean first contribution can be an architecture/runtime Pareto frontier and reproducible harness. A universal optimal ratio or a new overall RL SOTA requires substantially broader evidence. A recurrent core also cannot recover task information the encoder discarded. Include visual detail demand and memory demand as independent experimental axes.

## Which benchmarks answer which question?

| Suite | Best use here | Protocol and limitations |
|---|---|---|
| Original Procgen | Main encoder efficiency/generalization experiment | 16 games, 64×64 RGB observations, procedural train/test levels. Fix game version, difficulty, level sets, and training budget. Distinguish finite-level generalization from unlimited-level sample efficiency. |
| Original ALE Atari, broad suite | External validity and comparison with mature pixel RL systems | Choose an exact published protocol; 57-game, 200M-emulator-frame comparisons are common in systems work. Full suite is expensive. Sticky actions, action set, wrappers, and frame accounting materially change the task. |
| Atari 100K | Data efficiency and comparison with BBF-like agents | 26 games, 100K agent decisions, ordinarily 400K emulator frames at action repeat four. A small interaction budget can still entail heavy training/replay compute. Different learner families confound a CNN-only claim. |
| POPGym Arcade | Direct test of how partial observability changes the encoder/core split | Pixel observations and paired fully/partially observable variants. Promising controlled memory study; JAX environment integration is additional work, not something already provided by this checkout. |
| Memory Gym | Longer memory challenges with visual observations | Mortar Mayhem, Mystery Path, Searing Spotlights; finite/endless variants. Fix difficulty, episode limit, memory horizon, and evaluation distribution. |
| DeepMind Control from pixels | Later transfer to continuous control | Use a specific DrQ-v2-style protocol; action repeat, camera, resolution, augmentation, actor/critic encoder sharing, replay, and gradient routing must be fixed. |
| Native PufferLib games / Affine Lock | Cheap integration diagnostics and future controlled tasks | Useful for local performance engineering, but a rendered game is not automatically a pixel-observation benchmark. A custom pixel task establishes a new protocol and cannot inherit ALE/Procgen scores. |

## PufferLib-native candidates requested for tracking

These configs already exist in this checkout and are practical to run on a single 1x5060 box:

| Environment | Native config | Observation contract (from source) | CNN benchmark ready? | Why this is useful |
|---|---|---|---|---|
| 2048 | [config/g2048.ini](../config/g2048.ini) | `OBS_SIZE = 16`; flattened board tile values via `update_observations`/`compute_observations` [g2048.h](../ocean/g2048/g2048.h) | No | Cheap state-space baseline, deterministic transitions, and recurrence stress under small state |
| Tetris | [config/tetris.ini](../config/tetris.ini) | `OBS_SIZE = 234`; board occupancy + timing/selection scalars [tetris.h](../ocean/tetris/tetris.h) | No | Useful for action-latency and sparse reward behavior with moderate observation size |
| Go | [config/go.ini](../config/go.ini) | `OBS_SIZE = 326`; bit-grid plane encoding + self/opponent history bits [go.h](../ocean/go/go.h) | No | Deterministic and controlled with potentially rich planning structure |
| Connect4 | [config/connect4.ini](../config/connect4.ini) | `OBS_SIZE = 42`; bitboard-based flattened occupancy encoding [connect4.h](../ocean/connect4/connect4.h) | No | Fast, deterministic benchmark for low-noise encoder/core comparisons |
| Soccer (boxy pixel variant, to build) | [config/soccer.ini](../config/soccer.ini) | Current task is vector state features (`OBS_SIZE` from engineered ball/player/context channels) [soccer.h](../ocean/soccer/soccer.h) | Not yet; needs pixelized observation contract | Candidate for your "boxy pixelly" request; likely the cleanest route to a native visual task |
| Trash Pickup (maybe) | [config/trash_pickup.ini](../config/trash_pickup.ini) | `OBS_SIZE = 605`; local crop spatial channels [trash_pickup.h](../ocean/trash_pickup/trash_pickup.h) | Partial | Closest to an image-style input among requested set, though it is still structured channels rather than RGB |

Sources: [Procgen paper](https://arxiv.org/abs/1912.01588), [official Procgen implementation](https://github.com/openai/procgen), [Cleanba](https://arxiv.org/abs/2310.00036), [BBF](https://arxiv.org/abs/2305.19452), [POPGym Arcade](https://arxiv.org/abs/2503.01450), [Memory Gym](https://arxiv.org/abs/2309.17207), [DrQ-v2](https://arxiv.org/abs/2107.09645). These are complementary protocols, not interchangeable leaderboard entries.

## Current native pixel development panel (September 26, 2026)

| Task | Pixel path | Evidence status |
|---|---|---|
| [Connect4CNN](../ocean/connect4cnn/README.md) | Direct in-memory board/glyph rasterization | GPU learning/confirmation evidence exists; quality advantage over Nature remains uncertain |
| [PongCNN](../ocean/pongcnn/README.md) | Direct in-memory court rectangles and score bars | GPU learning/replication exists; recipe dependence and pooled-evaluation limitations remain |
| [FlappyCNN](../ocean/flappycnn/README.md) | Direct in-memory pipes/bird rectangles | Single-seed stock-learner pilots: quality pixels 56.12s / 355K process SPS / 52.25 pipes; original stock state 9.73s / 2.048M SPS / 48.53 pipes; other CNN baselines pending |

These are native PufferLib development tasks, not original ALE/Procgen score protocols. [Candidate ranking and next gates](NATIVE_PIXEL_ENV_CANDIDATES.md) explain the third task's selection. None requires a graphics context to generate policy images. This panel is separate from the state-environment inventory above.

## Full Procgen game catalog

Here, **Procgen means the specific original 16-game benchmark**, not procedurally generated environments in general. All games use 64×64 RGB observations. The descriptions below summarize the [official game catalog](https://github.com/openai/procgen#environments); pilot selection is our proposal.

| Game / environment ID | Main activity | Initial use |
|---|---|---|
| BigFish / `bigfish` | Eat smaller fish; avoid larger fish | **Pilot:** object size and relative position |
| BossFight / `bossfight` | Shoot a boss and dodge projectiles | Full-suite confirmation |
| CaveFlyer / `caveflyer` | Fly through caves to an exit | Full-suite confirmation |
| Chaser / `chaser` | Collect orbs while avoiding maze pursuers | Full-suite confirmation |
| Climber / `climber` | Climb platforms and collect stars | Full-suite confirmation |
| CoinRun / `coinrun` | Traverse platforms to collect a coin | **Pilot:** platform navigation |
| Dodgeball / `dodgeball` | Defeat opponents with thrown balls | Full-suite confirmation |
| FruitBot / `fruitbot` | Collect fruit while avoiding obstacles | Full-suite confirmation |
| Heist / `heist` | Use keys and locks to reach a gem | Full-suite confirmation |
| Jumper / `jumper` | Jump through platforms to a carrot | Full-suite confirmation |
| Leaper / `leaper` | Cross roads and rivers | Full-suite confirmation |
| Maze / `maze` | Navigate a maze to cheese | **Pilot:** spatial navigation |
| Miner / `miner` | Dig for diamonds; avoid falling boulders | Full-suite confirmation |
| Ninja / `ninja` | Use charged jumps to reach a mushroom | Full-suite confirmation |
| Plunder / `plunder` | Identify and shoot enemy ships | Full-suite confirmation |
| StarPilot / `starpilot` | Fly and shoot through scrolling hazards | **Pilot:** moving objects and projectiles |

These pilot roles are hypotheses about useful coverage, not measured task complexity or memory requirements. Maze's inclusion alone does not establish a controlled memory experiment.

**Integration status:** original Procgen is a proposed target, not a verified working environment in this checkout. Its official README documents Python 3.7–3.10 support, while our research environment uses Python 3.12. Do not assume its packages will install into this venv. The implementation exposes a C interface through `gym3.libenv`, which is a possible native integration route to investigate. Neither route has been validated here. [Official implementation and installation documentation](https://github.com/openai/procgen)

Procgen also offers a game-specific `memory` distribution mode. Investigate this before assuming a second suite is required, but audit the changes: larger worlds and altered level generation can change more than observability. The original paper already compares single frames, four-frame stacks, and an LSTM in Appendix H; our encoder/core study should build on that prior work. [Local Procgen paper, Appendices B and H](papers/procgen.md)

### Important local observation

The native [Breakout environment](../ocean/breakout/breakout.h) defines `OBS_SIZE 118` and fills observations with paddle/ball state and brick states. Its [configuration](../config/breakout.ini) uses `frameskip = 3` and even exposes frameskip to sweeping. This is not original ALE Breakout and is not presently a pixel-CNN benchmark. Its speed cannot establish that Nature CNN or IMPALA-CNN is fast. The native [memory task](../ocean/memory/memory.h) has `OBS_SIZE 1`: useful for checking recurrent behavior, but not visual encoding. Inspect Affine Lock's actual observation contract before assigning it a CNN role.

This also means the earlier pasted suggestion to ask what CNN shrank in a high-throughput Breakout run rests on an unverified premise. First establish which environment, observation type, and architecture that run actually used.

## Recommended first suite

**Primary pilot: original Procgen CoinRun, StarPilot, BigFish, and Maze, easy difficulty.** This is a deliberately small development panel spanning different visual layouts and control demands; the selection is a proposal, not a claim of statistical representativeness. Use all 16 games for later confirmation or explicitly limit the conclusion to the four-game panel. Procgen's observation format is convenient for encoder comparisons, and its held-out levels let us detect a fast encoder that merely overfits.

Use a finite training set such as seeds 0–199, a separate development-validation level set, and a locked final-test level set. Record exact ranges and generator version. The commonly used easy/200-level/25M-step protocol is a reasonable confirmation target, but match the selected comparison paper precisely before making a published-score comparison. The original benchmark also contains harder and longer-budget settings; “Procgen” alone does not specify one experiment. The 2020 competition, for example, imposed its own 8M-step and holdout-environment rules. [Procgen competition report](https://proceedings.mlr.press/v133/mohanty21a.html)

**Memory follow-up: one POPGym Arcade task in both its fully observable and partially observable forms**, selected only after confirming that both baselines learn it. Add a second task before generalizing about memory. This is cleaner than declaring one Atari game “reactive” and another “memory-heavy.” Ordinary Atari observations can be partially observable; stacking four frames reduces some velocity ambiguity but does not guarantee full observability. [POPGym Arcade paper](https://arxiv.org/abs/2503.01450)

**Later external test: an ALE development panel followed by the full chosen suite.** A possible six-game panel is Breakout, Pong, Seaquest, Qbert, MsPacman, and SpaceInvaders. Use actual ALE ROM environments; label this panel as development-only. Atari 100K should be a separate study if we add an appropriate data-efficient learner, rather than the default target for an on-policy throughput project.

Practical fallback if integrating original Procgen is initially expensive: use frozen image/trajectory fixtures for encoder correctness and timing while building the native pixel-observation path. Run a custom native pixel task for integration. Keep these diagnostics visibly separate from the external benchmark results.

## What to try to beat first

The first target is a **measured baseline on our hardware**, followed by a matched published protocol. We have not established current leaderboard scores or a SOTA claim. Nature, IMPALA, and Impoola are architecture controls; Cleanba, EnvPool, and PlayTrain provide systems comparisons with different integration and protocol requirements.

| Priority | Target and comparison | What would count as a useful win? |
|---|---|---|
| 0: establish a trustworthy baseline | Faithful Nature and IMPALA encoders; original pixel environment; numerical and learning checks | Reproducible baseline report. This is infrastructure, not a superiority claim. |
| 1: implementation speed | Same encoder and learner semantics, optimized reference versus PufferLib implementation; start with the four Procgen pilot games | Lower end-to-end time to a predefined return, with comparable learning curves, fixed resources, and repeatability checks. Kernel speed alone is a narrower result. |
| 2: CNN efficiency | Nature, IMPALA, **Impoola**, and candidate CNNs under the same backend and learner | Better held-out return at equal time/compute, or less time to the same return, with uncertainty and no hidden increase in tuning budget. |
| 3: publishable breadth | Frozen finalists and controls across all 16 original Procgen games under one matched protocol | The pilot advantage survives broader games, fresh seeds, and locked evaluation. Report per-game regressions as well as aggregate gains. |
| 4: encoder/core allocation | Independent encoder and recurrent sizes; validated memory conditions in Procgen or paired POPGym Arcade tasks | Evidence showing when moving compute to the core helps, including cases where it does not. |
| 5: transfer | Original ALE panel, then the chosen full suite; pixel continuous control later | The improvement survives a second task distribution without silently changing evaluation rules. |

**First architecture result to aim for:** improve the return-versus-time frontier over reproduced IMPALA and Impoola baselines on the four-game Procgen pilot, retaining Nature as the cheap control. Then confirm on all 16. This is more specific than aiming to beat an unspecified “Procgen score.” Impoola's author implementation documents a default generalization setting of 200 training levels and 25M steps; reproduce the paper's complete settings and metric before comparing reported numbers. [Impoola implementation](https://github.com/raphajaner/impoola), [local paper](papers/impoola.md)

For a combined PufferLib + new CNN result, use this comparison matrix where feasible:

| | Baseline CNN | Candidate CNN |
|---|---|---|
| Reference backend | A | B |
| PufferLib backend | C | D |

Compare A→B and C→D for architecture effects, A→C and B→D for implementation effects, and A→D for the combined outcome. Hold the environment and learner semantics fixed for attribution. If the native path also changes the optimizer, data collection, or update recipe, label that comparison as a whole-agent result and add ablations; it cannot isolate backend speed.

Keep two separate result ledgers: **locally reproduced baselines** with hardware, runtime, return curves, seeds, and exact configurations; and **published results** with paper version, game suite, protocol, resources, metric, and reported value. Populate numerical targets from verified tables or actual baseline runs. A score from Atari 100K, a throughput result on cloned games, and a Procgen held-out-level score cannot be ranked together.

## Exact semantics to freeze before comparing runs

Write a machine-readable protocol manifest before the first architecture sweep. At minimum include:

| Area | Fields to record |
|---|---|
| Environment | Package/source commit, game and ROM identifier/hash where applicable, difficulty/mode, action mapping, reward function, level distribution, reset behavior, time limits |
| Observation | Actual pixels vs state features; resolution; RGB/gray; channels order; uint8/float; normalization; crop/resize algorithm; max pooling; frame stack; previous-action/reward inputs; augmentation |
| Time | Emulator frames, environment transitions, agent decisions, action repeat, policy updates, learner transitions processed; actual totals after batch rounding |
| Termination | True termination vs time-limit truncation, bootstrap rule, life-loss pseudo-termination, autoreset/final observation handling, recurrent state reset |
| Learning | Algorithm, optimizer, learning-rate schedule, batch/minibatch size, rollout horizon, effective replay ratio, advantage computation, clipping, reward normalization, entropy schedule |
| Recurrence | Cell family, width/depth, input projection, state carry, truncation length, burn-in if any, train/inference sequence layout, policy version that produced each rollout |
| Runtime | GPU model/count, CPU model and assigned cores, RAM, versions and build flags, precision, graph capture, process/thread counts, concurrent workloads, memory footprint |

For Atari, retain seeded sticky actions when the selected benchmark requires them. Determinism means repeatability of the same seeded experiment, not removing stochasticity from the task. Explicitly set frame skipping at one layer so the emulator and wrapper do not accidentally repeat actions twice. Record random no-op starts, life-loss handling, episode limits, reward clipping, and full versus minimal action sets. [ALE evaluation protocol paper](https://arxiv.org/abs/1709.06009), [current ALE specifications](https://ale.farama.org/env-spec/)

Changing the frame stack while changing recurrent size confounds input information, encoder input channels, and memory capacity. Begin with a fixed observation protocol and add a separate frame-stack ablation. Likewise, increasing recurrent width without increasing the training horizon does not prove the wider network was trained to use longer memories.

## Measure learning and implementation separately

Keep these output panels for every candidate:

1. Evaluation return versus agent decisions, with raw per-game curves.
2. Evaluation return versus elapsed wall time, plus cold-start and steady-state timing definitions.
3. Evaluation return versus cumulative neural-network FLOPs; label analytical estimates as estimates.
4. Encoder/core/head parameter counts, actor forward time, learner forward/backward time, optimizer time, environment/render time, transfers, and peak memory.
5. Repeated-run determinism and accuracy results.

For time to target, choose the threshold and evaluation schedule from baselines/development data before looking at candidate final-test results. Report the first scheduled evaluation meeting the rule, the number of seeds that reach it, and uncertainty; non-reaching runs are right-censored at their allocated budget. Do not silently omit them or replace missing success with the budget endpoint. If a sustained-target rule requires two evaluations, state it beforehand. Avoid per-run best-checkpoint selection on the test set.

A useful accounting identity is:

```text
total network compute = actor forwards
                      + learner forwards and backwards
                      + auxiliary/target/world-model computation, if present
                      + evaluation forwards
```

Break every term into encoder, projection, recurrent core, and heads. State whether one fused multiply-add counts as two FLOPs. Do not assume backward is exactly twice forward: operation types, activation gradients, checkpoint recomputation, target networks, padding, and fused implementations matter. FLOPs exclude environment simulation, memory traffic, and launch latency; measured wall time captures those costs. Include optimizer and preprocessing time even if their arithmetic is excluded from “network FLOPs.”

Use measured actor batch sizes and learner sequence shapes. Batch-one latency is a different objective from throughput at thousands of environments. Include replay/update intensity: an encoder evaluated once per observation by actors may be trained repeatedly on that observation.

Plot empirical non-dominated points for return/time, return/compute, and return/memory; retain confidence intervals because a nominal frontier based on noisy point estimates can be misleading. Only fit a two-component compute-allocation law after sufficient budgets/tasks support it. Hilton-style intrinsic performance is a minimum-compute envelope over model families, not another name for raw return. [Scaling laws for single-agent RL](https://arxiv.org/abs/2301.13442)

## Determinism is an experimental result

Test three levels separately:

- **Kernel repeatability:** fixed tensors and seed yield the same outputs/gradients repeatedly under the same binary/hardware/configuration; otherwise quantify the difference.
- **Run repeatability:** same seed, fixed rollout composition, update order, RNG streams, and reset rules produce matching action/reward traces and checkpoint hashes where bitwise reproducibility is intended.
- **Statistical reproducibility:** independent training seeds give consistent conclusions, with uncertainty. This remains necessary even when individual runs are perfectly deterministic.

Do not assume a fixed seed is sufficient: reduction order, atomics, library algorithms, environment scheduling, and asynchronous policy versions can change trajectories. Assign RNG streams by stable logical environment identity, not worker completion order. Preserve optimizer, environment, and recurrent/RNG state if claiming exact resumed training. An identical checkpoint file hash requires stable serialization metadata as well as identical parameters.

Cleanba shows why actor/learner scheduling can change data composition and learning even when nominal hyperparameters match. Its controlled policy-version lag is directly relevant to this checkout's pipelining. Cross-hardware reproducibility of learning curves should not be described as guaranteed bitwise identity across GPU architectures. [Cleanba paper](https://arxiv.org/abs/2310.00036); local converted source: [cleanba.md](papers/cleanba.md).

No system changes are needed to specify or audit this methodology. Any future local experiments must respect the user's prohibition on changing CUDA, drivers, system libraries, or other virtual environments.

## What the current native sweep actually does

Relevant implementation: [config/default.ini](../config/default.ini), [src/pufferl.cu](../src/pufferl.cu), and [src/protein.cu](../src/protein.cu).

- Defaults sweep policy hidden size/layers and many learner/vector settings. They do not already constitute a controlled encoder/core experiment. Sweeping all defaults together cannot isolate the CNN's contribution.
- `sweep.downsample` controls the number of learning-curve summary points sent to Protein. It does **not** downsample input images or reduce the training budget by that factor. Summary points are binned means, with the final point replaced by the final recorded score/cost/steps.
- Trial `cost` is recorded from training `uptime`, and trial steps from `agent_steps`. This is a measured time objective, not a FLOP constraint. Track complete process startup/evaluation separately if using an end-to-end wall-clock claim.
- Bare `sweep.metric` names become `env/<name>`; the default is `score`. Training episode metrics are not an independent validation evaluator. A research runner must explicitly consume validation evaluations for architecture selection.
- Protein supports Pareto pruning and `early_stop_quantile`; the latter is used in fitting a score-versus-log-cost quantile model. The inspected scheduler receives results after a worker completes, so this setting should not be presented as implemented live ASHA-style trial termination. Likewise `max_suggestion_cost` filters predicted candidate costs, not a guaranteed wall-time kill limit. A hard per-trial resource cap needs an explicit runner mechanism. Multiple points from one learning curve are correlated observations, not independent seeds.
- The sweep chooses concurrent trials from `sweep.gpus / train.gpus` on disjoint GPU blocks, and by default `sweep.gpus = -1` selects all visible devices. Research manifests should set the resource allocation explicitly. Shared CPU/memory contention can still bias timing.
- The scheduler observes workers in completion order. Its fixed initial RNG seed does not by itself guarantee identical adaptive trial suggestions when job runtimes/order change. Save the suggestion manifest and observation order; use a fixed ordered grid for a deterministic first comparison.
- Native learner minibatch count is calculated as an integer from `replay_ratio * batch_size / minibatch_size`, where local rollout batch size is `total_agents * horizon`. Thus actual processed-transition ratio can differ from the requested float because of truncation. Log actual minibatch counts and processed transitions; validate that a small budget does not yield zero updates.
- `base.async = 1` is documented as a one-epoch-stale actor snapshot. `reset_every_horizon = 0` carries recurrent state by default. Keep these settings fixed and verify semantics in the actual architecture/backend used.

Separate architecture families into explicit run manifests until the configuration and checkpoint contracts support them. A string like `impala` is not automatically a valid native sweep dimension. The scaffold should reject unsupported combinations before launching a long run.

## A staged search that we can afford and trust

The following is a proposed starting budget, to be revised after measuring the first complete baseline. It is not authorization to start training now.

### Stage 0: correctness and cost reconnaissance

Implement faithful Nature and IMPALA references, plus current native baseline where compatible. Verify observation contracts, output shapes, gradients, parameter counts, serialization, recurrent resets, and a small end-to-end update. Time realistic actor/learner shapes. Compare the same architecture across implementations before attributing a speed difference to architecture. Collect one complete learning curve per baseline on a cheap task to estimate meaningful pilot budgets.

### Stage 1: narrow, fixed development grid

Start with two encoder families × two widths × three core choices: feedforward/no-memory, and one recurrent cell at widths 128 and 256. That gives 12 configurations. Define “width” explicitly as the channel tuple and keep the projection output dimension fixed for this initial comparison. Core width can otherwise silently resize the encoder FC projection and change what is being measured.

Use the four-game Procgen development panel and three predetermined training seeds: 144 pilot runs. A provisional 1M decisions per run is 144M decisions total, useful for smoke testing/early trends only; increase it if learning has barely begun. Reuse those curves for budget calibration, not a final ranking. Record all failed/diverged/OOM runs and their consumed time. Do not promote a candidate merely because it won one seed.

Give every family the same initial learner configuration and tuning allowance. This estimates the architecture effect under a common recipe. Later give finalists equal limited tuning budgets to test whether rankings survive reasonable optimization; these answer different questions and both should be labeled.

### Stage 2: methodical expansion

Advance approximately four configurations plus baseline controls. Add one change at a time: pooling/projection strategy, residual depth, channel allocation, resolution, separable convolutions, or encoder/core redistribution. Re-measure real time after each change. Treat reduced resolution as an information ablation, not just a free optimization. Include larger-core/smaller-encoder candidates on matched-compute contours, with a stated matching tolerance.

For budget-aware racing, promote using predetermined validation criteria and common rung budgets; keep at least some exploration slots for slower-starting candidates. Do not assume an early-return ranking predicts long-budget performance. Finalists must run unpruned at a common confirmation budget. Retraining from scratch versus resuming changes consumed compute and sometimes schedules; log which was used.

### Stage 3: confirmation and locked evaluation

Freeze architectures and tuning rules, then train all finalists at the full chosen budget on broader games with new training seeds. Five seeds per configuration/task is a reasonable initial planning number, not a guarantee of adequate power; add seeds when confidence intervals leave the practical decision unresolved. A 4-configuration × 16-game × 5-seed × 25M-step confirmation already costs **8 billion agent decisions**. Estimate GPU-hours from actual pilot throughput before committing to that scale.

Use held-out levels and, where possible, games that did not guide architecture decisions. The final test set is accessed once after selection under a documented evaluation protocol. Repeatedly checking it and adjusting the architecture makes it another validation set. Report both per-task raw scores and normalized aggregate scores using published normalization constants; include IQM, stratified-bootstrap intervals, performance profiles, and probability-of-improvement analyses when appropriate. More evaluation episodes reduce within-policy noise but do not substitute for independent training seeds. [Statistical evaluation paper](https://arxiv.org/abs/2108.13264), [rliable implementation](https://github.com/google-research/rliable)

### Runner outline

```text
load immutable protocol, candidate definitions, resource limits, seed lists
expand a fixed trial manifest; validate shapes, backend support, and budgets
for each trial in the declared schedule:
    materialize resolved configuration and source/build identity
    run correctness gate if this architecture/backend pair is new
    launch isolated trial; capture status, logs, checkpoints, elapsed resources
    evaluate at fixed milestones on declared validation levels/seeds
    append actual steps, updates, time, FLOPs estimate, memory, and returns
after a complete rung:
    summarize across paired tasks/seeds and apply declared promotion rule
freeze finalists; launch fresh-seed unpruned confirmation
evaluate locked test set and generate tables, curves, uncertainty, frontier
```

Each trial needs an immutable ID derived from a canonical resolved configuration plus seed, protocol ID, code/build identity, and backend. Retrying a crash must not overwrite the failed attempt. Save stdout/stderr, exit status, checkpoint selection rule, actual stopping reason, and precision/numerical checks. Keep a structured append-only result record in addition to human-readable reports. Compare candidates on shared seed lists, while recognizing that policies with different actions do not experience identical trajectories.

Charge the entire search: discarded architectures, compilation, validation, crashes, pilot runs, and hyperparameter tuning. Report both the winning model's training cost and total research/search cost. Selective disclosure of only the final fast run obscures whether the procedure was economical.

## Recency check and sources worth following

This is a focused scan as of 2026-09-11, not an exhaustive survey or verified current leaderboard.

- **PlayTrain, 2026-09-08:** an immediately relevant new systems comparator. Table 1 uses **four H100 GPUs and 92 CPU cores on one node**, with 64×64 RGB and frame skip one. Under the IMPALA learner, the reported 24-game geometric-mean throughput is approximately 1.07M agent decisions/s with Nature-CNN and 0.35M with IMPALA-CNN. These are whole-node results, not single-GPU numbers. Match architecture, buffering, timing boundary, and hardware before quoting speedups. Its Atari/Procgen-inspired clones are different tasks from the original suites. [Paper](https://arxiv.org/abs/2609.09059), [local text, Table 1 and section 3.1](papers/playtrain.md).
- **POPGym Arcade, 2025:** unusually close to the actual encoder/memory question because it offers paired observability conditions with pixels. Audit its rendering resolution and integration cost before choosing the first task. [Paper](https://arxiv.org/abs/2503.01450), [official code](https://github.com/bolt-research/popgym-arcade).
- **On the Role of Computation in RL, 2026:** relevant to the distinction between parameter size and repeated computation within a decision. It does not directly establish an optimal CNN/recurrent FLOP ratio. [Paper](https://arxiv.org/abs/2602.05999), [local text](papers/computation-rl-2026.md).
- **Cleanba and EnvPool:** strong existing systems baselines. EnvPool accelerates environment execution; Cleanba couples efficient environments with a reproducible distributed learner. Compare against optimized baselines when evaluating a native CNN contribution. [Cleanba](https://arxiv.org/abs/2310.00036), [EnvPool](https://arxiv.org/abs/2206.10558), [local EnvPool text](papers/envpool.md).

Additional locally converted sources: [ALE evaluation protocols](papers/ale-protocols.md), [POPGym Arcade v8](papers/popgym-arcade.md), [Memory Gym v6](papers/memory-gym.md), and [original POPGym v1](papers/popgym.md). POPGym Arcade v8 is titled *Investigating Memory in Model-Free RL with POPGym Arcade*; earlier versions/search results use *POPGym Arcade: Parallel Pixelated POMDPs*. Original POPGym emphasizes small observations and memory-model comparisons; it is valuable context, but its ordinary vector observations cannot substantiate a CNN claim.

## Decision before implementation

The first concrete milestone should be a faithful two-encoder reference, a supported original-pixel environment path, and one end-to-end baseline report containing return/time/compute and a repeated-seed determinism check. That will reveal whether the immediate bottleneck is image generation, transfer, convolution, projection, recurrent computation, or learner overhead. It also provides a credible starting point for a clean PufferLib contribution without assuming custom CUDA must be the first or largest win.
