# Robocode 5c Vanilla Self-Play Analysis

Date: 2026-07-28

## Purpose

This document analyzes the current PufferLib 5c Robocode implementation as a
reference for a pure, native Dogfight self-play experiment.

The target meaning of "vanilla self-play" is:

- stock PufferLib core;
- two ordinary agents exposed by the environment;
- PufferLib's built-in current-policy and frozen-policy routing;
- PufferLib's built-in checkpoint pool and opponent rotation;
- no Dogfight coordinator in the training path;
- no ELO requirement;
- no Dogfight-specific trainer or PPO fork.

This is a source analysis, not a completed Robocode benchmark. Commands in this
document are source-derived starting points and were not executed as part of
this analysis. Future agents must confirm them against the checkout rather than
treating them as permanent CLI guarantees.

## Executive conclusion

The current Robocode design is simple:

1. Robocode exposes two symmetric agent slots.
2. Most environments run current policy versus current policy.
3. A configured fraction run current policy versus one frozen historical
   policy bank.
4. PufferLib periodically saves the current policy into a FIFO pool.
5. PufferLib uniformly samples a historical checkpoint for the frozen bank on
   a fixed step cadence.
6. Opponent replacement occurs only after every affected environment reaches
   an episode boundary.
7. A final checkpoint can be evaluated against a small subset of the pool.

There is no rating-aware matchmaking in the current C++ path. Win rate does not
trigger snapshots, promotion, or opponent changes. Difficulty can still rise
because the pool contains progressively newer policies, but selection is
uniform and cadence-driven.

The self-play topology is directly applicable to Dogfight. Robocode's reported
throughput is not directly transferable. Robocode uses a small 2D simulation,
compact observations, discrete actions, thousands of simultaneous matches, and
no RK4 aircraft dynamics. Dogfight can adopt the batching, lifecycle, and
historical-opponent design while remaining substantially slower per
environment step.

The repository contains no Robocode benchmark log or documentation supporting
the recalled claim of approximately 100M training SPS. That number must be
treated as unverified until measured on this workstation. It may refer to an
environment-only benchmark, a different revision, different hardware, or a
different agent-count definition.

## Source and version provenance

The reference files are:

- `config/robocode.ini`
- `config/default.ini`
- `ocean/robocode/robocode.h`
- `src/pufferl.cu`
- `build.sh`
- `pufferlib/pufferl.py`
- `pufferlib/selfplay.py`

The current Robocode files have no changes relative to the pinned Joseph 5c
base:

```text
ebf5ed03cc3524076b6c1a4033bd69cec0b3db22
```

Relevant history:

| Commit | Meaning |
|---|---|
| `5cd76d4d` | Reports a good self-play result: "51 vs minimal, 67 vs surver, 38/44 with the 2 harder policies" |
| `0d851dea` | Ports Robocode forward |
| `fd8de357` | Refactors the current C++ trainer and Robocode configuration |
| `c908f2e8` | Performs additional trainer/configuration cleanup |

The successful-result commit predates material trainer and configuration
refactors. It is evidence that the general approach learned useful behavior,
not proof that the current configuration reproduces that result unchanged.

The current Dogfight experiment branch topology is:

| Branch | Purpose |
|---|---|
| `dogfight5c` | Preserved prior committed line at `a8fc98fb` |
| `dogfight5c-clean-core-baseline` | Stock-core baseline and evaluation work at `fe1c1332` |
| `dogfight5c-vanilla-selfplay` | Robocode-style experiment based on `fe1c1332` |

Starting the vanilla experiment from `fe1c1332` is intentional. Starting from
`a8fc98fb` would restore the earlier Dogfight-specific PufferLib-core changes
and would not be a clean vanilla comparison.

## Which trainer path is authoritative

There are two self-play implementations in the tree:

- The native C++/CUDA executable built from `src/pufferl.cu`.
- The Python orchestration path in `pufferlib/pufferl.py` and
  `pufferlib/selfplay.py`.

The native build compiles the selected environment header directly into one
executable:

```text
NVCC + src/pufferl.cu + ENV_HEADER -> ./puffer
```

The command shape used by the current native workflow is:

```bash
./build.sh robocode
./puffer train robocode
```

These commands are source-derived and were not run during this analysis.

The Dogfight work has used the same `./puffer` C++ executable path. Therefore,
the C++ behavior in `src/pufferl.cu` is authoritative for the proposed
Dogfight vanilla-self-play branch. The Python implementation is useful context
but must not be silently mixed into claims about the C++ trainer.

## Robocode environment contract

### Agent topology

Robocode exposes:

```c
Agent agents[2];
int num_agents = 2;
int num_bots = 0;
```

At initialization it declares:

```c
env->agents[0].policy = 0;
env->agents[1].policy = 1;
```

These declarations do not mean slot 1 is always frozen. The vectorizer decides
whether to honor policy 1 based on whether an environment belongs to the
historical-opponent region.

### Observation space

Each agent receives 22 floats:

| Group | Width | Contents |
|---|---:|---|
| Ego | 14 | Position, headings, velocity, energy, sampled dynamics multipliers, and reward coefficients |
| Other | 8 | Ego-frame relative position, relative headings, opponent energy, aim error, and presence flag |

The other-agent features are zero when radar has not found a target.

Important design choices:

- Relative position is rotated into the observing robot's body frame.
- Each slot computes observations from its own perspective.
- Domain-randomization multipliers are included in observations.
- Reward coefficients are included in observations.
- The opponent feature block includes an explicit validity flag.

Including randomized parameters in observations makes the task conditioned
rather than partially hidden. That is a strong candidate insight for Dogfight
if physics or reward parameters are randomized.

### Action space

Robocode has five discrete action heads:

```text
acceleration: 4
body turn:    9
gun turn:    11
radar turn:  11
firepower:   6
```

This is structurally different from Dogfight's continuous flight controls.
Dogfight should not discretize controls merely to copy Robocode. The applicable
insight is the fixed, compact action ABI and fully batched inference, not the
specific action representation.

### Reset

`puf_reset`:

- resets the shared episode tick;
- clears bullets;
- samples non-colliding positions;
- resets robot kinematics, energy, gun heat, and logs;
- samples per-agent domain-randomization multipliers;
- copies per-slot reward coefficients;
- computes a fresh initial observation.

It deliberately does not clear `boundary_reached`. That flag belongs to
PufferLib's self-play opponent-alignment protocol and must survive the immediate
environment reset.

### Step and terminal behavior

Both agents advance in one `puf_step` call. The environment:

- zeros both reward and terminal outputs;
- advances bullets;
- resolves collisions and damage;
- applies both agents' actions;
- checks deaths after important resolution phases;
- terminates both agents together;
- treats a simultaneous disable as a draw;
- treats `max_ticks` exhaustion as a draw;
- logs the result;
- immediately resets and emits fresh observations.

This joint transition is important. A two-agent environment must not reset one
aircraft while the other continues the old episode.

### Outcome and logging

At episode end:

- slot-0 win gives slot 0 a match score of `1.0`;
- slot-0 loss gives slot 0 a match score of `0.0`;
- draw gives both slots `0.5`;
- both terminal flags are set;
- the aggregate episode count increases by two agent records.

The slot scores are multiplied by `num_agents` before aggregation so
`slot_0_score / n` is a direct win rate after the generic logger averages by
agent count.

Metric meanings are easy to confuse:

| Metric | Meaning |
|---|---|
| `perf` | Scripted bots killed; normally zero during two-agent self-play |
| `score` | Damage dealt, not match win rate |
| `slot_0_score` | Slot-0 win/draw score |
| `slot_1_score` | Slot-1 win/draw score |
| `draw_rate` | Draw fraction |

Dogfight evaluation should use explicit outcome metrics rather than assume
`perf` or shaped return is a win rate.

## Exact C++ policy routing

The C++ vectorizer creates contiguous physical rollout rows for the current
policy and each frozen bank.

For Robocode's one-bank configuration:

| Environment region | Slot 0 | Slot 1 | Environment tag |
|---|---|---|---:|
| Normal self-play | Current policy, trainable | Current policy, trainable | `0` |
| Historical self-play | Current policy, trainable | Frozen bank 1, rollout only | `1` |

For normal environments, the vectorizer overrides both declared policies to
policy 0. For historical environments, it respects the environment's declared
policy IDs `0` and `1`.

The vectorizer binds each agent's observation, action, reward, and terminal
pointers directly to the appropriate physical row. Robocode does not manually
load or invoke a frozen policy.

Inference loops over contiguous bank slices:

1. Run the primary network over all primary rows.
2. Run frozen network 1 over its contiguous rows.
3. Copy the resulting actions back through the bound environment pointers.

Frozen policy banks use the same policy-construction path as the primary
network but have separate immutable weights, activations, and recurrent state.

Before prioritized replay is built, PufferLib explicitly zeros advantages on
every frozen row. This prevents frozen actions and log probabilities from
entering primary-policy PPO updates.

## Current Robocode configuration

The important current values are:

```ini
[vec]
total_agents = 16384
num_buffers = 4
num_threads = 4
num_frozen_banks = 1
frozen_bank_pct = 0.1
frozen_bank_hidden_size = 256
frozen_bank_num_layers = 4

[selfplay]
enabled = 1
max_size = 100
seed = 42
opp_timeout_steps = 100000000
eval_games = 4096
eval_pool_size = 8

[env]
num_agents = 2
num_bots = 0
dr = 0.6
max_ticks = 3000

[policy]
hidden_size = 256
num_layers = 4

[train]
gpus = 1
total_timesteps = 2_000_000_000
learning_rate = 0.0072914747025563794
gamma = 0.9996207050319893
gae_lambda = 0.8883938238847433
minibatch_size = 8192
horizon = 128
prio_alpha = 1.0
prio_beta0 = 1.0
```

Inherited defaults that materially affect the run:

```ini
[base]
cudagraphs = 1
async = 1
reset_every_horizon = 0

[torch]
network = MinGRU

[train]
anneal_lr = 1
replay_ratio = 1.0
clip_coef = 0.2
vf_coef = 2.0
vf_clip_coef = 0.2
max_grad_norm = 1.5
ent_coef = 0.001
momentum = 0.95
vtrace_rho_clip = 1.0
vtrace_c_clip = 1.0
```

Recurrent state carries across rollout horizons and resets on real terminals.
The frozen bank maintains its own recurrent state.

## Configuration math

With 16,384 agent rows and two agents per environment:

```text
total environments = 16,384 / 2 = 8,192
agent rows per buffer = 16,384 / 4 = 4,096
environments per buffer = 4,096 / 2 = 2,048
historical environments per buffer = floor(0.1 * 2,048) = 204
historical environments total = 204 * 4 = 816
```

The current C++ implementation applies `frozen_bank_pct` to environments, not
directly to agent rows. Each historical environment has one frozen row:

```text
historical environment fraction = 816 / 8,192 = 9.96%
frozen rollout-row fraction = 816 / 16,384 = 4.98%
current-policy rollout rows = 15,568 / 16,384 = 95.02%
```

The rollout batch is:

```text
16,384 agents * horizon 128 = 2,097,152 agent steps per epoch
```

The configured two billion steps produce:

```text
floor(2,000,000,000 / 2,097,152) = 953 train epochs
aligned trained steps = 1,998,585,856
```

`checkpoint_interval = 100` therefore means approximately:

```text
209,715,200 agent steps between interval checkpoints
```

The run produces an initial pool checkpoint, nine interval checkpoints, and a
final checkpoint, for approximately 11 pool entries. The configured pool cap
of 100 is not reached in the default two-billion-step run.

## Pool and opponent lifecycle

The current C++ lifecycle is:

1. Create the primary and frozen policies.
2. Save the initial primary policy at global step zero.
3. Add that checkpoint to the in-memory pool.
4. Uniformly sample the pool and load one checkpoint into each frozen bank.
5. Perform rollout and training epochs.
6. Add normal interval/final checkpoints to the pool.
7. When a bank has been active for `opp_timeout_steps`, uniformly sample a
   pending replacement path.
8. Clear the boundary flags for environments tagged with that bank.
9. Continue training until every tagged environment has completed an episode.
10. Load the pending checkpoint into the bank.
11. Clear the flags and begin the next timeout interval.

With one bank, every historical environment on a rank faces the same frozen
checkpoint at a given time. The system does not sample a different historical
opponent for every environment.

Opponent selection is:

```text
uniform random checkpoint from the current pool
```

It is not:

- PFSP;
- ELO weighted;
- weakest-unmastered selection;
- nearest-skill selection;
- win-rate-triggered promotion.

The C++ pool evicts the oldest single entry when it reaches its cap. The current
two-billion-step Robocode run does not reach the cap.

Opponent replacement is polled on the trainer's logging cadence as well as the
configured step cadence. The step threshold is therefore a lower bound, not an
exact swap instant.

## Why boundary alignment matters

Changing a frozen policy in the middle of an episode corrupts the transition
contract:

- the hidden state belongs to the previous opponent;
- the episode return mixes two opponent generations;
- the learner observes a discontinuous adversary without a terminal;
- match statistics no longer describe one matchup.

Robocode solves this with two environment fields:

```c
int tag;
int boundary_reached;
```

`tag > 0` identifies the frozen bank used by that environment.
`boundary_reached` is set at episode end and deliberately survives reset.
PufferLib loads a new bank only after all environments with that tag have set
the flag.

Dogfight must preserve the same rule. Every self-play scenario also needs a
finite episode bound; one non-terminating tagged environment can indefinitely
delay a bank rotation.

## Reward and task design

Robocode combines:

- sparse `+1/-1` kill rewards;
- dense damage-inflicted reward;
- dense damage-taken penalty;
- wall-collision damage;
- optional radar spotting reward, currently zero;
- random speed, handling, and power multipliers.

The current coefficients are not strictly zero-sum:

```text
range damage inflicted = +0.017177545879507577 per damage
damage taken = -0.004918136926337855 per damage
melee damage inflicted = +0.007448376232810955 per damage
```

Both learning slots use the same defaults, so the game is seat-symmetric even
though the shaped reward sum need not be zero.

Robocode has no staged curriculum in the current configuration. It obtains
variation from:

- random positions;
- partial radar contact;
- per-agent dynamics randomization;
- current-policy co-adaptation;
- historical opponents.

This is a much easier exploration problem than six-degree-of-freedom aerial
combat. Pure Dogfight self-play from scratch may still collapse into poor but
mutually compatible behavior. That is an experiment to measure, not an
assumption that Robocode's no-curriculum result will transfer.

## Final pool evaluation

After training, the C++ trainer can evaluate the final checkpoint against up to
`eval_pool_size` historical checkpoints:

```text
up to 8 opponents
4096 games per opponent
mean slot-0 score
```

The current loop walks the pool in stored order and stops after the requested
number of opponents. With the default run this emphasizes older pool entries,
not a random or difficulty-ranked sample.

Match mode:

- puts policy A in slot 0;
- puts policy B in slot 1;
- routes every environment through the frozen-bank layout;
- disables self-play pool mutation;
- uses explicit slot-0 outcome score.

Robocode's random initial conditions are approximately seat-symmetric.
Dogfight has strongly asymmetric spawn geometries, so Dogfight evaluation must
also swap seats or mirror geometry. A one-seat mean is insufficient.

## Why Robocode can be fast

The likely throughput contributors are:

- The environment header is compiled directly into the native executable.
- No Python code runs in the per-step hot path.
- The simulation is compact 2D arithmetic with small fixed agent counts.
- Observation width is only 22 floats.
- Actions are five small discrete heads.
- There are 8,192 simultaneous matches.
- Four buffers divide work into large batches.
- Four configured threads provide one CPU worker per buffer.
- Host rollout arrays use pinned memory.
- Primary and frozen rows are physically contiguous for large GPU forwards.
- CUDA graphs are enabled.
- Asynchronous actor/learner pipelining is enabled.
- Recurrent state persists without reconstructing episode history.
- Pool disk I/O and opponent management occur only at coarse intervals.
- Rendering is absent during training.

The main point is not any single micro-optimization. The system keeps thousands
of homogeneous environments and large model batches in flight while avoiding
Python dispatch and per-environment model calls.

Dogfight currently trains around roughly 0.8M to 1.1M SPS in the observed
4,096-agent runs on this workstation. A 10x wall-clock improvement would be
material even if Dogfight remains far below Robocode. The correct goal is to
find Dogfight's batching and CPU-simulation ceiling, not to require an
unverified Robocode number.

## SPS claim status

No current repository documentation, configuration comment, or commit message
found in this analysis reports Robocode at 100M SPS.

The repository does contain the successful self-play commit `5cd76d4d`, but its
message reports opponent scores, not throughput.

Do not state that current Robocode trains at 100M SPS until a controlled
benchmark records:

- source commit;
- build mode;
- hardware;
- precision;
- `total_agents`;
- `num_buffers`;
- `num_threads`;
- policy architecture;
- environment-only versus end-to-end training SPS;
- warmup exclusion;
- elapsed wall time and aligned agent steps.

## Historical success versus current configuration

The successful-result configuration differs materially from the current file.

The older configuration included:

- `snapshot_interval = 200000000`;
- ELO fields;
- `sweep.league = True`;
- replay ratio about `3.3166`;
- clip coefficient `0.01`;
- explicit value, entropy, optimizer, and V-trace values;
- minibatch size `4096`.

The current configuration:

- removes ELO and league mode;
- couples pool insertion to normal checkpoint saves;
- uses final pool evaluation;
- uses minibatch size `8192`;
- inherits many optimizer defaults after the trainer refactor.

This means:

- the current implementation is more purely "vanilla";
- the structural self-play result has historical support;
- the current exact config is not proven by that old commit message alone;
- Dogfight should copy contracts and measure behavior, not copy a mythology.

## C++ and Python self-play differences

The Python helper is not identical to the current C++ trainer.

Material differences include:

| Area | Current C++ `./puffer` | Python helper |
|---|---|---|
| Pool eviction | Drop one oldest entry | Stride-evict older half to preserve coverage |
| Snapshot control | Normal checkpoint interval | Separate snapshot interval support |
| External pool | Not present in the C++ `Selfplay` struct | Explicit path/glob pool support |
| Shared state | Shared checkpoint paths and per-rank logic | JSON publication/synchronization |
| `frozen_bank_pct` | Fraction of environments in vectorizer | Used to size frozen agent rows |

For the current Robocode numbers, the percentage interpretation differs
substantially:

```text
C++: about 10% historical environments, about 5% frozen agent rows
Python helper: about 20% historical environments, about 10% frozen agent rows
```

This divergence should be treated as a maintenance risk. Dogfight's vanilla
experiment must record that it uses the C++ `./puffer` path.

## Test evidence and gaps

The current source includes runtime assertions for:

- valid frozen-bank count;
- valid bank policy IDs;
- matching bank layout across buffers;
- non-empty frozen bank slices;
- required frozen policy architecture;
- frozen-row exclusion from advantages.

No dedicated current Robocode native-self-play contract test was found under
`tests/`. The repository has a league test, but that is not a substitute for
the C++ Robocode routing and boundary protocol.

Dogfight should add local contract tests before relying on the topology.

## Applying the design to Dogfight

### What should be copied

- Two ordinary native agents.
- Slot policy declarations `0` and `1`.
- Current-current training in most environments.
- One frozen historical bank in a small environment fraction.
- Joint episode termination and reset.
- Explicit slot outcome metrics.
- Persistent bank tags and episode-boundary flags.
- Frozen-row exclusion from learner updates.
- Uniform checkpoint pool as the first baseline.
- Large homogeneous batches.
- CUDA graphs and asynchronous actor/learner execution.
- Match evaluation against fixed checkpoints.

### What should not be copied blindly

- Robocode reward coefficients.
- Discrete control heads.
- A 0.6 physics-randomization range.
- A 256x4 policy without a memory/VRAM measurement.
- `checkpoint_interval = 100` without converting it to agent steps.
- One-seat final evaluation.
- The assumption that self-play alone solves initial exploration.
- The unverified 100M SPS claim.

### Required pure-vanilla Dogfight mode

The branch should provide one unambiguous training mode with:

```text
selfplay.mode = native
selfplay.enabled = 1
env.num_agents = 2
scripted internal opponent disabled
coordinator disabled
one frozen bank
both aircraft controlled through ordinary Agent slots
stock src/algo.cu
stock src/pufferl.cu
```

The environment's only self-play responsibilities should be:

- expose both aircraft;
- compute both perspectives;
- consume both action rows;
- emit both rewards and terminals;
- report unbiased outcomes;
- provide `tag` and `boundary_reached`;
- reset both aircraft together.

Checkpoint selection and frozen inference remain PufferLib's job.

## Proposed Dogfight experiment sequence

### V0: Contract tests

Add Dogfight-local tests for:

- `agents[0].policy == 0`;
- `agents[1].policy == 1`;
- both current-policy slots act and receive gradients in tag-0 environments;
- the frozen slot cannot contribute learner advantages in tag-1 environments;
- both terminal flags pulse together;
- reset observations are fresh and finite;
- `boundary_reached` persists through reset;
- every scenario has a finite terminal bound;
- seat-swapped scenarios produce mirrored observations and rewards;
- fixed seeds reproduce spawn and outcome traces.

The frozen-gradient exclusion itself is generic core behavior and should be
verified without modifying core.

### V1: Throughput canary

Keep environment behavior fixed and measure a small matrix:

```text
total_agents: 4096, 8192, 16384
num_buffers:  2, 4, 8 where divisible
num_threads:  enough for at least one worker per buffer
async:        1
cudagraphs:   1
horizon:      128
```

Record:

- total SPS;
- environment time;
- model time;
- copy time;
- training time;
- GPU utilization;
- VRAM;
- episode completions;
- deterministic checkpoint hash for repeated pinned runs.

Do not compare learning runs with different total steps or different effective
update counts.

### V2: Pure self-play from scratch

Run the closest Robocode-style baseline:

- symmetric two-aircraft spawn distribution;
- no scripted opponent;
- no stage advancement;
- 90% current-current environments;
- 10% current-history environments;
- one uniformly sampled frozen checkpoint;
- no domain randomization initially;
- bounded episodes;
- fixed seed.

This is the branch's defining experiment even if it learns poorly.

### V3: Evaluate what actually learned

Use independent gates:

- both-seat matches against initial and intermediate pool checkpoints;
- fixed scripted curriculum stages;
- AutoAce when appropriate;
- deterministic headless episode sets;
- `DISPLAY=:0` visible evaluation;
- human judgment of control stability, energy management, and tactical flight.

Self-play win rate against itself is expected to remain near 50% and is not an
absolute skill metric.

### V4: Only then try variants

If pure self-play is stable but weak, compare:

- curriculum checkpoint followed by vanilla self-play;
- Dogfight-local progressive spawn distributions during vanilla self-play;
- a larger historical environment fraction;
- more than one frozen bank;
- different checkpoint and opponent cadence;
- reward changes supported by fixed-scenario tests.

Keep each variant on a separate commit or branch and change one causal factor
at a time.

## Source-derived configuration sketch

This is a proposal, not a verified working command:

```ini
[base]
checkpoint_interval = 100
cudagraphs = 1
async = 1
reset_every_horizon = 0

[vec]
total_agents = 16384
num_buffers = 4
num_threads = 4
num_frozen_banks = 1
frozen_bank_pct = 0.1
frozen_bank_hidden_size = 128
frozen_bank_num_layers = 4

[selfplay]
mode = native
enabled = 1
max_size = 100
seed = 42
opp_timeout_steps = 100000000
eval_games = 4096
eval_pool_size = 8

[env]
num_agents = 2
curriculum_enabled = 0
role_randomization = 1
domain_randomization = 0
vertical_spawn_prob = 0

[policy]
hidden_size = 128
num_layers = 4

[train]
gpus = 1
total_timesteps = 2000000000
horizon = 128
minibatch_size = 8192
```

The primary and frozen architectures must match unless checkpoint compatibility
is independently established.

If Dogfight remains at 4,096 agents with horizon 128, one epoch is only 524,288
agent steps. Matching Robocode's approximate 209.7M-step snapshot cadence would
require a checkpoint interval near 400, not 100.

## Performance expectations for Dogfight

Possible gains:

- more simultaneous environments;
- fewer small inference calls;
- better overlap from four or more buffers;
- one native executable;
- no coordinator in the hot path;
- no per-step Python;
- no per-environment opponent model invocation;
- frozen bank inferred as one contiguous batch.

Hard limits:

- two six-degree-of-freedom aircraft per environment;
- RK4 integration;
- aerodynamic calculations and trigonometry;
- longer episodes;
- continuous actions;
- more complex terminal and reward logic;
- rendering assets, although rendering remains disabled in training.

A lower SPS system can still be much better if it learns more skill per sample.
The comparison must report both:

```text
skill versus wall-clock time
skill versus agent steps
```

## Primary risks

- Pure self-play may converge to mutually weak behavior.
- Long or heterogeneous Dogfight episodes may delay frozen-bank rotation.
- Asymmetric spawns can create seat bias.
- Non-zero-sum shaping can reward degenerate mutual behavior.
- Uniform pool sampling can over-sample obsolete opponents.
- One bank provides limited opponent diversity at any instant.
- Current final pool evaluation emphasizes early stored checkpoints.
- A larger policy or agent count may reduce rather than increase SPS.
- The current C++ path has less pool sophistication than the Python helper.
- The historical successful Robocode run used different optimizer settings.

## Recommended next implementation

The next code change on `dogfight5c-vanilla-selfplay` should be narrow:

1. Add a Dogfight-local test defining the exact Robocode-style two-agent,
   tag, boundary, and policy-slot contract.
2. Add a clearly named vanilla native-self-play configuration profile.
3. Do not change `src/algo.cu` or `src/pufferl.cu`.
4. Build and run the full Dogfight test suite.
5. Run a short deterministic current-current/frozen-bank canary.
6. Measure SPS before changing physics, rewards, or PPO settings.
7. Run headless matches and a visible `DISPLAY=:0` flight evaluation.
8. Keep the result on this experimental branch until both learning and flight
   quality justify promotion.

The first question is not whether vanilla self-play produces an ace. The first
question is whether stock PufferLib can route, train, freeze, rotate, and
evaluate two real Dogfight agents correctly and substantially faster without
damaging aircraft behavior.
