# Dogfight 5c: Clean-Core Gaps, Recovery Options, and Merge Path

Date: 2026-07-28

## Purpose

Dogfight previously reached the scripted AutoAce portion of the curriculum
while this branch contained substantial changes to PufferLib core. Those core
changes were useful for research, but they made a Dogfight pull request harder
to review and merge.

The current decision is:

1. Keep Joseph's pinned 5c core stock on the merge-oriented branch.
2. Keep almost all Dogfight work under `ocean/dogfight` and
   `config/dogfight.ini`.
3. Preserve the old core-enabled result on a separate benchmark branch.
4. Recover genuinely generic capabilities through small, independently tested
   upstream PufferLib pull requests.
5. Do not quietly reintroduce the old core patch into the Dogfight pull
   request.

Pinned Joseph base:

```text
ebf5ed03cc3524076b6c1a4033bd69cec0b3db22
```

Last known core-modified working point:

```text
a8fc98fb
```

Safety branch containing that point:

```text
dogfight5c-core-working
```

At the time this document was written, the working-tree versions of
`src/algo.cu` and `src/pufferl.cu` matched the pinned Joseph base. They can
still appear modified relative to the current branch commit until the core
restoration itself is committed.

## Executive Summary

Restoring stock PufferLib did not prove that stock 5c cannot train Dogfight.
The opposite was demonstrated:

```text
stock core + conservative Dogfight defaults: poor fixed-stage-0 learning
stock core + proven Phase 2 profile:          100% fixed-stage-0 eval
```

However, restoring stock core removed more than custom self-play code. It also
removed:

- true training warm-start;
- exact runner global-step delivery;
- one globally aggregated curriculum controller;
- stronger exclusion of frozen-policy trajectories from PPO;
- generation-safe frozen-opponent rotation;
- self-play cohort evidence and configuration guards;
- native render and the BF16-compatible evaluation horizon.

The Dogfight-local eval wrapper restores the last item operationally without
editing PufferLib core. The other losses must be treated separately.

## What Warm-Start Means

Warm-start begins a new training run by loading an existing checkpoint into
the primary trainable policy before the first rollout.

A warm-start keeps:

- neural-network weights;
- the behavior already learned by the checkpoint.

A warm-start normally resets:

- optimizer state;
- replay or rollout buffers;
- recurrent hidden state;
- random-number streams;
- wall-clock and global-step counters;
- curriculum controller state, unless explicitly restored separately.

Warm-start is not the same as resume.

Resume attempts to continue the exact previous run, including optimizer,
schedule, random state, step count, and other training state. Warm-start
creates a new experiment using only the learned policy as its starting point.

Dogfight needs warm-start for transitions such as:

```text
scripted curriculum checkpoint
    -> AutoAce-focused training
    -> native self-play
    -> historical-opponent or league training
```

The removed PufferLib patch made `base.load_model_path` load the primary
policy inside `run_train`. It also synchronized the actor copy in asynchronous
mode. Stock pinned 5c loads `base.load_model_path` for evaluation, but not for
training. Therefore, a stock-core training command that includes
`base.load_model_path` must not be assumed to warm-start.

This is one of the strongest explanations for a regression after switching to
native self-play. A competent curriculum policy may previously have seeded
self-play, while the clean-core run starts both sides from an untrained policy.

## What "Smart Curriculum Advancement" Previously Did

The core-modified runner owned one `PufCurriculumState` for the entire
training process.

At rollout and logging boundaries it:

1. delivered the true global training step to every Dogfight environment;
2. aggregated completed-episode wins and episode counts across all envs;
3. evaluated a mastery window only after warmup, interval, and evidence gates;
4. compared the relevant stage win rate with the mastery threshold;
5. advanced one authoritative curriculum target;
6. applied that same target to every environment;
7. handled log-counter clearing without double-counting evidence.

That was smart, globally coordinated curriculum advancement.

The current no-core fallback retains the threshold and target logic, but each
`Dogfight` environment owns its own `local_curriculum` state. Each env sees
only its own completed episodes. Consequences include:

- `min_eval_episodes=50` means 50 episodes per env, not 50 globally;
- targets can advance at different times;
- some envs can remain permanently behind;
- evidence collection depends strongly on `vec.total_agents`;
- `global_step_stride` is an approximation that must match the vector layout;
- dashboard averages can hide a heterogeneous population of targets;
- deterministic advancement is harder because envs run on worker threads.

Therefore, we did not lose all curriculum intelligence. We lost the
authoritative global evidence scope and synchronization that made the
intelligence trustworthy.

## Exact PufferLib Core Changes Removed

The old patch changed two core files:

```text
src/algo.cu     109 lines added or changed
src/pufferl.cu  537 lines added or changed
total           563 insertions, 83 deletions
```

### Training and curriculum

| Removed capability | Effect after restoration |
|---|---|
| Runner-delivered true global step | Dogfight uses a configured local stride |
| One runner-owned curriculum state | Curriculum evidence is per environment |
| Global log aggregation before mastery decisions | No single authoritative win/loss window |
| Training load of `base.load_model_path` | New training runs start from random weights |
| Primary/actor weight synchronization after load | Relevant only if warm-start is restored |

### PPO and frozen self-play rows

| Removed capability | Effect after restoration |
|---|---|
| Compact prioritized-replay domain containing trainable rows only | Stock behavior again owns frozen-row handling |
| Mapping compact sampled indices back to physical rollout rows | Removed with the compact priority domain |
| Explicit `mb_trainable_mask` applied to every PPO loss and gradient | Lost defense-in-depth against frozen-row updates |
| Priority buffers sized only for primary rows | Priority buffers use stock sizing |

Pinned 5c already tries to suppress frozen rows by zeroing their advantages.
The removed patch made the exclusion structural and added an all-loss safety
mask. This distinction matters only when frozen banks are enabled. It does not
explain poor one-agent, no-self-play fixed-stage learning.

### Native self-play lifecycle

| Removed capability | Effect after restoration |
|---|---|
| External historical-opponent seed | Less control over the first frozen opponent |
| Sampling that excludes the currently loaded checkpoint | Greater risk of uninformative rematches |
| Synchronous generation barrier | Frozen-policy changes rely on stock rotation behavior |
| Reset/truncate affected games at a swap | Generation boundaries are less explicit |
| Clear frozen recurrent state after loading weights | State can be harder to reason about across generations |
| Clear corresponding primary recurrent rows on forced reset | Removed with atomic rotation |
| Track bank generation and swap truncations | Lost diagnostic evidence |
| `off`, `native`, and `coordinator` mode validation | Fewer preflight guards |
| One-GPU, synchronous, topology, and ownership guards | Invalid configurations fail less clearly |
| Current-vs-current and current-vs-history cohort metrics | Harder to tell what self-play is actually learning |

### Checkpoints and evaluation

| Removed capability | Current replacement or consequence |
|---|---|
| Checkpoint byte-size validation before load | Stock load behavior |
| Primary-weight loading helper | Stock eval load path |
| Disable frozen self-play banks for ordinary eval | Must be supplied explicitly in commands |
| BF16-compatible eval horizon | Restored by `ocean/dogfight/build_eval.sh` |
| Native `render` command | Restored by the Dogfight-local eval wrapper |
| Detailed match evidence | Reduced evidence in stock commands |

The eval wrapper is acceptable as a temporary Dogfight-local tool. It copies
`src/pufferl.cu` to a generated build file and does not modify the source
checkout.

## Current Evidence

### Failed conservative fixed-stage baseline

Multiple stock-core runs using the conservative defaults failed to master the
early task. Visible policies showed persistent control problems, including a
right-roll tendency.

### Rejected multidiscrete experiment

A Dogfight-local multidiscrete action adapter removed the persistent right
roll, but the resulting policy tended to nose down and achieved only about
12.7% fixed-stage-0 headless performance. The adapter was reverted.

### Successful stock-core continuous baseline

Run:

```text
no_core_continuous_phase2profile_fixed0_32m_20260728_a
```

Checkpoint:

```text
checkpoints/dogfight/no_core_continuous_phase2profile_fixed0_32m_20260728_a/0000000033554432.bin
```

Results:

```text
training crossed 90% near 25.7M steps
fixed-stage-0 headless perf = 1.0000
fixed-stage-0 headless score = 1.000
human eval: strong on immediate intercepts, weak when it needed to build energy
```

This is decisive evidence that stock 5c can train the continuous Dogfight
environment when given the proven profile.

### Interrupted mixed curriculum run

Run:

```text
no_core_continuous_phase2profile_curriculum2_64m_20260728_a
```

The process aborted with `SIGABRT` after writing checkpoints through:

```text
checkpoints/dogfight/no_core_continuous_phase2profile_curriculum2_64m_20260728_a/0000000037748736.bin
```

The latest visible training logs before truncation showed approximately 60%
aggregate performance and an average stage near 0.9. Policy entropy rose from
about 7 to about 17 during the run. This correlation does not yet prove that
entropy caused the abort.

Headless vector evaluation of the saved checkpoint reported 100% on fixed
stages 0, 1, and 2. Human-visible serial stage-2 evaluation finished around
72%. That disagreement must be resolved before using either value as a sweep
objective.

Stage 2 was genuinely fixed stage 2:

```text
CURRICULUM_VERTICAL
opponent 200-400 m ahead
opponent 200-400 m above or below
lateral offset up to 50 m
```

The human reported acceptable but not great behavior and no stuck ailerons.

## Why Mastery Probably Dropped

The leading explanations, in current evidence order, are:

1. The conservative default profile is poor for Dogfight.
2. Training warm-start disappeared when core was restored.
3. Globally aggregated mastery became per-environment mastery.
4. Native self-play replaced a competent scripted teacher with two weak
   policies and a different reward path.
5. The successful Phase 2 profile was designed around
   `vec.total_agents=4096`, while the mixed run used 1024 so local controllers
   could collect enough episodes.
6. Changing 4096 to 1024 altered the effective optimizer-update cadence.
7. Continuous-policy entropy grew substantially in the longer mixed run.
8. The long run aborted before its requested endpoint.
9. Vector and serial evaluation currently disagree.

A blind hyperparameter sweep would mix these causes and could optimize an
invalid metric.

## Configuration Changes Recommended

Do not immediately overwrite `config/dogfight.ini` with an unproven final
profile. First preserve two explicit configurations:

| Profile | Purpose |
|---|---|
| Reproduction profile | Reproduce the successful Phase 2 stock-core result |
| Candidate production profile | Derived from controlled sweeps and long-run stability |

The reproduction profile should use:

```ini
[vec]
total_agents = 4096
num_buffers = 4
num_frozen_banks = 0
frozen_bank_pct = 0

[selfplay]
mode = off
enabled = 0

[env]
num_agents = 1
curriculum_enabled = 1
curriculum_randomize = 0
curriculum_target = 0.9
fixed_stage = -1

[policy]
hidden_size = 64
num_layers = 3

[train]
learning_rate = 0.0025
minibatch_size = 4096
gamma = 0.996
gae_lambda = 0.999
ent_coef = 0.02
clip_coef = 0.06
vf_coef = 4.6
vf_clip_coef = 1.5
max_grad_norm = 3.4
momentum = 0.9896
prio_alpha = 0.4
prio_beta0 = 0.82
replay_ratio = 0.95
vtrace_rho_clip = 0.1
vtrace_c_clip = 2.5
```

Important caveats:

- `ent_coef=0.02` is a reproduction value, not yet an accepted long-run
  default.
- Self-play must remain off for the scripted curriculum baseline.
- Frozen banks must remain disabled when self-play is off.
- Do not lower mastery thresholds simply to make stages advance.
- Do not rely on `global_step_stride` as the final global curriculum design.

After measurement and curriculum coordination are fixed, run a narrow sweep
over:

```text
train.ent_coef
train.learning_rate
train.clip_coef
```

Hold environment code, seed, rollout geometry, total agents, evaluation
episodes, and mastery criteria fixed during that sweep.

## How to Recover Removed Capabilities Later

### Benchmark-only recovery

Use a separate worktree or checkout of the safety branch. Do not toggle core
files inside the merge-oriented working tree.

Example:

```bash
git worktree add ../5c-core-benchmark dogfight5c-core-working
```

Before using this example, confirm that the target path does not exist and that
the branch is not already checked out elsewhere.

The benchmark worktree can answer:

- Does global curriculum aggregation reproduce the AutoAce climb?
- Does true warm-start preserve curriculum skill entering self-play?
- Does structural frozen-row exclusion improve native self-play?
- Does generation-safe rotation prevent policy regressions?
- Which removed feature has measurable value in isolation?

Do not treat the entire old patch as the final upstream solution.

### Small upstream PufferLib pull requests

Split generic changes so Joseph can review them independently:

1. Generic training warm-start using `base.load_model_path`, with topology,
   checkpoint-size, actor-copy, and recurrent-reset tests.
2. Generic environment lifecycle/global-step hook, designed with Joseph before
   implementation.
3. Generic curriculum aggregation callback only if PufferLib wants to own this
   abstraction.
4. Eval horizon and native render bug fixes.
5. Frozen-row priority/loss isolation, but only after reproducing the issue in
   another official self-play environment such as Robocode.
6. Generation-safe historical-opponent rotation, also proven with a generic
   self-play test.

This separation prevents the Dogfight PR from becoming a disguised rewrite of
the trainer.

### Dogfight-local temporary options

Possible benchmarking tools that do not edit core source include:

- the existing generated eval wrapper;
- an external experiment orchestrator;
- a generated training wrapper used only on a benchmark branch;
- a process-global Dogfight curriculum accumulator.

The last two options are not preferred merge designs. A process-global
accumulator must solve OpenMP synchronization, deterministic evaluation
boundaries, multi-buffer initialization, and reset semantics. A generated
training wrapper is useful for experiments but duplicates trainer behavior.

## What to Fix First

### First acceptance fix: trustworthy evaluation

Reconcile vector and serial evaluation before tuning.

The evaluator must guarantee:

- fixed stage really remains fixed;
- self-play and frozen banks are disabled unless explicitly requested;
- topology matches the checkpoint;
- a fixed seed or an explicit seed set;
- an exact, reported number of completed episodes;
- separate results for stages 0 through the current maximum;
- numerical headless and visible `DISPLAY=:0` evaluation;
- deterministic repeatability when deterministic mode is requested.

No sweep should use a metric that disagrees by 28 percentage points between
headless and visible execution.

### First training-system fix: authoritative curriculum evidence

Benchmark the old global controller against the clean local controller. Use
that evidence to choose one of two merge-quality designs:

1. a small generic PufferLib lifecycle hook accepted upstream;
2. a deterministic Dogfight-local curriculum schedule that does not pretend
   per-environment evidence is globally aggregated.

Do not silently keep the current per-environment controller as if it were
equivalent to the removed runner controller.

### Next fix: warm-start

Recover warm-start as a small generic upstream feature. This is the cleanest
way to hand a competent scripted-curriculum policy to native self-play.

### Then: stable profile and sweep

Reproduce the 4096-agent profile twice with matching traces and checkpoints.
Run a narrow stability sweep. Require a full-length run without aborting.

### Then: native self-play

Enable self-play only after the scripted policy passes the AutoAce and flight
quality gates. Start against known frozen checkpoints or favorable asymmetric
spawns. Measure current-vs-current and current-vs-history separately.

## Merge-Ready Path to a Human-Competitive Plane

### Gate 1: Environment correctness

- Dogfight3 flight tests pass.
- Action direction, control bounds, energy, G loading, stalls, and recovery
  remain covered.
- No persistent aileron, elevator, or rudder bias.
- Deterministic seeded traces match when required.

### Gate 2: Evaluation truth

- Exact fixed-stage evaluator is accepted.
- Headless and visible results agree within expected sampling error.
- Every accepted checkpoint has recorded commands, topology, seed, and hashes.

### Gate 3: Scripted curriculum

- One authoritative curriculum target.
- Independent fixed-stage gates for stages 0 through 19.
- Rehearsal prevents forgetting earlier stages.
- Energy-building cases are explicitly evaluated, not inferred from easy
  intercept wins.
- Long run completes without entropy explosion or trainer abort.

### Gate 4: AutoAce

- Stage 20 is reached by evidence, not forced configuration.
- Policy beats AutoAce reliably from varied geometry.
- Visible flight remains stable.
- Domain randomization does not destroy baseline mastery.

### Gate 5: Warm-started native self-play

- Start from the accepted scripted/AutoAce checkpoint.
- Keep current-vs-current and current-vs-history evidence separate.
- Use favorable spawns for weak policies and harder asymmetric spawns as
  competence improves.
- Maintain frozen anchors to detect catastrophic forgetting.

### Gate 6: Human-competitive evaluation

- Evaluate against a frozen checkpoint ladder.
- Evaluate against AutoAce and scripted geometry.
- Evaluate energy disadvantage, vertical fight, recovery, and defensive starts.
- Run repeated human-visible evaluations.
- Record actual human matches when controls are available.

### Gate 7: Pull-request separation

Dogfight PR:

```text
ocean/dogfight
config/dogfight.ini
Dogfight tests, commands, and documentation
no unrelated PufferLib trainer changes
```

Generic PufferLib PRs:

```text
warm-start
optional lifecycle/global-step hook
eval fixes
generic self-play correctness fixes proven outside Dogfight
```

This is the cleanest route to both goals: a Dogfight environment Joseph can
merge and a training system capable of producing a policy that can eventually
beat strong human opponents.

## Decision Rule for Future Agents

Do not choose between stock core and the old patch by intuition.

For every recovered capability:

1. state the hypothesis;
2. add a failing contract or reproducible benchmark;
3. change one behavior;
4. build and run the Dogfight tests;
5. train with a recorded command;
6. run exact fixed-stage evaluation;
7. run visible `DISPLAY=:0` evaluation;
8. compare against the clean baseline;
9. keep generic changes separate from the Dogfight PR.

