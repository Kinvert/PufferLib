# Porting Dogfight 3 to PufferLib 5c

Research date: 2026-07-27

Status: research and migration design only. No Dogfight or PufferLib implementation was changed as part of this report.

This is the canonical handoff copy in the target 5c checkout. Relative links to Dogfight source refer to files that will appear here as the port proceeds; until then, resolve them against `/home/claude/dogfight3/pufferlib/ocean/dogfight` at the pinned Dogfight revision.

## Executive decision

Start with a clean, pinned PufferLib `5c` checkout and bring Dogfight across in small, test-gated pieces. Do **not** merge the Dogfight 3 fork into 5c, and do **not** transplant the older partial Dogfight 5 port wholesale.

The recommended end state is:

1. One native Dogfight environment implementation with a fixed capacity of two `Agent` slots.
2. A curriculum/autopilot run created with `num_agents = 1`.
3. A separate self-play run created with `num_agents = 2`, where both aircraft are ordinary PufferLib agents:
   - slot 0 uses the current policy;
   - slot 1 uses either the current policy or a frozen policy according to 5c's native routing.
4. A weights-only checkpoint handoff from curriculum to self-play, accompanied by an external manifest.
5. No Dogfight-specific opponent side-channel buffers or activation flag; slot 1 uses ordinary `Agent` buffers.
6. No copied Dogfight-specific PPO training loop.
7. No attempt to load the old PyTorch LSTM checkpoints directly into the new native MinGRU policy.

That direction uses 5c the way it is now designed instead of rebuilding the 3.0 workarounds. It also preserves the parts of the existing work that are genuinely valuable: the flight simulation, observations, spawn curriculum, reward/accounting intent, learned curriculum schedule, self-play promotion policy, anchors/league concepts, and the large set of golden tests developed in the partial port.

There are, however, several **5c-native correctness gates that should be fixed before a production Dogfight self-play run**:

- native training currently ignores `base.load_model_path`, so a curriculum checkpoint cannot initialize a self-play run;
- frozen rows are assigned zero raw advantage but can still be sampled and can still produce PPO/value/entropy gradients;
- the current opponent "boundary" handshake can load a new opponent while some battles are already in their next episode, and it does not clear that bank's recurrent state;
- the async loop can prefetch a complete next rollout under the old opponent before rotation is checked, so a repaired boundary check still needs explicit quiescing/draining;
- current multi-GPU self-play has run-ID/wait and final-evaluation deadlock paths;
- native checkpoints are untyped raw weight arrays and do not contain optimizer, training, curriculum, or pool state.

These should be small, generic PufferLib fixes with focused tests. Reintroducing Dogfight's 3.0 shared-memory and trainer hacks would hide the problems rather than solve them.

## Source versions examined

The word "5c" is not enough to reproduce this analysis. The branch is moving quickly and the packaging metadata at the examined tip still reports 4.0/4.0.0 while the native dashboard calls itself 5.0. All conclusions about 5c below are pinned to a commit.

| Source | Exact revision | Role in this report |
|---|---|---|
| Dogfight 3 working repository | `5a3effce470ce19fd6c4c6385a515853a4bfc0cb` | Current Dogfight implementation and self-play system |
| Dogfight branch's PufferLib 3 merge base | `a9c02db78a896af109e1743dca893ed8869cf14e` | Separates Dogfight/PufferLib modifications from upstream 3 |
| Clean upstream PufferLib `5c` | [`ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`](https://github.com/PufferAI/PufferLib/commit/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22) | Target ABI, native learner, environment examples, and native self-play |
| Older partial Dogfight 5 port | `10da7b5abac0f67f69f302c85511baf99a22446c` plus a dirty research working tree | Test/donor material only |
| Base of that partial port | `c3e28db186bed5a341fda705faa2a33fd5dc97ab` | Older, divergent May 2026 5.0 implementation |
| Known-good older Dogfight source used by the partial port | `171482a9b9889bebaa427ab658c95c0fc391c031` | Behavioral oracle for the 26-value observation path |

The clean 5c research checkout is at `/home/claude/5c-research`. The older partial port is at `/home/claude/dogfight5`; it already contains uncommitted research changes and was treated as read-only.

Re-pin `5c` and repeat the small-env and self-play audit immediately before implementation. Several relevant native APIs changed during July 2026 alone.

## What Dogfight 3 currently is

### Environment contract

Dogfight is a batched C simulation exposed through a custom Python extension and a `PufferEnv` wrapper.

| Property | Current behavior |
|---|---|
| Simulation | Two P-40-like aircraft, 6-DOF dynamics, RK4 integration |
| Simulation rate | `DT = 0.02`, or 50 Hz |
| Exposed Puffer agents | One per physical duel |
| Physical controllers | Learning player plus an internal autopilot/AutoAce or externally supplied opponent |
| Action | Five continuous values: throttle, elevator, aileron, rudder, trigger |
| Declared action bounds | `[-1, 1]` |
| Gun model | Approximately 500 m range, 5-degree cone, 10-tick cooldown |
| Observation scheme 0 | 22 values, pilot-centric |
| Observation scheme 1 | 26 values, opponent-aware |
| Reset | The C environment resets itself when an episode terminates |
| Timeout | Reported as terminal, not truncation |

The most important architectural fact is that the second aircraft is **not** represented as a second Puffer agent. The Python environment reports one learning agent per duel. Opponent state and control are carried beside PufferLib's normal agent arrays.

The observation implementation is already largely symmetric: it computes a perspective from `(self, other)` and can swap those inputs to construct the opponent view. That is a good foundation for the direct two-agent port.

### Current process and data flow

The active self-play path is [`train_dual_selfplay.py`](train_dual_selfplay.py), not the older [`train_selfplay.py`](train_selfplay.py). The latter imports a removed `PolicyPool` interface and should be treated as dead code.

The active topology is:

```text
main Python process
├─ current policy inference on ordinary Puffer observations
├─ frozen/current opponent policy inference on opponent-observation SHM
├─ writes opponent-action SHM
├─ sends ordinary player actions to vector workers
└─ runs a copied dual-rollout PPO update
       ├─ player rollout
       └─ opponent rollout

vector worker
└─ Dogfight C vector binding
   ├─ reads player action from ordinary Puffer buffer
   ├─ reads opponent action through Dogfight-specific override storage
   ├─ steps both aircraft
   ├─ writes player obs/reward/terminal normally
   └─ writes opponent obs/reward to Dogfight-specific shared buffers
```

The changes in [`pufferlib/vector.py`](../../vector.py) allocate `opponent_observations`, `opponent_rewards`, `opponent_actions`, and a global `selfplay_active` flag for every multiprocessing vector environment, whether or not the environment needs them. The Dogfight binding then connects those arrays to extra pointers in each C environment.

The arrangement worked around the one-agent public interface in PufferLib 3. It is exactly the kind of workaround that should disappear in 5c, because 5c represents multiple agents and policy ownership directly.

The common large configuration makes the duplication significant: 8 outer workers × 1,024 duels yields 8,192 ordinary player rows. At horizon 64 that is 524,288 player transitions; concatenating the opponent rollout produces a nominal 1,048,576-transition dual batch. In 5c capacity planning, `total_agents` already includes both directly exposed seats, so these numbers must not be doubled a second time.

### Curriculum-to-self-play lifecycle

The intended lifecycle in the active trainer is:

1. Train against progressively harder scripted opponents.
2. Save a stage-10 milestone.
3. Continue the curriculum.
4. Save a stage-20 milestone.
5. Set a shared activation flag and begin neural self-play, initially against an older milestone.
6. Periodically save new checkpoints.
7. Rotate through a top-N window of unlocked opponents.
8. Promote only after sustained performance and clean-combat gates.

The promotion/ratchet policy is substantially more careful than uniform historical sampling. Its important ideas are:

- at least 10 total decisive gun kills in the evaluation window;
- at least 55% of decisive gun kills belonging to the learner;
- at least 80% clean fights, to prevent "winning" through crashes or invalid episodes;
- two consecutive successful rotations before advancing;
- stage milestones, periodic snapshots, self-play snapshots, and optional external/anchor opponents;
- recovery behavior when the learner collapses;
- two-sided anchor evaluation in the later league flow.

These concepts are worth retaining. Their current implementation is not.

### Checkpoint and league behavior

`CheckpointQueue` is a Dogfight-driven addition under the PufferLib package. In the standard self-play path it holds stage milestones and periodic/self-play snapshots and supports lagged/top-N selection. The league code adds better long-term concepts:

- atomic JSON manifests;
- a frozen generation-zero reference;
- candidates and promotion/rejection;
- both-seat anchor matches;
- Elo-like accounting and opponent scheduling.

The current queue does not reconstruct itself robustly from files, its bookkeeping can drift when checkpoints are cleaned up, and several documented PFSP options are not active in the common path. The league path also has architecture-metadata and anchor-overwrite hazards. The right approach is to preserve the policy and data-model ideas, then rebuild the coordinator around typed native checkpoints and the native match interface.

The intended PFSP weighting is of the usual form `(1 - win_rate) ^ exponent`, emphasizing opponents the learner has not mastered. Preserve that as an explicit, deterministic selection policy only after win-rate evidence is persisted and tested; do not assume the presence of the option means the active standard trainer is using it.

## What had to be changed in PufferLib 3

The current fork contains four categories of PufferLib-level changes.

### 1. Global vector shared-memory additions

`pufferlib/vector.py` unconditionally creates opponent observation, reward, action, and activation arrays and passes them through the multiprocessing worker stack. This is the largest framework intrusion and the clearest sign that the old public representation was wrong for this use case.

### 2. A custom opponent side channel

[`binding.c`](binding.c) exposes functions to:

- retrieve opponent-perspective observations;
- set opponent actions;
- attach opponent observation/reward buffers;
- toggle opponent override and self-play behavior;
- push global curriculum/shaping values into C environments.

These bypass the normal `Agent` contract. They also make behavior depend on vectorization topology and on whether a particular wrapper propagated a particular optional buffer.

### 3. A copied training implementation

`train_dual_selfplay.py` duplicates the normal rollout/advantage/minibatch/PPO path so that player and opponent trajectories can be concatenated. This creates permanent synchronization risk with the upstream algorithm and is the source of one serious correctness issue discussed below.

### 4. Generic changes motivated by Dogfight

The fork also contains checkpoint/sweep conveniences and continuous-policy stability changes. Some of those ideas now exist natively in 5c—for example, current 5c clamps continuous `logstd` to a safe range. They should not be carried as an undifferentiated patch set.

## Correctness issues found in the current Dogfight 3 path

These findings matter because the port should preserve intended behavior, not accidentally preserve every implementation defect.

### Frozen-opponent trajectories are trained as if they came from the learner

In standard self-play, the opponent policy can be a historical checkpoint, but its observations/actions/log-probabilities/rewards are concatenated into the learner's PPO batch. The current policy then recomputes probabilities for actions generated by a different policy. The resulting PPO ratio is not an on-policy correction and can poison the update.

The league path recognizes this for external opponents and trains only on the player side, but standard historical self-play generally does not.

This is the most consequential training error in the old system. The target behavior is:

- pure current-vs-current battles: both rows may train the current policy;
- current-vs-frozen battles: only the current-policy row may train;
- frozen rows may be used for environment evolution and opponent evaluation, never for current-policy loss or normalization.

### Stored opponent rewards are not the simulated opponent rewards

The C simulation has separate reward calculations for the two aircraft, and those rewards are not always exact negatives. The trainer nevertheless stores `-player_reward` for the opponent rollout. Examples of non-zero-sum terminal rewards include:

- a crashing opponent gives the player `+0.25` while the opponent receives `-1`;
- a crashing player receives `-1` while the opponent receives `+0.25`;
- timeout and some invalid-flight outcomes penalize both aircraft.

The port must publish an explicitly specified final reward for each slot. It should never infer slot 1 merely by negating slot 0; the dense slot-1 contract needs the versioned symmetric definition described below.

### Opponent terminal observations can be stale

The reset path computes the primary observation but does not consistently refresh the opponent observation. The opponent view is normally computed after a non-terminal step. This can leave the first or terminal-adjacent opponent observation zero or from the previous episode.

In the two-agent port, reset must compute both perspectives after all state initialization. Both slots must receive fresh initial observations.

### Episode ordering is asymmetric

Current combat and crash checks are order-dependent:

- the opponent hit is checked before the player's hit;
- externally controlled opponent firing is not gated exactly like player firing;
- the opponent ground collision is checked before the player's, with outcome logic that can favor one side;
- simultaneous events therefore depend on branch order.

This was partly hidden by always calling the learning aircraft "player." Direct two-agent self-play needs an explicit simultaneous-resolution contract. At minimum:

1. derive both attempted control/fire effects from the same pre-resolution state;
2. apply equivalent cooldown and trigger rules;
3. collect hit/crash/invalid events for both sides;
4. resolve the joint terminal outcome once;
5. assign both rewards from that joint outcome.

Randomizing which physical aircraft is controlled by logical slot 0 at reset is also recommended. It cancels residual spawn and update-order bias while keeping policy ownership stable.

### Curriculum controls do not reliably reach worker environments

With the normal multiprocessing configuration, the Python driver environment is not one of the stepped worker environments. Several setters are called on that driver:

- global step;
- opponent probability;
- vertical/guided curriculum controls.

Some values are propagated through special shared state, some are not. The opponent probability can remain at its C default in workers and then jump abruptly. Global shaping step is also approximated per worker and can be off by the outer worker factor. Nested vector layouts do not propagate every special field.

This argues for either segmented curriculum runs or one generic, explicit native lifecycle/control interface—not another driver-environment side channel.

### Recurrent state is reset at rollout boundaries, not episode boundaries

The 3.0 trainer's LSTM handling does not consistently zero per-agent recurrent state at every terminal. State can bleed between episodes. Current 5c has a much better per-terminal carry-state path; that behavior should be tested and used.

### RNG is not thread-safe or reproducible

The simulation uses global `rand`/`srand` paths in places while environments are stepped concurrently. This creates data races and makes results depend on thread timing and vector layout. Every environment needs its own RNG state, seeded deterministically from the run seed and environment identity.

The partial port already developed per-environment RNG work and tests; that is valuable donor material.

The audit also found mutable header globals such as `_realistic_rk4_stage` and `_realistic_step_count` that are written during integration even when debug output is disabled. Those are data races under 5c's OpenMP stepping and must become local/per-environment debug state or disappear. AutoAce, autopilot initialization, spawn selection, and domain randomization all need to be checked; converting only the top-level reset RNG is not sufficient.

### Curriculum edge cases

Several details require a parity test before trusting the stage transition:

- the Python initial target and the C initial target do not match until a later setter runs;
- the public Python setter/documentation is stale and clamps at stage 17 even though the C setter and adaptive internal path support stage 20;
- the stage-20 transition checks a floating threshold that can truncate to stage 19;
- evaluation randomization can bypass a requested fixed stage;
- some vertical/guided controls appear inactive in the common multiprocessing path.

The migration should name stages semantically and test the actual geometry/difficulty produced at each stage rather than relying only on a reported scalar.

Two concrete state leaks also deserve fixtures:

- some spawners mutate `max_steps`, but reset does not reliably restore the configured base value before selecting the next episode;
- the evaluation-spawn branch can return without assigning the stage expected by the controller-selection logic.

Store an immutable `configured_max_steps`, restore it at the start of every reset, and assign the intended evaluation controller/stage explicitly.

### Other issues worth fixing during parity work

- action recovery/autopilot may replace a sampled neural action after its log-probability was recorded, without masking that transition;
- score, return, clean-fight, and aggregation conventions are not uniform;
- some rendered labels still assume an older observation width;
- [`dogfight.c`](dogfight.c) is stale relative to the extension path;
- W&B run identity and checkpoint metadata are not sufficient to reconstruct a run;
- the Python fallback observation width disagrees with the C default in one path.

None of these justifies a rewrite of the physics. They justify a test-first adapter and a deliberately versioned behavior contract.

## PufferLib 5c's current native architecture

### Build and execution model

The examined 5c branch has moved its main path from Python extension orchestration to a monolithic native executable:

```bash
bash build.sh dogfight
./puffer train dogfight
./puffer eval dogfight
./puffer match dogfight
```

Configuration overrides use section-qualified keys, for example:

```bash
./puffer train dogfight train.total_timesteps=1000000
```

The build includes the selected environment header directly in `src/pufferl.cu`. It checks for the native environment declarations and compiles the learner, vector runtime, policy, and environment together. For this path, an old `binding.c` beside an environment may be irrelevant or stale.

Important build distinctions:

- normal native training uses CPU environment stepping plus the GPU learner;
- `--gpu` means that the environment itself is compiled for GPU execution;
- current native self-play explicitly rejects a GPU environment;
- `--cpu` is not a general CPU-training replacement for the CUDA/NCCL native learner;
- `--local` is useful for a sanitizer-backed standalone environment build.

Dogfight should begin as a CPU-stepped environment with the native GPU learner. A GPU-environment port is a later optimization and currently cannot use the native self-play pool.

### Required environment ABI

At the pinned 5c revision, `src/pufferenv.h` defines:

```c
typedef struct Agent {
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
```

The environment header must also define:

```c
typedef ... obs_t;
#define OBS_SIZE ...
#define NUM_ATNS ...
#define ACT_SIZES {...}
```

For the initial Dogfight port:

```c
typedef float obs_t;
#define OBS_SIZE 26
#define NUM_ATNS 5
#define ACT_SIZES {1, 1, 1, 1, 1}
```

All action heads of size one are interpreted as continuous. Mixed discrete and continuous heads are not supported by the current native policy path.

Every CPU-native `Env` also needs fields that the current vector implementation accesses directly; self-play additionally consumes the final two:

```c
Agent agents[2];
int num_agents;
Log log;
unsigned int rng;
int tag;
int boundary_reached;
```

The trainer wires each `Agent`'s pointers. Environment code should read and write through those pointers, not allocate parallel opponent buffers. `total_agents` is the number of agent rows, not the number of battles.

Two lifecycle/layout details are critical:

- `puf_init` runs before the trainer binds observation/action/reward/terminal buffers. It may parse configuration and allocate internal state, but it must not reset or publish an observation. The trainer calls the initial `puf_reset` after binding.
- rows are regrouped by policy bank and each `Agent` pointer is assigned independently. The two aircraft in one duel may be far apart in memory. Never obtain slot 1 by pointer arithmetic from slot 0 and never assume the two slots are contiguous.

A low-risk adapter may retain old local aliases such as `observations` and `actions`, as G2048 does, provided a helper synchronizes each alias from the corresponding `Agent` pointer at reset/step and never invents an opponent offset.

The mechanical buffer mapping is:

| Dogfight 3 concept | Native 5c destination |
|---|---|
| player observation pointer | `(float *)env->agents[0].observations` |
| player action array | `env->agents[0].actions` |
| player reward | `*env->agents[0].rewards` |
| player terminal | `*env->agents[0].terminals` as a float |
| opponent observation side channel | `(float *)env->agents[1].observations` |
| opponent action override | `env->agents[1].actions` |
| opponent reward side channel | `*env->agents[1].rewards` |
| missing opponent terminal side channel | `*env->agents[1].terminals` |

Continuous Dogfight does not need environment-produced action masks in the initial port. Timeout/crash endings remain terminal floats; there is no need to revive the old unused truncation buffer.

With two agents, `vec.total_agents = 4096` means 2,048 fights, not 4,096. Configuration must also satisfy the current buffer-layout divisibility constraints for `num_buffers * num_agents`.

At the start of each `puf_step`, clear both rewards and both terminal outputs. On episode end, write both final rewards/terminals, add exactly one episode log, set the trainer-owned boundary signal when applicable, reset internal episode state, and publish both next-reset observations **without clearing the terminal pulse for the completed transition**. `puf_reset` itself should not clear a `boundary_reached` value owned by the opponent-swap protocol.

### Vector layout and concurrency

5c steps CPU environments through a mixture of OpenMP and pthread-managed buffers. Consequences for Dogfight:

- no global mutable RNG;
- no process-wide mutable curriculum value unless it is read-only during a rollout and explicitly synchronized;
- render/client pointers must remain per environment or be confined to the render instance;
- all per-duel simulation state belongs in `Env`;
- the same bank layout must be valid for each async vector buffer.

At the pinned revision, native CPU construction assigns `env.rng = local_env_index` before `puf_init`; it does not mix `base.seed` or rank into that value. This makes environment randomness identical across nominal run seeds and repeats streams on each distributed rank. Fix this generically by deriving an environment seed from `(base seed, rank/global environment identity)` before `puf_init`, then test both repeatability and cross-rank uniqueness.

The native trainer has alignment constraints:

- BF16 horizons must be multiples of 8;
- FP32 horizons must be multiples of 4;
- minibatch size must be divisible by horizon.

These are configuration validation requirements, not environment behavior.

### Logging ABI

The native aggregation path treats a `Log` as a contiguous sequence of floats, sums instances, divides fields by the aggregate `n`, and then calls `puf_log`.

Therefore Dogfight's native `Log` should:

- contain only `float` fields;
- not contain pointers, integers, packed flags, or assumed padding;
- hold raw additive episode totals/counts;
- leave ratios to `puf_log` or to downstream reporting when possible;
- define exactly how `n` scales for a two-agent episode.

Robocode's slot-score convention is a useful model: each game contributes one total unit of outcome credit, while the aggregation scale makes `slot_0_score` directly readable as a win rate in match mode.

The generic match path expects `slot_0_score`, `slot_1_score`, and `draw_rate`. It also currently compares requested `num_games` to aggregated `n`; Robocode increments `n` per agent, so a request for 4,096 "games" produces roughly 2,048 two-agent episodes. Dogfight should define one convention explicitly, and PufferLib should distinguish episode count from agent-record count in its stopping/reporting logic.

### Observation precision

Some current environments conditionally store observations as native BF16 for throughput. Dogfight should use float observations first:

- golden fixtures can compare exact float behavior;
- the old policy was trained from float observations;
- it avoids mixing numerical migration with ABI migration.

Once parity and learning are established, benchmark BF16 observation storage separately. It should be an optimization experiment with tolerance-based trace tests, not part of the first working port.

### Continuous actions

The native policy samples an unsquashed Normal distribution. At the pinned revision:

- `logstd` is clamped to a safe range;
- means and samples have broad finite guards;
- the output is not automatically transformed to `[-1, 1]`.

Dogfight's physics already clamps control inputs. Preserve the policy's raw sampled action and log-probability, and treat input limiting as environment execution semantics. Record the clamp rate during early runs; a high rate indicates that a bounded distribution or policy-side transformation should be considered deliberately.

The trigger's exact threshold/continuous semantics must be frozen in a test. It should not accidentally change because the first native policy has a different action scale.

The current firing decision is `trigger > 0.5`; preserve that for the parity baseline.

Use one canonical local executed-action vector: copy the five raw samples, apply the declared finite/clamp policy once, and use that same executed vector for physics, shaping, control-rate telemetry, and trigger logic. The current code clamps the four flight controls inside physics but can use raw values in shaping, which makes the recorded action contract internally inconsistent. Keep the untouched raw vector only for policy/log-probability accounting.

### Policy and recurrent state

Dogfight 3 uses the Python default encoder—linear layer with bias followed by GELU—then a PyTorch LSTM path. The native 5c default uses:

- a native encoder;
- stacked MinGRU/highway recurrent layers;
- a fused policy/value decoder;
- native optimizers and raw native checkpoints.

The old `.pt` LSTM weights are not structurally compatible with this model. A new native training lineage is required.

The partial Dogfight 5 work found that restoring the linear+bias+GELU input encoder materially improved curriculum progress compared with the then-current native default. Current 5c has an environment-specific encoder registry in `src/ocean.cu`. A small registered Dogfight encoder is therefore a justified candidate, but it must be evaluated against the current 5c default at the pinned revision.

Current 5c's recurrent rollout handling is significantly better than the old path:

- state persists across horizons by default;
- state is zeroed per terminal;
- each async rollout slot snapshots its own initial state;
- `reset_every_horizon = 0` is the default.

Those properties need explicit Dogfight terminal and opponent-swap tests. They should not be replaced with the older partial port's recurrent patches, which targeted a different May implementation.

### Native checkpoint format

The native `.bin` format is a flat FP32 weight array. It has no embedded:

- magic/version/schema;
- environment or source revision;
- observation semantics;
- action contract;
- architecture dimensions;
- optimizer state;
- global step/epoch;
- LR schedule position;
- curriculum state;
- RNG state;
- self-play pool or opponent assignment.

This is adequate as an internal weight blob, not as a durable Dogfight artifact. Every kept checkpoint should have an atomic sidecar manifest containing at least:

```json
{
  "format": "pufferlib-native-fp32-weights-v1",
  "pufferlib_commit": "ebf5ed03cc3524076b6c1a4033bd69cec0b3db22",
  "dogfight_behavior_version": "...",
  "obs_width": 26,
  "obs_schema": "opponent-aware-v1",
  "action_schema": "throttle,elevator,aileron,rudder,trigger",
  "encoder": "...",
  "recurrent": "mingru",
  "hidden_size": 256,
  "num_layers": 1,
  "weight_count": 0,
  "global_agent_steps": 0,
  "curriculum_stage": null,
  "parent": null,
  "sha256": "..."
}
```

Exact resume would additionally require optimizer, scheduler, RNG, counters, and pool state. A weights-only handoff must be labeled as a warm start, because optimizer and schedule state restart.

The current loader verifies that it can read the destination's expected prefix, but it does not reject trailing bytes. A smaller frozen architecture can therefore appear to load a prefix of a larger, structurally incompatible checkpoint. Validation must `stat` the file and require exactly `parameter_count * sizeof(float)` before reading, in addition to checking the manifest.

Publishing a payload and sidecar requires an ordering contract because two renames are not atomic as a pair. Write and atomically rename the validated payload first; publish the manifest last as the commit marker. Loaders and "latest checkpoint" discovery must ignore orphan payloads and verify the manifest's size/hash. A single versioned checkpoint container is the alternative.

## Lessons from the requested 5c environment comparisons

### Comparison table

| Environment | Current 5c state | Lesson for Dogfight |
|---|---|---|
| Breakout | Current native header ABI, one `Agent`, fixed macros, per-env RNG, native reset/step/log; optional separate GPU implementation | Shows the minimal one-agent native shape and build/header conversion |
| G2048 | Current native header wraps preserved `c_reset`/`c_step` logic with a thin `puf_*` adapter and aliases `Agent` buffers | Best structural model for porting tested simulation code without rewriting it |
| Robocode | Current two-agent native environment with policy IDs, per-slot rewards/observations, match scores, `tag`, and `boundary_reached` | Closest direct model for Dogfight self-play |
| Boxoban | Header/binding remain on an older interface; current native build precheck does not accept it as-is | Negative example; do not copy merely because the directory exists |
| Sokoban | No standalone current implementation was found; only unrelated Nethack references | No port template is available under this name |

### Breakout

The useful Breakout conversion landed around `6e9adeb7` and illustrates the mechanical native shape:

- declarations and required constants live in the environment header;
- `Env` owns one `Agent`;
- `puf_init` reads numeric config and assigns policy 0;
- `puf_reset`/`puf_step` write directly to agent buffers;
- RNG is environment-local;
- terminal handling self-resets.

This is the right model for the first compile-only Dogfight adapter, but it says little about self-play.

### G2048

The current G2048 conversion, around `d7cf8325`, is particularly relevant because it did not discard its tested core API. It keeps `c_reset` and `c_step`, adds `puf_reset` and `puf_step`, and uses a small `sync_agent_buffers` helper so old core fields alias the new `Agent` storage.

Dogfight should follow the same pattern initially:

1. retain flight dynamics and observation helpers;
2. add a thin native adapter;
3. prove golden parity;
4. only then refactor controller selection and terminal resolution for symmetry.

Trying to redesign physics, rewards, observations, native ABI, curriculum, and self-play in one commit would make regressions impossible to localize.

### Robocode

Robocode is the authoritative self-play example:

- `Agent agents[2]`;
- slot 0 initially has policy ID 0;
- slot 1 initially has policy ID 1;
- both slots expose ordinary observation/action/reward/terminal pointers;
- a completed game marks both terminals and records per-slot outcome;
- `tag` identifies which historical bank owns an environment;
- `boundary_reached` participates in opponent rotation.

Dogfight should copy this public representation, not its game-specific state machine. In particular, the current Robocode boundary behavior exposes a native trainer flaw: it marks the boundary, immediately resets, and continues, while the trainer treats the flag as evidence that the environment is still aligned. That cannot be copied uncritically.

### Boxoban and Sokoban

The Boxoban directory is evidence of API transition, not a working target pattern. Its binding still resembles the older vector interface, and the current build's required-`obs_t` precheck rejects it. No separate Sokoban environment exists in the examined tree.

This is also why the clean 5c commit must be pinned. Directory presence is not enough to establish that an environment exercises the current native path.

## How native 5c self-play works

### Routing

For each vector buffer, 5c divides environments into:

- **pure environments**, where every agent slot is forced to policy 0;
- **historical environments**, where each slot's declared `Agent.policy` is used.

With a two-agent environment declared as `[policy 0, policy 1]`:

| Battle type | Slot 0 | Slot 1 | Rows eligible for current-policy training |
|---|---|---|---|
| Pure | Current policy | Current policy | Both |
| Historical | Current policy | Frozen bank 1 | Slot 0 only |

Rows are physically regrouped by policy bank and the environment's `Agent` pointers are repointed into that layout. Inference runs each bank over its own row range with separate weights, activations, and recurrent state.

One naming trap: `frozen_bank_pct` is effectively applied to environment/battle count in the current allocator. For a two-agent game, a value of 0.10 means roughly 10% of battles are historical, not 10% of total rows. Only one of the two rows in each such battle is frozen.

For example, 4,096 rows represent 2,048 two-agent battles. At `frozen_bank_pct = 0.10`, integer rounding selects 204 historical battles: 3,892 rows remain in the primary bank and only 204 rows—about 4.98%—are actually frozen.

Robocode hardcodes only policy IDs 0 and 1. More than one frozen bank requires the environment to assign higher policy IDs to some slots/environments; simply raising `num_frozen_banks` can otherwise create empty-bank assertions. Dogfight should start with exactly one frozen bank.

### Rollout and training intent

The intended 5c behavior is better than Dogfight 3's copied dual PPO:

- each bank performs the inference that generated its own actions/log-probabilities;
- current-policy rows share the current learner;
- frozen-bank rows should be excluded from learner optimization;
- pure self-play supplies two trainable rows per battle;
- historical self-play supplies one trainable row per battle.

The implementation contains an explicit kernel that writes zero advantage on frozen rows. This directly targets the old "train on frozen trajectories" bug.

Unfortunately, the exclusion is incomplete at the pinned revision:

1. at `prio_alpha = 0`, the priority kernel ordinarily evaluates zero frozen advantage as `pow(0, 0) = 1`, giving frozen rows full uniform weight before any epsilon;
2. for positive alpha, priority normalization still adds `1e-6` to every row, including zeroed frozen rows;
3. if one is selected, minibatch-wide advantage normalization transforms its raw zero into `(0 - mean) / std`;
4. that row can then produce policy gradient, value loss, and entropy gradient.

The same normalizer adds epsilon to every numerator but only one epsilon to the denominator, so its final CDF is not exactly normalized either.

Required generic fix:

- build the sample population from explicit trainable/current-policy row indices;
- give frozen rows exactly zero sampling probability after all normalization;
- compute advantage moments only over trainable samples;
- mask all policy, value, and entropy terms as defense in depth.

Required tests:

- `alpha = 0` and `alpha = 1`;
- all-zero and nonzero current advantages;
- millions of sampled indices with no frozen index;
- gradients invariant to arbitrary observations/actions/rewards inserted into frozen rows;
- final CDF value exactly representing the trainable distribution.

Until this is fixed, native self-play does not fully solve the old PPO-contamination problem.

### Historical checkpoint pool

The current native pool:

- saves the initially created current policy as pool entry zero;
- adds checkpoints at fixed `checkpoint_interval` epochs;
- evicts oldest in-memory entries with FIFO behavior;
- samples uniformly;
- supports up to eight frozen banks;
- performs a small final match against early pool entries.

It can sample the checkpoint already loaded in a bank because the bank does not track/exclude its current path. When alternatives exist, rotation should exclude the current ID and log both current and pending generations. FIFO eviction removes an entry from the in-memory pool but does not remove its file. Final pool scoring walks the oldest eligible entries, not a random or evenly spaced sample.

It does not currently provide:

- PFSP or difficulty-weighted sampling;
- clean-fight promotion gates;
- stage-10/stage-20 activation;
- a top-N ratchet;
- persistent pool reconstruction;
- fixed anchor classes;
- Elo/league promotion;
- full resume state.

`checkpoint_interval` is counted in training epochs, not agent steps. Dogfight configuration should not copy an old "steps" number into that field without conversion.

Uniform FIFO is sufficient for the first correctness/learning baseline. It should not be mistaken for feature parity with the Dogfight trainer.

### Native self-play versus residual Python self-play

The tree also contains `pufferlib/selfplay.py` and Python league machinery with richer concepts such as external pools, stride-style eviction, shared JSON, and multi-rank coordination. Those are separate from the native implementation described above. At the pinned commit, the clean build does not produce the old Python extension that this path expects; the canonical buildable trainer is `./puffer`.

Do not combine promises from the Python implementation with behavior of native `src/pufferl.cu`. If that Python path is intentionally revived later, treat it as a separate architecture decision and test it independently.

The repository's `tests/test_selfplay.py` exercises that separate Python implementation, not the native bank/pool code. The native behavior described here currently lacks an equivalent direct test suite, another reason to land the P0 tests before trusting it.

### Opponent swap behavior

When a bank's timeout expires, the trainer chooses a pending checkpoint and clears `boundary_reached` for tagged environments. It later loads the pending checkpoint after every tagged environment has set that flag.

The flag is currently sticky evidence that an environment **ended at least once**, not a barrier:

1. an environment reaches terminal;
2. it sets `boundary_reached = 1`;
3. it immediately resets and keeps stepping;
4. other, longer episodes have not yet ended;
5. after the last environment finally ends, earlier environments may be midway through another episode;
6. the trainer loads new frozen weights into the shared bank.

The load replaces weights but does not clear the frozen bank's recurrent state. One physical fight can therefore see two opponent checkpoints, and the new checkpoint can inherit hidden state produced by the old checkpoint.

Required generic fix:

- arm the bank change;
- park each tagged environment at its first terminal after arming;
- when all relevant environments are parked, load and validate the new weights;
- zero that bank's recurrent state;
- reset/unpark those environments together.

An alternative is a generation-tagged reset protocol that guarantees no post-terminal transition is collected before the swap. A sticky bit without parking is insufficient.

For an initial synchronous implementation, a simpler trustworthy alternative is to rotate only between completed rollouts: explicitly truncate/reset every environment using the bank, record those swap truncations separately, load the new validated weights, clear all affected frozen and primary recurrent rows, publish fresh reset observations, and resume. This sacrifices some episode completions but is easier to make atomic. It must never masquerade as a natural terminal or silently discard evaluation outcomes.

A natural-terminal parked design cannot merely skip `puf_step`. The CPU vector path has already run inference, copied a rollout row, and advanced recurrent state before that callback. It must either discard all alignment rollouts or propagate a transition-validity mask through GAE, priority sampling, advantage normalization, and every loss. Environments that self-reset when setting the boundary flag are already at their next initial state and must not be reset a second time on release.

Required test:

- for the initial synchronous design, assert that every affected partial episode is explicitly counted as a swap truncation and all affected environments restart atomically at tick zero;
- assert both frozen-bank state and affected primary recurrent rows are zero;
- if natural parking is later implemented, deliberately stagger episode lengths, prove alignment rollouts are discarded/masked, and assert that an early environment takes no trainable post-terminal step while waiting;
- assert an internally self-reset parked environment is not reset again on release;
- record checkpoint generation in a test observation/log and prove that no episode contains two generations.

There is a second ordering problem in async mode. The training loop starts and finishes the next rollout before it reaches the opponent-rotation state machine. A pending/ready rollout and live environment state can therefore already have been produced under the old frozen opponent when new weights are loaded. The rotation code also sits after a dashboard/log-throttle `continue`, so the state machine advances on logging cadence rather than every epoch.

Safest initial rule:

- reject or configure `selfplay.enabled && base.async` and run trusted Dogfight self-play with `base.async = 0`;
- move rotation/state-machine work ahead of display throttling;
- before supporting async self-play, explicitly stop prefetch, park environments, drain or discard any ready slot produced under the old generation, load weights, zero recurrent state, reset/release, and only then restart prefetch.

Test both sync and eventual async paths with generation-tagged rollouts. No ready rollout, live environment, or recurrent-state tensor from generation N may be consumed as generation N+1.

### Runtime activation and warm start

Native `run_train` reads `selfplay.enabled` once at startup. There is no in-process curriculum-to-self-play transition. This supports the recommended two-run design.

The required handoff is currently missing: `run_train` creates a randomly initialized policy and, when self-play is enabled, immediately saves that random initialization as pool entry zero. `base.load_model_path` is consumed by eval/match, not by native training.

The generic fix should:

1. create and validate the requested architecture;
2. load a primary checkpoint when `base.load_model_path` is set;
3. synchronize the actor/inference weights;
4. only then create the initial self-play snapshot and frozen bank;
5. clearly distinguish weights-only warm start from full resume.

A test should verify that the first pool checkpoint hashes to the supplied curriculum checkpoint and that incompatible sizes/schema are rejected.

### Multi-GPU caveats

The current multi-GPU path should not be used for the first Dogfight self-play run:

- a default run ID is generated independently inside each post-fork rank;
- non-owner ranks can wait indefinitely for a checkpoint path owned under another ID;
- final pool evaluation is run only by rank 0 but re-enters distributed/NCCL construction with `world_size > 1` without the non-owner ranks participating in that initialization;
- final evaluation constructs a new learner per opponent, while the current close path does not free the major allocations.

Generic remedies:

- resolve and broadcast one run ID before forking ranks;
- use bounded, error-aware waits;
- run final pool matches after distributed ranks join with a fresh single-rank context, or make all ranks participate;
- implement a real destructor or isolate repeated matches in subprocesses.

Start Dogfight native self-play on one GPU. Add a two-GPU timeout/failure test before scaling it out.

### Architecture compatibility

A frozen bank may be configured with a different hidden size/layer count, but pool checkpoints are current-policy checkpoints and the file is an untyped float array. Loading a raw prefix into a different bank architecture is not a meaningful model conversion.

For the first implementation:

- require every policy in a pool to have an identical environment schema and architecture;
- validate exact weight count and manifest fields before load;
- use heterogeneous frozen architectures only after there is an explicit conversion/adapter design.

The current Robocode sweep can vary primary architecture while leaving frozen-bank dimensions fixed, so this is not merely hypothetical. Pool construction should enforce equality or mirror the chosen primary architecture into every frozen bank.

## Audit of the older partial Dogfight 5 port

The `/home/claude/dogfight5` work is useful, but it targeted an older May 2026 5.0 design that is not the current 5c lineage. Its binding/vector/state/self-play APIs should not define the new port.

### Material worth carrying over

| Donor material | Why it is useful | How to carry it |
|---|---|---|
| Environment-local RNG conversion | Removes races and enables deterministic traces | Reapply to the clean native `Env`, preserving tests |
| 26-wide observation compatibility work | Establishes that the old "scheme 2" source corresponds to the current opponent-aware 26-value path | Convert fixtures to the final semantic schema |
| Fixed-state observation tests | Detect sign, perspective, normalization, and ordering regressions | Vendor expected values into new native tests |
| One-step and 12-step trace fixtures | Protect physics/reward parity | Run against a thin native adapter before refactoring |
| Terminal tests | Cover player kill, opponent kill, OOB/crash, timeout, and reset | Expand to both slots and simultaneous outcomes |
| Multi-reset and state round-trip tests | Catch stale buffers and incomplete reset | Preserve fixture/test intent; do not port the duplicated runtime mirror |
| Curriculum geometry/progress tests | Verify actual stage behavior | Adapt to the chosen segmented/native controller |
| Native build/config tests | Catch ABI drift early | Rewrite for the pinned current build |
| Linear+bias+GELU encoder | Empirically improved the earlier native port | Register minimally and A/B test on current 5c |
| Action clamp/contract tests | Protect continuous-control semantics | Reuse with raw-policy/executed-action distinction |
| Warm-start test and investigation | Found the absent train load path | Reimplement as a generic current-5c test/fix |

The deterministic snapshot/round-trip **test intent** remains valuable, but the partial port's runtime `State` mirror manually duplicates roughly a hundred mutable fields and is a drift hazard. Do not port that mirror. If snapshot debugging is later required, define a narrow Dogfight-local value snapshot that deliberately excludes framework pointers, rendering clients, and aggregate logs. Current 5c no longer exposes the generic `State` curriculum API that the partial port targeted.

### Material not worth carrying over

- old vector/binding ABI glue;
- a one-agent opponent setter/override interface;
- Python `_C` wrappers for the old runtime;
- broad edits to core trainer files;
- the duplicated runtime `State` mirror;
- the old state-curriculum hook;
- duplicated sweep plumbing;
- recurrent-state fixes for the superseded May trainer;
- hard-coded local test paths;
- repository-wide setup/tooling churn;
- the old `binding.c` as the final two-agent design.

The partial binding still models one public agent and a hidden opponent. That is a transitional scaffold, not the target.

The initial bulk port was commit `507dbae4c39e0b826c4da6cd4ed89b62f9404497`; the later GELU work was `1f1e7a0460d8a9fd8c07fa751504f1ac95b3fbbe`. These are useful provenance points for extracting individual files or tests, not cherry-pick candidates.

### Empirical result and its limit

The partial port's `df39_investigation.md` records a fresh native curriculum run reaching approximately stage 11.9 in roughly 57.7 million steps at about 1.5–1.7 million SPS after restoring the GELU encoder and matching the intended configuration.

That is useful evidence that:

- the physics/observation port can learn;
- encoder details matter;
- the donor tests and configuration work are valuable.

It is not evidence that stage 20, curriculum handoff, or self-play works. A longer run also showed that changing `total_timesteps` changes the cosine schedule, so runs cannot be compared by copying every other hyperparameter and extending only the horizon.

## Recommended target design

### One simulation, two creation modes

Use one compiled Dogfight implementation with `Agent agents[2]` capacity and a numeric configuration mode.

#### Curriculum/autopilot mode

- `num_agents = 1`;
- slot 0 is policy 0;
- slot 1 is unbound capacity only; callbacks may access only slots `< num_agents`;
- the second physical aircraft remains controlled internally by autopilot/AutoAce;
- only slot 0 is exposed to PufferLib;
- native self-play is disabled;
- stages are controlled by fixed/segmented configuration initially.

This mode gives maximum behavioral continuity with Dogfight 3 and makes physics/observation/curriculum comparison simple.

#### Self-play/match mode

- `num_agents = 2`;
- slot 0 declares policy 0;
- slot 1 declares policy 1;
- both observations, actions, rewards, and terminals use ordinary `Agent` buffers;
- pure native environments force both slots to current policy 0;
- historical environments keep slot 0 current and route slot 1 to frozen bank 1;
- both slots receive fresh observations at reset;
- both slots terminate together.

`num_agents` is fixed when the vector environment is created. Curriculum and self-play should therefore be separate processes/runs, even if they use the same binary and header.

### Why not one always-two-agent process?

It is possible to expose two agents from the beginning, ignore or override slot 1 during curriculum, then attempt to activate it later. That is not recommended:

- self-play mode is read once at native trainer startup;
- early frozen pools would contain curriculum-era/random snapshots;
- ignored neural actions would have invalid log-probability/transition semantics;
- global stage control is still unsolved;
- two rows per duel would consume capacity while only one is meaningful;
- logging and promotion boundaries become harder to interpret.

A clean checkpoint handoff is simpler and easier to test.

### Internal code boundaries

Preserve the current domain headers where practical:

- `flightlib.h`: aircraft dynamics and integration;
- `dogfight_observations.h`: perspective-based observation construction;
- `dogfight_spawn.h`: stage/spawn geometry;
- `autopilot.h` and `autoace.h`: scripted controllers;
- `dogfight_render.h`: rendering;
- `dogfight.h`: native `Env`, episode orchestration, and `puf_*` entry points.

Within the environment, separate:

1. physical state integration;
2. controller/action selection;
3. joint event detection;
4. joint terminal/reward resolution;
5. observation publication;
6. episode logging/reset.

The first port should use a thin adapter around the existing tested core, as G2048 does. Refactor into the boundaries above only after golden parity passes.

### File disposition

| Current file/area | Port treatment |
|---|---|
| `flightlib.h` | Keep dynamics/integration; refactor RNG signatures and debug globals |
| `autopilot.h` / `autoace.h` | Keep control logic; finish per-env RNG conversion and move shot resolution to symmetric episode logic |
| `dogfight_spawn.h` | Keep geometry; replace global randomness, restore base max steps, and define symmetric self-play spawns |
| `dogfight_observations.h` | Keep generalized per-plane math; publish directly to each active `Agent` |
| `dogfight_render.h` | Adapt to `puf_render`/`puf_close`, 26 labels, and current resource paths |
| `dogfight.h` | Major native ABI, lifecycle, two-agent, logging, and joint-resolution refactor |
| `binding.c` | Do not port to the native path |
| `dogfight.c` | Defer or rewrite as a small standalone/native demo |
| `dogfight.py` / `train_dual_selfplay.py` | Preserve as behavioral/control-policy references, not runtime dependencies |
| `p40.glb` | Preserve content; current 5c resource packaging expects an environment resource path such as `resources/dogfight/p40.glb` |

### Configuration disposition

Move direct simulation values into `[env]` and cast current `Dict` numeric values explicitly. Current `dict_get` returns a numeric value and fails on missing keys; Robocode's item-loop helper is the pattern for optional defaults. Do not copy an older pointer-returning `dict_get(...)->value` idiom.

Likely native environment keys include:

- agent/mode: `num_agents`, `max_steps`, observation semantics;
- curriculum/spawn: enabled/fixed stage, randomization, eval spawn, vertical level/probability;
- rewards/shaping: aim, closing, negative-G, speed, control-rate, low-altitude, energy terms;
- recovery: enabled, altitude/speed/bank thresholds, trigger probability;
- domain randomization.

Worker counts, total timesteps, eval cadence, model/pool selection, checkpoint cadence, and sweep settings belong to native trainer sections, not `puf_init`. Old State-buffer, clipping, Python-only curriculum, and copied-PPO keys should be removed unless a current native feature deliberately reintroduces them.

For reproducibility, current native policy/action RNG reads `[base].seed`, not `[train].seed`; the latter is inert on this path even though a checked-in Robocode config uses it. Set `[base].seed` and `[selfplay].seed` explicitly, and—after the generic CPU environment-seeding fix—derive each `Env.rng` from the base seed and global environment identity.

### Observation contract

Compile one fixed width: 26 floats.

- semantic schema `pilot-22`: write the first 22 defined values and zero the last four deterministically;
- semantic schema `opponent-aware-26`: write all 26;
- do not reuse ambiguous numeric "scheme" labels in durable manifests;
- test both physical perspectives from the same state;
- store semantic field names and order in the checkpoint manifest.

The exact index map is:

| Indices | Both schemas |
|---|---|
| 0–2 | body-frame forward speed, sideslip, climb rate |
| 3–5 | roll, pitch, yaw rate |
| 6–9 | angle of attack, altitude, G-force, own energy |
| 10–12 | own world-up vector x/y/z |
| 13–16 | target azimuth, elevation, range, closure |
| 17–20 | energy advantage, aspect, opponent pitch rate, opponent roll rate |

The tail is **not** semantically prefix-compatible:

| Index | `pilot-22` in a 26-wide buffer | `opponent-aware-26` |
|---|---|---|
| 21 | timer | opponent world-up x |
| 22 | zero padding | opponent world-up y |
| 23 | zero padding | opponent world-up z |
| 24 | zero padding | opponent speed |
| 25 | zero padding | timer |

This fixed allocation avoids changing runtime buffer width, but it does **not** make the two semantic schemas or their checkpoints interchangeable. The manifest must distinguish them, and an implementation must not accidentally move the pilot-22 timer from index 21.

### Action contract

Use five continuous heads:

```c
#define NUM_ATNS 5
#define ACT_SIZES {1, 1, 1, 1, 1}
```

For every transition distinguish:

- raw policy sample used for log-probability;
- executed/clamped controls used by physics;
- any scripted override.

Do not include a transition in PPO if recovery logic replaces the learning agent's action after sampling, unless the override is represented as part of the policy/action transform with a valid probability model.

Freeze the current polarity as part of the ABI:

| Index | Control and sign |
|---|---|
| 0 | throttle: raw `[-1, 1]` maps to physical `[0, 1]` |
| 1 | elevator: positive means push/nose down |
| 2 | aileron: positive means roll right |
| 3 | rudder: positive means yaw left |
| 4 | trigger: fire when greater than `0.5` |

### Reward and outcome contract

There is no single existing dense "actual opponent reward" to copy:

- C computes and clamps the player's dense reward to `[-1, 1]`;
- the C opponent side channel starts from the negative clamped player reward, then adds its own altitude/energy terms and negative player energy advantage, without a final clamp;
- the active Python dual trainer ignores that side-channel value and stores simply the negative clamped player reward.

For the parity baseline, preserve slot 0's golden dense reward. For true two-agent mode, compute the same dense reward function independently from each slot's perspective, then clamp each final published reward to `[-1, 1]`. That symmetric slot-1 definition is an intentional behavior-version change, not historical parity, and needs its own fixtures/manifest version.

Current native 5c also clamps transposed rollout rewards to `[-1, 1]` before GAE. Publishing finite already-clamped per-slot rewards makes that learner clamp a no-op and keeps environment episode-return logging consistent with training. If raw component sums are useful, log them separately from the final learner reward.

Do not force every reward component to be zero-sum merely because the game is competitive.

Treat this published-versus-learner reward contract as a Dogfight P0 parity gate before any two-agent learning run.

Separately define a clean match outcome:

- gun kill;
- opponent crash;
- player crash;
- simultaneous kill/crash;
- timeout/draw;
- invalid/supersonic outcome.

Training reward and match score need not be identical. Promotion should use the explicit match outcome and clean-fight metrics rather than the sign of shaped episode return.

After native parity, decide whether the crash bonus/penalty table should be made zero-sum. That would be a behavior-version change and should not be hidden inside the ABI port.

### Physical-role randomization

At reset, randomly map logical slots to the two physical aircraft using the environment RNG.

- policy ownership stays logical: current is slot 0, frozen is slot 1;
- observations are computed from the mapped aircraft's perspective;
- incoming actions are mapped to physical controls;
- rewards/outcomes are mapped back to logical slots;
- logs record enough information to measure side bias.

This protects self-play from first-check, spawn, and integration-order asymmetries while they are being eliminated.

### Curriculum control

The lowest-risk initial controller is segmented training:

1. run a fixed stage or bounded group of stages;
2. evaluate explicit mastery metrics;
3. atomically select the next stage;
4. warm-start a new native run from the prior weights;
5. record the transition in the manifest.

This requires the generic training warm-start fix but no environment-specific trainer hook.

If recreating the continuously adaptive curriculum is later important, add one generic native environment lifecycle callback or control block that is:

- updated only at a safe rollout/episode boundary;
- propagated to every stepped environment;
- deterministic and recorded;
- usable by environments other than Dogfight;
- covered by multi-buffer and multi-rank tests.

Do not send curriculum state to an unstepped driver instance.

Aim/closing shaping can initially be fixed or disabled in final self-play, as the league path already tended to do. If exact global-step annealing is required, expose global agent steps through the same generic lifecycle mechanism.

### Self-play progression

Implement self-play in layers:

1. current-vs-current only;
2. one frozen bank with a small historical battle fraction;
3. uniform FIFO native pool;
4. verified checkpoint boundary swaps;
5. fixed native anchors and both-seat match;
6. clean-fight promotion gates;
7. top-N/PFSP selection;
8. persistent league/Elo scheduling.

The first four validate the framework. The later four recover Dogfight's useful game-specific policy without forking PPO.

## Required PufferLib-generic work

The following changes should precede production self-play. They are not Dogfight special cases.

| Priority | Generic change | Why required |
|---|---|---|
| P0 | Honor and validate `base.load_model_path` in native training before the initial pool snapshot | Enables curriculum-to-self-play handoff and meaningful warm starts |
| P0 | Exact trainable-row sampling/loss mask | Prevents frozen-policy data from updating the current policy |
| P0 | True opponent-swap barrier plus frozen-state reset | Prevents mid-episode policy changes and recurrent-state inheritance |
| P0 | Quiesce/drain async rollout state during opponent rotation, or reject async self-play | Prevents prefetched old-generation transitions from crossing a swap |
| P0 | Require primary/frozen architecture equality for native self-play | Prevents bootstrap failure or silent raw-prefix corruption |
| P0 for multi-GPU | Resolve run ID once, bound waits, and restructure final pool eval | Prevents hangs/deadlocks |
| P1 | Typed checkpoint manifest and exact size/schema validation | Prevents silent incompatible loads |
| P1 | Real native resource destruction or subprocess-isolated repeated matches | Prevents final-eval memory growth |
| P1 | Persistent self-play pool manifest | Enables restart, audit, and stable anchors |
| P1 | Exclude the currently loaded opponent when alternatives exist | Makes a requested rotation an actual change |
| P1 | Separate episode count from per-agent log count | Makes match/eval game budgets accurate |
| P1 | Seed CPU environments from run seed plus global identity | Makes runs reproducible without duplicated rank streams |
| P1 | Make learner reward clipping explicit/configurable and log clip counts | Keeps published, logged, and learned reward semantics auditable |
| P1 | Clone evaluation configuration before `run_eval` mutates it | Prevents repeated/final matches from leaking configuration state |
| P2 | Generic environment lifecycle/control hook | Needed only for continuous in-process curriculum/shaping |
| P2 | Public opponent-selection/coordinator interface | Lets Dogfight add gates/PFSP without modifying PPO |

The P0 tests should land alongside the generic fixes before Dogfight relies on them.

## Migration plan

Each phase should be a small reviewable commit or commit series, with no later-phase behavior folded into an earlier one.

### Phase 0 — Freeze the contracts and fixtures

Deliverables:

- pin the 5c commit;
- document current 22/26 observation fields by semantic name;
- document five action meanings and trigger threshold;
- freeze current terminal reward/outcome table;
- freeze slot-0 dense reward fixtures, record both historical slot-1 formulas, and version the proposed perspective-symmetric slot-1 formula;
- identify the known-good Dogfight source snapshot;
- copy/rewrite the partial port's golden fixtures into portable tests;
- record representative curriculum and self-play configs.

Exit gate:

- each fixture runs against its individually recorded oracle revision; numeric fixtures derived at `171482a9b9889bebaa427ab658c95c0fc391c031` are not silently re-baselined to Dogfight 3 HEAD;
- every checkpoint/reference fixture records source revision and semantic schema.

### Phase 1 — Minimal native one-agent scaffold

Deliverables:

- native `Env` and `Log`;
- required macros and `puf_*` callbacks;
- numeric config loading;
- direct slot-0 agent buffers;
- standalone allocation/free path;
- clean `bash build.sh dogfight` and `--local` sanitizer build.

Keep the internal autopilot opponent and make no intentional physics/reward changes.

Exit gate:

- reset/step/render smoke tests;
- no old `vecenv.h`, Python `_C`, or opponent SHM dependency in the native build;
- sanitizer-clean deterministic short run.

### Phase 2 — Deterministic behavioral parity

Deliverables:

- replace global RNG with per-environment RNG;
- fix native CPU seeding to mix run seed with global environment identity before `puf_init`;
- fixed 26-wide output with deterministic zero padding for the 22-value schema;
- exact one-step and multi-step differential traces;
- terminal/reset tests;
- all-float additive log structure.

Exit gate:

- one-step and 12-step golden traces pass within declared tolerance;
- two equal seeds reproduce across thread counts and async buffer arrangements;
- different seeds diverge;
- the public slot-0 reset observation is finite and fully initialized; an internal test helper may also check the opposite perspective before Phase 4 binds slot 1.

### Phase 3 — Native curriculum baseline and warm start

Deliverables:

- current-5c Dogfight encoder A/B test;
- generic native training load-path fix;
- manifest-last checkpoint publication with a validated sidecar;
- segmented/fixed-stage curriculum runner or configuration protocol;
- stage geometry and progress tests.

Exit gate:

- reproduce an early learning milestone such as stage 10/11 under a pinned step and LR schedule budget;
- loading a checkpoint reproduces its weight hash in the first self-play pool entry;
- incompatible manifest/weight count fails loudly;
- extending total timesteps is accompanied by an explicit LR-schedule decision.

### Phase 4 — Direct symmetric two-agent environment

Deliverables:

- direct buffers for both slots;
- perspective-correct observations for both;
- slot-0-parity plus versioned perspective-symmetric slot-1 dense rewards;
- both terminals together;
- simultaneous event resolution;
- logical-to-physical role randomization;
- per-slot match/clean-fight logs.

Exit gate:

- action and observation permutation tests;
- reward/outcome table for every terminal cause;
- both final slot rewards are finite, clamped, and covered by dense/terminal fixtures;
- simultaneous shots and simultaneous ground events are order-invariant;
- swapping logical roles produces swapped observations/rewards within tolerance;
- no stale terminal/reset observation;
- no opponent side-channel API.

### Phase 5 — Correct native self-play baseline

Prerequisite: land the applicable P0 native fixes—training warm start, exact trainable-row masking, primary/frozen architecture equality, the true swap barrier/state reset, and async quiescing (or run with async explicitly disabled).

Deliverables:

- one frozen bank;
- modest historical battle fraction;
- uniform FIFO pool;
- recurrent carry state enabled;
- current-vs-current and current-vs-frozen metrics;
- checkpoint generation instrumentation in tests.

Exit gate:

- frozen rows are never sampled and contribute zero gradient;
- pure battles train both rows and historical battles train only slot 0;
- no episode observes two opponent generations;
- no prefetched/ready rollout crosses opponent generations;
- frozen hidden state is zero after a load;
- one-GPU native self-play runs stably and improves against fixed snapshots;
- slot/physical-side win rates show no unexplained bias.

### Phase 6 — Recover the useful Dogfight ratchet

Deliverables:

- fixed stage/native anchor classes;
- clean-fight and minimum-decisive-game gates;
- top-N or PFSP opponent selection;
- persistent pool manifest;
- recovery policy that does not create invalid PPO actions.

Keep this in a coordinator around public checkpoint/match operations. Do not copy the old PPO implementation.

Exit gate:

- restart reconstructs the exact pool and opponent ranks;
- selection distribution matches a deterministic test;
- promotion requires the documented gates;
- failed promotion cannot mutate frozen anchors;
- every training row still identifies the policy that generated it.

### Phase 7 — Native league and long-run evaluation

Deliverables:

- new native `.bin` anchors;
- both-seat match scheduling;
- architecture/ABI filters;
- atomic candidates/promotions;
- Elo or another explicit rating model;
- reproducible evaluation seeds and confidence intervals.

Old `.pt` anchors remain usable through a legacy evaluator for comparison. If they are strategically important, distill their behavior into a native policy later; do not pretend their LSTM weights are convertible.

Some native `.bin` files produced by the partial port may have a parameter count compatible with a current Dogfight encoder/MinGRU configuration. Quarantine them rather than deleting or trusting them. Compatibility requires:

- exact byte count, not merely a successful prefix read;
- identical encoder/recurrent/decoder parameter ordering;
- identical observation/action semantics;
- fixed-observation forward-output parity;
- recurrent multi-step parity.

The default remains retraining. A matching file size alone is not a model-conversion proof.

Exit gate:

- candidate evaluation survives restart;
- anchors are immutable and hash-verified;
- both roles are evaluated;
- promotion can be replayed from manifest and match logs.

### Phase 8 — Performance work

Only after behavioral and self-play stability:

- benchmark BF16 observations;
- profile environment vs learner;
- optimize hot observation/physics loops;
- consider GPU environment execution only after 5c supports it with self-play;
- validate multi-GPU after its P0 fixes and timeout tests.

## Mandatory TDD and development loop

The port must follow test-driven development rather than relying on eventual
training performance to expose simulation regressions. For each production
behavior change, first add or port a focused failing contract test, implement
the change, and then run the affected regression tiers.

The required working cycle is:

1. **Build:** compile the narrowest sanitizer/local target and the native
   float/full target required by the current phase.
2. **Test:** run the new test and the affected flight, ABI, lifecycle,
   observation, action, reward, terminal, reset, and determinism suites.
3. **Train:** run a short canary with pinned code, configuration, seed, hardware,
   optimizer schedule, and step budget; retain its metrics and checkpoint hash.
4. **Evaluate:** test against fixed scripted/checkpoint opponents and record
   outcome and flight-health metrics.
5. **Render:** after `c_render`/`puf_render` is operational, run a user-visible
   evaluation with `DISPLAY=:0 ./puffer eval dogfight` and inspect the aircraft
   motion, control polarity, spawn, altitude/speed/energy behavior, combat, and
   terminal/reset sequence.

Do not proceed to training after a failed build or regression suite, and do not
advance a phase on an unexplained training/evaluation regression.

Phase 0 must inventory and preserve the relevant Dogfight 3 and partial-port
flight tests with exact provenance. These tests are permanent gates, including
fixed-state observations, control polarity, one-step and 12-step RK4 traces,
longer integration sanity, finite state, spawn/reset, energy/altitude behavior,
gun/cooldown behavior, crash handling, and terminal outcomes. Learning progress
does not excuse a broken flight contract.

Deterministic simulation traces must match exactly or within an explicitly
recorded tolerance across repeated same-seed runs and supported vector layouts.
Training canaries must state which artifacts are expected to match exactly and
which use a declared tolerance or learning envelope. If a deliberate major
change invalidates a baseline, the implementation may be red while still
unlanded, but the change must be explained and behavior-versioned, the old
fixture retained where practical, and the new pinned canary repeated at least
twice before its reviewed results become the baseline for future runs. Golden
fixtures must never be silently updated.

## Test and acceptance matrix

### Build and ABI

- native build accepts the Dogfight header at the pinned commit;
- `--local` standalone build runs under sanitizers;
- `OBS_SIZE`, `NUM_ATNS`, and `ACT_SIZES` match runtime allocations;
- `num_agents` is one in curriculum config and two in self-play config;
- `puf_init` does not access unbound agent buffers;
- agent pointers are non-null, non-aliased where required, and remain valid after vector regrouping;
- a canary test binds deliberately non-contiguous slot pointers;
- total agent rows and minibatch/horizon constraints are validated before training;
- no legacy opponent SHM symbols are needed.

### Observation

- fixed-state expected values for every semantic field;
- both physical perspectives;
- 22-value semantic mode has four exact zero tail values;
- 26-value opponent-aware mode matches the known-good source;
- all reset and step observations are finite;
- normalization/range checks;
- physical-role swap produces the corresponding perspective swap;
- float baseline before BF16 tolerance tests.

### Actions

- each of five indices maps to the documented control;
- controls clamp exactly at the physics boundary;
- trigger threshold/cooldown is symmetric;
- raw sample and executed action are separately observable in tests;
- no post-log-probability learner-action replacement enters PPO;
- clamp-rate metric catches a badly scaled native policy.

### Physics and rewards

- fixed one-step trace;
- fixed 12-step trace;
- long no-action integration sanity;
- player hit, opponent hit, player crash, opponent crash, OOB/invalid, supersonic, timeout;
- simultaneous hits;
- simultaneous ground collision;
- both reward values checked directly, never inferred by negation;
- both published slot rewards are finite and within `[-1, 1]`;
- slot-0 dense parity and perspective-symmetric slot-1 dense fixtures;
- raw reward components, final published reward, and logged episode return cannot be confused;
- terminal and outcome reason checked independently;
- order-invariance when slot/physical mapping changes.

### Reset and determinism

- both initial observations refreshed;
- both rewards cleared;
- terminal remains attached to the completed transition before reset data replaces live state;
- cooldowns, episode returns, opponent controller state, and role mapping reset correctly;
- same seed produces the same trace across repeated runs;
- same seed is stable across supported thread/buffer configurations;
- no `rand`/`srand` calls remain in stepped code.

### Curriculum

- initial target/stage matches configuration;
- every named stage produces expected spawn geometry and opponent settings;
- final stage is reachable and not truncated through integer conversion;
- fixed-stage eval is actually fixed;
- mastery metrics are based on completed episodes;
- checkpoint handoff records stage and source;
- shaping schedule uses true global agent steps if enabled.

### Logging

- `Log` contains only floats;
- raw counts and totals aggregate correctly over environments/buffers;
- `n` convention is tested for one- and two-agent modes;
- ratios are computed after aggregation;
- slot-0/slot-1 score plus draws obey the intended accounting;
- clean gun kill is distinct from crash/invalid "win."

### Native routing and PPO

- pure environment routes both slots to current policy;
- historical environment routes only the declared frozen slot;
- row regrouping preserves each environment's buffer pointers;
- frozen indices have exactly zero sample probability for all supported priority settings;
- frozen data has no effect on advantage moments or any gradient;
- pure rows from both slots do update;
- historical current rows do update;
- arbitrary frozen trajectory corruption leaves learner gradients unchanged.

### Recurrent state

- terminal zeros the correct logical slot state;
- nonterminal horizon carry uses the saved initial state;
- async buffers do not exchange initial states;
- role randomization does not exchange logical policy state;
- opponent checkpoint load zeros the frozen bank state;
- parked environments resume at a clean episode boundary.

### Checkpoints and pool

- exact weight byte count and SHA-256;
- trailing checkpoint bytes are rejected;
- schema/architecture mismatch is rejected;
- warm start loads before initial self-play snapshot;
- initial, periodic, FIFO, and sample behavior;
- manifest write is atomic;
- restart reconstructs pool, parents, counters, and anchors;
- no live anchor can be evicted or overwritten;
- no one episode contains two checkpoint generation IDs.

### Integration

- short curriculum learning smoke test;
- pinned-budget curriculum milestone reproduction;
- current-vs-current symmetry test;
- current-vs-fixed opponent learning test;
- historical self-play stability test;
- both-seat match test;
- restart/continuation test;
- long-run finite obs/action/value/logstd check;
- multi-GPU self-play and final eval complete under a timeout before production use.

## Configuration defaults for the first trustworthy run

These are design defaults, not final tuned values:

- clean 5c commit pinned in the manifest;
- CPU-stepped environment, one GPU learner;
- float observation storage;
- 26-wide opponent-aware observation semantics;
- native recurrent carry enabled (`reset_every_horizon = 0`);
- native async disabled (`async = 0`) for self-play until rotation drains/quiesces prefetched state;
- one frozen bank;
- small historical battle fraction;
- uniform FIFO selection initially;
- no runtime action-recovery override on learner rows;
- fixed or disabled shaping during final self-play;
- single architecture for all pool entries;
- sidecar manifests mandatory;
- one-GPU execution until multi-GPU lifecycle tests pass;
- final native pool evaluation disabled for the initial self-play run until its rank/NCCL/resource-lifecycle fixes pass.

Do not copy a `total_timesteps` value without also fixing the intended number of optimizer updates and LR/entropy schedules.

## Risks and explicit decisions still to make

### Preserve or revise terminal rewards

Recommendation: preserve slot 0's dense-reward parity and the current explicit terminal table, add the versioned perspective-symmetric dense formula for slot 1, and use a separate clean outcome for match/promotion. Revisit zero-sum terminal design only as another versioned experiment.

### Exact curriculum controller

Recommendation: segmented fixed-stage/mastery runs first. Add a generic native lifecycle callback only if segmented runs materially impede learning or reproduce the old schedule poorly.

### Encoder

Recommendation: port the small linear+bias+GELU encoder and A/B it against the current native default. Do not assume the older May result transfers unchanged.

### Old anchors

Recommendation: retain old `.pt` models as legacy behavioral opponents/evaluation oracles. Establish new `.bin` anchors through native training. Consider distillation only after the native environment and league are stable.

### PFSP/league timing

Recommendation: do not make PFSP or Elo a prerequisite for the first native self-play baseline. First prove routing, masking, boundary swaps, recurrence, and symmetry. Add opponent-selection sophistication afterward.

## Final recommendation

The best path is not "port the fork" and not "rewrite Dogfight." It is:

1. pin a clean current 5c;
2. wrap the proven Dogfight simulation in the current native ABI;
3. use the partial port's fixtures to establish deterministic one-agent parity;
4. add generic native warm-start support;
5. convert the duel to two ordinary agent slots and make joint resolution symmetric;
6. fix 5c's generic frozen-row and boundary-swap correctness issues;
7. validate simple uniform native self-play;
8. rebuild the useful clean-fight/ratchet/league policy around native checkpoints and match APIs.

This removes the unfortunate PufferLib 3 modifications rather than recreating them, keeps the best parts of the existing self-play design, and gives each risky change an observable acceptance gate.

## Primary code references

Dogfight 3:

- [`dogfight.py`](dogfight.py): Python environment wrapper and current control propagation
- [`binding.c`](binding.c): custom C vector/opponent side channel
- [`dogfight.h`](dogfight.h): episode simulation and reward/terminal logic
- [`dogfight_observations.h`](dogfight_observations.h): perspective observation construction
- [`dogfight_spawn.h`](dogfight_spawn.h): curriculum spawn geometry
- [`flightlib.h`](flightlib.h): dynamics
- [`train_dual_selfplay.py`](train_dual_selfplay.py): active self-play trainer and ratchet
- [`league.py`](league.py), [`league_manifest.py`](league_manifest.py), and [`anchor_eval.py`](anchor_eval.py): later league/anchor design
- [`pufferlib/vector.py`](../../vector.py): PufferLib 3 shared-memory modifications

Pinned PufferLib 5c:

- [`src/pufferenv.h`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/src/pufferenv.h): native environment ABI
- [`src/pufferl.cu`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/src/pufferl.cu): vector routing, recurrent state, checkpoints, and native self-play
- [`src/algo.cu`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/src/algo.cu): priority sampling, PPO normalization/loss, continuous distribution
- [`src/ocean.cu`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/src/ocean.cu): environment-specific native policy registration
- [`build.sh`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/build.sh): current native build selection
- [`ocean/breakout/breakout.h`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/ocean/breakout/breakout.h): minimal native environment
- [`ocean/g2048/g2048.h`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/ocean/g2048/g2048.h): thin adapter around retained C core
- [`ocean/robocode/robocode.h`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/ocean/robocode/robocode.h): direct two-agent/self-play example
- [`config/default.ini`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/config/default.ini) and [`config/robocode.ini`](https://github.com/PufferAI/PufferLib/blob/ebf5ed03cc3524076b6c1a4033bd69cec0b3db22/config/robocode.ini): native recurrent/self-play defaults and example
