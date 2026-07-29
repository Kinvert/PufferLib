# Dogfight 5c: Next-Agent Handoff and Implementation Plan

Read this file first, then read [`PUFFERLIB_5C_PORT_RESEARCH.md`](PUFFERLIB_5C_PORT_RESEARCH.md) completely before changing code.

This repository is the implementation target. The purpose of this handoff is to let a new agent continue without rediscovering the repository history, accidentally reviving obsolete PufferLib 3/4 APIs, or trusting the work-in-progress 5c self-play path before its correctness problems are fixed.

The current clean-core decision, the capabilities lost when the research core
patch was removed, benchmarking recovery options, and the merge-ready roadmap
are documented in
[`CLEAN_CORE_GAPS_AND_MERGE_PATH.md`](CLEAN_CORE_GAPS_AND_MERGE_PATH.md).
Read that document before restoring or proposing any `src/algo.cu` or
`src/pufferl.cu` change.

## Mission

Port the proven Dogfight 3 simulation and training behavior onto a clean, pinned PufferLib 5c base, then implement trustworthy native two-agent self-play without restoring the old opponent shared-memory or copied-PPO hacks.

The governing strategy is:

1. preserve Dogfight 3 as a runnable behavioral oracle;
2. implement in this clean 5c checkout;
3. extract selected game code and golden fixtures from exact source commits;
4. establish deterministic one-agent/autopilot parity first;
5. convert the duel to two ordinary native `Agent` slots;
6. fix the generic 5c self-play correctness blockers;
7. validate simple native historical self-play;
8. recover promotion gates, PFSP, anchors, and league behavior only afterward.

Do not merge the Dogfight 3 branch, apply its PR as a patch, or cherry-pick the older bulk port.

## Repositories and source authority

| Path | Revision/condition | Authority |
|---|---|---|
| `/home/claude/5c-research` | PufferLib `5c` at `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22` | **Write here.** This is the new port |
| `/home/claude/dogfight3` | Dogfight at `5a3effce470ce19fd6c4c6385a515853a4bfc0cb` | Primary simulation/behavior/history oracle; treat as read-only |
| `/home/claude/dogfight3` merge base | `a9c02db78a896af109e1743dca893ed8869cf14e` | Separates Dogfight/PufferLib changes from upstream 3 |
| `/home/claude/dogfight5` | Committed head `10da7b5abac0f67f69f302c85511baf99a22446c`, dirty working tree | Selective tests/RNG/GELU ideas only; never merge or copy blindly |
| known-good older Dogfight source | `171482a9b9889bebaa427ab658c95c0fc391c031` in the Dogfight 3 repository | Observation/behavior provenance used by the partial-port investigation |

Dogfight 4 is not a base or donor by default. It was an unsuccessful PufferLib 4 attempt. Consult it only if a specific behavior is missing elsewhere and can be proven independently with a fixture.

The Dogfight 3 checkout is more authoritative than the PR diff because it contains the final resolved source and its commit history. Use the PR/history to understand intent, not as an application mechanism.

Useful read-only commands:

```bash
git -C /home/claude/dogfight3 show \
  5a3effce470ce19fd6c4c6385a515853a4bfc0cb:pufferlib/ocean/dogfight/flightlib.h

git -C /home/claude/dogfight3 diff \
  a9c02db78a896af109e1743dca893ed8869cf14e \
  5a3effce470ce19fd6c4c6385a515853a4bfc0cb \
  -- pufferlib/ocean/dogfight pufferlib/vector.py

git -C /home/claude/dogfight5 show \
  10da7b5abac0f67f69f302c85511baf99a22446c:ocean/dogfight/dogfight.h
```

Use `git show` from the pinned revisions when provenance matters. Do not assume the dirty `/home/claude/dogfight5` working tree is committed source.

## First actions in this repository

Before implementation:

1. Confirm the only uncommitted files are the two handoff documents under `ocean/dogfight/`.
2. Confirm `HEAD` is still `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`.
3. Create the implementation branch from that exact revision:

   ```bash
   cd /home/claude/5c-research
   git switch -c dogfight5c
   ```

4. Commit the research and handoff documents as the provenance commit.
5. Record any decision to update/rebase 5c. Do not casually `git pull`: this branch changed rapidly in July 2026, and the ABI/self-play audit is pinned to `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`.
6. If rebasing to a newer 5c tip, repeat at least the ABI, G2048, Robocode, checkpoint-loading, priority-sampling, recurrent-state, and boundary-swap audit before carrying the conclusions forward.

Do not modify either donor repository. Do not overwrite unrelated dirty work. Do not start with a bulk file transplant.

Use `dogfight5c` as the integration branch. Before the first generic change in Phase 2, put CPU seeding, warm-start, and self-play correctness fixes on a separate `5c-selfplay-correctness` branch based on the audited 5c SHA, then stack/rebase the Dogfight integration branch on top. Keep `src/pufferl.cu` ownership serialized and commit each generic invariant with its tests separately. After any controlled upstream rebase, use `git range-diff`, rerun the full native test matrix, and record the new audited SHA.

## What to carry and what to reject

### Carry selectively

| Dogfight source | Treatment |
|---|---|
| `flightlib.h` | Preserve 6-DOF/RK4 physics; refactor RNG/debug globals |
| `autopilot.h` | Preserve scripted controller behavior; refactor initialization RNG |
| `autoace.h` | Preserve tactical controller; refactor all randomness and remove asymmetric shot ownership |
| `dogfight_spawn.h` | Preserve stage geometry; replace global RNG and fix max-step/eval-stage leaks |
| `dogfight_observations.h` | Preserve generalized `(self, other)` perspective math; bind to native agent buffers |
| `dogfight_render.h` | Defer until training parity or adapt minimally to `puf_render`/resource paths |
| simulation portions of `dogfight.h` | Port behind a thin native adapter first; then refactor joint resolution |
| `p40.glb` | Preserve binary content; place in the current 5c resource layout |
| `/home/claude/dogfight3/pufferlib/config/ocean/dogfight.ini` | Mine environment/reward/curriculum values as priors; rewrite into current 5c sections |
| committed partial-port golden tests | Rewrite harness/API while preserving expected values and provenance |
| partial-port per-environment RNG and GELU work | Reapply selectively to current 5c APIs and test |

### Do not carry

- Dogfight 3 `binding.c` as an implementation base;
- `pufferlib/vector.py` opponent shared-memory changes;
- `opponent_observations`, `opponent_rewards`, `opponent_actions`, or `selfplay_active` side channels;
- the copied `train_dual_selfplay.py` PPO implementation;
- the dead `train_selfplay.py` API;
- the partial port's old `vecenv.h`, `bindings.cu`, Python `_C`, or State-buffer integration;
- the duplicated runtime `State` mirror;
- broad core patches from commit `507dbae4c39e0b826c4da6cd4ed89b62f9404497`;
- Dogfight 4 framework plumbing;
- old `.pt` LSTM checkpoints as native MinGRU weights;
- old configuration keys merely because they existed.

Keep old trainers, checkpoints, and league code available as behavior/policy references. They are not runtime dependencies of the new port.

## Target repository shape

The eventual native port will likely contain:

```text
config/dogfight.ini
ocean/dogfight/dogfight.h
ocean/dogfight/flightlib.h
ocean/dogfight/autopilot.h
ocean/dogfight/autoace.h
ocean/dogfight/dogfight_spawn.h
ocean/dogfight/dogfight_observations.h
ocean/dogfight/dogfight_render.h
ocean/dogfight/dogfight.c              # optional standalone harness
ocean/dogfight/dogfight.cu             # later, only for custom native encoder
ocean/dogfight/tests/...
resources/dogfight/p40.glb
```

Do not add a native GPU environment in the initial port. Native self-play at the pinned 5c revision supports CPU-stepped environments with a GPU learner, not `PUFFER_GPU_ENV`.

## Non-negotiable native ABI

Use the current direct `Agent` ABI from `src/pufferenv.h`, not any checked-in stale `binding.c`.

The header must define:

```c
typedef float obs_t;
#define OBS_SIZE 26
#define NUM_ATNS 5
#define ACT_SIZES {1, 1, 1, 1, 1}
```

The CPU `Env` needs, at minimum:

```c
Agent agents[2];
int num_agents;
Log log;
unsigned int rng;
int tag;
int boundary_reached;
```

The current CPU vector code accesses `tag` and `boundary_reached` even outside enabled self-play, so define them unconditionally.

Required callbacks:

```c
void puf_init(Env* env, Dict* kwargs);
void puf_reset(Env* env);
void puf_step(Env* env);
void puf_render(Env* env);
void puf_close(Env* env);
void puf_log(Log* log, Dict* out);
```

Lifecycle rules:

- `puf_init` runs before agent buffers are bound. Parse config and allocate internal state only; do not reset or write observations there.
- The trainer binds each `Agent` pointer independently. Slots in the same duel may be non-contiguous because rows are regrouped by policy bank. Never derive slot 1 by pointer arithmetic from slot 0.
- Only slots `< num_agents` are bound/valid. Native terminals are `float*`; do not retain Dogfight 3's old byte-terminal alias.
- Begin each `puf_step` by clearing active rewards and terminals.
- On terminal, publish final rewards/terminals, log once, set the boundary signal if appropriate, reset internal episode state, and compute next observations without clearing the terminal pulse for the completed transition.
- `puf_reset` must not clear `boundary_reached`; the trainer owns that protocol.
- Keep `Log` all-float and additive. Compute ratios in `puf_log`, not as running ratios inside `Log`.

Use G2048 as the thin-adapter model and Robocode as the direct two-agent representation model. Do not treat Boxoban, Chess, or residual Python self-play as current native templates.

## Two creation modes

Use one implementation with capacity for two agents but two explicit run modes.

### Curriculum/autopilot mode

- `num_agents = 1`;
- slot 0 is native policy 0;
- `agents[1]` is unbound capacity only; no callback may dereference any slot `>= num_agents`;
- the second physical aircraft is internal autopilot/AutoAce;
- native self-play is disabled;
- this is the first parity and curriculum-training target.

### Native self-play/match mode

- `num_agents = 2`;
- slot 0 declares policy 0;
- slot 1 declares policy 1;
- both use ordinary observation/action/reward/terminal buffers;
- pure environments are current-vs-current;
- historical environments are current slot 0 vs frozen slot 1;
- logical agent identity is randomly mapped to physical aircraft at reset;
- both slots terminate and reset together.

Agent count is fixed at vector creation. Do not attempt an in-process one-agent-to-two-agent transition.

## Frozen behavior contracts

### Observations

Compile a 26-float buffer.

Common indices:

| Indices | Meaning |
|---|---|
| 0–2 | body-frame forward speed, sideslip, climb |
| 3–5 | roll, pitch, yaw rates |
| 6–9 | AoA, altitude, G-force, own energy |
| 10–12 | own world-up x/y/z |
| 13–16 | target azimuth, elevation, range, closure |
| 17–20 | energy advantage, aspect, opponent pitch rate, opponent roll rate |

Tail:

| Index | `pilot-22` carried in width 26 | `opponent-aware-26` |
|---|---|---|
| 21 | timer | opponent world-up x |
| 22 | zero | opponent world-up y |
| 23 | zero | opponent world-up z |
| 24 | zero | opponent speed |
| 25 | zero | timer |

The timer moves. The two schemas are not prefix-compatible after index 20 and their checkpoints are not interchangeable.

Start training with opponent-aware 26 semantics. Keep pilot-22 fixtures only for regression/legacy evaluation unless the user deliberately requests that mode.

### Actions

| Index | Contract |
|---|---|
| 0 | throttle: raw `[-1,1]` maps to `[0,1]` |
| 1 | elevator: positive is push/nose down |
| 2 | aileron: positive rolls right |
| 3 | rudder: positive yaws left |
| 4 | trigger: fires above `0.5` |

Native 5c samples an unsquashed Normal. Preserve the raw sample/log-probability, derive one canonical finite/clamped executed vector, and use that executed vector consistently for physics, shaping, telemetry, and trigger logic.

### Rewards and outcomes

Preserve slot 0's dense reward as the parity oracle. There are two incompatible historical slot-1 dense definitions:

- the C side channel starts with negative clamped slot-0 reward, then adds independent opponent terms without a final clamp;
- the active Python trainer ignores that value and stores only negative clamped slot-0 reward.

For true two-agent mode, define slot 1 by applying the same dense function from its own perspective. Clamp each final published slot reward to `[-1,1]`. Version this as a deliberate symmetric reward contract; it is not historical slot-1 parity.

Freeze and test the existing terminal table separately:

- slot-0 kill: `{+1, -1}`;
- slot-1 kill: `{-1, +1}`;
- slot-1 crash: `{+0.25, -1}`;
- slot-0 crash: `{-1, +0.25}`;
- supersonic invalid outcome: `{-1, -1}`;
- timeout: `{-0.5, -0.5}`.

Training reward and match outcome are separate concepts. Promotion uses explicit gun-kill/crash/draw/clean-fight outcomes, not merely reward sign.

## Implementation phases and gates

Do not collapse these into one large port commit.

### Phase 0 — Provenance and golden fixtures

Deliver:

- these two documents committed on `dogfight5c`;
- exact source revision recorded for every imported fixture;
- named observation/action/reward schemas;
- portable fixed-state observation, one-step, 12-step, terminal, and reset fixtures;
- a written list of intended configuration values.

Gate:

- each fixture still runs against its individually recorded oracle revision; many numeric fixtures originate at `171482a9b9889bebaa427ab658c95c0fc391c031`, not Dogfight 3 HEAD;
- no implementation code imported yet without provenance.

### Phase 1 — Minimal one-agent native scaffold

Deliver:

- direct native header/callbacks;
- `config/dogfight.ini` with self-play disabled and `num_agents = 1`;
- internal scripted opponent;
- thin aliases/adapters around the existing tested simulation;
- standalone allocation/free support;
- no rendering requirement for initial training smoke.

Build commands:

```bash
./build.sh dogfight --local
./build.sh dogfight --float
./build.sh dogfight
```

Gate:

- sanitizer-backed reset/step smoke passes;
- full native build succeeds;
- no legacy `vecenv.h`, Python binding, or opponent-side-channel dependency.

### Phase 2 — Deterministic parity

Deliver:

- every RNG call converted to environment-owned deterministic state;
- generic 5c CPU seeding fixed to mix `[base].seed`, rank/global environment identity before `puf_init`;
- mutable integration debug globals removed/localized;
- 26-wide observation publication;
- slot-0 physics/reward/terminal parity;
- reset/max-step/eval-stage leak fixes;
- all-float additive logging.

Gate:

- golden one-step and 12-step traces pass;
- same seed reproduces across supported thread/buffer arrangements;
- different run seeds and global environment identities diverge;
- public reset observation is fresh/finite;
- terminal pulse, return, and immediate reset behavior are correct.

Do not modify expected values to make a port pass unless the behavior change is independently justified and versioned.

### Phase 3 — Native curriculum baseline

Deliver:

- fixed/segmented curriculum control;
- stage geometry tests, including stages 17–20;
- Dogfight linear+bias+GELU encoder registered through current `src/ocean.cu`, after an A/B baseline **(deferred for now; preserve the work item, but do not implement it without the evidence described below)**;
- generic training warm-start fix;
- manifest-last checkpoint publication with exact byte count, schema, and payload hash.

Encoder decision record (2026-07-28): the stock current-5c encoder with
`policy.hidden_size=64` and `policy.num_layers=3` mastered stages 0 through 11
under the pinned 57.7M-step baseline. That satisfies the early-learning gate
without changing PufferLib policy topology. Keep the proposed A/B experiment
available, but prioritize environment and flight fidelity and keep nearly all
port changes in `ocean/dogfight`. Reopen the encoder work only after a
reproducible later-stage learning plateau indicates that the stock policy,
rather than the environment or curriculum, is limiting progress. Record the
new evidence before implementation.

Gate:

- reproduce an early pinned learning milestone such as stage 10/11;
- supplied weights load before self-play's initial snapshot;
- initial snapshot hash matches the supplied curriculum checkpoint;
- incompatible/trailing checkpoint bytes fail loudly;
- learning-rate schedule/update count is held constant in comparisons.

### Phase 4 — Symmetric direct two-agent environment

Deliver:

- both native agent slots bound directly;
- both perspective observations;
- versioned symmetric dense rewards;
- two-phase symmetric fire/crash/terminal resolution;
- identical cooldown/trigger rules;
- logical-to-physical role randomization;
- `slot_0_score`, `slot_1_score`, `draw_rate`, and clean-fight logs.

Gate:

- forced actions independently control the intended aircraft;
- swapping physical roles swaps observations/rewards/outcomes;
- simultaneous shot/crash tests are order-invariant;
- both reset observations are fresh;
- A-vs-A slot score is approximately balanced over seeded trials;
- no opponent side channel exists.

### Phase 5 — Fix native self-play before trusting it

These are generic PufferLib issues, not Dogfight workarounds. Keep each fix isolated and tested.

1. **Training load before bootstrap**
   - `run_train` currently ignores `base.load_model_path`;
   - load/validate primary weights and synchronize actor weights before saving pool entry zero.

2. **Exact frozen-row exclusion**
   - zeroed frozen advantages still receive priority;
   - at `prio_alpha = 0`, `pow(0,0)` ordinarily gives full uniform weight;
   - for positive alpha, epsilon still makes frozen rows sampleable;
   - if selected, advantage normalization and value/entropy terms create gradients;
   - construct sampling and moments only from explicit trainable/current rows, with a loss mask as defense in depth.

3. **Architecture equality**
   - require primary and frozen hidden size/layer count equality for the native pool;
   - mirror sweep choices;
   - require exact file size, not a readable prefix.

4. **Atomic opponent swap**
   - sticky `boundary_reached` is not a barrier;
   - park each tagged environment at its first terminal after a swap is armed;
   - once all are parked, load weights, zero that bank's recurrent state, and release/reset together;
   - prove no episode contains two opponent generations.

   Use a synchronous truncation protocol first: with async disabled, rotate only between completed rollouts; explicitly truncate/reset every environment using that bank; count swap truncations separately; load the new bank; clear the entire frozen-bank state and the affected primary recurrent rows; publish fresh observations; then resume. This is less elegant than natural-terminal parking but has a much smaller correctness surface. Never silently change weights in live episodes.

   Natural-terminal parking is a later alternative, not a `puf_step` no-op. The vector path still performs inference, records a row, and advances RNN state before calling `puf_step`. A correct parked design must either discard every alignment rollout until all relevant environments are parked or carry a transition-validity mask through GAE, priority sampling, advantage moments, and every loss. Environments such as Robocode have already reset internally when they set the boundary flag, so release must not accidentally reset them twice.

5. **Async generation barrier**
   - current async training prefetches and finishes the next rollout before rotation is checked;
   - a ready rollout and live state can therefore belong to the old opponent generation even after the visible boundary check is repaired;
   - rotation processing also currently occurs after a dashboard/log-throttle `continue`;
   - initially reject `selfplay.enabled && base.async` and set `[base].async = 0`;
   - move the rotation state machine ahead of display throttling;
   - only re-enable async after prefetch is explicitly stopped and old ready slots are drained/discarded before swap.

6. **Initial execution limitations**
   - use one GPU and one frozen bank;
   - set `[base].async = 0`;
   - disable final pool evaluation initially;
   - do not enable GPU-environment mode.

Gate:

- no frozen index is sampled for alpha 0 or 1 over millions of test draws;
- gradients are invariant to arbitrary frozen-row contents;
- primary/frozen mismatch is rejected before CUDA execution;
- synchronous swap tests prove atomic truncation/reset and zero affected primary/frozen recurrent state;
- if natural parking is later implemented, staggered-episode tests prove invalid alignment rows never train and release does not double-reset;
- no fight observes two checkpoint IDs.
- no prefetched or ready rollout crosses checkpoint generations.

### Phase 6 — Simple native self-play

Deliver:

- one frozen bank;
- modest historical-battle fraction;
- uniform FIFO pool;
- current-vs-current and current-vs-history metrics;
- checkpoint generation logging;
- one-GPU stability/learning baseline.

Remember that `frozen_bank_pct` is applied to battles/environments in the pinned allocator. With 4,096 rows, 2,048 two-agent battles, and 0.10, only 204 rows (about 4.98%) are frozen.

Gate:

- pure battles train both current rows;
- historical battles train current slot 0 only;
- self-match is side-balanced;
- the learner improves against fixed snapshots without NaN/Inf or unexplained clamp saturation.

### Phase 7 — Recover Dogfight's useful self-play policy

Only after Phase 6:

- persistent pool manifest and restart;
- exclude the currently loaded opponent when alternatives exist;
- fixed native anchors;
- both-seat matches;
- minimum 10 total decisive gun kills;
- learner decisive-kill share at least 55%;
- clean-fight rate at least 80%;
- two successful rotations before promotion;
- top-N/PFSP selection using persisted evidence;
- recovery behavior that never substitutes learner actions after log-probability capture;
- league/Elo candidates and atomic promotion.

Do not fork or copy PPO to implement these. Build a coordinator around native checkpoints, match operations, and explicit metadata.

### Runtime self-play choice

Keep both implementations. They solve different layers of the problem, and the
Dogfight coordinator must not replace or fork PufferLib's trainer.

`config/dogfight.ini` uses `selfplay.mode` as the authoritative switch:

| Mode | Owner and behavior |
|---|---|
| `off` | Disable frozen banks for ordinary curriculum or fixed-opponent work. |
| `native` | Use PufferLib's official in-process self-play: current-v-current battles, one frozen FIFO bank, checkpoint insertion, uniform historical-opponent sampling, synchronous rotation, and optional final pool evaluation. This is the Dogfight default. |
| `coordinator` | Use the persistent Dogfight manifest/PFSP/promotion system around native training and `match` operations. The coordinator must select `base.load_enemy_model_path`; the runtime pins that opponent with `opp_timeout_steps=0`, and `selfplay.eval_games` must remain zero because the coordinator owns both-seat evaluation. |

Retain `selfplay.enabled` for compatibility with upstream-style configs.
When `selfplay.mode` is absent, `enabled=1` resolves to `native` and
`enabled=0` resolves to `off`. When an explicit mode is present, the mode wins.
Standard `eval` and `render` disable training-only frozen banks; `match`
constructs its opponent bank explicitly.

The pinned 5c native references work as follows:

| Environment | Native self-play behavior at the audited revision |
|---|---|
| Robocode | Two policy slots per battle; ordinary battles are current-v-current and tagged battles put the current policy in slot 0 and a historical FIFO policy in slot 1. Frozen rows are excluded from PPO. Its INI enables one 10% frozen bank, a size-100 FIFO, 100M-step opponent timeout, and final evaluation against up to eight pool entries. |
| Chess | The same native memory-only FIFO mechanism, with randomized colors, one 10% frozen bank, a size-500 pool, and a 4B-step opponent timeout. |
| G2048 | Thin native adapter reference only, not a self-play policy reference. |

PufferLib's native pool is deliberately simple and process-local. It does not
persist a manifest, Elo, PFSP evidence, fixed anchors, both-seat promotion
records, or atomic league promotion. Those remain the coordinator's Phase 7
responsibility. The coordinator should launch bounded native training segments;
it should not duplicate PPO, checkpoint loading, inference, or match execution.

See `ocean/dogfight/PHASE7_STATUS.md` for dated commands that worked on this
branch. They are evidence to retry, not commands to accept as gospel after 5c
changes.

### Phase 8 — Scaling and optimization

After correctness:

- BF16 observation experiment;
- rendering/resource integration;
- hot-loop profiling;
- real native destruction or subprocess-isolated repeated matches;
- multi-GPU run-ID, bounded-wait, NCCL final-eval fixes and timeout tests;
- GPU environment only if native self-play gains explicit support.

## Current 5c source anchors

At `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`, inspect these before editing:

| Concern | Source |
|---|---|
| native Agent ABI | `src/pufferenv.h` |
| environment inclusion/build | `build.sh`, `src/pufferl.cu` near the environment header include |
| CPU environment init/binding | `src/pufferl.cu` around 1791–1915 |
| inference/routing | `src/pufferl.cu` around 814–881 and 1850–1915 |
| frozen advantage zeroing | `src/pufferl.cu` around 1112–1127 and 1347–1365 |
| priority epsilon/CDF | `src/algo.cu` around 1160–1320 |
| minibatch advantage normalization/loss | `src/algo.cu` around 1541–1689 and 1806–1815 |
| checkpoint loader | `src/pufferl.cu` around 1612–1641 |
| primary/frozen policy construction | `src/pufferl.cu` around 1954–2098 |
| native training/bootstrap | `src/pufferl.cu` around 3135–3205 |
| async rollout prefetch ordering | `src/pufferl.cu` around 3233–3259 |
| dashboard/log throttle ordering | `src/pufferl.cu` around 3285–3289 |
| opponent rotation | `src/pufferl.cu` around 3347–3372 |
| final pool evaluation | `src/pufferl.cu` around 3440–3474 |
| multi-rank launch | `src/pufferl.cu` around 3603–3642 |
| minimal ABI example | `ocean/breakout/breakout.h` |
| thin preserved-core adapter | `ocean/g2048/g2048.h` |
| direct two-agent example | `ocean/robocode/robocode.h` |

Line numbers are revision-specific. Re-search symbols after any rebase.

## Configuration warnings

- Native policy/action RNG reads `[base].seed`, not `[train].seed`.
- Historical pool sampling reads `[selfplay].seed`.
- At the pinned revision CPU `Env.rng` ignores both until the generic seeding fix.
- `checkpoint_interval` is in epochs, not agent steps.
- `total_agents` counts agent rows, not fights.
- horizons must meet current precision alignment constraints.
- minibatch size must be divisible by horizon.
- all continuous action heads have size one.
- native train currently does not perform a full resume; a weight warm start resets optimizer/schedule/counters unless those are explicitly added.
- native `.bin` files have no embedded type/schema and the loader accepts a valid prefix unless exact size is checked.
- `[torch]` names and some older policy/config fields are residual Python-facing metadata and may not affect the native policy.
- trusted self-play must use `[base].async = 0` until opponent rotation explicitly quiesces/drains prefetched rollout state.

## Checkpoint policy

Every retained native checkpoint needs an atomic sidecar manifest containing:

- PufferLib commit;
- Dogfight behavior/reward version;
- observation width, field schema, and semantics;
- action order/polarity/trigger;
- encoder/recurrent/decoder architecture;
- exact parameter count and byte size;
- SHA-256;
- global agent steps and curriculum stage;
- parent checkpoint;
- optimizer/scheduler/pool resume status.

Two files cannot be renamed atomically as a pair. Publish the validated weight payload first and rename the manifest last as the commit marker. Loaders and "latest" discovery must ignore orphan `.bin` payloads and verify the manifest's exact byte count and hash before loading. A single versioned checkpoint container is an acceptable alternative.

Old PyTorch LSTM `.pt` files are legacy evaluation oracles, not native load candidates. Partial-port `.bin` files are quarantined until exact size, parameter order, fixed-observation forward parity, and recurrent multi-step parity all pass.

## Working discipline

### Mandatory TDD and build-test-train-eval loop

Dogfight development is test-driven. Before changing production behavior, add or
port the smallest test that expresses the intended contract and observe it fail
for the expected reason. Implement only enough to make that test pass, then run
the broader affected regression tier. Characterization tests imported from
Dogfight 3 or the partial port must record their exact source revision and
semantic schema.

Every behavior-affecting increment follows this loop:

1. **Build** the narrowest relevant sanitizer/local target, then the native
   float/full target required by the phase.
2. **Test** the new contract plus the affected ABI, lifecycle, observation,
   action, flight-dynamics, reward, reset, and determinism regressions.
3. **Train** a short pinned-config, pinned-seed canary before longer curriculum
   work. Record the source revision, behavior version, configuration, seed,
   hardware, metrics, and checkpoint hash.
4. **Evaluate** the canary against the phase's fixed scripted or checkpoint
   opponents. Once `c_render`/`puf_render` is available, also run a visible
   evaluation on the workstation with:

   ```bash
   DISPLAY=:0 ./puffer eval dogfight
   ```

5. **Advance** only when the current loop is green and its artifacts are
   attributable to the exact code/configuration that produced them.

Stop the loop at the first failing stage. Do not use a training run to work
around a failed build or regression test.

Flight behavior is a permanent critical gate. During Phase 0, inventory and
vendor the relevant Dogfight 3 and partial-port flight tests before porting
their implementation. The standing suite must cover fixed-state observations,
action/control polarity, one-step and 12-step RK4 traces, longer integration
sanity, spawn/reset state, finite aircraft state, energy/altitude behavior,
gun/cooldown behavior, crashes, and terminal outcomes. A change that makes the
plane learn while violating these contracts is a regression, not progress.

Determinism baselines are versioned contracts. Environment/physics traces must
reproduce exactly or within their declared numeric tolerance across repeated
same-seed runs and supported vector layouts. Training canaries must declare
which outputs are expected to be exact and which use a tolerance or learning
envelope. A major intentional behavior change may temporarily make an old
baseline red during unlanded work, but before the change advances it must:

1. explain and version the behavior change;
2. retain the old fixture for its old behavior version when practical;
3. run the new pinned canary at least twice under the same declared conditions;
4. establish a new reviewed baseline that future runs must match.

Never silently rewrite a golden value, accept unexplained flight drift, or defer
determinism repair to a later phase.

- Make small commits with one behavioral purpose.
- Keep generic PufferLib fixes separate from Dogfight environment changes.
- Prefix or otherwise clearly name generic versus environment commits.
- Run the narrowest relevant tests after every change.
- Preserve both donor repositories.
- Never silently update a golden fixture.
- Never declare self-play correct from a smoke run alone.
- Never infer a slot's reward by negating the other.
- Never assume two agent buffers are contiguous.
- Never load a checkpoint based only on a successful `fread`.
- Never let a frozen policy change mid-episode.
- Never train on an action produced by a different policy or replaced after its log-probability was recorded.

## Stop and reassess if

- the target 5c revision has changed without re-audit;
- the repository contains unexpected dirty files;
- a change requires reviving old VecEnv/Python opponent APIs;
- a golden slot-0 physics/observation trace changes unexpectedly;
- an imported test depends on hard-coded donor paths instead of a vendored fixture;
- a generic framework change cannot be tested independently;
- a frozen row reaches any learner loss;
- a checkpoint architecture cannot be proven identical;
- an opponent swap cannot guarantee one generation per episode;
- a proposed shortcut couples curriculum, two-agent conversion, and PPO changes in one step.

## Definition of success

The port is complete only when:

1. Dogfight builds and trains through current native `./puffer`;
2. deterministic one-agent behavior matches the pinned oracle where parity is intended;
3. intentional behavior changes are named, versioned, and tested;
4. both aircraft are ordinary native agents in self-play;
5. no Dogfight-specific PufferLib vector or PPO fork exists;
6. frozen opponent data cannot update the learner;
7. opponent generations change only at a real episode barrier with cleared recurrent state;
8. curriculum weights reliably initialize self-play;
9. match and promotion metrics are side-balanced and outcome-based;
10. the recovered self-play/league system is restartable, auditable, and checkpoint-schema safe.

The first implementation objective is deliberately smaller: **a sanitizer-clean, deterministic, one-agent native Dogfight scaffold that reproduces pinned slot-0 behavior.** Do that before adding native self-play.
