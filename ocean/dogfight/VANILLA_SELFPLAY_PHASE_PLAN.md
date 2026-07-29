# Dogfight Vanilla PufferLib Self-Play Plan

## Objective

Build a merge-ready Dogfight environment that follows PufferLib 5c's native
Robocode self-play structure while preserving Dogfight's RK4 flight model,
continuous controls, observations, and mandatory human-visible flight checks.

The primary engineering goal is the shortest wall-clock time to a strong policy,
not SPS in isolation. `20M agent SPS` is a stretch target, not a promise or a
reason to weaken flight physics. Intermediate targets are `2M`, `5M`, and `10M`
agent SPS.

The detailed source analysis is in
[`ROBOCODE_VANILLA_SELFPLAY_ANALYSIS.md`](ROBOCODE_VANILLA_SELFPLAY_ANALYSIS.md).

## Current Safety Point

- Experimental branch: `dogfight5c-vanilla-selfplay`
- Clean-core baseline commit: `fe1c1332`
- Protected baseline branch: `dogfight5c-clean-core-baseline`
- Earlier Dogfight branch remains available at commit `a8fc98fb`.
- Joseph's pinned 5c base remains available at commit
  `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`.
- The current worktree is not literally clean because this plan, the Robocode
  analysis, and generated root launchers are untracked.
- The generated root files `dogfight` and `puffer` must never be committed.
- No experiment from this branch should be merged wholesale. Merge-ready changes
  should be reviewed and cherry-picked as small Dogfight-local commits.

## Non-Negotiable Constraints

- Do not modify PufferLib core for the vanilla implementation.
- Keep implementation changes in `ocean/dogfight` and configuration changes in
  `config/dogfight.ini`.
- Preserve RK4 and the existing flight equations unless a separately reviewed
  physics change is explicitly approved.
- Do not reduce observation quality merely to imitate Robocode's observation
  shape.
- Do not replace continuous aircraft controls merely to imitate Robocode's
  discrete action heads.
- Support one GPU. Do not design the primary path around multi-GPU training.
- Keep the existing Dogfight coordinator available, but disable it in vanilla
  self-play mode.
- Judge learning against fixed external scenarios and saved opponents. A
  self-play policy's win rate against itself is not evidence of improvement.
- Never trade stable, credible aircraft behavior for a favorable throughput
  number.

## Structure We Will Copy From Robocode

- A native compiled two-agent environment.
- Policy row `0` and policy row `1` representing the two aircraft.
- Most matches use current-policy versus current-policy training.
- A configurable minority of matches use current-policy versus frozen-history
  rollout.
- PufferLib's existing policy pool, frozen bank, checkpoint cadence, opponent
  timeout, asynchronous collection, and CUDA graph paths.
- Environment `tag` and `boundary_reached` fields for safe opponent replacement
  only at episode boundaries.
- Joint episode termination and reset semantics.
- Symmetric seat assignment and evaluation from both seats.
- Configuration-driven experiments rather than trainer patches.

## Standard TDD and Acceptance Loop

Every behavior-changing phase follows this order:

1. Add or update a focused test that fails for the intended reason.
2. Make the smallest Dogfight-local implementation change.
3. Build the native environment.
4. Run focused tests, then the complete Dogfight test suite.
5. Run a fixed-seed training canary and compare deterministic artifacts or the
   documented statistical tolerance.
6. Run the phase's required training duration.
7. Run headless evaluation against fixed scenarios and opponents.
8. Run evaluation with `DISPLAY=:0` and have a human assess the aircraft.
9. Record the exact working commands, commit, configuration, seed, SPS, elapsed
   time, model path, and visible flight result.

Do not accept a phase when flight tests regress, an aircraft develops a stuck or
biased control, deterministic behavior changes without explanation, or visible
flight becomes materially worse.

## Measurement Contract

- Report PufferLib's end-to-end `agent SPS` after warm-up.
- Also report `match steps/s`, which is approximately half of agent SPS for a
  two-aircraft environment.
- Record environment, inference, transfer, and learner timing when available.
- Record peak CPU memory and GPU memory.
- Use the same hardware, seed set, policy, horizon, and logging level for direct
  throughput comparisons.
- Use median steady-state SPS rather than a startup peak.
- Measure skill per wall-clock hour as well as raw SPS.
- Keep profiler and diagnostic logging disabled during the final throughput
  measurement, but retain a separate diagnostic run.

Approximate time for two billion agent steps:

| Agent SPS | Ideal elapsed time |
| ---: | ---: |
| 1M | 33m 20s |
| 2M | 16m 40s |
| 5M | 6m 40s |
| 10M | 3m 20s |
| 20M | 1m 40s |

These are ideal throughput calculations and exclude startup, evaluation,
checkpointing, and stalls.

## Phase 0: Lock the Baseline and Benchmark Protocol

### Work

- Preserve `fe1c1332` as the immutable clean-core rollback point.
- Confirm that PufferLib core matches Joseph's expected source on this branch.
- Establish one canonical build command and one command each for focused tests,
  full tests, training, headless evaluation, and visible evaluation.
- Establish a short performance canary and a full learning run.
- Capture current Dogfight SPS, wall-clock time, highest mastered fixed stage,
  fixed-scenario results, and visible flight notes.
- Define golden RK4 flight traces covering level flight, pitch, roll, energy
  building, recovery, and representative control transitions.
- Define which outputs must be bitwise deterministic and which training metrics
  use a statistical tolerance.

### Gate

- Stock core is confirmed.
- All baseline flight tests pass.
- Repeated performance canaries agree within `5%` steady-state SPS.
- Fixed-seed environment traces match exactly or within an explicitly justified
  floating-point tolerance.
- A human-visible baseline evaluation is recorded.

## Phase 1: Match the Native Two-Agent Contract

### Work

- Make both aircraft ordinary native agents with policy rows `0` and `1`.
- Produce observations, rewards, actions, termination, and reset data for both
  perspectives without Python work in the hot loop.
- Add and validate `tag` and `boundary_reached` semantics expected by PufferLib's
  frozen policy bank.
- Ensure opponent replacement cannot occur in the middle of a Dogfight episode.
- Ensure a joint terminal condition cannot leave one aircraft training against a
  stale or reset opponent.
- Verify seat mirroring so neither row receives a systematic geometry or control
  advantage.
- Disable the custom coordinator only when the vanilla self-play mode is selected.

### Tests

- Two-agent observation and action routing.
- Reward perspective and zero-sum invariants where intended.
- Joint termination and reset.
- Boundary transition behavior.
- Tag behavior for current-current and current-history matches.
- Seat-mirrored spawn equivalence.
- Existing flight dynamics and RK4 traces.

### Gate

- Both current-policy rows contribute training data in current-current matches.
- The historical opponent row acts but is excluded by PufferLib's existing
  frozen-policy behavior.
- There are no core changes.
- Build, tests, deterministic canary, headless evaluation, and visible evaluation
  pass.

## Phase 2: Run the Smallest Pure Vanilla Self-Play Canary

### Work

- Add a clearly named vanilla configuration in `config/dogfight.ini`.
- Disable scripted opponents, AutoAce selection, custom curriculum advancement,
  and the custom self-play coordinator for this configuration.
- Start with one frozen bank and Robocode-like current-current/current-history
  routing.
- Start near Robocode's `0.1` historical-environment fraction.
- Use a bounded, symmetric spawn distribution and a finite episode timeout.
- Keep the existing policy size initially so topology and model size are not
  changed in the same experiment.
- Start at a memory-safe agent count, expected to be `4096`, before scaling.
- Confirm pool snapshots are created, loaded, and changed only at safe boundaries.

### Gate

- A short run completes without crash, deadlock, stale episode, or invalid action.
- The pool grows on the configured cadence.
- Frozen opponents rotate after episode boundaries.
- Current-current matches remain seat-symmetric.
- Two identical fixed-seed canaries meet the defined repeatability rule.
- Visible flight remains physically credible.

## Phase 3: Scale the Existing Path Before Rewriting It

### Work

- Sweep total agents through memory-safe values such as `4096`, `8192`, and
  `16384`.
- Sweep native buffers through values such as `2`, `4`, and `8`.
- Keep native threads at least equal to active buffers when CPU capacity permits.
- Compare asynchronous collection enabled and disabled.
- Compare CUDA graphs enabled and disabled.
- Tune horizon and minibatch only after measuring rollout and learner balance.
- Keep physics, observations, actions, rewards, and spawn distribution fixed.
- Reject configurations that gain peak SPS by causing stalls, excessive memory
  pressure, or worse wall-clock learning.

### Gate

- Select a reproducible single-GPU configuration.
- Reach at least `2M agent SPS`, or document measured evidence showing the current
  bottleneck prevents it.
- Preserve all flight traces and learning-canary behavior.
- Record a profile-backed bottleneck before beginning code-level optimization.

## Phase 4: Optimize Only the Dogfight Native Hot Path

### Work

- Profile before changing code.
- Remove redundant relative-geometry, trigonometric, quaternion, and observation
  calculations shared by the two perspectives.
- Compute pairwise state once per match when both agents can reuse it.
- Hoist constants and episode-invariant calculations out of the RK4 loop.
- Remove allocations, formatting, logging, and avoidable branches from step and
  reset hot paths.
- Improve cache locality and contiguous access where profiling demonstrates a
  benefit.
- Allow compiler vectorization and safe parallel buffer execution without changing
  numerical meaning.
- Batch reset initialization when it does not change seeded spawn results.
- Retain RK4 step size, integration order, force model, and control dynamics.
- Land one measured optimization per commit so regressions can be bisected.

### Gate

- Each optimization has a failing performance or redundancy test before the
  implementation where practical.
- Each optimization reports SPS before and after under the measurement contract.
- Golden RK4 traces remain within the predeclared tolerance.
- All flight tests and visible evaluations pass.
- Target ladder is `5M` as strong, `10M` as excellent, and `20M` as a stretch.
- If `20M` requires reduced physics fidelity, stop below `20M`.

## Phase 5: Establish a From-Scratch Learning Baseline

### Work

- Train pure native self-play from random initialization using symmetric,
  bounded, initially manageable spawn geometry.
- Use current-current matches for most samples and current-history matches for
  anti-forgetting pressure.
- Save initial, early, middle, recent, and final policy snapshots.
- Evaluate every candidate from both seats against fixed curriculum stages,
  scripted opponents, and saved snapshots.
- Include energy-building, recovery, and adverse-geometry scenarios, not merely
  close favorable engagements.
- Run a short canary first, then a full run sized from measured wall-clock cost.

### Gate

- Skill improves against fixed external tests, not only against the latest
  self-play opponent.
- Performance is not explained by timeout exploitation or one favorable seat.
- The policy retains early-stage control while improving on harder geometry.
- Human-visible evaluation shows controlled flight, energy management, and useful
  pursuit behavior.

## Phase 6: Add Progressive Spawn Difficulty Only If Needed

### Work

- Keep PufferLib's native self-play and frozen pool unchanged.
- Implement difficulty through Dogfight-local spawn distributions rather than
  trainer or core modifications.
- Begin with advantageous pursuit geometry, then mix neutral and adverse geometry.
- Mirror roles and seats so the policy learns both advantage conversion and
  disadvantage recovery.
- Prefer a deterministic step-based mixture for the first implementation.
- Add performance-controlled advancement later only if the fixed schedule
  demonstrably fails.
- Preserve fixed evaluation suites outside the training distribution.

### Gate

- Advancement improves contiguous fixed-stage mastery.
- Later difficulty does not erase level flight, recovery, or favorable-spawn
  competence.
- Both seats and both geometric roles remain balanced.
- No PufferLib core or custom trainer coordination is required.

## Phase 7: Tune Historical Self-Play Pressure

### Work

- Sweep historical-environment fraction around the initial `0.1` value.
- Tune checkpoint cadence in agent steps rather than assuming Robocode's absolute
  interval transfers to Dogfight.
- Tune opponent timeout and pool size.
- Begin with one frozen bank; add banks only if the current environment-policy
  contract supports them without core changes and measurements justify them.
- Compare final policies against early, middle, recent, and final pool members.
- Look for cycling, forgetting, overfitting to a single opponent, and policy lag.
- Do not add ELO or PFSP unless uniform sampling fails in a measured way.

### Gate

- The final policy improves or holds performance across the historical suite.
- Pool use improves fixed-scenario wall-clock learning over current-current only.
- Frozen rollout cost does not negate its learning benefit.
- Results reproduce across the agreed seed set.

## Phase 8: Full Acceptance and Merge Preparation

### Work

- Run the complete build, test, full-train, headless-eval, and `DISPLAY=:0`
  visible-eval loop.
- Run the full fixed flight suite, both-seat combat suite, historical opponent
  suite, and deterministic trace suite.
- Record final SPS, match steps/s, wall-clock training time, peak memory, highest
  mastered stage, and opponent results.
- Verify PufferLib core has no branch changes.
- Document every command that actually worked and label source-derived commands
  as examples rather than guarantees.
- Separate required merge-ready changes from experimental configurations and
  diagnostics.
- Prepare small commits limited to Dogfight and its configuration.

### Gate

- Stock PufferLib core.
- Reproducible training from a documented command.
- No flight-test regression.
- Human-visible aircraft behavior is acceptable.
- Clear skill improvement against fixed and historical opponents.
- Throughput and wall-clock-to-skill are documented honestly.
- The resulting commit series is suitable for Joseph to review without accepting
  unrelated trainer changes.

## Decision Rules

- If Phase 2 cannot execute native self-play correctly, fix the environment
  contract before any throughput work.
- If Phase 3 is learner-bound, tune batching and policy cost before optimizing
  RK4.
- If Phase 3 is environment-bound, continue to Phase 4 with profiler evidence.
- If pure self-play learns from scratch, keep Phase 6 optional.
- If pure self-play stalls, use progressive symmetric spawn mixtures before
  reintroducing a coordinator.
- If SPS rises while fixed-scenario mastery per hour falls, reject the SPS change.
- If an optimization changes flight traces or visible handling, reject it unless
  the physics change is separately understood, tested, and approved.
- If any required solution needs PufferLib core changes, stop and document the
  missing extension point rather than silently patching core.

## Expected Outcome

The intended result is not a literal copy of Robocode. It is Dogfight using the
same scalable native self-play topology:

- Native two-aircraft rollouts.
- Mostly current-current training.
- A small current-history fraction.
- PufferLib's official frozen pool and boundary handling.
- Dogfight-local spawn progression when required.
- No custom core patches.
- RK4 flight fidelity protected by tests and visible evaluation.

The first meaningful milestone is a correct vanilla self-play canary. The second
is a measured multi-million-SPS configuration. The final milestone is improved
fixed-opponent and human-visible combat performance per wall-clock hour.
