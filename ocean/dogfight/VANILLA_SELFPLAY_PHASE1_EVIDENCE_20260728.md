# Vanilla Self-Play Phase 1 Evidence

Date: 2026-07-28

## Decision

Phase 1 is complete. Dogfight's native two-agent environment now implements the
same tagged historical-match boundary signal used by Robocode. The change is
Dogfight-local and does not alter flight physics, actions, observations,
rewards, spawns, one-agent curriculum behavior, or PufferLib core.

## Contract Change

Robocode sets `boundary_reached = 1` when an episode ends in an environment with
`tag > 0`. It does not set the signal for current-current matches and does not
clear the signal during environment reset. PufferLib's self-play coordinator
owns clearing the signal after processing the boundary.

Dogfight already had:

- two independently bound native agent slots;
- policy declarations `0` and `1`;
- both perspective observations and dense rewards;
- independently routed actions;
- joint terminal pulses and reset;
- order-invariant simultaneous hits and crashes;
- symmetric cooldown behavior;
- deterministic logical-to-physical role randomization;
- role-aware score and gun-kill accounting.

Dogfight was missing the tagged terminal signal. The shared native two-agent
terminal path now raises it only when `env->tag > 0`.

## Tests Added

The two-agent regression now verifies:

- `puf_init` declares policy rows `0` and `1`;
- `puf_init` does not touch agent buffers before vector binding;
- a tagged historical match raises `boundary_reached`;
- `puf_reset` preserves the trainer-owned signal;
- a pure current-current match does not raise the signal.

Observed focused result:

```text
1 passed, 21 deselected
```

Observed full result:

```text
46 passed, 3 skipped in 23.10s
```

Both the native trainer and Dogfight-local evaluator rebuilt successfully.

## Deterministic Training Guard

Observed working command:

```bash
bash ocean/dogfight/train_reproduction.sh \
  vanilla_phase1_one_agent_guard_s42_20260728 1048576 0
```

The Phase 0 and Phase 1 one-agent checkpoints had the same SHA-256:

```text
6d18e8b2793e18d80accc797c69fd78e023fa7a6dad0ea86379b62bd65c042e6
```

This proves that the tagged two-agent terminal change did not alter the
fixed-seed one-agent training trajectory.

## Acceptance Evaluation

Headless command:

```bash
bash ocean/dogfight/eval_checkpoint.sh headless \
  checkpoints/dogfight/vanilla_phase0_stage0_32m_s42_20260728/0000000033554432.bin \
  0 16 42
```

Headless result:

```text
14/16 wins
performance = 0.875
score = 0.812
```

Visible command:

```bash
DISPLAY=:0 bash ocean/dogfight/eval_checkpoint.sh visible \
  checkpoints/dogfight/vanilla_phase0_stage0_32m_s42_20260728/0000000033554432.bin \
  0 8 42
```

Visible result:

```text
8/8 wins
```

Human observation: the run was extremely fast, but the policy appeared to win
efficiently. No persistent right-roll or other control bias was observed.

## Phase 2 Entry Conditions

- Keep PufferLib core unchanged.
- Use PufferLib's official policy pool and frozen-bank path.
- Disable the custom coordinator and environment curriculum in the vanilla
  profile.
- Begin with fixed stage-0 geometry and deterministic role randomization.
- Prove pool creation, historical routing, and safe boundary rotation in a
  bounded canary before judging learning quality.
