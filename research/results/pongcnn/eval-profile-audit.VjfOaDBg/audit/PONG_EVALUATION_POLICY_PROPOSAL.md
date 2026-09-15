# Proposed Pong evaluation v2 — not used in the original experiment or audit

## Problem in v1

The evaluator runs 64 environments continuously, pools completed-match logs,
and stops at the first rollout boundary where the total is at least the requested
count. A finished environment immediately starts another match. A slow environment
can therefore contribute fewer completed matches, or none, while others contribute
many. This is a completion-conditioned sample, not a fixed allocation of matches
to independently seeded episode identities. At a finite stopping time it can
depend on the association between match duration and outcome. Aggregate progress
cannot establish the size or direction of that effect.

No periodic full-vector reset occurs in standalone evaluation: `eval_make` sets
`base.reset_every_horizon=0`, and `env_start` initializes the vector once. Each
individual environment resets after reaching `max_score`. `vec_log(clear=0)`
accumulates the completed-match statistics. The existing `episode_length` uses
the tick counter since the last point reset, not total match decisions.

## Separately versioned policy to validate before a future comparison

- Preallocate exactly 512 episode IDs, eight per each of 64 slots. Derive each
  episode seed deterministically from the evaluation seed and episode ID; record
  the mapping. Do not take additional episodes from fast slots.
- Stop a slot after its eighth assigned episode; mask its actions, environment
  steps and statistics while other assigned episodes finish. Preserve the model's
  recurrent reset semantics. Test that masked slots cannot affect active slots.
- Record whole-match decisions, rally decisions, points and outcome separately
  for every episode. Never label a partial match a completed match.
- Predeclare a per-match decision cap and optional rally diagnostic threshold
  before examining comparative quality. The numeric cap remains a protocol design
  choice; this audit does not select or apply one. At the cap, mark the episode
  truncated without awarding a win, changing scores, or forcing an outcome.
- Report completed episode fraction and decisions first. For the original
  complete-match point-fraction estimand, report identification bounds with each
  truncated episode's unknown final fraction in [0,1]. A mean of completed matches
  alone must be labeled completion-conditioned. Do not silently use current
  partial-match scores as final outcomes.
- Keep a separate process wall cap and distinguish infrastructure failure from
  deliberate episode truncation. A process that fails cannot supply missing
  episode outcomes. Archive every assigned episode and its status.
- Evaluate v1 and v2 on a preregistered saved-checkpoint panel as distinct
  estimators; never splice v2 results into the original v1 frontier.

Acceptance checks: episode ID/seed repeatability, exact allocation and no duplicate
IDs, terminal/recurrent reset parity, cap boundary and truncation labeling, score
parity with v1 on explicitly matched completed episode trajectories, and unchanged
Pong rules/observations/model math. Any future quality study must freeze its policy
and stopping rules before launch.
