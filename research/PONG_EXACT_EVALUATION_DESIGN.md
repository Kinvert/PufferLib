# Pong exact-match evaluator: inspected contract

October 5, 2026. Design below is now implemented in the
[native adapter and suite/audit tool](../ocean/pongcnn/DETERMINISTIC_EVAL.md).
Four targets compile; host simulation/manifest/audit checks pass. GPU acceptance
remains pending; the supervised launcher now passes preparation/audit checks.
No new policy result. Preserve both
GPU holds, original game/training code and historical pooled-v1 results.
Existing native Pong isn't ALE Atari Pong.

## What the current code actually records

Inspected `ocean/pongcnn/pongcnn.h` and its original state reference:

- `puf_step` increments `tick` once per agent decision, before the physics
  frame-skip loop. A point can return early from that loop.
- `reset_round` resets `tick` and `n_bounces` after **every point**, not only
  after a match. Thus `add_log`'s `episode_length` is the last rally's decision
  count; it is not whole-match decisions or total physics frames.
- Each point produces exactly +1 (agent/right) or -1 (opponent/left); the
  match ends at `max_score` points on either side. Stock threshold is 21.
- The terminal path logs score difference and right/(left+right), then calls
  `puf_reset`. Reading live point counters afterward gives the next match's
  zeros. Point-reset paths return immediately, so this does **not** have
  Breakout's frame-skip spillover into a replacement game.
- Each `reset_round` consumes one `rand_r` draw for initial vertical direction.
  The terminal auto-reset also consumes a draw for the replacement world;
  reseeding each assigned match prevents that from setting its next identity.
- `puf_reset` doesn't explicitly clear every control/helper field; the exact
  start must clear `paddle_dir`/`win` and per-episode log/action/reward/terminal
  receipts in addition to seeding and resetting the world.

Pooled-v1 point-fraction scores aren't invalidated merely by the length issue,
but their duration field cannot stand in for exact whole-match duration. The
earlier timeouts/selection limitations remain; don't relabel old data as exact.

## Proposed evaluation-only implementation

Keep `puf_step` physics/opponent/rewards unchanged. One assigned match per
active slot, never replacement matches, with explicit identity-derived game
and Philox action seeds. Fixed drawing per suite; all three discrete actions,
any matching native encoder/core shape, separate checkpoint/build receipts.

Track whole-match agent decisions externally, never infer them from live
`tick`. Before stepping, retain both point totals. After stepping, use the
point reward to derive ending totals and cross-check live counters on a
nonterminal transition. On a terminal transition, cross-check derived totals
against the native ending log's difference/fraction and threshold. Sum point
rewards independently; it must equal final right-minus-left. Capture the log
before reusing/reseeding the slot. Preserve the initial image/world/RNG and
action digest, exactly N IDs and the independent last-wave replay contract.

Start with a common **agent-decision cap**, not a purported exact physics-frame
cap: early point returns mean `decisions * frameskip` overcounts actual frames.
Exact physics-frame measurement would require separately instrumenting the
native loop. Do not change game rules or set fake terminal/loss rewards when
the evaluation cap is reached. Natural match completion on the cap decision
takes precedence. Record completed/capped separately and retain every ID.

## Capped matches and the estimand

Native `perf` is a completed match's fraction of points won, not match win rate.
Preserve that distinction. At a cap, the observed partial point fraction is
not the unknown final match fraction; zero points makes it undefined. Never
drop that match, assign it a convenient win/loss, or average only completions.

For a first-to-M match capped at right=r, left=l (both below M), the possible
final point fraction has exact deterministic bounds:

```
lower = r / (M + r)       # opponent reaches M without another agent point
upper = M / (M + l)       # agent reaches M without another opponent point
```

For completed matches both bounds equal right/(right+left). Average the lower
and upper values over **all assigned matches** to bound the suite mean. These
are censoring bounds, **not confidence intervals**. A zero-point capped match
contributes [0,1]. Additional seed/frontier uncertainty needs its own calibrated
analysis; bounds alone don't establish dominance.

Also report accumulated net points through the common decision cap, completed
match outcomes when known, point totals and decision lengths. That bounded
return is defined even for zero-point caps and includes every assigned match,
but is a different estimand from final point fraction. Freeze the primary
estimand, cap/selection rules and analysis before collecting comparative data.
Do not select the friendliest one after seeing model outcomes. A calibrated
long enough suite with few/no caps is preferable for the final-fraction claim.

## Acceptance before comparison

1. CPU environment/raster fixtures against original state: independent
   initial RNG, dirty prior state, scoring on either side, nonterminal point
   resets, terminal scores on both sides, whole-match counter beyond the last
   rally, cap with/without points and terminal-at-cap. No CPU model execution.
2. Exhaustive finite-score fixtures for the censoring-bound arithmetic and
   impossible/tampered scores; source-pinned host manifests across fixed
   drawings/binary families and final-ID independent starts.
3. Native adapter builds and external immutable-suite/audit/launcher tests;
   supervised failure/timeout paths retain all partial data and never produce
   a successful partial aggregate.
4. Separately scheduled GPU checkpoint/recurrent-reset/repeat/eager/graph,
   full/final-wave quota and multiple-core acceptance. No new GPU execution
   is authorized by this design. Keep encoder-5 qualification separate.
5. Cap/common-recipe calibration, frozen paired development seeds and full
   checkpoint frontiers. Preserve failures, losing tasks and contrary evidence.

[Multi-task goal and fixed graph](MULTI_ENV_ROBUSTNESS.md).
[Historical Pong limitations](PONG_5090_RESULTS.md).
