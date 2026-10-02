# Exact-episode evaluator readiness — 5090 assessment

Reviewed revision: `c49ad33e9ab899b8f08198d4a154f532cfebe1d2`.
This is a source/readiness assessment, not an implementation or a launch approval.
No shared core paths were edited for this appearance validation. The original
5090 checkout, local commit and uncommitted audit changes remain separate.

## Current code and missing guarantees

| Requirement | Current behavior | Required v2 change/test |
|---|---|---|
| Exact episode allocation | `eval_loop` checks pooled `env/n` only after a full rollout. Environments automatically restart after completion. | Enforce each slot's quota at every environment decision inside the worker loop, not only at rollout boundaries. Record every assigned episode ID exactly once. |
| Independent game seed | CPU construction sets `envs[num_envs].rng = num_envs` before `puf_init`. `base.seed` does not create independent game RNG streams. | Define and version an environment-seed mapping from evaluation seed and episode ID. Apply it at the correct episode reset boundary. Record actual mapped seeds. |
| Policy/action seed | `rng_init` initializes persistent Philox states from process seed and row index. Sampling updates those states continuously. | Explicitly reset each episode's policy RNG stream. Test reproducibility under worker scheduling, uneven match durations and inactive slots. Do not claim that changing `base.seed` already implements per-episode streams. |
| Recurrent state | `zero_term_state` clears recurrent rows from uploaded terminal flags before inference. | Test natural terminals, truncations and first assigned episode separately; avoid skipping/double-applying reset or carrying state into the next episode. |
| Inactive slots | The CPU worker calls `puf_step` for every environment at every horizon step. Inference/action sampling also runs for every row. | Freeze completed-quota slots and exclude their records; preserve active-slot observations/actions/RNG trajectories. CUDA graph compatibility must be tested. |
| Terminal outcome capture | Connect4 `finish_game` and Pong terminal branches log and reset during the step. Pong reset clears scores. | Record the episode outcome and complete duration before auto-reset destroys terminal state. Preserve v1/training behavior when v2 is disabled. |
| Duration bounds | Connect4 tests enforce the finite-game bound; Pong `tick` resets every point. | Connect4 >21 decisions is an error, not a draw. Add a separate Pong whole-match decision counter. Calibrate a Pong cap using development duration data under a recorded ceiling, then freeze it. |
| Appearance identity | Appearance hashes an independent seed and initialization slot; it persists across resets. | Preserve and record slot/representation identity with each assigned episode. Reseeding game RNG must not silently reassign appearance. |

The 32-step rollout matters: a slot can complete more than one short Connect4
match before the current outer evaluator sees its aggregate count. Merely changing
the stop condition from `>=` to `==`, or discarding extra completed records after
the rollout, does not implement exact allocation or inactive-slot isolation.

## Proposed implementation sequence to coordinate with G240

1. Agree on one owner of shared `src/pufferl.cu` changes and the versioned record
   schema before both machines edit core. Keep the mode opt-in; v1 records remain
   historical and immutable. A single-GPU CPU-environment implementation is the
   first bounded target; broader environment/backend support is not assumed.
2. Define episode IDs, exact slot quotas, game/policy seed derivation, appearance
   preservation and terminal-versus-cap precedence. A natural terminal on the
   cap decision should have an explicitly tested treatment. Capture outcomes
   before auto-reset; never fabricate a partial Pong point fraction as final.
3. Add CPU fixtures for zero/uneven quotas, duplicate/missing IDs, all-complete,
   all-truncated and mixed statuses, exact cap boundaries, whole-match counters,
   and Connect4's 21-decision bound. Missing records are errors.
4. Add small saved-checkpoint GPU tests for recurrent resets, explicit action RNG
   streams, slot masking and graph/eager behavior. Vary one/two CPU workers and
   completion patterns. Verify checkpoint bytes remain unchanged. No retraining
   is needed to validate evaluator semantics.
5. Validate on a frozen successful/timed-out checkpoint panel under predeclared
   process limits. Keep process timeout distinct from episode truncation, retain
   every assigned episode/status, and report completion fractions and [0,1]
   identification bounds for unknown final point fractions. Do not replace
   original 300-second outcomes with v2 results.

## Gates that remain after the appearance canary

The canary can establish deterministic appearance assignment, finite/reloadable
weights and one/two-worker repeatability on its small tested panel. It cannot
establish exact episode quotas, independent per-episode RNG, per-appearance
robustness or learning quality. The proposed mixed-training robustness panel must
evaluate every fixed appearance separately; it does not replace the primary
fixed-representation-0 Connect4 comparison.

Before a full-frontier confirmation, still validate checkpoint *completion*
timestamps and the proposed common 51-checkpoint cadence, finalize exact held-out
allocation, calibrate simultaneous full-frontier inference and justify/freeze the
paired training-seed count. Forty seeds remains a resource estimate. The original
Pong long-match/truncation issue remains unresolved, regardless of this canary's
result. No long experiment follows automatically from these checks.
