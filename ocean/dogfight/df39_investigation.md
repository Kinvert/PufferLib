# df39 Training Investigation

Date: 2026-06-04

## Constraints

- Do not make the Dogfight curriculum easier than Dogfight3.
- Treat Dogfight3 as the behavioral reference for curriculum stages, observations, physics, rewards, and self-play behavior.
- Prefer Dogfight-local fixes. Do not change PufferLib core unless Dogfight-local options have been checked and ruled out.

## W&B Findings

df39 local artifacts and W&B agree that the best runs are now on commit `1f1e7a04`:

- `qbktakds`: target `9.9`, mastery stage `10`, soft quality `0.3678`, kill rate `0.0417`, action saturation `0.7194`.
- `dp2u2tx8`: target `9.9`, mastery stage `10`, soft quality `0.4105`, kill rate `0.5667`, action saturation `0.4432`.
- Earlier df39 best before the GELU encoder was `bxmro3re` on `0825ada5`, target `4.9`.

df38 reached target `8.9` on commit `ae750d44` in multiple runs. Representative top runs include `acj17oyc`, `k9wof24s`, `raewv8jo`, `y5aa27z9`, all at target `8.9`.

df36 used different metric names. It logs `environment/stage` and `environment/avg_stage` instead of `env/curriculum_target`. Top df36 runs reached stage `20` on commit `171482a9`; the best score query found `d8jn9wiv` at stage `20`, score `0.9139`, base-stage kills `0.9569`. High-ELO df36 references also include commit `51514907`.

## Open Metric Anomaly

The df39 runs `dp2u2tx8` and `qbktakds` advanced to target `9.9`/mastery stage `10` despite apparently large signed control-bias metrics such as `base_stage_signed_bias_rudder`. This needs a deeper audit before treating those runs as clean evidence of learning quality.

Questions to answer:

- Are the signed-bias metrics computed on the same episode/time window as curriculum advancement, or are they an aggregate over a later/current base-stage evaluation window?
- Are the metric names in W&B unambiguous, or is TUI truncation hiding which control axis/window is being inspected?
- Can high signed rudder/aileron bias still coexist with enough stage-specific kills because the stage geometry/reward allows a biased tactic, especially before self-play/anchor eval exists?
- Did the old stage-9 ramp let a biased policy advance in df39 in a way that the Dogfight3 fixed curriculum would not?
- Does `curriculum_soft_quality` over-credit stage advancement relative to true kill/win behavior when action bias is extreme?

Next audit step: pull full W&B history for `dp2u2tx8` and `qbktakds`, inspect the exact steps where curriculum target advanced, and compare `curriculum_target`, `base_stage_kills`, `base_stage_ground`, `base_stage_timeouts`, action saturation, and signed-bias metrics at those same steps. Then trace the local metric code that emits each `base_stage_signed_bias_*` value so the interpretation is grounded in the actual denominator/window.

## Reference Differences Checked

- Curriculum stage enum and `STAGES` table match Dogfight3, including stage 9 bank `30`, stage 17 bank `60`, stage 18 crossing geometry, and stage 20 AutoAce.
- Dogfight5 had an active stage-9 bank ramp through config (`stage9_bank_deg=-1`, `stage9_bank_curriculum=1`). Dogfight3 did not train with that easier stage-9 ramp; stage 9 comes directly from `STAGES[9].bank == 30`.
- df36 W&B configs often show `obs_scheme=2`, but local Dogfight3 only accepts schemes `0` and `1`; invalid scheme `2` falls back to scheme `0`. That means the df36 stage-20 reference likely trained on effective `OBS_PILOT` scheme `0`.
- Dogfight5 config was using `obs_scheme=1`, which adds opponent up vector and opponent speed. That is a likely behavior mismatch, not a curriculum difficulty issue.
- Dogfight5 native binding previously advertised fixed observation width `26`. Scheme 0 writes `22` values, so that trained a 26-input native encoder on four structural zeros. The binding now exposes the df36-effective scheme-0 width directly (`22`) and clamps native Dogfight training to `OBS_PILOT`.

## Changes Made From This Investigation

- Set `config/dogfight.ini` default `obs_scheme = 0` to match the likely effective df36 observation path.
- Set `config/dogfight.ini` `stage9_bank_deg = 30.0` and `stage9_bank_curriculum = 0` so active training uses Dogfight3 stage-9 difficulty.
- Replaced the earlier fixed-26 padding mitigation with a native `OBS_SIZE 22` binding for the active df36-effective scheme-0 path.
- Updated `test_observation_padding.c` to prove scheme 0 writes only its declared 22 values instead of clearing/writing a 26-wide tail.
- Set Dogfight's omitted-key stage-9 defaults to Dogfight3 behavior (`30.0` bank, no substep ramp). Explicit `stage9_bank_deg = -1` still keeps the diagnostic ramp reachable, but it is no longer the default in C init, native binding fallback, or Dogfight-specific pufferl curriculum setup.
- Changed the native Dogfight binding to advertise `OBS_SIZE 22`, matching the df36-effective scheme-0 observation count. The direct C observation regression now proves scheme 0 writes only its declared 22 values instead of clearing/writing a 26-wide tail.
- Set Dogfight's direct native `policy.action_init_scale` default to `1.0`. Dogfight3's policy constructor used `0.01`, but `DogfightRecurrent` immediately reinitialized wrapped policy weights with orthogonal gain `1.0`, making `1.0` the effective df36 action-head scale.
- Moved Dogfight reset/spawn/domain-randomization draws from process-global `rand()`/`rndf()` to `env->rng`, matching the PufferLib 5 state-memory pattern used by Boxoban/G2048. The added terminal roundtrip test first failed because restored pre-kill states reset into different future geometry, then passed after the local RNG fix.

## Not Changed

- Did not change the `STAGES` curriculum table or make any curriculum step easier.
- Did not change Dogfight physics constants or integration equations in `flightlib.h`; only the reset-time random source was made env-local.
- Did not change action scaling, continuous-action distribution code, or optimizer code.
- Did not change general PufferLib core behavior. The only non-`ocean/dogfight` edit was Dogfight-specific fallback handling in `pufferlib/pufferl.py`.
- Did not enable self-play or anchor-rating optimization for df39 stage-climb sweeps.

## Current Culprit Ranking

1. Observation mismatch: df36 effective scheme 0 vs df39 scheme 1. This is a real behavior mismatch and should be tested first.
2. Native input width mismatch: df36-effective scheme 0 is 22 observations; the port previously gave the native policy 26 inputs. This is now fixed in the Dogfight binding.
3. Stage-9 easing: df39 target `9.9` happened while stage 9 was easier than Dogfight3. Treat those results as not directly comparable to df36 until the fixed 30-degree stage-9 path is swept. The active config and omitted-key defaults now both use 30 degrees.
4. Native policy architecture: commit `1f1e7a04` restored Dogfight3-style `Linear + bias + GELU` encoder behavior and df39 improved after it. Keep this under scrutiny because it lives in `src/ocean.cu`, but it uses the existing custom encoder extension point.
5. State-memory reset RNG: before the env-local RNG fix, a restored preterminal state could produce a different post-terminal reset because reset/spawn used process-global `rand()`. This is now fixed for the stage-climb reset/spawn path and covered by `test_state_roundtrip.c`.
6. Remaining likely areas: action scaling/logprob path, exact native encoder/recurrent weight initialization vs Dogfight3 recurrent wrapper, observation parity, step/reset semantics, reward terms, and physics parity. AutoAce stage-20 tactical randomness still needs a separate state-memory audit if sweeps target stage 20. No curriculum step changes should be used as a fix.

## Next Tests

- Run a short df39 sweep after this config/obs fix and compare against the old top: target must clear 7-9 without the stage-9 ramp.
- Run a short `max-runs 2` state-buffer sweep smoke before trusting larger state-memory sweeps.
- Add parity tests against Dogfight3 for scheme 0 observations under scripted states.
- Add physics/step parity probes for neutral action traces and scripted action traces against Dogfight3.
- Audit `flightlib.h` differences before changing physics.

## Verification

- `git diff --check`: clean.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py ocean/dogfight/tests/test_curriculum_progress.py -q`: `12 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `60 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_port_build.py::test_dogfight_gpu_vec_creates_and_resets -q`: skipped in this environment.
- `python -m pytest ocean/dogfight/tests/test_native_policy_architecture.py::test_dogfight_native_binding_uses_df36_effective_scheme0_width ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k 'native_binding or observation_padding'`: `2 passed`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py ocean/dogfight/tests/test_native_policy_architecture.py ocean/dogfight/tests/test_port_build.py::test_dogfight_training_backend_builds -q`: `9 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `61 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_policy_init.py -q`: `7 passed`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k state_roundtrip`: failed before the env-local RNG fix because restored terminal futures diverged in observations/plane state, then passed after the fix.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `6 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `61 passed, 1 skipped`.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `python -m pufferlib.pufferl train dogfight --train.gpus 1 --train.total-timesteps 5000000`: native trainer completed, TUI showed GPU usage and normal Dogfight curriculum metrics.
