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

## Reference Differences Checked

- Curriculum stage enum and `STAGES` table match Dogfight3, including stage 9 bank `30`, stage 17 bank `60`, stage 18 crossing geometry, and stage 20 AutoAce.
- Dogfight5 had an active stage-9 bank ramp through config (`stage9_bank_deg=-1`, `stage9_bank_curriculum=1`). Dogfight3 did not train with that easier stage-9 ramp; stage 9 comes directly from `STAGES[9].bank == 30`.
- df36 W&B configs often show `obs_scheme=2`, but local Dogfight3 only accepts schemes `0` and `1`; invalid scheme `2` falls back to scheme `0`. That means the df36 stage-20 reference likely trained on effective `OBS_PILOT` scheme `0`.
- Dogfight5 config was using `obs_scheme=1`, which adds opponent up vector and opponent speed. That is a likely behavior mismatch, not a curriculum difficulty issue.
- Dogfight5 native binding currently advertises fixed observation width `26`. Scheme 0 writes `22` values, so the unused tail must be zeroed to avoid stale observations.

## Changes Made From This Investigation

- Set `config/dogfight.ini` default `obs_scheme = 0` to match the likely effective df36 observation path.
- Set `config/dogfight.ini` `stage9_bank_deg = 30.0` and `stage9_bank_curriculum = 0` so active training uses Dogfight3 stage-9 difficulty.
- Added zero-padding for fixed-width native observations when a smaller Dogfight observation scheme is active.
- Added `test_observation_padding.c` to prove scheme 0 does not leak stale values into the unused 26-wide native observation tail.

## Not Changed

- Did not change the `STAGES` curriculum table or make any curriculum step easier.
- Did not change Dogfight physics in `flightlib.h`.
- Did not change action scaling, continuous-action distribution code, or optimizer code.
- Did not change PufferLib core in this investigation.
- Did not enable self-play or anchor-rating optimization for df39 stage-climb sweeps.

## Current Culprit Ranking

1. Observation mismatch: df36 effective scheme 0 vs df39 scheme 1. This is a real behavior mismatch and should be tested first.
2. Stage-9 easing: df39 target `9.9` happened while stage 9 was easier than Dogfight3. Treat those results as not directly comparable to df36 until the fixed 30-degree stage-9 path is swept.
3. Native policy architecture: commit `1f1e7a04` restored Dogfight3-style `Linear + bias + GELU` encoder behavior and df39 improved after it. Keep this under scrutiny because it lives in `src/ocean.cu`, but it uses the existing custom encoder extension point.
4. Remaining likely areas: action scaling/logprob path, observation parity, step/reset semantics, reward terms, and physics parity. No curriculum step changes should be used as a fix.

## Next Tests

- Run a short df39 sweep after this config/obs fix and compare against the old top: target must clear 7-9 without the stage-9 ramp.
- Add parity tests against Dogfight3 for scheme 0 observations under scripted states.
- Add physics/step parity probes for neutral action traces and scripted action traces against Dogfight3.
- Audit `flightlib.h` differences before changing physics.

## Verification

- `git diff --check`: clean.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `python -m pytest ocean/dogfight/tests -q`: `59 passed, 1 skipped`.
