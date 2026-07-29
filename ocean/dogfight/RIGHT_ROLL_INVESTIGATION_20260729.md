# Persistent Right-Roll Investigation

Date: 2026-07-29

Branch: `dogfight5c-vanilla-selfplay`

Base: `fe1c1332 dogfight: preserve clean-core curriculum baseline`

## Scope

This investigation used only Dogfight-local code and configuration. No
PufferLib core source files were changed.

The failure was a learned policy that commanded positive aileron for targets
on both sides. In visible evaluation this appeared as continuous right roll.
This was not accepted as merely poor combat performance because it obscured
whether physics, observations, action routing, or learning had regressed.

## Causes Eliminated

The following were checked before changing the learning objective:

- Stage-0 resets were balanced between left and right target offsets.
- Mirrored states produced mirrored opponent-aware observations.
- Mirrored aileron and rudder actions produced mirrored flight dynamics.
- Player and opponent actions, rewards, and policy rows were routed correctly.
- Dogfight uses five continuous Gaussian action heads. The behavior was not a
  missing neutral bin in a categorical action space.
- Seed sweeps could reverse the learned direction, ruling out a hard-coded
  rightward physics force or fixed action sign.
- The pure-pursuit teacher produced mirror-symmetric controls.

One real but separate reset bug was fixed during the investigation:
`c_reset()` now selects the curriculum stage through
`get_curriculum_stage(env)`. Before that fix, native runs intended for other
stages reset into stage 0.

## Measured Teacher Signal

A 10,000-reset stage-0 diagnostic showed balanced, unambiguous teacher labels:

| Policy row | Target azimuth | Samples | Mean azimuth | Mean teacher aileron |
| --- | --- | ---: | ---: | ---: |
| Front pilot | Negative | 5,027 | -0.041571 | +0.972945 |
| Front pilot | Positive | 4,973 | +0.041764 | -0.973008 |
| Rear pilot | Negative | 4,973 | -0.958236 | +1.000000 |
| Rear pilot | Positive | 5,027 | +0.958429 | -1.000000 |

The labels were not biased. The front pilot's normalized azimuth signal was
much smaller than the rear pilot's, making contextual learning harder but not
ambiguous.

## Failed Learning Profiles

The original one-step flight-school reward averaged squared error across
throttle, elevator, aileron, and rudder. It improved aggregate reward while the
policy continued to ignore target-conditioned aileron:

```text
64M full-control flight school
azimuth < 0 mean aileron: +0.162209
azimuth > 0 mean aileron: +0.117514
```

Increasing imitation scale did not fix the direction. A 16M-step
aileron-isolated run also remained undertrained:

```text
16M aileron-only flight school
azimuth < 0 mean aileron: +0.149514
azimuth > 0 mean aileron: +0.120685
```

This ruled out reward scale and easy-control masking as complete explanations.
The isolated task also needed substantially more optimizer updates.

## Successful Diagnostic

The flight-school score was changed to isolate aileron error:

```text
reward = scale * (1 - 0.5 * (policy_aileron - teacher_aileron)^2)
```

Each lesson remains a one-step terminal contextual-control episode. Training
that objective for 64M environment steps provided 256 optimizer updates:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
DOGFIGHT_BOOTSTRAP_IMITATION_SCALE=1 \
  bash ocean/dogfight/train_vanilla_selfplay.sh \
  df5c-aileron-school1-stage0-64m-s42 \
  67108864 4096 42 67108864
```

Checkpoint:

```text
checkpoints/dogfight/df5c-aileron-school1-stage0-64m-s42/0000000067108864.bin
```

The 32-episode headless gate then produced target-conditioned signs:

```text
azimuth < 0 mean aileron: +0.045997
azimuth > 0 mean aileron: -0.033200
average signed bias:       +2.055742
```

A subsequent normal-speed visible gate showed reasonably even control with a
small remaining right bias rather than the previous continuous right roll.

## Interpretation and Limitations

The persistent right roll was a policy-collapse and optimization problem, not
a flight-physics or action-sign defect. The successful checkpoint combined:

- A balanced pure-pursuit teacher.
- An aileron-only contextual objective that could not be masked by easier
  controls.
- Enough optimizer updates for the network to use the signed target feature.

This checkpoint is not a trained dogfighter. It intentionally teaches only
target-conditioned aileron, so mediocre throttle, elevator, rudder, energy
management, and combat performance are expected. Do not cite its win rate as a
self-play result.

The next training design must preserve this left/right skill while adding the
remaining flight controls and only then transitioning into native self-play.
Every transition checkpoint must repeat the headless sign gate and a visible
flight gate so later training cannot silently reintroduce the collapse.

## Validation and Human Gate

The source change passed:

```text
63 passed, 3 skipped
native Dogfight trainer built
Dogfight-local eval wrapper built
```

Visible evaluations should run until the human presses Escape. The large game
count intentionally prevents the evaluator from closing first:

```bash
DISPLAY=:0 DOGFIGHT_RENDER_DELAY_SECONDS=0.15 \
  bash ocean/dogfight/eval_checkpoint.sh visible \
  checkpoints/dogfight/df5c-aileron-school1-stage0-64m-s42/0000000067108864.bin \
  0 1000000 42
```

`DOGFIGHT_RENDER_DELAY_SECONDS=0.15` is approximately normal playback speed for
this evaluator. It changes render pacing only and does not change simulation
time steps or policy behavior.
