# Learned Physics Surrogate Research Idea

## Status

This is a preserved research idea, not part of the current merge-ready vanilla
self-play implementation. RK4 remains the authoritative Dogfight physics model.

## Core Idea

Use the existing RK4 simulator as a teacher to generate supervised transition
data. Train a small neural network to approximate:

```text
(aircraft state, controls, timestep, environment parameters)
    -> next-state delta
```

The inputs would include the complete state required to make the transition
Markovian, not only convenient measurements such as angle of attack, airspeed,
and attitude. Likely inputs include position, orientation, linear and angular
velocity, control commands, throttle or engine state, timestep, and relevant air
or aircraft parameters.

The output should probably be a state delta or derivatives rather than an
unconstrained absolute state. During RL rollout, many aircraft transitions could
then be evaluated as batched matrix operations, potentially on the same GPU as
the policy, instead of executing four full force evaluations for every RK4 step.

## Possible Development Path

1. Generate a broad dataset from RK4 trajectories, including normal flight,
   stalls, spins, high angle of attack, recovery, control transients, and combat
   maneuvers.
2. Train a small feed-forward or residual dynamics model on one-step state
   transitions.
3. Validate multi-step rollouts, because low one-step error can still accumulate
   into unstable or implausible flight.
4. Benchmark true end-to-end training SPS, including data movement and inference
   cost.
5. Train policies with a mixture of surrogate and RK4 transitions if pure
   surrogate rollouts drift or become exploitable.
6. Evaluate every resulting policy in authoritative RK4 physics and visible
   flight scenarios.

## Main Risks

- Small prediction errors can compound over long trajectories.
- The surrogate may fail on stalls, spins, or other rare nonlinear regimes.
- An RL policy may discover and exploit surrogate errors that do not exist in
  RK4.
- CPU-to-GPU transfer or model inference could erase the expected speedup.
- A policy trained only on approximate dynamics may transfer poorly to RK4.
- Training data must cover states produced by developing policies, not only
  states from competent or scripted flight.

## Safer Variants

- Learn RK4 state deltas while retaining explicit normalization and quaternion
  constraints.
- Learn only the expensive aerodynamic force and moment calculation, then retain
  the conventional integrator.
- Learn a residual correction around a cheaper analytic integrator.
- Periodically execute RK4 steps to re-anchor surrogate trajectories.
- Use uncertainty or out-of-distribution detection to fall back to RK4.
- Iteratively add RK4 trajectories from states where the learned policy drives
  the surrogate.

## Success Criteria

- Meaningful end-to-end SPS improvement, not merely a fast isolated forward pass.
- Stable long-horizon trajectories over the full flight envelope.
- No material loss on fixed RK4 flight tests or combat evaluations.
- Policies trained with the surrogate continue to fly credibly when evaluated
  entirely with RK4.
- The surrogate remains an optional Dogfight-local research path and does not
  weaken the authoritative simulator.
