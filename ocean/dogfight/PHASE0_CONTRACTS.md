# Dogfight 5c Phase 0 Contracts

This file freezes the behavior to protect before implementation begins. Runtime
code is not imported in Phase 0.

## Source authority

| Role | Repository | Revision |
|---|---|---|
| Target and write location | `/home/claude/5c-research` | PufferLib 5c `ebf5ed03cc3524076b6c1a4033bd69cec0b3db22`, branch `dogfight5c` |
| Current behavior authority | `/home/claude/dogfight3` | `5a3effce470ce19fd6c4c6385a515853a4bfc0cb` |
| Older comparison oracle | `/home/claude/dogfight3` | `171482a9b9889bebaa427ab658c95c0fc391c031` |
| Golden-test donor | `/home/claude/dogfight5` | `10da7b5abac0f67f69f302c85511baf99a22446c` |
| Test-design donor only | `/home/claude/dogfight4` | `21281f623abeb44c525b070cdd20b17519470195` |

Dogfight 3 HEAD defines intended historical behavior. The older comparison
oracle is useful because its flight, spawn, reward, terminal, reset, and
curriculum formulas match HEAD, while its observation scheme numbers differ.
Dogfight 4 never supplies expected physics values.

## Observation contract

- Storage: 26 `float32` values for the initial native port.
- Default semantics: `opponent-aware-26`.
- Indices: body velocity `0..2`, angular rates `3..5`, angle of attack,
  altitude, G and own energy `6..9`, own up vector `10..12`, target azimuth,
  elevation, range and closure `13..16`, energy advantage, aspect and opponent
  pitch/roll rates `17..20`, opponent up vector `21..23`, opponent speed `24`,
  timer `25`.
- Normalizers: speed `250 m/s`, angular rate `3 rad/s`, angle of attack
  `0.5 rad`, altitude `5000 m`, and range `2000 m`.
- Timer: `tick / (max_steps + 1)`.
- Legacy `pilot-22` uses the same fields through index 20 and places its timer
  at index 21. When carried in a 26-wide buffer, indices `22..25` are zero.

At the older oracle, opponent-aware observations were scheme 2. At Dogfight 3
HEAD the identical implementation is scheme 1. Durable artifacts use semantic
names, not numeric scheme identifiers.

## Action and simulation contract

- Five continuous actions: throttle, elevator, aileron, rudder, trigger.
- Raw policy range is nominally `[-1, 1]`.
- Throttle maps `(raw + 1) / 2`.
- Positive elevator pushes the nose down.
- Positive aileron rolls right.
- Positive rudder yaws left.
- Trigger fires when greater than `0.5`.
- Gun cooldown is 10 ticks.
- Gun range is `1 < distance <= 500 m`.
- Gun cone uses strict `dot > cos(0.087)`, approximately five degrees.
- Simulation step is `0.02 s`, or 50 Hz.

The native environment will preserve the raw action for policy accounting and
derive one finite, clamped executed action used consistently by physics,
shaping, telemetry, and firing.

## Reward and outcome contract

The parity baseline preserves Dogfight 3 slot-0 dense reward with its checked-in
training configuration. Components include closing, aim, negative G, stall,
rudder, control-rate, low-altitude, time, own-energy, and energy-advantage
terms. The final published dense reward is clamped to `[-1, 1]`.

Terminal rewards are frozen separately:

| Outcome | Slot 0 | Slot 1 |
|---|---:|---:|
| Slot-0 gun kill | `+1` | `-1` |
| Slot-1 gun kill | `-1` | `+1` |
| Slot-1 ground/ceiling exit | `+0.25` | `-1` |
| Slot-0 ground/ceiling exit | `-1` | `+0.25` |
| Either aircraft exceeds `340 m/s` | `-1` | `-1` |
| Timeout | `-0.5` | `-0.5` |

Timeout is a terminal, not a truncation. Promotion and evaluation will use an
explicit outcome reason rather than inferring an outcome from reward sign.

## Reset and determinism contract

Historical Dogfight 3 automatically resets inside a terminal step: reward and
terminal describe the completed episode while the observation contains the
next episode's initial state. Phase 1 must preserve that transition alignment
through the native ABI and publish a fresh finite observation.

Dogfight 3 uses process-global `rand`/`srand`; that implementation is neither
thread-safe nor vector-layout deterministic. The 5c port intentionally replaces
it with environment-owned RNG seeded from the run seed and global environment
identity. Fixed-state physics must retain golden parity. Seeded reset sequences
will establish a new, explicitly versioned deterministic contract rather than
pretending to reproduce the old data race.

## Historical hazards that are not target behavior

- Python's initial curriculum target `0.9` is not synchronized to C's `0.0`.
- Episode-specific `max_steps` can leak into later resets.
- Most terminal rewards are omitted from the historical logged return.
- Opponent reset observations can be stale.
- Simultaneous fire and ground events are resolved asymmetrically.
- Invalid observation schemes disagree on Python and C buffer width.
- Frozen opponent trajectories can enter the old copied PPO update.

Parity tests preserve these facts where needed to understand the oracle. Any
fix is named, behavior-versioned, and tested; fixtures are never silently
rewritten.

## Phase 0 fixture policy

The imported C fixtures are exact copies from the pinned Dogfight 5 donor.
Several embed numerical values labeled only `Dogfight3` or `df36` and do not
state the generating Dogfight 3 commit. Those fixtures are useful but remain
quarantined in `FIXTURE_PROVENANCE.json` until their generator or oracle
revision is proven.

Every golden update requires:

1. an explanation of the behavior change;
2. exact source and generator revisions;
3. the regeneration command and numeric tolerance;
4. repeated same-seed agreement where determinism applies;
5. explicit review and approval.

