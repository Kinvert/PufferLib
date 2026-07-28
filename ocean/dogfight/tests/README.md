# Dogfight Test System

This directory starts red by design. The imported C regression fixtures define
the contracts that the Phase 1 native scaffold must satisfy; they require
`ocean/dogfight/dogfight.h`, which has not been ported yet.

## Provenance

The C tests are copied unchanged from:

```text
/home/claude/dogfight5
10da7b5abac0f67f69f302c85511baf99a22446c
```

Their behavior authority is Dogfight 3 at
`5a3effce470ce19fd6c4c6385a515853a4bfc0cb`. Three donor
source-compatibility tests explicitly compare against
`171482a9b9889bebaa427ab658c95c0fc391c031`, but the embedded numerical C
fixtures do not record their generating oracle SHA. See
`FIXTURE_PROVENANCE.json`; unpinned numerical fixtures are quarantined rather
than silently attributed to that older revision.

Dogfight 4 revision `21281f623abeb44c525b070cdd20b17519470195`
supplies test scenarios only. Its thresholds and physics values are not
oracles.

## Test tiers

### Every edit

- fixed-state opponent-aware observations;
- one-step and 12-step physics/reward traces;
- action bounds and control polarity;
- terminal rewards and immediate-reset alignment;
- repeated state/trajectory determinism;
- finite observations and aircraft state.

### Before training

- all every-edit tests;
- curriculum spawn geometry and episode caps;
- long integration, oscillation, smoothness, speed, climb, glide and turn
  envelopes;
- energy ordering and G limits;
- autopilot recovery from bank, pitch, inverted, dive, speed and altitude
  disturbances.

### Major physics or controller changes

- combat recovery matrix;
- full observation edge/crossover suite;
- AutoAce tactical behavior and firing discipline;
- visible sustained-turn and recovery evaluation through `DISPLAY=:0`.

The long-horizon flight scenarios will be adapted from Dogfight 4 only after
Dogfight 3 supplies approved expectations. Tests that merely print `FAIL` must
be converted to nonzero hard failures before becoming gates.

## Candidate commands

These commands came from the older Dogfight repositories. Future agents must
try them against the current branch, bound expensive attempts, inspect current
5c CLI/config behavior, and record the command that actually works. They are
starting points, not gospel.

Dogfight 5 test command:

```bash
.venv/bin/python -m pytest ocean/dogfight/tests -q
```

Dogfight 5 bounded training canary:

```bash
python -m pufferlib.pufferl train dogfight --train.gpus 1 --train.total-timesteps 5000000
```

Dogfight 5 bounded sweep:

```bash
timeout 300s bash -lc 'source .venv/bin/activate && python -m pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1 --sweep.max-runs 2'
```

Current 5c checkpoint-loaded display candidate, synthesized from the current
Nethack eval form and Dogfight 4 display setup:

```bash
DISPLAY=:0 timeout 20s ./puffer eval dogfight --load-model-path=latest
```

An explicit checkpoint path is preferred over `latest` for recorded gates.

