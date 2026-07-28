# Phase 2 Flight-Test Gate

Date: 2026-07-28

Flight behavior must be validated before interpreting reward, curriculum, or
training results. A model that scores well while flying incorrectly is not a
valid result.

## Authority boundary

- Physics oracle: pinned Dogfight3 source at
  `/home/claude/dogfight3`, commit
  `5a3effce470ce19fd6c4c6385a515853a4bfc0cb`.
- Test-design donor: `/home/claude/dogfight4/ocean/dogfight/tests`.
- Dogfight4 is not a physics oracle. Its harness organization and diagnostics
  are reused, but physics changes must be compared against pinned Dogfight3.

The following files were copied unchanged from the Dogfight4 test-design donor:

```text
109d3003fa0f28895469418deae9850ae6f0154f8e6ede8b2f86295271725f28  test_common.h
81bb7a058401adbb3381b4fadcf1b0d345bd3c08af6310749766ba630525d598  test_flight_autoace.c
89276fc20330cab9f86cd3225a5a11ca490f293091fa7ca9eeb6da0e5b8e978a  test_flight_energy.c
fe6c11021ad909d990a2323561033c693425e73f3a5967fd4767e287d2e72e49  test_flight_obs_dynamic.c
fc2121d4122140a614ecf05e2843d2a8ea8e84bd557b456ac2749d527869d63f  test_flight_obs_static.c
824b5b8171a45992708b91acba93fe01510e0e5573f0e3d80b3df8d25cdfbc77  test_flight_physics.c
2df8cd16574344e945693876b60dd541ecf1962e52f575697299a25e597f024e  test_flight_recovery.c
```

## Initial characterization

All six structured suites compiled against the Phase 1 5c headers and exited
with zero hard failures:

- AutoAce pursuit, defense, firing, and energy execution.
- Energy behavior under sideslip, climb, dive, turn, loop, split-S, zoom,
  throttle, and G loading.
- Static and dynamic observation bounds and continuity.
- Basic P-51-like speed, stall, climb, glide, turn, and control direction.
- Recovery from dives, rolls, inversion, speed, altitude, heading, and rate
  disturbances.

Important soft diagnostics remain visible:

- Rudder-only turn reports negligible heading change.
- High-speed aggressive pitch and roll tests report instability.
- Several recovery cases report oscillation even though they satisfy the hard
  recovery bounds.

These are inherited Dogfight3 behaviors, not 5c port regressions. The standalone
Dogfight4 oscillation harness produced exact output against current 5c and
pinned Dogfight3, including its nonzero exit status.

## Active command

The six suites are part of the normal C regression gate:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/claude/dogfight5/.venv/bin/python \
  -m pytest -p no:cacheprovider \
  ocean/dogfight/tests/test_fixture_provenance.py \
  ocean/dogfight/tests/test_c_regressions.py -q
```

This command was observed working on this checkout. Future agents must adapt
dependency paths if the local virtual environment changes.

## Required next gate

Before training longer:

1. Run this complete flight suite.
2. Keep pinned Dogfight3 parity for unintended changes.
3. Treat deliberate physics improvements as separate TDD changes with updated
   fixtures and a visible `DISPLAY=:0` evaluation.
4. Require stable visible flight and reliable curriculum progression through
   stages 0, 1, and 2 before expanding the curriculum.
