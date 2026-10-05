# Draw-correction validation receipts

CPU environment/configuration checks only; no neural inference/training or GPU
execution. See [the correction report](../../../CONNECT4_DRAW_CORRECTION.md).

- `red-test.txt`: new full-board assertion fails on the old state environment.
- `environment-tests.txt`: corrected state/pixel fixtures, ten representations,
  mixed assignments, deterministic parity and ASan/UBSan checks.
- `artifact-tests.txt`: twelve artifact/configuration checks, including the new
  rules label, rejection of old execution manifests and early-draw rejection.
- `build.txt`: normal native float32 trainer build, not a runtime check.
- `legacy-draw-rows.txt`: all 23 zero-reward terminal rows from the pre-fix pixel
  trace. The state trace has the same rows. Columns are seed, transition, action,
  player bits, opponent bits, last opponent bit, RNG, tick, pending reset, reward,
  terminal, cumulative wins, score, return, episode lengths, episodes, invalids.
  Differences of cumulative episode lengths show every such draw lasted one
  decision. Full traces had 4,096 transitions and 596 terminals each.
- `legacy-traces.sha256`: original full-trace identities, preserved locally under
  `build/connect4cnn/draw-before-fix.P82rSG5z/` before tests overwrote normal output.
- `checked-sources.sha256`: source identities for the corrected checks/build.
- `binary.sha256`: compiled native float32 trainer identity; GPU runtime pending.
- `trace-summary.txt`: old/new random-action trace counts. The corrected trace
  has 578 terminal episodes and no draws in 4,096 transitions; the dedicated
  near-full fixture separately verifies genuine 21-decision draw termination.

Reproduce environment checks from the repository root using the existing shared
Raylib dependency (no new renderer installation):

```bash
bash ocean/connect4cnn/tests/run_all.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B ocean/connect4cnn/tests/test_claim_tools.py
node research/tests/test_claim_chart.cjs
```

Historical full-trace paths are G240-local, not inputs needed to validate the
corrected source. The original defect is independently reproducible from the
pre-fix commit `1f76a116` or the captured confirmation revision documented above.
