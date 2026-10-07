# Architecture discovery on deterministic drawing mixtures

October 6, 2026. Preparation and scalar configuration checks only; neither GPU
is scheduled by this change. Six existing games are enough for this phase.

Discovery can now explicitly use the full drawing catalogs used by the
multi-game candidate panel. The helper previously rejected Flappy catalog 1,
despite accepting its seven fixed drawings. Flappy catalog 0 still selects
the historical four-ID mixture; Pong catalog 0 still selects five. Fixed mode
accepts all current IDs. Native game, RNG, encoder and learner code is unchanged.

```bash
# GPU-free: resolve config, freeze recipe and receipts; no build/query.
.venv/bin/python ocean/connect4cnn/sweep.py \
  --environment connect4cnn \
  --recipe research/recipes/general_cnn_discovery.ini \
  --appearance-seed 35173 --max-runs 128 \
  --prepare-only --wandb disabled
```

The same command accepts `pongcnn`, `flappycnn`, `breakoutcnn`, `snakecnn` and
`mazecnn`. `--appearance-seed` sets mode 1, fallback ID 0 and the explicit full
catalog selector for Pong/Flappy. It is a fixed experiment setting, never a
PROTEIN coordinate. Without this option, the recipe's existing fixed or legacy
mixture settings are preserved. A simultaneous `sweep.env.representation` is
rejected because that field is inactive in mixed mode; no silent removal.

Assignment stays native: hash/rejection arithmetic on appearance seed and global
slot, once during initialization, consuming no world RNG. It persists through
resets rather than changing every episode/frame. Reload must preserve slot
count/order. Architecture proposals still come from native **single-game**
PROTEIN; an adaptive cross-game objective is not implemented by this option.

The discovery recipe retains 18 numeric CNN dimensions, encoder 4, H128/L1
and 13,312,000 decisions. LR/clip/etc. come from the game's `compare.ini`, fixed
across its proposals. Adequate-learning/memory/runtime calibration is pending.
The original recipe is now copied before resolution and hashed separately from
the full resolved INI. Protocol appearance metadata records mode/seed/catalog/
fallback; vector-runtime acceptance remains false. Source receipts are not
complete toolchain certification.

## Actual preparation evidence

[Retained inputs, native CSVs and verification](results/discovery-mixtures-20261006/README.md)
cover six fresh CLI preparations at two proposed runs each, **zero executed**.
The existing scalar C helper parses each captured full INI and exports 64 slot
assignments without a game/image/weights/model/CUDA. Independent unsigned
arithmetic matches every row. Counts at predeclared seed 35173 are:

| Game | Drawing IDs | Slots per ID, in ID order |
|---|---|---|
| Connect4CNN | 0–9 | 2, 10, 8, 4, 8, 4, 9, 7, 6, 6 |
| PongCNN | 0–6 | 10, 6, 8, 9, 8, 9, 14 |
| FlappyCNN | 0–6 | 10, 6, 8, 9, 8, 9, 14 |
| BreakoutCNN | 0–4 | 6, 19, 15, 10, 14 |
| SnakeCNN | 0–5 | 7, 9, 11, 10, 15, 12 |
| MazeCNN | 0–5 | 7, 9, 11, 10, 15, 12 |

All 41 drawings occur; exposure is unequal. Other seeds/slot counts may omit
drawings. Do not tune appearance seeds based on policy scores. Twenty-six
configuration/sidecar tests pass, including unchanged per-game learners,
repeated preparation, uint32 limits, legacy/full Flappy guards and inactive
dimension rejection. Offline verification reconstructs complete resolved INIs
from captured inputs and independently recomputes CSVs. These aren't CPU neural
tests or live vector/learning acceptance.

## Following discovery

After separately scheduled qualified native discovery, import **every completed
proposal**, including dominated/duplicate candidates, through
[CROSS_GAME_CNN_SWEEP.md](CROSS_GAME_CNN_SWEEP.md). Declare the same
`candidate_panel.py prepare --appearance-seed 35173` setting, paired learner
seeds and calibrated per-game recipes/budgets. Include Nature/IMPALA/Impoola
through [the baseline panel](CANDIDATE_BASELINE_COMPARISON.md), then evaluate
all fixed drawings at every allocated checkpoint on exact episode identities.
Different slot geometry changes exposure even at the same seed: retain counts.

Keep discovery training scores/costs separate from exact cross-game validation.
All current drawings are exposed: trained-catalog robustness, not withheld
appearance generalization. Preserve failures/full curves; don't pool unrelated
game metrics or drawings as independent seeds. Architecture superiority, GPU
math, memory/reload/evaluator acceptance, learning/cap calibration and selection-
safe frontier inference remain unproved. Both GPU holds remain; remote agents
need committed Kinvert source and fresh local preparations. No push occurred.
