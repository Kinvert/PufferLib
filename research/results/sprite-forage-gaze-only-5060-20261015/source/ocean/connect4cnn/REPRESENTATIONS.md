# Connect4CNN representations

`env.representation` is an integer selector, 0–9. Default `representation_mode=0` uses this fixed ID for every environment. Mode 1 assigns an ID deterministically per environment slot, using `representation_seed`; that assignment stays fixed across all matches/resets. Missing settings default to zero for old configs. Invalid IDs, modes and seeds fail at initialization.

All presets preserve the 7-column × 6-row game and float32 `[1,36,44]` input. Cell pitch means the width/height allocated per game location. Smaller boards are centered in the same black canvas: **this changes occupied pixel size, not tensor dimensions or encoder FLOPs**. Changing image dimensions remains separate work.

| ID | Cell pitch (width × height) | Piece appearance | Board placement |
|---:|---|---|---|
| 0 | 6×6 | Solid 6×6 squares, no interior gaps | Original 42×36 board, one black column at either side |
| 1 | 6×6 | Centered 4×4 squares | Two black pixels between adjacent pieces, in both directions |
| 2 | 6×6 | Centered 2×2 squares | Four black pixels between adjacent pieces |
| 3 | 6×6 | Filled diameter-6 raster disks | Original board footprint |
| 4 | 6×6 | Centered diameter-4 raster disks | Two-pixel gaps plus curved corners |
| 5 | 6×6 | Player X, opponent O/ring | Original board footprint |
| 6 | 4×4 | Solid 4×4 squares, no interior gaps | Centered 28×24 board |
| 7 | 2×2 | Solid 2×2 squares, no interior gaps | Centered 14×12 board |
| 8 | 1×1 | One pixel per game location | Centered 7×6 board |
| 9 | 2×4 | Solid narrow rectangles, no interior gaps | Centered 14×24 board |

Empty/background pixels are zero. Player pixels are 1 and opponent pixels 0.5 in every preset, including X/O, so ownership is unambiguous. Disk and ring boundaries are integer masks on a coarse grid, not antialiased graphics. Glyph masks are computed once at initialization, then observations use native buffer writes with no allocation, RNG, image library, or rendering context. ID 0 retains the original observation loop. Resets yield an empty black board in every preset. The human viewer continues to show the standard game; it is not an exact policy-image preview.

## Deterministic mixed appearances

```ini
[env]
representation = 0
representation_mode = 1
representation_seed = 12345
```

Mode 1 hashes the explicit unsigned 32-bit appearance seed and native environment slot with the versioned integer function in `appearance.h`. The native CPU environment constructor initializes `env.rng` to the slot before `puf_init`; selection reads that value without modifying it. Appearance does not use wall time, global `rand()`, worker order or the opponent RNG stream. It is selected once, not per frame, match or rollout. Equal seed/slot/count gives equal assignment for every encoder. Changing thread count does not reassign slots. Rank-local slots repeat across GPU ranks; only the one-GPU workflow is currently validated.

`base.seed` controls model initialization/action sampling, **not this appearance seed**, and currently does not change the native CPU game's initial per-slot RNG. Record both seeds explicitly. Keep slot count/order and appearance settings on checkpoint reload: weight-only checkpoints do not contain environment assignment. Changing the appearance seed intentionally creates a new assignment; changing the fixed ID in mixed mode has no effect. The sweep preparer rejects sweeping that inactive ID. Modes/seeds are fixed experiment controls, not optimizer targets.

Generate an assignment receipt after the CPU tests build the native helper:

```bash
build/connect4cnn/test_appearance connect4cnn 12345 64 > build/connect4cnn/appearance.csv
```

The first 16 IDs for seed 12345 are `0,0,7,8,6,4,4,4,6,4,7,3,8,8,1,5`. Golden-vector and interleaving checks pin this mapping. Finite random assignments need not balance categories or cover every ID. A mixed-training robustness study should evaluate fixed weights on **every fixed ID separately** with the same episode quotas; a pooled mixed score alone does not establish per-appearance robustness. Training separately per ID, mixed training, and transfer to unseen appearances are distinct experiments. The primary paper comparison remains fixed representation 0; mixed panels are separately labeled.

## Sweeping fixed appearances

Fixed appearance:

```ini
[env]
representation = 5
representation_mode = 0
representation_seed = 0
```

Add this section to an architecture recipe to include representation in the search:

```ini
[sweep.env.representation]
distribution = int_uniform
min = 0
max = 9
scale = auto
```

`sweep_representation.ini` is a ready-to-use representation-only recipe with the existing Flex quality shape fixed, a fixed 13,312,000-decision budget and a 20-trial cap. No long sweep is launched merely by adding the recipe. Short plumbing command:

```bash
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh \
  --recipe ocean/connect4cnn/sweep_representation.ini \
  --canary --max-runs 3 --wandb disabled
```

The standard flexible architecture recipe explicitly fixes representation 0. Add the above sweep section when appearance should vary too. The temporary preparation layer accepts representation-only or joint searches for encoder families 1/2/3/4; native C/CUDA PROTEIN performs the search and training. CSV/sidecar payloads retain representation, mode and seed separately from the architecture hash. Pareto flags group by fixed ID or mixed seed, never merging those experiment types. Old results without these settings mean fixed representation 0.

## Interpreting the results

These IDs are categorical presets encoded as integers; numerical distance between IDs has no scientific meaning. The current optimizer may treat that coordinate as ordered. An adaptive sweep is not guaranteed to sample every ID equally, or at all. Search can pick an easy rendering instead of finding an encoder that is robust across renderings. Keep the full sampled history. Report Pareto flags are computed within each representation; native PROTEIN still optimizes its joint score/cost objective.

For robustness, freeze an architecture and give each representation the same seeds/budgets, then compare all curves. Training separately on each representation tests learning sensitivity; evaluating fixed weights on a different representation tests policy transfer and is a separate experiment. All encoders should receive the same selected representation for an architecture comparison. Compact one-pixel boards retain the game state in the input, but existing strides/downsampling may make those pieces difficult for a given encoder to use.

This provides inexpensive appearance variation within one game. It does not establish transfer across game rules or replace eventual multi-environment evaluation. Store the chosen representation with checkpoints and use the same setting on reload unless intentionally testing a representation shift.

## CPU validation

Native validation also passed: `sweep.wstneiqe` ran three fixed-architecture trials at 32,768 decisions each, sampling representation IDs 0, 2 and 4. All completed, all 160,736-parameter checkpoints were finite, and saved native INIs/CSV/sidecar payloads retained the selected ID while the architecture hash stayed identical. Sweep process wall was 2.720 seconds excluding build. W&B was disabled; no new online or learning-scale sweep was launched. These scores/timings are plumbing evidence only. Small receipts are archived in `research/results/connect4cnn/sweep.wstneiqe/`.

`bash ocean/connect4cnn/tests/run_all.sh` checks all presets against independent literal pixel masks for every cell/owner, dirty-buffer overwrite, reset/terminal handling and clipping/padding. For each preset, 4,096 seeded game transitions match original Connect4 and repeated traces match. Invalid IDs fail explicitly. Checks run with ASan/UBSan and passed September 14, 2026. Encoder shapes and kernels are unchanged.
