# PongCNN appearances

The native game, opponent, actions, rewards and 1x36x44 input shape remain unchanged. `env.representation_mode=0` selects one `env.representation` ID for every slot. Mode 1 assigns a fixed appearance per slot from `env.representation_seed`, using the same [deterministic mapping contract as Connect4CNN](../connect4cnn/REPRESENTATIONS.md#deterministic-mixed-appearances). `env.representation_mix_catalog=0` preserves the historical five categories and assignments; catalog 1 opts into all seven. Neither consumes game RNG. Fixed mode accepts IDs 0–6 independently of the catalog.

| ID | Appearance | Preserved information |
|---:|---|---|
| 0 | Original rectangle raster | Original pixels exactly |
| 1 | Swap left/right paddle intensities 0.5 and 0.75 | Ball, geometry and scores unchanged |
| 2 | Reflect the court horizontally | All court pixels retained; action IDs unchanged |
| 3 | Reflect the court vertically | All court pixels retained; action IDs unchanged |
| 4 | Invert court intensities with `1 - pixel` | Reversible contrast change |
| 5 | Checker texture with class-specific intensity bands | Every original court pixel is recoverable |
| 6 | Four-neighbor contour texture with the same bands | Filled interiors and original pixels retained |

The top two score rows stay unchanged in every preset. Transforms operate on the existing raster and are reversible; no ball/paddle state, velocity or collision geometry is changed. The viewer still displays the original game, not the transformed observation. Transforms add native observation work but no allocation, graphics calls or RNG consumption. Encoder FLOPs stay unchanged. No speed advantage is claimed.

For IDs 5/6, `floor(4*pixel)` recovers class index 0/1/2/3; map these to
original court values 0/0.5/0.75/1. The two values within each band encode
checker position or a class boundary. Score rows bypass the transform.
See [the texture design and verification](../../research/PONG_TEXTURE_ROBUSTNESS.md)
for literal values, edge handling, overhead and retained evidence.

```ini
[env]
representation = 0
representation_mode = 1
representation_seed = 12345
representation_mix_catalog = 0
```

Use identical settings and 64 slots for all encoders in a mixed comparison. Do not vary the appearance seed/catalog along with architecture and silently pool results. For fixed evaluation panels use mode 0 and IDs 0 through 6, preserving all outcomes. Extra appearance seeds test assignment sensitivity; they do not replace training seeds or unseen game identities. The first 16 legacy catalog-0 mixed IDs at seed 12345 remain `0,0,2,3,1,4,4,4,1,4,2,3,3,3,1,0`. Catalog 1 changes assignments and must be reported separately.

CPU tests compare 49,152 transitions per panel with original Pong, including game state/logs/RNG, rewards and terminals. All seven fixed IDs and both catalogs with mixed seeds 0/12345/4294967295 pass literal pixel, score-row, clipping, dirty-buffer, reset and ASan/UBSan checks. The comparison hash excludes only appearance and unused tail padding. The shared native helper writes assignment receipts:

```bash
bash ocean/connect4cnn/tests/run_all.sh
bash ocean/pongcnn/tests/run_all.sh build/pongcnn/FRESH_ENVIRONMENT_ID
build/connect4cnn/test_appearance pongcnn 12345 64 > build/pongcnn/appearance.csv
```

These tests do **not** solve Pong's sparse match completion. Pooled evaluation v1 can overrepresent fast-completing slots; its `episode_length` still counts the last rally rather than the whole match. Keep failed evaluations unresolved and retain diagnostic progress/timeout logs. [The dedicated exact adapter](DETERMINISTIC_EVAL.md) now implements assigned episodes, whole-match counters and censoring bounds; its GPU runtime/reset/reload/cap-calibration gates remain pending. Appearance variation must not hide this limitation. Both GPU execution holds remain.
