# Native PongCNN evidence

These are small receipts copied from the matching `build/pongcnn/<campaign>` directories. [Experiment log](../../EXPERIMENT_LOG.md) and [Pong search protocol](../../PONG_HYPER_SWEEP.md) explain their scope. Logs may be renamed `.txt` for Git; contents are preserved. Large binaries, checkpoint arrays, W&B SDK caches, full training dashboards and source-copy directories remain local and ignored. Source/binary/config hashes and native metric histories are retained. These files do not supply a downloadable trained-model bundle.

- `environment-20260914`: initial CPU parity and pixel checks.
- `canary.w5tqV8nE`: five-model native GPU training and checkpoint reload.
- `pilot.p0ydqKv4`: fixed-quality model, 13.312M-decision common-recipe negative learning pilot; its binary/source receipts are in `canary.w5tqV8nE`.
- `hypers.l6li6thO`: initial reporting quote-normalization failure after two valid native trials. `launch.txt` preserves the failure; `REPORT.md` was regenerated afterward with the corrected audit for diagnosis.
- `hypers.hF7MhfRo`: corrected eight-trial native sweep/evaluation/online W&B canary.

The subsequent `hypers.jwyApeXS` learning-scale discovery is running locally; archive its small receipts when complete. Do not mix canary scores into learning comparisons. All Pong `perf` values mean episode-averaged fraction of points won, not match win rate or ALE benchmark performance.
