# Committed Connect4CNN benchmark evidence

These small files are copies of the original local `build/connect4cnn/<run>/` artifacts. Reports, CSVs, recipes, resolved configurations/metrics, commands, hardware records, and source/binary hashes are preserved unchanged. Evaluation logs have only been renamed from `.log` to `.txt` so they are not hidden by the repository's log ignore rule.

| Run | Purpose |
|---|---|
| [compare.we7qgdcg](compare.we7qgdcg/REPORT.md) | Full-budget modified-recipe state/tiny CNN comparison, three seeds each |
| [compare.9egh6y44](compare.9egh6y44/REPORT.md) | Stock training configuration in float32, three seeds |
| [compare.2s__8t8l](compare.2s__8t8l/REPORT.md) | Full-budget adapted Nature CNN, three seeds |
| [compare.16ohu06z](compare.16ohu06z/REPORT.md) | Corrected state/tiny comparison smoke test |
| [compare.otwm9q9m](compare.otwm9q9m/REPORT.md) | Superseded score-reporting diagnostic, retained for provenance |
| [compare.z10zc3xf](compare.z10zc3xf/REPORT.md) | Nature training smoke test |
| [compare.ltxw6alr](compare.ltxw6alr/REPORT.md) | Nature same-seed training repeat |
| [compare.ij5_xpo7](compare.ij5_xpo7/REPORT.md) | IMPALA/Impoola training smoke tests |
| [compare.sqngvlom](compare.sqngvlom/REPORT.md) | IMPALA/Impoola same-seed repeats |
| smoke.a0cFzy / smoke.EMKokT | Earlier tiny-CNN smoke/repeat receipts, configs, and evaluation outputs |

The interpretation and hyperparameter comparison live in [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md). Absolute paths in command/protocol receipts describe the original machine. CSV checkpoint paths refer to the original run directory, not files included in this archive.

Large checkpoints, executables, full training dashboards, and full source snapshots remain in the ignored local build directories. This archive retains the original source hashes and copied config snapshots; it is not a backup of the checkpoint binaries. Implementation source and test code are committed under `ocean/connect4cnn/`, with the shared integration under `src/`. Benchmark receipts refer to the upstream base revision plus source hashes because these experiments preceded the first local implementation commit.

Before committing future results, preserve the same small evidence files here under the unique run ID and update Markdown links to this archive. Keep earlier results and diagnostic records intact.
