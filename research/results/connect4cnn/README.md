# Committed Connect4CNN benchmark evidence

These small files are copies of the original local `build/connect4cnn/<run>/` artifacts. Reports, CSVs, recipes, resolved configurations/metrics, commands, hardware records, and source/binary hashes are preserved unchanged. Evaluation logs have only been renamed from `.log` to `.txt` so they are not hidden by the repository's log ignore rule.

| Run | Purpose |
|---|---|
| [confirm.ol9tcj5k](confirm.ol9tcj5k/analysis/REPORT.md) | Complete frozen five-seed confirmation: 30 jobs, 390 evaluations, audited checkpoints/configs, full uncertain time/step frontiers |
| [sweep.wstneiqe](sweep.wstneiqe/REPORT.md) | Representation selector native canary: three IDs, fixed CNN, finite checkpoints; all-ten-preset CPU fixtures/parity receipts |
| [confirm-canary._0y3shor](confirm-canary._0y3shor/REPORT.md) | Frozen confirmation plumbing: all six models, 24 evaluations, six online uploads passed |
| [confirm-canary.hdq_fms8](confirm-canary.hdq_fms8/REPORT.md) | Preserved first confirmation canary: missing Flex native config keys; three reference jobs passed |
| [sweep._u86vi03](sweep._u86vi03/REPORT.md) | Completed 128-trial CNN2 sweep; all final held-out evaluations, winner checkpoint curve, and interactive full Pareto-front comparison |
| [compare.we7qgdcg](compare.we7qgdcg/REPORT.md) | Full-budget modified-recipe state/tiny CNN comparison, three seeds each |
| [compare.9egh6y44](compare.9egh6y44/REPORT.md) | Stock training configuration in float32, three seeds |
| [compare.2s__8t8l](compare.2s__8t8l/REPORT.md) | Full-budget adapted Nature CNN, three seeds |
| [compare.16ohu06z](compare.16ohu06z/REPORT.md) | Corrected state/tiny comparison smoke test |
| [compare.otwm9q9m](compare.otwm9q9m/REPORT.md) | Superseded score-reporting diagnostic, retained for provenance |
| [compare.z10zc3xf](compare.z10zc3xf/REPORT.md) | Nature training smoke test |
| [compare.ltxw6alr](compare.ltxw6alr/REPORT.md) | Nature same-seed training repeat |
| [compare.ij5_xpo7](compare.ij5_xpo7/REPORT.md) | IMPALA/Impoola training smoke tests |
| [compare.sqngvlom](compare.sqngvlom/REPORT.md) | IMPALA/Impoola same-seed repeats |
| [compare.l6d5sbk2](compare.l6d5sbk2/REPORT.md) | Full-budget IMPALA/Impoola common-recipe comparison, three seeds each |
| [sweep.r3qepq1g](sweep.r3qepq1g/REPORT.md) | Initial native PROTEIN custom-CNN canary, 12 trials; before final tooling refinements |
| [sweep.7s4ovb72](sweep.7s4ovb72/REPORT.md) | Finalized custom-CNN canary, repeatability and offline W&B evidence |
| [sweep.co1g6diu](sweep.co1g6diu/REPORT.md) | Completed learning-scale discovery: 24 trials, 8 shapes, online W&B; training scores only |
| [sweep.9sl0k5ll](sweep.9sl0k5ll/REPORT.md) | Nature numeric-ID control canary, 12 budget trials |
| [sweep.lvfycy5l](sweep.lvfycy5l/REPORT.md) | Compact strided family canary, 12 trials/9 shapes, numerical and reload receipts |
| [sweep.4h5iffsm](sweep.4h5iffsm/REPORT.md) | Completed Nature budget sweep: 12 trials, limited longer-budget coverage |
| [sweep.od0_6e75](sweep.od0_6e75/REPORT.md) | Completed compact discovery: 24 trials, 10 shapes, faster SPS but learning comparison unconfirmed |
| [sweep.jfu9hfqr](sweep.jfu9hfqr/REPORT.md) | Expanded flexible family: 12-shape canary and numerical/memory/repeatability/reload validation |
| [sweep.jd06qqgp](sweep.jd06qqgp/REPORT.md) | One-knob kernel sweep with four fixed-budget trials |
| smoke.a0cFzy / smoke.EMKokT | Earlier tiny-CNN smoke/repeat receipts, configs, and evaluation outputs |

The interpretation and hyperparameter comparison live in [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md). Absolute paths in command/protocol receipts describe the original machine. CSV checkpoint paths refer to the original run directory, not files included in this archive.

Large checkpoints, executables, full training dashboards, and full source snapshots remain in the ignored local build directories. This archive retains the original source hashes and copied config snapshots; it is not a backup of the checkpoint binaries. Implementation source and test code are committed under `ocean/connect4cnn/`, with the shared integration under `src/`. Benchmark receipts refer to the upstream base revision plus source hashes because these experiments preceded the first local implementation commit.

Before committing future results, preserve the same small evidence files here under the unique run ID and update Markdown links to this archive. Keep earlier results and diagnostic records intact.

The `sweep.*` archives additionally retain the isolated effective configs, native PROTEIN stdout as `sweep.txt`, and sidecar JSON payloads. They do not contain the local W&B binary run files. Canary scores are training metrics at very short budgets; they are not comparable to the held-out learning baselines above. Validation receipts live under the finalized canary's `validation/` directory.
