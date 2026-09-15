# RTX 5090 Pong replication — reported results

Received September 15, 2026 from the user's separate 5090 agent. **The full original raw archive is not yet present on G240.** A subsequent [evaluation/profile handoff](PONG_EVALUATION_AUDIT_RESULTS.md) now provides the original protocol/results/curves/review plus raw diagnostic receipts; its package and diagnostic logs were checked locally. Original checkpoint bytes and the complete original evaluation logs remain remote. Protocol intent and exact learner recipes: [5090 handoff](PONG_5090_HANDOFF.md).

Reported source commit: `1d9e9e0dca88706c7a8495644c8dc584a4446f11`. Four fixed models, two shared learner recipes, five seeds per model/recipe, 4,194,304 decisions per run. Native PufferLib PongCNN; point fraction is points won, not match win rate or an ALE Pong benchmark score. Actual resolved configs, seeds and source changes still require archive review.

## Completion and timing

- 40/40 training runs completed; 312/320 evaluations completed.
- Eight evaluations hit the fixed 300-second timeout. All 320 checkpoint hashes and finite-weight checks reportedly passed on the 5090.
- Total elapsed: **3h13m2s**; summed training **1h22m4s**; evaluation **1h50m11s**; builds **16.6s**. Other orchestration accounts for the remaining elapsed time; verify exact timing boundaries from receipts.
- Evaluation consumed about 57% of total elapsed time. Eight capped evaluations account for roughly 40 minutes, so timeouts alone do not explain all evaluation cost. No timeout cause is established by this summary.

## Reported final results

Brackets are pointwise 95% bootstrap intervals reported by the remote agent. Each row has five training seeds; seeds reaching the target may have done so before the final checkpoint.

| Recipe | Model | Final point fraction | Seeds reaching >=95% | Mean training seconds | Process SPS | Native SPS |
|---|---|---:|---:|---:|---:|---:|
| A | Ours quality | Unresolved | 3/5, one unknown | 8.69 | 483,277 | 498,604 |
| A | Nature | Unresolved | 3/5, one unknown | 9.36 | 448,386 | 461,506 |
| A | IMPALA | 79.64% [39.66–99.96] | 4/5 | 131.92 | 31,810 | 31,875 |
| A | Impoola | 99.91% [99.85–99.95] | 5/5 | 132.61 | 31,651 | 31,716 |
| B | Ours quality | 29.90% [0.36–69.55] | 1/5 | 13.52 | 310,340 | 316,616 |
| B | Nature | Unresolved | 4/5, one unknown | 14.79 | 284,628 | 290,016 |
| B | IMPALA | 59.37% [19.68–99.07] | 3/5 | 338.47 | 12,411 | 12,421 |
| B | Impoola | 0.00% [0.00–0.00] | 0/5 | 335.42 | 12,510 | 12,520 |

For each unresolved row, one final evaluation and one seed's target attainment remain unknown. Possible five-seed final mean bounds are **68.42–88.42% for A/ours**, **69.46–89.46% for A/Nature**, and **79.66–99.66% for B/Nature**. These bounds allow the missing final score anywhere in [0,1]; they are **not confidence intervals**. Failed evaluations are not zero scores and must not be silently dropped. The other timed-out checkpoints also remain part of the full curve evidence.

## Interpretation and next analysis

Ours used approximately **7.2% less mean training time than Nature under A**, and **8.6% less under B**, from the rounded reported times. This is a descriptive timing advantage, not proof of score superiority. Nature performed substantially better under B: even its lower possible observed mean exceeds ours' reported mean, although inference about the wider training-seed population remains separate. Under A, ours/Nature final means remain unresolved and their possible ranges overlap.

Impoola was the most reliable observed model under A, reaching the target in all five seeds; under B it scored zero in all five final evaluations. Recipe interactions and seed variability are material. The near-perfect single-seed discovery results were not reliable predictions of fresh-seed performance.

**Do not infer the whole Pareto frontier from this final table.** Review all checkpoint curves, timings, paired-seed analyses and censored observations when evidence is available. Do not assert a new CNN advantage or SOTA based solely on training speed. Five-seed pointwise intervals do not prove frontier-wide dominance.

Before allocating another training campaign, inspect the evaluation timeout mechanism and full time/step curves. Potential long matches versus evaluation implementation overhead must be distinguished from logs/code, not assumed. Preserve the frozen results; any revised evaluator, cap or retry policy belongs to an explicitly labeled follow-up and must apply consistently across models. No retraining, retry or system modification was launched for this intake.

## Remote evidence references

These paths currently refer to the 5090 machine:

- Reviewed report: `build/pongcnn/confirm-5090.full.JXYKB06r/review/REPORT.md`.
- Full curves: `build/pongcnn/confirm-5090.full.JXYKB06r/curves.html`.
- Archive: `build/hardware-artifacts/confirm-5090.full.JXYKB06r.kYluzMzA.tar.gz`.
- Reported archive SHA256: `ef80a26d3fd6d88d0d2f6fb59613d72fde3093af760c42424989cb0777234a0e`.

Preserve exact source/config/runtime identities and verify this digest when the archive arrives. A remote commit identifier alone does not establish that the G240 checkout contains those changes or that the recorded experiment ran from a clean tree.
