# Full curves for arbitrary CNN candidates and references

October 6, 2026. The existing HTML viewer had four fixed model names. That
omitted custom candidate names used by the cross-game panel, even when their
CSV observations were retained. It now uses the declared model catalog, with
stable colors and names for every assigned candidate/reference. Missing or
losing models remain in the legend and coverage table. Hiding a model changes
the display only; it cannot recompute stored frontier flags or seed means.

`research/candidate_viewer.py` bridges the existing candidate-panel audit to
the shared self-contained viewer. It is offline reporting glue, not a learner,
optimizer, native launcher or GPU query. Both execution holds remain.

## Prepare an empty report

```bash
.venv/bin/python research/candidate_viewer.py prepare \
  --plan LOCAL_PREPARED_PANEL/plan.json \
  --out build/candidate-view/FRESH_ALLOCATION_VIEW
```

This checks the frozen panel and displays its complete allocation. It cannot
create scores, means, process timing, SPS or frontier marks. The actual
five-model/two-seed six-game preparation displays 820 missing evaluation cells,
41 conditions and all five models, including Nature/IMPALA/Impoola. It remains
unexecuted; the short example budgets/caps are not learning recommendations.

## Report retained native results

```bash
.venv/bin/python research/candidate_viewer.py report \
  --plan LOCAL_EXECUTED_PANEL/plan.json \
  --out build/candidate-view/FRESH_AUDITED_VIEW
```

The tool first runs the existing **offline** candidate-panel auditor. It checks
the raw assigned-episode CSVs, native completion receipts, training/checkpoint
clocks, configs, source/binary/weight identities and exact quotas. A failed audit
retains a failure receipt and produces no score view. The CLI doesn't accept an
external score/analysis file or bypass the auditor. The original audit outputs
remain under `audit/`; viewer outputs are additive.

The report includes:

- `curves.html`: standalone conditions, means/seeds, seconds/decisions, model
  controls, every checkpoint decline, missing gaps, parameter counts, episodes
  and checkpoint hashes.
- `analysis.json`: the complete allocated/observed cells, source/input hashes,
  declared model catalog, separate score units and qualification flags.
- Observation/paired-mean/missing CSVs and a short `REPORT.md`.

Training process cost is counted once per job, even when that checkpoint targets
several drawings. Native checkpoint time remains the curve coordinate; process
seconds/SPS stay separately labeled. Native SPS is not collected by this reader;
no process value substitutes for it. Lookup tables keep rendering preparation
linear in the allocated/observed rows rather than repeatedly scanning the full
campaign for each checkpoint.

Games, training mixtures and evaluation drawings stay separate. Means require
all assigned seeds; incomplete conditions have no frontier marks. Pong retains
lower/upper censoring curves, never midpoints or confidence intervals. Smoke,
preparation and failed campaigns have no frontier marks; the visible scope is
labeled. No model hiding, best seed or best-game selection changes the stored
comparisons. Descriptive development marks don't certify statistical dominance.

## Actual checks and limits

The completed historical two-candidate smoke now renders all 82 audited drawing
observations across all 41 conditions, with the 12 original training costs counted
once. Its short scores still select no architecture and establish no learned
advantage. There are **no new policy executions**, and no reference scores were
retroactively added. The new five-model preparation renders zero observations.

Python artifact/scalar checks cover allocation, full paired curves, missing
seeds, cost reuse, smoke/failed-front suppression, mandatory audit, checkpoint
identity and failure retention. Node tests exercise the actual embedded display
logic with synthetic minimal-DOM fixtures, including a fifth missing candidate,
hyphenated reference names, unsafe text/prototype names, hiding, seeds, gaps and
Pong bounds. These aren't browser layout or neural validation.

The 5090 must receive reviewed committed Kinvert source and use locally
regenerated packets/artifacts. G240 build paths are not remote launch commands.
Independent neural math, runtime/memory/evaluator acceptance, learner/cap
calibration, cross-game adaptive native search, held-out selection and frontier
inference remain separate work. This improves evidence visibility, not SOTA.
