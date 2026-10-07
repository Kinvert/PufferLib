# Existing CNN checkpoints: exact evaluator acceptance

October 6, 2026. Kinvert authorizes short local RTX 5060 smokes. This follow-up
uses the completed [quality/Nature six-game smoke](NATURE_MULTIGAME_5060_SMOKE.md)
to check the five newer evaluators, without training another policy. Connect4
keeps its separate existing checker; encoder 5 and slow baselines are excluded
from this local allocation. It does not schedule the separate RTX 5090.

## Export from actual completed evidence

`research/candidate_acceptance_cases.py` invokes the existing complete offline
panel audit before exporting cases for `research/eval_acceptance.py`. It checks
selected binary bytes against the captured build registry and selected INI,
checkpoint and suite bytes against audited input hashes. It keeps every
declared non-Connect4 policy and training seed; unsupported families fail
rather than disappearing. A failed/incomplete panel cannot export successfully.

The exported checkpoint is the **last scheduled checkpoint**, not a winner
chosen from scores. Default coverage is drawing 0 for each policy/game; explicit
`--all-drawings` retains every fixed drawing target from that same checkpoint.
The existing checker accepts at most 64 cases; oversized exports fail rather
than truncate. Multiple training seeds may therefore need explicit separate
allocations. Original inputs remain immutable. The export is local artifact
glue, not a CNN/learner/optimizer or permission to launch inference.

```bash
# Offline export; no native process or GPU query.
.venv/bin/python research/candidate_acceptance_cases.py \
  --plan LOCAL_COMPLETED_PANEL/plan.json --out FRESH_EXPORT
# Add --all-drawings for all declared fixed drawing targets.

# Host-only manifests and full native parameter/byte registration.
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/eval_acceptance.py prepare \
  --cases FRESH_EXPORT/cases.json \
  --policy-metadata LOCAL_NATIVE_METADATA_BUILD \
  --out FRESH_PACKET --timeout 30
.venv/bin/python research/eval_acceptance.py inspect \
  --packet FRESH_PACKET/packet.json
```

Preparation and inspection preserve false execution/math/claim flags. An
explicitly scheduled local run uses the existing native supervised adapters,
with shared reservation/idle queries and per-process timeouts; it never trains:

```bash
.venv/bin/python research/eval_acceptance.py run \
  --packet FRESH_PACKET/packet.json --out FRESH_EXECUTION
# Offline completed-receipt check:
.venv/bin/python research/eval_acceptance.py audit \
  --packet FRESH_PACKET/packet.json --execution FRESH_EXECUTION
```

These placeholder paths are not 5090 commands. Remote agents must receive
committed source and regenerate against their own matching builds and actual
checkpoints. No push or campaign launch follows automatically from this file.

## Local bounded allocation

Ten cases: quality and Nature, drawing 0, for Pong/Flappy/Breakout/Snake/Maze.
Actual training seed 58173; final checkpoint at 65,536 decisions; matching
full training INIs and native binary/source/build receipts from the completed
RTX 5060 smoke. Development evaluation seed 68173, 17 IDs in 16 slots.
Pong decision cap 512 and Breakout frame cap 2,048 are unchanged.

Each case checks graph, fresh-process repeat, eager, and global episode ID 16
alone in a fresh process with the original 16-slot neural batch size. Total:
40 native inference processes / 520 assigned episode executions. The independent
tail must reproduce the original tail's complete episode/action receipt, not
merely its score. No private held-out data or performance selection is used.

Local export: `build/eval-acceptance/quality-nature-5060-export-v2-20261006`.
Local packet: `build/eval-acceptance/quality-nature-5060-packet-v2-20261006`.
Execution destination: `build/eval-acceptance/quality-nature-5060-execution-20261006`.

The initial export failed before GPU work because its new binary proof lookup
expected an input-list entry instead of the existing audited build binding.
That failed export and attempted preparation log remain unchanged. The corrected
export uses a fresh destination; production native source/checkpoints are
unchanged. Nine synthetic artifact/control-flow tests pass; they are not
policy/evaluator runtime evidence.

## Scope of a successful comparison

The completed comparisons cover these two actual policies/configurations on these five
drawing-zero development suites for quota/repeat/eager/independent-tail receipts
at 16 slots. They do not certify every drawing, batch size, recurrent width,
thread schedule, baseline architecture, encoder forward/gradient math, full
learner/reload/concurrency, adequate learning, censoring calibration, held-out
selection or a Pareto/SOTA advantage. Broader qualification remains necessary.
The existing all-drawing smoke's repeats remain separate, without an independent
tail claim for drawings 1+. Retain every failure and contrary outcome.

## Completed local execution

**Ten of ten cases passed**, with all 40 native inference processes and 520
assigned episode executions retained. Graph/repeat/eager episode/action rows
and independent-tail rows compare exactly for each case. The existing offline
audit also passes against all frozen inputs, mode configurations, native output
hashes and recorded parameter/process receipts. Ten full-policy layout checks
repeat before evaluation; no checkpoint was trained or modified.

Sum of native inference process times: **27.949727103 seconds**. The interval
from first native process launch to last completion is **44.325992075 seconds**;
this is not the complete shell/preparation/build/review time. There is no
throughput claim from these evaluation-only measurements. Every hardware
receipt identifies RTX 5060, UUID `GPU-56b0df25-b4c9-46eb-e498-bc5c71904097`,
driver 591.86. Current matching native builds use CUDA compiler 12.8.93.

[Durable source, reports and evidence](results/candidate-evaluator-5060-20261006/README.md)
retain the failed host export/attempted prepare separately from the successful
fresh allocation. The successful allocation is finished; do not restart it.
Actual all-drawing export is separate host-only groundwork for 62 cases;
none of those additional drawing-tail checks have executed.
