# Shared native evaluator acceptance

Current local follow-up: [audited candidate checkpoints](CANDIDATE_EVALUATOR_ACCEPTANCE.md)
provides an offline export and a bounded quality/Nature RTX 5060 allocation.
Kinvert's renewed authorization permits short local smokes; the historical
hold statements and zero-execution evidence below describe earlier packets.
The RTX 5090, encoder-5, long-search and scientific qualification gates remain.

October 6, 2026. This is preparation and test infrastructure for comparing one
fixed encoder across the six native pixel games and their 38 drawings. **Both
GPU execution holds remain.** No command in this document implicitly schedules
inference, training, profiling or an architecture sweep.

`research/eval_acceptance.py` prepares existing checkpoints for the five
pending exact evaluators: FlappyCNN, PongCNN, BreakoutCNN, SnakeCNN and MazeCNN.
Connect4 retains its separate accepted evaluator and
`ocean/connect4cnn/tests/verify_deterministic_eval.py`. Do not run Connect4
through this tool or substitute its accepted behavior for the other games.

Opt-in [packet-v2 native layout preflight](CHECKPOINT_LAYOUT_GATE.md) now
checks full parameter registration and raw checkpoint length before GPU work
when `prepare --policy-metadata LOCAL_BUILD` is supplied. It supports the
legacy configurable encoder 4 plus Nature/IMPALA/Impoola; state/other graphs
and encoder 5 are excluded. All cases must pass: no silent fallback to v1.
Use actual merged training INIs/action declarations and retain finite-byte/
hash/provenance checks; equal size does not certify a policy. Existing v1
packets and direct game launchers retain their separately documented checks.

Python is external acceptance/preparation/audit glue. Every policy evaluation
uses the existing native C/CUDA binary and each game's supervised launcher;
there is no Python learner, CPU policy fallback or new search framework.

## What a case tests

Each case specifies one matching native binary, actual checkpoint, full executed
training INI, architecture family and **development** suite. Prepare a suite
with two allocation waves and a partial final wave, such as 65 episode IDs and
64 slots. Held-out suites are rejected: evaluator debugging must not expose
publication evaluation outcomes.

The four modes are:

| Mode | Allocation and purpose |
|---|---|
| `graph` | Original two-wave suite, CUDA graphs enabled. |
| `repeat` | Identical suite in a fresh native process. |
| `eager` | Identical suite with graphs disabled. |
| `tail` | Last wave's global IDs evaluated alone in a fresh process, retaining the original slot count. |

For each mode the game auditor checks every assigned ID, seeds, native starts,
appearance, counters, score semantics and censoring/end reason. Comparisons
require identical per-episode receipt rows, including action hashes, between
graph/repeat/eager, and between the original last wave and the independent
tail. Keeping the same slot count isolates reset/history effects without
changing the neural batch shape. It does **not** test all possible batch sizes,
thread counts or order permutations.

Complete prepared/executed INI contents must agree after relocating only the
three output paths. Policy/core, world, vectorization, evaluator, learner and
mode settings cannot drift. Weights, native binaries, suites, tools and outputs
have SHA256 receipts; native process clocks and float32 parameter-byte counts
are checked. Each supervised adapter retains reservation/query failures,
timeouts and partial outputs. The shared runner produces no successful
`acceptance.json` when any case fails.

## Prepare without a GPU

Use this checkout's existing Python 3.12 `.venv` and normal process-local native
runtime paths. No install, GPU discovery, model execution or new weights are
needed. First create development suites with each game's `exact_eval.py suite`
command; the game contracts below list its world/cap/level arguments. These
commands use host-only native manifest modes.

Create a JSON list with exactly these fields for each case:

```json
[
  {
    "name": "flappy-quality",
    "task": "flappycnn",
    "family": "flex",
    "binary": "local-build/default",
    "config": "actual-training.ini",
    "checkpoint": "actual-final-checkpoint.bin",
    "suite": "development-suite/suite.json"
  }
]
```

Paths resolve relative to the JSON file, not the current working directory;
absolute paths also work. Names must be unique simple alphanumeric/hyphen/
underscore identifiers. The allowed families are `state`, `tiny`,
`experimental`, `nature`, `compact`, `flex`, `impala`, `impoola`; **Flex2 retains
its separate encoder-5 numerical/reload gates and is excluded.** One case is
not acceptance for untested families, recurrent sizes or appearances. Do not
use another game's checkpoint or a shortened hand-written INI.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/eval_acceptance.py prepare \
  --cases build/eval-acceptance/cases.json \
  --out build/eval-acceptance/FRESH_PREPARATION --timeout 600
```

Preparation freezes all four mode configurations, original and independent
tail manifests, input/tool hashes and a `packet.json`, with
`status=prepared-not-executed`. It rejects source changes during preparation.
It does not copy full weights or binaries: originals must remain unchanged
and available when the packet is executed. Parameter-byte preflight verifies
finite nonempty float32 bytes, **not** weight compatibility or CNN math.
Compiled-source/build provenance remains a separately required receipt.
The GPU-free `inspect` command rechecks original inputs, frozen files/tools,
all four preparation flags/configurations and independent native tail starts:

```bash
.venv/bin/python research/eval_acceptance.py inspect \
  --packet build/eval-acceptance/FRESH_PREPARATION/packet.json
```

## Scheduled GPU step, currently on hold

Only after Kinvert explicitly schedules a GPU window, regenerate a fresh local
packet against that machine's current native builds and actual weights. Then:

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/eval_acceptance.py run \
  --packet build/eval-acceptance/FRESH_PREPARATION/packet.json \
  --out build/eval-acceptance/FRESH_EXECUTION
```

Each mode has its own bounded native process, shared benchmark reservation and
idle/hardware queries through the existing adapter. The deadline is per mode,
not per panel. No checkpoints are trained or modified. Source changes require
fresh preparation; preserve failed packets rather than editing them or
selectively ignoring disagreeing modes. A live interrupted/failed native job
must be resolved before another job is scheduled.

After completion the GPU-free audit reparses every CSV and comparison:

```bash
.venv/bin/python research/eval_acceptance.py audit \
  --packet build/eval-acceptance/FRESH_PREPARATION/packet.json \
  --execution build/eval-acceptance/FRESH_EXECUTION
```

Offline audit is artifact consistency, not independent proof that a GPU ran.
Synthetic control-flow fixtures can pass it; preserve their explicit identity
and never present them as policy results. No acceptance report certifies
forward/gradient math, held-out exclusion, learning quality, confidence bands,
frontier dominance or SOTA. Independent encoder math/reload and baseline
efficiency gates still apply.

## Current coverage and next allocation

The host/synthetic tests use all five actual native manifest adapters and
separately test action drift, configuration drift, quota/tail/path constraints,
immutable inputs, failed executions and offline output tampering. They execute
no neural model. Run them only with the matching local build fixtures described
in `research/tests/test_eval_acceptance.py`; this fixture inventory is G240
specific and is not a remote setup script.

Fourteen host/synthetic tests pass. [Retained source, logs and real-input
preparations](results/eval-acceptance-20261006/README.md) include the final
actual Flappy packets and their relocated host inspection. No native GPU
acceptance was executed.

Existing Flappy quality and stock-state final checkpoints provide real-input
preparation cases without additional training. They do not provide Nature,
IMPALA or Impoola coverage, nor acceptance for the other games. Future scheduled
canaries must produce and retain matching checkpoints for each required
family/game/configuration before this acceptance runner is used on them.

Once runtime gates pass, calibrate a common learner within each task using
development data and equal tuning resources, then freeze budgets/seeds and
retain all checkpoint curves. The [multi-task plan](MULTI_ENV_ROBUSTNESS.md)
separates those steps from architecture selection and paper confirmation.

Game contracts: [Flappy](../ocean/flappycnn/DETERMINISTIC_EVAL.md),
[Pong](../ocean/pongcnn/DETERMINISTIC_EVAL.md),
[Breakout](../ocean/breakoutcnn/DETERMINISTIC_EVAL.md),
[Snake](../ocean/snakecnn/DETERMINISTIC_EVAL.md),
[Maze](../ocean/mazecnn/DETERMINISTIC_EVAL.md).
