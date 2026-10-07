# Paired development evaluation across native pixel tasks

October 6 [mixed-training bindings](MIXED_PIXEL_TRAINING.md) now use binding-v3:
mixed training seed/catalog metadata, null single-training-drawing label and
fixed-ID evaluation of every checkpoint. They require explicit all-drawing
evaluation; matched mode is rejected. All host/configuration checks pass; no
policy or GPU executes, and the legacy contracts below remain unchanged.

October 6 [same-policy appearance transfer](PIXEL_APPEARANCE_TRANSFER.md)
adds opt-in binding-v2 via `--evaluation-appearances all`. Training drawing
and evaluation drawing are recorded separately; the same checkpoint targets
every drawing without retraining. Default matched mode preserves binding-v1
below. Actual six-game host preparation and 50 host/configuration checks pass;
no policy, transfer performance or GPU execution is implied.

October 6, 2026. This connects the six-task fixed-architecture preparation to
dedicated evaluation suites. It executes **no policy, training, GPU query or
dataset generation**. Both GPU execution holds and encoder-5 gates remain.
Python is external configuration/audit glue; the learners and policies remain
native C/CUDA. This is not a replacement native search framework.

## What is now connected

`audit_pixel_panel.py` checks every planned training cell against its captured
default/game/common recipe and declared overrides, including complete INI
semantics, numeric model selectors, the frozen quality graph, common core,
initialization, rotated job order, decisions/checkpoint cadence, exact build
commands, source hashes and evaluation gates. The separately loaded game INI
must remain the exact empty override emitted by preparation. Merely rehashing a modified
one-model **or all-model** learner/world does not bypass recipe closure.
Original absolute output paths remain recorded when inspecting a relocated
archive; inspection does not execute them.

`pixel_evaluation_plan.py prepare` consumes that audited panel and an explicit
three-build-per-game registry. It freezes the panel, full per-job training
INIs, tool snapshots, binary SHA/host identity and development suites. Each
training job binds to a suite, policy/build family, paired evaluation block,
all declared checkpoint steps and its still-open evaluation gate. It contains
no campaign launcher and does not load weights or invent checkpoint results.

Evaluation block allocation is `evaluation_seed + training_seed_index`, with
episode IDs `0..N-1`. Neither drawing, architecture, checkpoint nor observed
score affects that allocation. Four models share one suite per
`(game, drawing, training seed)`; every drawing of that game shares world/RNG
identities for that seed. Native host manifests additionally verify all four
job configurations/build selections against the canonical starting world
**and raster**. World comparison across drawings retains every manifest
column except drawing ID and observation hash, including RNG and Maze levels.

Connect4 has an existing independent empty-board/episode-seed manifest and
accepted dedicated GPU evaluator. It lacks a native host manifest command;
the new packet explicitly records **declared empty boards and identity seeds**,
not a newly executed native spawn/raster proof. The five other games have
native host manifests, but their GPU evaluator behavior remains unaccepted.
Identical starts do not promise identical action sequences for different
architectures or batch sizes.

## GPU-free use

First prepare a fresh fixed panel as described in
[MULTI_ENV_ROBUSTNESS.md](MULTI_ENV_ROBUSTNESS.md). Then:

```bash
.venv/bin/python research/audit_pixel_panel.py \
  --panel build/pixel-robustness/FRESH_PANEL/protocol.json
```

Compile the normal three float32 families per requested game with that panel's
build-only script when compilation is appropriate. Do not launch training or
evaluation. The registry is JSON keyed by exactly the requested task names,
with `default`, `impala` and `impoola` paths for each. Paths are absolute or
relative to the registry file. Example for a Connect4-only panel:

```json
{
  "connect4cnn": {
    "default": "FRESH_PANEL/bin/connect4cnn-default",
    "impala": "FRESH_PANEL/bin/connect4cnn-impala",
    "impoola": "FRESH_PANEL/bin/connect4cnn-impoola"
  }
}
```

That example assumes the registry is in `build/pixel-robustness/`. For a
six-task panel, include all six tasks and all three families. Never use G240
binary paths on the separate 5090 host. Default builds select encoder 4 for
quality and encoder 2 for Nature through their full INIs; the other builds
select their compiled encoder with selector 0.

```bash
# Process-local existing dependencies; no system changes/GPU discovery.
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/pixel_evaluation_plan.py prepare \
  --panel build/pixel-robustness/FRESH_PANEL/protocol.json \
  --registry build/pixel-robustness/LOCAL_REGISTRY.json \
  --out build/pixel-robustness/FRESH_EVALUATION_BINDINGS

# Offline packet/config/manifest audit: executes no native command.
.venv/bin/python research/pixel_evaluation_plan.py inspect \
  --plan build/pixel-robustness/FRESH_EVALUATION_BINDINGS/plan.json

# Optional current executable-byte check, still no execution/GPU query.
.venv/bin/python research/pixel_evaluation_plan.py inspect \
  --plan build/pixel-robustness/FRESH_EVALUATION_BINDINGS/plan.json \
  --check-binaries

.venv/bin/python -m unittest research.tests.test_pixel_evaluation_plan \
  research.tests.test_pixel_robustness -v
```

Defaults: 1,000 assigned IDs, 64 slots, development seed 56123; Pong cap
16,384 decisions and Breakout cap 8,192 physics frames. **Those administrative
caps and learning budgets are uncalibrated.** The other games retain their
native ending horizons. Maze uses the declared full level table, not an
unseen-level split. This tool offers no publication/held-out mode. Changes in
quota, batch, caps or block require a fresh packet, never editing old suites.

Only native `eval_exact_info` and `eval_exact_manifest` are invoked. Inputs,
registered executable bytes and tool sources are checked before/after
preparation; copied packet/configs/suites are checked offline. Existing output
directories are refused. Preparation failures retain `failure.json` and
partial artifacts; even a written plan is rejected if failure is present.
Compilation/source closure, numerical correctness and runtime acceptance
aren't inferred from executable metadata/hash equality.

## Retained checks and remaining work

[Local receipts](results/pixel-evaluation-binding-20261006/README.md) preserve
the successful actual six-task packet: 38 suites of 1,000 declared starts,
152 job bindings, 18 binary identities, 112 per-job native host comparisons,
and offline inspection. The initial fourteen tests are preserved unchanged;
[a loader-override follow-up](results/pixel-evaluation-binding-audit-20261006/README.md)
demonstrates the original gap and brings the passing suite to fifteen. The full
fixture covers 304 two-seed jobs. Connect4 metadata fixtures in those tests
are explicitly synthetic and do not run a policy. Actual native receipts
are separate from fixture output.

Next, qualify the five pending evaluators in an explicitly scheduled GPU
window using [EVAL_ACCEPTANCE.md](EVAL_ACCEPTANCE.md), then calibrate common
learning recipes with equal allowances and freeze a paired development
campaign. Bind every real checkpoint through its executed full INI and actual
matching build. Retain complete score/time curves, censored outcomes, failed
runs and all drawings. A future campaign controller must collect process/native
timing and checkpoint completion receipts and use those supervised adapters;
this packet alone neither launches that controller nor certifies the paper's
inference/replication gates or a multi-game frontier advantage.
