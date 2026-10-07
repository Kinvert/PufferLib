# Shared CNN candidates versus fixed native references

Current follow-up: [quality versus Nature RTX 5060 smoke](NATURE_MULTIGAME_5060_SMOKE.md)
has actual matched six-game/all-drawing policy executions.
[Audited case export](CANDIDATE_EVALUATOR_ACCEPTANCE.md) sends completed final
checkpoints to the existing evaluator acceptance checker without retraining.
Historical preparation/zero-execution statements below remain specific to
their older five-model packet. The new short-local authorization does not
authorize a 5090 or long-search campaign.

October 6, 2026. This follows the actual
[six-game candidate smoke](results/candidate-panel-20261006/README.md).
That smoke trained only two of our configurations. The new opt-in panel-v2
adds Nature, IMPALA and Impoola to arbitrary-candidate comparisons. It doesn't
retroactively add baseline results to the completed smoke or authorize a run.

One candidate architecture is reused across all six games. Policies initialize
randomly and train separately per game; this is architecture generalization,
not a pretrained policy/backbone or one set of weights playing every game.
Only the CNN is a discovery dimension. LR/clipping/etc. remain per-game recipes,
shared across every candidate and reference in that game. H128/L1, float32,
64 actor slots/H32/M2048, decisions and checkpoint cadence are matched.

## Native builds and exact pairing

`candidate_panel.py build --baselines` creates a v2 registry with three fresh
normal native binaries per game: default (our encoder 4 and Nature selector 2),
IMPALA and Impoola (compiled family, selector 0). It captures owned source,
compiler/flags/commands/environment and binary identities. All use the same
production PufferLib/cuBLAS helpers; no isolated profiler optimization is enabled.
Vendor/system/link closure remains incomplete. Building and `eval_exact_info`
host metadata perform no policy/GPU query.

`prepare --baselines nature_cnn impala_cnn impoola_cnn` adds fixed reference
records, not optimizer dimensions. Baseline INIs contain only the exact selector
and H128/L1 core: inactive custom-CNN knobs are removed. The reference family
cannot be relabeled to select the faster default build accidentally. Every
candidate/reference uses identical full per-game world/learner/budget/mixture.
The v2 inspector reconstructs each complete INI from captured default/game/
compare files, numeric per-game overlays and exact declared output paths. Merely
rehashing changes to every family's learner or a secondary game INI fails.

Training mixes all drawings through fixed native seed/slot assignment; Pong and
Flappy opt into catalog 1. Counts aren't balanced after observing outcomes.
Each original checkpoint is bound to all fixed drawings of its game. Candidates
and references share one immutable suite per game/training seed/drawing; all
drawings share world/RNG/level identities. All five non-Connect4 adapters compare
native family/config world **and raster** manifests before inference. Connect4
retains its separate declared empty-board/identity-seed proof. Host starts don't
constitute policy episodes or prove live mixed vector initialization.

## GPU-free preparation

Fresh destinations, existing venv/toolkit/NCCL only. Both current execution holds
remain; the previous local smoke was completed, not extended. No new GPU query,
model, dataset, training, system dependency change or push belongs to this work.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/candidate_panel.py build --baselines \
  --out build/candidate-baselines/BUILD_FRESH

# Short budgets only demonstrate allocation, not learning or launch readiness.
.venv/bin/python research/candidate_panel.py prepare \
  --registry build/candidate-baselines/BUILD_FRESH/registry.json \
  --candidate happy-cat-1=research/recipes/panel_smoke_quality.ini \
  --candidate mystic-tree-2=research/recipes/panel_smoke_small.ini \
  --baselines nature_cnn impala_cnn impoola_cnn \
  --seeds 57173 57174 --steps 65536 --checkpoint-steps 32768 \
  --episodes 17 --slots 16 --pong-max-decisions 512 --breakout-max-frames 2048 \
  --out build/candidate-baselines/PANEL_FRESH

.venv/bin/python research/candidate_panel.py inspect \
  --plan build/candidate-baselines/PANEL_FRESH/plan.json

# Offline missing ledger: no native binary, GPU query, policy or fabricated score.
.venv/bin/python research/candidate_panel.py coverage \
  --plan build/candidate-baselines/PANEL_FRESH/plan.json \
  --out build/candidate-baselines/COVERAGE_FRESH
```

Allocation in that example: **60 training jobs, 82 shared suites, 410 checkpoint
bindings, 820 planned evaluations, 13,940 planned episode executions**, zero
observations. There are 310 native family/config world/raster comparisons and
100 declared Connect4 target bindings. Two checkpoints are allocated to each
binding; appearances don't create independent training seeds or extra training.
Declared training is 3,932,160 decisions; this is not a calibrated campaign.

The older v1 candidate packets/default-only builds and four-fixed-model panel/
binding/report protocols keep their original formats. Opt-in v2 requires the
three-family v2 registry. Old source snapshots and the first failed smoke remain
unchanged. Don't execute an archived G240 path on the separate 5090.

## Full curves after separately scheduled execution

### Optional native policy layout before training

`prepare --policy-metadata LOCAL_METADATA_BUILD` now freezes the three existing
host registration tools and records every job's full encoder/head/MinGRU layout
before weights exist. Compile tools with
`NVCC_ARCH=sm_120 NVCC_PREPEND_FLAGS='--threads 1' bash research/build_policy_metadata.sh FRESH_DIR`,
source the normal runtime helper, and add that directory to the preparation
command above. This works with both candidate-only v1 and baseline v2 packets;
old packets are unchanged and don't gain this check retroactively.

Owned production constructor/core/encoder sources and game action headers must
match the training build and captured panel. The host-only `--describe-flex`
mode registers shapes with no CUDA allocation, weight initialization/loading,
model math or GPU query. Ordered tensors/component counts/action/core/payload
bytes and native stdout/process clocks/commands/input hashes are reparsed by
offline inspection, including after relocation. Relocated packets can't launch.

In a separately scheduled run, registration repeats **before** the GPU query
and trainer for each job. All emitted checkpoints must have the registered byte
count before finite-byte reads or evaluation; native checkpoint logs must agree
with that parameter count. Offline audit requires a declared training prefix,
rechecks every retained registration proof and rejects size/count drift.
Failed registration retains the failure without touching a GPU. Host registration
is outside the native training timer; process receipts keep its separate cost.

This catches some shape/count errors, not equal-size swapped checkpoint contents,
tensor math, full-model memory, runtime binary/toolchain provenance or learning
quality. Its flags explicitly keep numerical and full-learner-memory acceptance
false. Existing finite-byte/weight/family/config/source/suite/timing checks stay
required. Both GPU holds and all broader qualification/calibration/selection
gates remain. Fresh source-matching builds/packets are required after code changes.

Nine new host/configuration/synthetic guards plus the 20 existing candidate/front
checks pass (29 total); the 23 existing checkpoint/evaluator acceptance regressions
also pass using fresh metadata tools. The tests use native registrations but mock
game manifests/training/query failure paths; no policies execute. See
[retained evidence](results/candidate-policy-layout-20261006/README.md) for the
separate real native multi-game preparation and original preliminary loader error.

### Learning and full curves

After committed Kinvert delivery, the 5090 regenerates builds and configurations
locally. Review actual prior evidence, then qualify independent model math,
learner-batch memory/reload/evaluator behavior and calibrate per-game learners/
learning budgets/caps before scheduling a long run. Geometry is still deliberately
fixed; comparisons must not silently grant only a slow baseline a smaller batch.
Encoder 5 remains excluded until its own acceptance gates. These preparations
don't authorize occupying the busy 5090 or relaunching the completed 5060 smoke.

Once scheduled, `run --mode development --allow-gpu` uses the selected native
binary/policy family and existing exact adapter for each job. Use explicit whole-
campaign and per-process limits; retain failures and all checkpoint declines.
The same native training-time receipt is reused for every drawing of a checkpoint.

`audit` remains offline and verifies configs, copied suites, native completion
summaries, assigned raw CSVs, binary/weight hashes and process/checkpoint clocks.
It adds full `curves.json`, `paired-means.csv`, `conditions.csv`, and `missing.csv`:

- Every candidate/reference/checkpoint stays present in raw observations.
- Means require every assigned training seed, weighted equally; no best seed.
- Game, training mixture and evaluation drawing remain distinct conditions.
- Incomplete conditions have null descriptive frontier flags.
- Development-only flags describe score/time envelope membership across **all**
  assigned candidates/references/checkpoints; smoke/preparation flags stay null.
- Pong lower/upper censoring envelopes remain separate, not a midpoint or CI.
- No average of raw pipes, wins, points, length and goal fractions is invented.
- Descriptive means/fronts don't establish uncertainty, held-out selection,
  replicated dominance or SOTA; those remain separate publication work.

This first version still expands native single-game PROTEIN proposals, keeping
every completed candidate rather than selecting its source-game winner. A later
native cross-game search objective needs predeclared normalization or a vector
objective and honest discovery/selection cost accounting. The direct contribution
here is a fair, reproducible full-panel comparison path, not a new learned win.
