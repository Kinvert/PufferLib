# RTX 5060 mini benchmark: short training curves

October 6, 2026. Kinvert explicitly requested small local 5060 training runs
at the short-time end of the score/time tradeoff, avoiding long accuracy runs.
This is a new allocation; earlier smoke/numerical packets remain untouched.

**Completed and audited:** 24/24 jobs, 328/328 evaluations, 48/48 exact-byte
repeat/eager checks and 6,392 assigned episode executions; zero missing cells.
Campaign379.587885s, training processes88.709230s, evaluation processes201.206052s
(excluding builds). Ours has 9.18–22.23% higher process SPS across these small
paired runs. Learning is weak: Connect4/Pong/Flappy score zero across all
drawings; no quality winner or meaningful Pareto dominance is established.
Three native uptime values exceed whole-process durations; preserve their
logs and use monotonic process/checkpoint time for comparisons, not native
uptime-derived average SPS.
[Full result, SPS table and interactive curves](results/mini-short-5060-20261006/README.md).

## Frozen allocation before execution

- Ours quality (`happy-cat-1`): encoder 4, C16/K7/S4, flatten/projection64,
  H128/L1. Nature (`nature-cnn`): adapted three-valid-convolution encoder,
  same H128/L1. Both initialize randomly in each game/seed.
- All six existing games; no architecture/hyperparameter search. Per-game
  common learner recipes remain identical between models, 64 actor slots,
  rollout H32, minibatch2048, float32. Full native policy layouts checked.
- Two paired training seeds 59173/59174. Each job stops at **524,288 decisions**;
  scheduled checkpoints at **262,144 and 524,288**, retained regardless of score.
  Twenty-four serial jobs / 12,582,912 total decisions across the whole panel.
- Training mixes all 41 drawings through persistent native slot assignments,
  appearance seed35173; Pong/Flappy explicitly use full catalog1.
- All fixed drawings evaluated at both checkpoints: 328 evaluations, each with
  17 assigned deterministic episodes in 16 slots, eval seeds69173/69174.
  Final drawing0 additionally gets graph repeat/eager checks per job:
  48 checks. Total 6,392 assigned episode executions if all complete.
- Pong cap512 decisions retains censoring bounds, Breakout cap2048 physics
  frames retains partial score. Caps/episode IDs match both architectures.
- Campaign deadline600s, each trainer/evaluator at most20s; fail/retain on
  timeout, contention, changed source/config, bad weights or incomplete quota.
  Shared GPU reservation/idle checks; no interruption of competing processes.

`candidate_panel.py run --mode mini` explicitly routes this bounded allocation
to RTX5060; the historical development mode still requires5090 and the old
smoke bounds remain unchanged. Mini accepts only the frozen quality plus
Nature, at most two seeds/24 jobs/four checkpoints/1,048,576 decisions per job,
656 evaluations/33 episodes/32 slots and bounded process/campaign deadlines.
Actual allocation above is smaller than those ceilings. IMPALA/Impoola/encoder5
and new CNN shapes aren't executed. The baseline registry still compiles all
three native families for its existing schema; compilation isn't policy use.

## Local commands and evidence paths

Use the checkout's existing Python3.12 uv venv and process-local runtime helper;
no system CUDA/driver/Torch/dependency changes. Fresh destinations, no retry
of existing allocations. These commands describe this authorized local panel,
not automatic 5090 instructions.

```bash
source ocean/connect4cnn/runtime_env.sh
.venv/bin/python research/candidate_panel.py build --baselines \
  --out build/candidate-baselines/native-mini-5060-20261006
.venv/bin/python research/candidate_panel.py prepare \
  --registry build/candidate-baselines/native-mini-5060-20261006/registry.json \
  --policy-metadata build/policy-metadata/candidate-layout-20261006 \
  --candidate happy-cat-1=research/recipes/panel_smoke_quality.ini \
  --baselines nature_cnn --seeds 59173 59174 \
  --steps 524288 --checkpoint-steps 262144 \
  --appearance-seed 35173 --eval-seed 69173 --episodes 17 --slots 16 \
  --pong-max-decisions 512 --breakout-max-frames 2048 \
  --out build/candidate-baselines/mini-short-5060-20261006
.venv/bin/python research/candidate_panel.py inspect \
  --plan build/candidate-baselines/mini-short-5060-20261006/plan.json
.venv/bin/python research/candidate_panel.py run --mode mini --allow-gpu \
  --plan build/candidate-baselines/mini-short-5060-20261006/plan.json \
  --timeout 600 --train-timeout 20 --eval-timeout 20
.venv/bin/python research/candidate_panel.py audit \
  --plan build/candidate-baselines/mini-short-5060-20261006/plan.json \
  --out build/candidate-baselines/mini-short-5060-review-20261006
.venv/bin/python research/candidate_viewer.py report \
  --plan build/candidate-baselines/mini-short-5060-20261006/plan.json \
  --out build/candidate-baselines/mini-short-5060-view-20261006
```

Build/source/compiler/binary receipts, full per-game executed configurations,
world/raster/assignment suites, every checkpoint, native timing/SPS/performance
arrays and every exact episode remain in the panel. Native monotonic checkpoint
time defines curve x; full process SPS is separate and includes startup/writes.
Training is charged once per job, not repeatedly per drawing/evaluation.

## Interpretation

This samples two low-budget points per game/seed/model, not a complete Pareto
front or calibrated learning comparison. Mini frontier/selection/publication
flags remain false/null; the viewer still exposes every measured curve and
paired seed mean. Tiny episode quotas and two seeds provide only coarse
development evidence. Don't pick favorable games/drawings/checkpoints, turn
Pong bounds into midpoint scores, pool 5060/5090 timing, or call this SOTA.

The prior H256/B2048 gradient comparison remains failed; this panel uses the
unchanged H128 models, with no mathematical acceptance upgrade from training.
Full policy/optimizer/concurrency/reload math, steady-state timing, sufficient
per-game learning/cap calibration, baseline efficiency and statistical/held-out
confirmation remain open. No long sweep, 5090 job or push is authorized here.
