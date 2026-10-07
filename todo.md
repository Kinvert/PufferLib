# CNN robustness and paper roadmap

October 5, 2026. Ordered priorities agreed for discussion; this document does
not launch training or authorize new GPU campaigns.

## Goal and current evidence

Deliver a clean native PufferLib pixel-learning system with a reproducible
advantage on a clearly defined part of the score-versus-training-time Pareto
frontier, backed by a defensible paper. Beating Nature alone does not establish
SOTA or "faster than anyone else."

The latest RTX 5060 pilot ran four fixed models on corrected Connect4CNN,
representation 0, seed 173, 13.312M decisions each and 13 checkpoints each.
Ours finished in 179.246 seconds at 91.96% final wins; Nature in 196.779 seconds
at 76.19%; IMPALA in 1,293.430 seconds at 97.73%; Impoola in 1,281.174 seconds
at 76.98%. This is one seed with pooled-v1 evaluation and approximate checkpoint
times, not established dominance. Other renderings, Pong and Flappy were not
part of this panel. [Full report](research/results/connect4cnn/rtx-5060/hardware-5060.Mz7sDeEe/full/REPORT.md)
and [whole curves](research/results/connect4cnn/rtx-5060/hardware-5060.Mz7sDeEe/full/analysis/curves.html).

## Ordered todos

October 6 checkpoint groundwork: [shared acceptance packet-v2](research/CHECKPOINT_LAYOUT_GATE.md)
now supports native parameter-layout/size rejection before GPU work, including
configurable legacy CNNs and Nature/IMPALA/Impoola. Existing Flappy quality
weights prepare successfully without model execution. Regenerate fresh local
metadata/packets after delivery; still qualify neural math/memory/evaluator
behavior and calibrate matched learners. Direct training panels/game launchers
need their own deliberate integration rather than assuming this optional gate
covers them. Both GPU holds and encoder-5 restrictions remain.

October 6 completed preparation: [matched candidate/reference panel-v2](research/CANDIDATE_BASELINE_COMPARISON.md)
adds Nature/IMPALA/Impoola with captured identical per-game recipes and shared
starts/pixels. Eighteen targets compile; 60 jobs/82 suites/310 native comparisons
and 100 Connect4 declarations prepare successfully. All 820 checkpoint
evaluations remain missing. Full-curve reporting and artifact verification pass;
the prior smoke remains unchanged. Next prioritize scheduled independent
math/memory/evaluator acceptance and per-game learner/cap calibration, followed
by a reviewed Kinvert delivery and matched multi-seed development campaign.
Do not launch the uncalibrated 65K example as a quality comparison or describe
single-game native proposals as a cross-game adaptive optimizer.

October 6 active task: [one architecture across games](research/CROSS_GAME_CNN_SWEEP.md).
Per Kinvert, learner settings are per game and fixed across candidate CNNs. Build
the arbitrary-candidate/native-PROTEIN expansion and complete the explicitly
authorized bounded 5060 smoke (two candidates/six games/all 41 drawings), retaining
deterministic full quotas, failures and training clocks. Then review/calibrate and
deliver committed 5090 instructions for a separately scheduled larger panel.
Smoke scores cannot select the general architecture or establish a paper claim.

Completed smoke: 12/12 training, 82/82 drawings, 24/24 exact-byte repeat/eager
checks. [Actual retained evidence](research/results/candidate-panel-20261006/README.md).
The architecture-only recipe is verified for all six games (18 CNN dimensions,
per-game learners/core/budget fixed). Actual native import keeps all 12 proposals
and prepares 72/492 jobs/targets without executing them. Next qualify/calibrate
the planned larger 5090 work and deliver a reviewed source commit; don't relaunch
the completed smoke or infer a learned-quality winner from it.

Current paired-math [protocol v2](research/DENSE_ALIAS_ACCEPTANCE.md) covers
B2048 as well as rollout/small batches: 72 fixtures, zero executed, 39 host
tests and actual preparation pass. Preserve v1. Both GPU holds remain; do not
mistake its declared coverage for numerical or whole-policy acceptance.

October 6 [paired copy-candidate numerical preparation](research/DENSE_ALIAS_ACCEPTANCE.md)
is implemented, with exact quality H64 coverage and native marker/device-weight
receipts. Two libraries compile, 39 host/synthetic tests and actual preparation
pass; no library/GPU/model is executed. The next scheduled step is actual
fixed-graph GPU acceptance, then broader policy/concurrency/reload and fair
timing/learning. Both holds remain; preparation isn't a frontier result.

October 6 [isolated dense-input optimization](research/DENSE_PATCH_ALIAS_CANDIDATE.md)
is compiled/prepared for quality and Nature equally, with unchanged IMPALA/
Impoola controls. Paired registration/counts and host guard tests pass; owned
allocations stay unchanged. Before scheduling its complete timing panels,
execute the newly prepared paired GPU numerical acceptance including quality H64.
Both GPU holds remain; this candidate adds no frontier result and is outside
the production learner. Broader evaluator/calibration/learning gates remain.

October 6 [offline multi-game reporting](research/PIXEL_FRONTIER_REPORTING.md)
is implemented: exact native timing/episode audits, separate score units and
full paired-seed/appearance/checkpoint allocation. Seventeen focused tests and
the related 42-test suite pass; actual current report is honestly empty with
328 missing evaluations. Native execution stays unchanged. Next scheduled work
still accepts evaluators/full-model paths and calibrates matched learners before
learning curves; reporting readiness does not lift either GPU hold.

October 6 [native full-policy size/layout metadata](research/POLICY_CHECKPOINT_PREFLIGHT.md)
now compiles in three standalone targets and passes seven host/scalar tests,
including a 144-case six-game/four-model/core grid. The actual old Flappy quality
checkpoint matches its merged INI's count; default-only input is rejected.
This standalone tool is not yet integrated in launchers. Keep their finite-byte
checks and all immutable source/config/binary/weight bindings; size equality
doesn't qualify numerical/runtime/learning behavior. Both GPU holds remain.

Follow-up: self-contained HTML now exposes all four CNNs, all checkpoint/seed
curves, model coverage, missing/failing cells and shared-job costs. Twenty Python
and ten Node artifact/display fixtures pass, without a policy or GPU. The six
current environments are enough per Kinvert; prioritize measured acceptance and
matched comparisons rather than new game conversions. The actual report remains
empty, and no incomplete/seed/decision view receives frontier rings.

October 6 [Flappy geometric drawings](research/FLAPPY_GEOMETRIC_ROBUSTNESS.md)
expand the current panel to six games/41 drawings. Seven fixed Flappy IDs and
legacy/expanded mixtures pass game/raster/sanitizer checks; the old four pixel
streams match sampled pre-change hashes. All five native targets compile and
107 host tests pass. Actual mixed preparation retains 24 training jobs/164
fixed-target bindings, zero policies executed. Historical 38-drawing packets
retain their captured catalogs. Both GPU holds remain; scheduled evaluator,
memory/math/reload and shared-learner qualification still precede learning.

October 6 [mixed-training panels](research/MIXED_PIXEL_TRAINING.md) are prepared:
all CNNs share native slot assignments and each checkpoint targets every fixed
drawing. Counts, complete curves and mixture/learner seeds must remain distinct;
do not call this unseen-drawing transfer or count drawings as independent seeds.
Sixty host/configuration checks pass, no training ran. Both GPU holds remain;
scheduled evaluator/model/memory/reload and recipe calibration still come first.

October 6 [same-policy drawing transfer](research/PIXEL_APPEARANCE_TRANSFER.md)
is prepared: 24 drawing-0 jobs bind to all 38 drawings/152 targets without
retraining. Host starts and complete checkpoint allocation pass; no policy
ran. After scheduled evaluator acceptance and learner calibration, distinguish
fresh learning per drawing from transfer of identical learned weights. Don't
count target drawings as independent training seeds or duplicate their training
cost. Both GPU holds and final-test/inference gates remain.

October 6 [per-game budget support](research/TASK_BUDGETS.md) resolves the
single-budget plumbing restriction. All four CNNs remain matched within a
game; v3 preserves each game's complete checkpoint schedule and native scalar
receipts. Six-game host preparation passes; its large one-seed allocation is
an uncalibrated example, not an approved launch. Next qualify pending evaluator/
memory/math/reload and affordable per-game shared recipes before choosing
adequate execution/replication budgets. Both GPU holds remain.

October 6 memory constraint: [LEARNER_MEMORY_PREFLIGHT.md](research/LEARNER_MEMORY_PREFLIGHT.md)
shows stock Flappy IMPALA/Impoola training tensors alone exceed 16 GiB, so the
unchanged recipe cannot fit the 5060. An equally applied smaller-minibatch
stock candidate is prepared, not trained. Full-policy memory/math/reload gates
still precede timing; 5090 fit also remains unverified. Preserve separate
recipe fronts and the 96 earlier all-drawing cells; current twelve extra cells
calibrate drawing 0 rather than establish drawing robustness. Both holds remain.

October 6 shared learner preparation:
[SHARED_LEARNER_RECIPES.md](research/SHARED_LEARNER_RECIPES.md) implements
numeric per-game overlays while keeping CNN/core/world fixed. Stock and
small-batch Flappy candidates now have 96 prepared jobs, all four drawings,
three paired seeds, aligned budgets/cadence and identical native host starts.
34 host/config/scalar tests pass; no policy ran. Qualify exact evaluation first,
then baseline memory/math/reload at the chosen geometry and shared-recipe
learning adequacy in a scheduled GPU window. Do not launch these panels under
either hold or pool the two learner regimes into an unlabeled frontier.

October 6 learning adequacy preparation:
[native geometry and calibration plan](research/LEARNING_RECIPE_CALIBRATION.md)
now distinguishes decisions from optimizer updates and effective replay.
All paired jobs match; current templates remain uncalibrated. Use the existing
successful Flappy recipe as a candidate crossed with all four CNNs, preserving
identical actual budgets/learners within each candidate. First qualify pending
exact evaluators with available weights in a scheduled window; no GPU execution
or new learning is authorized by the scalar audit.

October 6 paired evaluation preparation:
[PIXEL_EVALUATION_BINDINGS.md](research/PIXEL_EVALUATION_BINDINGS.md) now
connects all six tasks/38 drawings/152 planned fixed-model jobs to immutable
development episodes and complete checkpoint schedules. Complete captured
recipe checks (including secondary game INIs), 15 host/configuration tests and actual 1,000-ID native-start
comparisons pass; Connect4's empty-board manifest is separately labeled.
This resolves suite binding, not GPU acceptance, budget/cap calibration,
matched learning or frontier uncertainty. Preserve both GPU holds and use the
scheduled evaluator gate next before any new learning campaign.

October 6 fairness groundwork: [shared GEMM acceptance](research/SHARED_GEMM_ACCEPTANCE.md)
compiles actual main/dW event/workspace wrappers with independent GPU dyadic
references; ten host/synthetic checks and the frozen 96-case preparation pass.
Exclusive process-group/full-panel supervision and offline auditing now have
host/synthetic checks. Actual GPU execution,
full-model qualification and comparable updated learning curves remain pending;
no production helper or measured result changed, and both GPU holds remain.

October 6 next-runtime preparation: [the shared acceptance tool](research/EVAL_ACCEPTANCE.md)
freezes existing checkpoint inputs for the five pending evaluators and checks
repeat/eager/independent-tail behavior at an unchanged slot count. Actual
Flappy quality/state packets need no retraining; missing matched baseline/game
checkpoints still need explicitly scheduled canaries. Host/synthetic checks
are complete preparation, not runtime or frontier evidence. Both holds remain.

October 6 current inventory supersedes the older expansion counts below:
[six native pixel tasks, now 41 drawings](research/MULTI_ENV_ROBUSTNESS.md), including
MazeCNN navigation with the same original local crop and shared level table.
Maze game/raster checks and four native builds pass, but no policy ran. Its
exact adapter now compiles, with identity-based levels/first-terminal capture,
14 host/synthetic supervision tests and 1,000 matching starts per drawing.
Checkpoint preparation retains failures and checked prefix-table expansion;
actual training/selection level exclusion and non-Connect4 GPU acceptance
remain pending. Current packets are 24 default / 152
all-drawing jobs per seed, **prepared only**. Both GPU holds remain.

Flappy exact implementation now compiles and passes host/simulation/audit
checks; [GPU acceptance instructions](ocean/flappycnn/DETERMINISTIC_EVAL.md)
are ready for a separately scheduled window. GPU validation remains pending;
this does not authorize training or relabel old pooled scores as exact.
Its checkpoint wrapper now passes 21 host/synthetic tests, supports preparation
without GPU queries and rejects source/manifest configuration drift. Existing
quality/state final weights and their executed INIs have fresh 65/64 acceptance
and 1,000-ID appearance packets, prepared only. First scheduled acceptance can
reuse those weights; no historical pilot needs relaunching. Nature/IMPALA/
Impoola still need actual matching Flappy checkpoints and acceptance.

October 5 expansion groundwork: [MULTI_ENV_ROBUSTNESS.md](research/MULTI_ENV_ROBUSTNESS.md).
BreakoutCNN now has a compiled exact adapter: first-terminal capture stops
frame-skip spillover, and administrative frame caps preserve partial scores.
Host/simulation/sanitizer/audit checks pass, without GPU policy execution.
Four-model configurations now cover five tasks/all 30 drawings. For item 4,
schedule Flappy exact GPU acceptance when available, then Breakout acceptance.
Its supervised checkpoint launcher now passes preparation/audit tests without
GPU use. Pong's whole-match/censoring adapter now compiles and passes host
checks; its supervised checkpoint preparation/audit checks now pass, while
GPU acceptance remains open. SnakeCNN/SnakeBench now share a separately
versioned one-agent local-view protocol with local RNG, clean resets and
explicit death/horizon terminals. Independent world/pixel checks and four
native targets compile. Exact evaluation now compiles with assigned identity,
death/horizon and all-episode auditing; host/game/sanitizer checks pass.
Its supervised checkpoint launcher now passes fourteen host/glue checks;
GPU inference/learning acceptance remain pending. No GPU launch is authorized.

1. [ ] **Validate exact evaluation and checkpoint timing on GPU.**
   **Further progress:** timing/recovery canary passes; all 52 existing
   checkpoints now have exact-suite full curves. Full-cadence overhead remains
   a practical monitoring/analysis limitation rather than a calibrated bound.
   [Follow-up methods and evidence](research/CONNECT4_DEVELOPMENT_REPLICATION.md).
   **October 5 progress:** dedicated frozen-suite evaluation now passes GPU
   quota/reload/RNG/reset/repeat/eager checks across eight configurations;
   [instructions and evidence](ocean/connect4cnn/DETERMINISTIC_EVAL.md).
   Checkpoint-timing overhead and partial-training recovery remain open.
   Verify exact episode quotas, episode IDs/RNG, recurrent resets, checkpoint
   reloads and repeatability. Check native completion timestamps against the
   launch clock. Retain usable completed checkpoints if training later fails,
   and correct the worker-count check's scope: the exact evaluator steps games
   serially. **Deliverable:** retained passing runtime receipts and explicit
   failure handling before using this evaluator for new comparisons.
   [Current implementation and open gates](research/CLAIM_PIPELINE.md).

2. [x] **Replicate corrected Connect4 with ours and Nature across five fresh,
   paired training seeds.**
   **Completed and independently audited:** all ten jobs and 510 exact checkpoint
   evaluations, no missing/failure cells. Quality/Nature final means 76.94%/71.36%,
   mean process seconds 183.382/202.153; ours wins four of five seeds with large
   variance. Descriptive frontier favors ours; certified inference remains open.
   [Results and limitations](research/CONNECT4_DEVELOPMENT_REPLICATION.md).
   **Launched:** `replication-20261005`, seeds 53101–53105, ten jobs and 510
   exact checkpoint evaluations. Automatic independent archive scheduled;
   retain the running status until completion/audit. Do not launch duplicates.
   Freeze both architectures, the common learner/core, budgets, evaluation and
   checkpoint cadence before launching. Preserve every seed, checkpoint and
   failure. The latest timings imply approximately **31 minutes of training**
   for ten runs, plus builds and evaluation. Five seeds is a proposed replication
   stage, not a calibrated publication sample size or promise of significance.
   **Deliverable:** whether the apparent advantage repeats across seeds, with
   complete curves and clearly labeled uncertainty. If this panel informs
   further selection, treat it as development evidence rather than final testing.

3. [ ] **Run a rendering-robustness panel.**
   Predetermine square-with-gaps, circles, X/O and smaller-board presets. Train
   both frozen networks from scratch with matched learner settings, seed lists
   and budgets for each rendering. Keep each representation's frontier separate;
   do not let a sweep choose the easiest appearance. **Deliverable:** whether
   the advantage survives visual changes. Training separately on each rendering
   does not establish zero-shot transfer; mixed-training/held-out-rendering
   evaluation is a separate later experiment with a frozen split.
   [Available representations](ocean/connect4cnn/REPRESENTATIONS.md).

4. [ ] **Run matched multi-seed comparisons on existing FlappyCNN and PongCNN.**
   Start with Flappy as the cheaper next candidate. Resolve Pong's evaluation
   limitations before comparative runs. Use identical learner settings across
   architectures within each task; task-specific recipes may differ, but any
   tuning allowance must be recorded and equal. **Deliverable:** evidence beyond
   Connect4 without first implementing another environment. These existing
   development tasks are not untouched final-test games.

5. [ ] **Audit baseline efficiency and add relevant fast baselines.**
   Check architecture fidelity and profile Nature/IMPALA when exclusive GPU time
   is available. Include a deliberately small conventional CNN and review relevant
   current baselines before making a SOTA claim. Separate encoder architecture,
   recurrent-capacity and implementation-speed effects. **Deliverable:** evidence
   that the result is not an artifact of a weak or inefficient reference.
   Source review can proceed alongside item 1; this gate must pass before final
   claims even though substantial profiling is listed here.

6. [ ] **Support RGB/general image sizes and integrate original Procgen.**
   Start with the original environment's native-interface feasibility probe.
   Validate 64×64 RGB input for all compared encoders. Develop on the proposed
   CoinRun, BigFish, StarPilot and Maze panel, reserving other game identities and
   final-test levels before tuning. **Deliverable:** an established external
   benchmark and a meaningful test of architecture generalization. Narrow claims
   if only a smaller predeclared panel is feasible.
   [Benchmark priorities](research/BENCHMARKS_AND_SWEEPS.md).

7. [ ] **Resume architecture search using verified infrastructure.**
   Validate encoder 5 on the 5090 before searching its operators. Start with
   bounded small networks on development tasks; expand choices based on measured
   weaknesses. Record search cost and preserve failed candidates. **Deliverable:**
   improved candidates selected without using final-test tasks. Keep pretraining
   a separate axis with random-initialization controls and compute accounting.
   [Verification](research/FLEX2_VERIFICATION.md) and
   [bounded search panels](NEXT_5090_FLEX2_SWEEP.md).

8. [ ] **Freeze finalists and execute the publication comparison.**
   Include Nature, IMPALA, Impoola and other relevant controls. Predeclare claim,
   tasks, practical effect size, replication, stopping/failure rules, tuning
   allowances and full-frontier analysis. Validate the uncertainty method first:
   the current candidate bands failed preliminary synthetic coverage checks.
   **Deliverable:** reproducible claims, figures, auditable artifacts, a paper
   draft and a clean PufferLib contribution. Do not add seeds until significant.
   [Paper plan](research/PAPER_PLAN.md) and
   [full-frontier protocol](research/CONNECT4_CLAIM_PROTOCOL.md).

## Immediate sequence and constraints

Recommended next sequence: **1 → 2 → 3**. This either establishes repeatability
and rendering robustness or identifies where the current advantage breaks.
No architecture improvement is required to start these checks.

- Retain the whole observed frontier, declines, failures and losing cases; no
  best-seed envelope or favorable time-window selection.
- Keep corrected and legacy Connect4 rules separate, and keep 5060 and 5090
  measurements separate. Match source/configuration/protocol for cross-host claims.
- Reserve exclusive GPU time for measured runs; the separate 5090 is currently
  busy. The completed local four-model pilot is not permission for every item here.
- No CPU neural-model substitutes or system CUDA changes. CPU environment,
  configuration and reporting checks remain useful.
- Keep production architecture construction/search/training native C/CUDA.
- Preserve [earlier engineering priorities](potential-todos.md), including memory
  estimation, native tooling and additional operators, as supporting work rather
  than treating this roadmap as cancellation of that backlog.
# October 6 short-budget benchmark follow-up

- Preserve the completed [5060 mini panel](research/results/mini-short-5060-20261006/README.md); do not automatically repeat it. All 24 jobs/328 evaluations/48 repeat checks pass, but Connect4/Pong/Flappy score zero at 524,288 decisions. Find the smallest useful learning budget with a separately declared short paired pilot before interpreting a quality frontier.
- Investigate native timing: three logged uptime values exceed full monotonic process durations. The captured trainer uses CLOCK_REALTIME. Preserve all raw metrics, qualify uptime-derived average SPS as unusable for this comparison, and plan a small native CLOCK_MONOTONIC timer change with separate compatibility/receipt checks; no system clock/driver changes. Cause of the observed discrepancies remains unestablished.
