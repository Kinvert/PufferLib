# CNN experiment history

## Completed 5090 campaign received as exports — October8

Pulled `6e30f9bb` in a separate G240 audit worktree; originalb102 sources remain
unchanged. The remote report records192 jobs/5,248 drawing-checkpoint evaluations,
12 native trials plus controls/references and4h54m28s campaign time. Kinvert relays
that the authorized36-additional-trial/12h continuation is now running on5090;
this analysis launches nothing and implements no continuation path.

[Independent export analysis](results/learning-feedback-5090-20261008T173307Z/independent-export-analysis-v2/README.md)
checks all seven tables, once-per-job costs, paired seeds, four-checkpoint coverage
and SPS calculations, then retains every model/point in descriptive per-game
frontiers and a self-contained viewer. Five artifact/scalar checks pass. Final
aggregate frontier: swift-fox-3, happy-cat-1, quality-reference. Quality-reference
scores0.353913 at631.885s versus Nature0.298506 at664.977s, but Nature has better
final Flappy/Snake means. Quiet-owl-11's Maze half-budget checkpoint scores0.234848
at50.841s versus Nature's full-budget0.229798 at120.559s; earlyPong also has a
promising separated censoring-bound pair. These are adaptive-seed discoveries,
not independently audited raw outcomes or confirmed superiority. Point-level
seed/drawing data and candidate configs remain on5090; raw audit and fresh
held-out multi-seed confirmation are still required for paper claims.

## Implemented and locally validated — staged real learning launcher 20261007

[Evidence and exact qualification](results/learning-launcher-20261007/README.md).
Fresh builds register full-budget paired controls/references. Actual RTX5060
checks pass three distinct native feedback proposals across all six games/41
drawings (18 trainings/123 evaluations/36 repeat-eager checks, 208.753s), with
independent CUDA checks before each trial and final GP replay of three scores.
Separate paired quality/Nature execution passes12/82/24 in119.342s. Seven encoder
check allocations total98 process/case instances/392 native calls including
H128/B2048; Nature's current allocation enforces its original stricter tolerance.
Final68 host/scalar/mock tests and4 native host tests pass. The launcher now stops
if any full-budget game control fails its declared learning rule, otherwise runs
12 native architecture trials and paired references with complete curves.
No full-budget 5090 run or learning/frontier win is claimed. Earlier failed H256
comparison remains failed. Read [the handoff](../NEXT_5090_LEARNING_FEEDBACK.md);
do not repeat the completed tiny canary or either completed local allocation.

## Implemented and GPU-free prepared — broader cross-game feedback 20261007

[Actual preparation receipts](results/feedback-development-preparation-20261007/README.md).
The working six-game aggregate/native-PROTEIN feedback now exposes fixed per-game
budgets/cadence and learner geometry, plus all 18 architecture controls with
swept depth. Fifty-one host/scalar/mock tests and four native host checks pass;
fresh bridge/six-game/three-metadata builds and feedback-v2 preparation pass.
A distinct one-stage C8 / three-stage fixture panel registers all 24 policies
and 82 suites across two seeds/six games/all41 drawings, with 656 planned
checkpoint evaluations and zero executed. Native scalar proofs show exact,
positive learner updates with an explicit Maze horizon override. Updated offline
inspection still reads the received feedback-v1 packet. No new GPU query,
training or numerical/learning/frontier result. Recipes/anchors are uncalibrated
examples; no search is scheduled and completed canaries must not be repeated.

## Received — successful native feedback canary on RTX 5090 20261007

[Intake/evidence](results/feedback-canary-5090-20261007/README.md).
Checksum-verified 5090 archive reports all three native-feedback trials complete:
18 training jobs, 123 all-drawing evaluations, 36 exact repeat/eager checks,
2,703 assigned episode executions. Execution 102.862s; training processes 9.109s;
once-per-job checkpoint cost 7.241s; optimizer processes 1.156s. Final unused
native suggestion replays three observations and reports three GP observations.
Local artifact checks independently pass all checkpoint/result/CSV hashes,
weight sizes/finiteness, quotas and repeat/eager byte comparisons, with no GPU.
All normalized proposals round to C16; duplicates are explicit. Same one-seed
architecture and almost-zero scores are plumbing evidence, not a quality/frontier
comparison. Remote source commit bb4d1c10683e06e8373a4f971bee8bc447761279 has
captured auxiliary Pong modifications; native/active feedback source matches
G240. Full math/calibration/learning/diversity/confirmation gates remain.

## Prepared — 5090 cross-game feedback delivery 20261007

[Handoff](../NEXT_5090_FEEDBACK.md). Kinvert authorized delivery to his research
fork so the 5090 can pull and run without code edits. The new launcher separates
GPU-free preparation from explicit bounded `canary5090` execution, rejects
restarts, reuses existing CUDA/NCCL/uv Python 3.12, and archives failures too.
The fresh-checkout reservation-directory bug is fixed. The remote canary adds
mandatory final drawing-zero repeat/eager checks: planned 18 training jobs,
123 drawing evaluations plus 36 checks, 2,703 assigned episodes if completed.

Actual local launcher preparation at
`build/cross-game-feedback/delivery-20261007/` passes 44 host/scalar/mock tests,
three native pre-CUDA validation tests, the native bridge, six normal game and
three metadata builds, and frozen-input preparation/inspection. No GPU query,
neural computation or feedback trial ran. A separate clean Git clone of
`b61127cb979354c9db3e829933dda3c4777a1919`, with a fresh uv Python 3.12 / NumPy
venv, also passes the entire launcher preparation, including ordinary Raylib
dependency download. [Verification receipts](results/feedback-delivery-20261007/README.md)
record 512 immutable inputs/291 owned sources and zero policy executions.
Remote GPU execution remains pending. This supersedes earlier no-delivery wording for the Kinvert
research handoff only, not the minimal eventual upstream PR or any math,
calibration/learning/frontier qualification.

## Prepared — temporary native cross-game feedback 20261006

[Contract](CROSS_GAME_PROTEIN_FEEDBACK.md) ·
[Evidence](results/cross-game-feedback-20261006/README.md).
Macro-gated native PROTEIN bridge plus an external panel supervisor now implement
the adaptive architecture loop. Fixed per-game learners, six games/all 41 drawings,
explicit fixed-anchor score normalization and once-per-job monotonic training
costs. A fresh optimizer process reconstructs the immutable feedback history and
exits before training; the final unused proposal observes the last result.

Native bridge, six normal default CNN targets and three metadata targets compile.
43 host/scalar/mock tests and three native-host tests pass. Mocked three-trial
supervision/feedback and offline audit reject changed history; native descriptor
and ten invalid ledger/control cases return before GPU queries. Actual recipe,
build/source/metadata preparation and offline inspection pass. Initial host-test
failures and first build are preserved. Declared tiny allocation is 18 training
jobs/123 evaluations/2,091 episodes if completed; **zero executed**. No new GPU
query, policy, dataset, learning or frontier result. Keep both holds and all
numerical/calibration/selection gates. No push: prototype is local research;
selected-architecture delivery will use separate minimal normal PufferLib code.

## Prepared — native parameter layouts in candidate panels 20261006

[Contract](CANDIDATE_BASELINE_COMPARISON.md) ·
[Retained evidence](results/candidate-policy-layout-20261006/README.md).
Opt-in `prepare --policy-metadata` now registers full encoder/head/MinGRU layouts
before training; owned production/action source binding and offline raw proof
inspection supplement existing complete recipe/suite checks. Scheduled runs
repeat registration before queries/trainers and enforce the expected raw float32
checkpoint size before reads/evaluation. Offline audit rejects training outside
the registered prefix and reparses every repeated host receipt.

Three fresh metadata targets and 18 normal native game targets compile. Actual
two-seed/five-model preparation passes 60 layouts/jobs, 82 shared suites, 310
native world/raster comparisons and 100 Connect4 declarations. All 820 allocated
evaluations remain unexecuted. Nine new host/config/scalar/synthetic checks plus
20 candidate/front regressions pass in 21.097s; the 23 existing acceptance checks
pass in 34.576s. Tests use mocked manifest/run failures and byte artifacts, not
policies. First tests failed to load NCCL without the normal process-local helper;
original stderr/failure log is retained, with no installation/system change.

Old baseline packet still inspects; the original 12-job/82-eval/24-repeat smoke
re-audits without gaining the new check. No GPU query, model/dataset/learning
run or push; both holds remain. Layout/bytes are not independent numerical math,
full-learner memory, equal-size weight provenance, runtime/toolchain certification,
adequate learning/calibrated caps or selection-safe superiority.

## Prepared — architecture discovery drawing mixtures 20261006

[Contract](DISCOVERY_PIXEL_MIXTURES.md) ·
[Retained INIs/source subset/native scalar CSVs](results/discovery-mixtures-20261006/README.md).
Fixed Flappy catalog-1 rejection and added explicit `sweep.py --appearance-seed`
for full per-slot catalogs. Native game/RNG/encoder/learner code is unchanged;
fixed/legacy defaults are preserved. Recipe freezes before resolution and is
hashed separately; protocol records appearance mode/seed/catalog.

Six CLI preparations retain 18 CNN-only dimensions, H128/L1, per-game learners
and 13.312M decisions, two proposed trials each and zero execution. All 384 native
scalar assignment rows match independent arithmetic; all 41 drawings occur,
with unequal exposure. Twenty-six configuration/sidecar checks and offline full
INI/hash/count reconstruction pass. Preliminary fixture/header-selector mistakes
are recorded separately. This isn't live vector/GPU/model/learning acceptance,
adaptive cross-game search or superiority evidence. Both holds remain; no policy,
dataset, system change or push.

## Reported — arbitrary-candidate full curves 20261006

[Contract](CANDIDATE_FRONTIER_VIEW.md) ·
[Retained views/source/receipts](results/candidate-viewer-20261006/README.md).
The shared HTML viewer previously recognized only four fixed model names;
custom candidate-panel IDs could be retained in CSVs but omitted visually.
It now reads the declared allocation catalog and preserves arbitrary names,
reference aliases, missing/losing models and stable colors. Hiding a model
doesn't recalculate stored front flags or remove allocation/coverage.

The new offline candidate bridge invokes the original raw assigned-episode/
clock/config/input auditor before rendering any scores. Actual five-model
preparation displays all 820 missing cells/41 conditions with zero observations;
the existing two-candidate smoke re-audits/displays all 82 observations, 12
training jobs and 24 repeat/eager checks. Costs total 9.177963895s once per job.
Smoke marks stay null; no baseline scores or training are added retroactively.
Pong bounds remain separate, and parameter/quota/checkpoint identities are kept.
Native SPS remains uncollected; process SPS isn't substituted for it.

Twenty-eight Python artifact/scalar tests pass in 3.939s, including the 20
existing reporting regressions; all 14 Node display/minimal-DOM checks pass.
Fixtures test allocation, mandatory audit/failure retention, missing seeds,
cost reuse, full declines/bounds, a fifth missing candidate and safe model text.
They don't test neural math or browser layout. Standalone artifact verification
matches the retained data/HTML/source and reconstructs scalar view fields;
native weights/binaries/raw runtime remain separately retained evidence.
No GPU query, policy, dataset, system change or push. Both holds and numerical/
runtime/memory/evaluator/calibration/selection/frontier inference gates remain.

## Prepared — native checkpoint-layout gate for shared acceptance 20261006

[Contract](CHECKPOINT_LAYOUT_GATE.md) ·
[Retained receipts](results/checkpoint-layout-20261006/README.md).
Three fresh native metadata targets compile with explicit sm_120. New opt-in
registration/stat modes support legacy configurable encoder 4; older modes
keep their frozen-quality gate. Native construction/registration creates shapes
only, with no CUDA allocation/query, weight initialization or neural computation.
Existing dynamic-link dependencies are not fully certified; production source
and individual game/candidate launchers remain unchanged.

Shared evaluator packet-v2 freezes tools/owned source/action headers and checks
raw checkpoint bytes against actual encoder/head/MinGRU registration before
GPU adapters. Scheduled execution repeats the host check; offline audit reparses
its native descriptor/process/input proof, not merely outer hashes. Unsupported
state/other graphs/encoder 5 fail without falling back to v1. Older packets and
their source snapshots are preserved, including source-matching verifier limits.

Final 23 host/configuration/scalar/synthetic acceptance checks pass in 33.910s;
seven existing metadata regressions pass in 10.040s with all 175 calls retained.
Independent scalar counts cover five pending game heads, a three-stage skip/
pool/GAP/projection graph and Nature/IMPALA/Impoola. Wrong length/core bytes,
matching-size NaNs, rehashed proof edits and pre-GPU failures retain evidence.
Test tensors/CSV actions are synthetic artifacts, not policy outputs.

Actual old Flappy quality final checkpoint prepares/inspects at 160,096
parameters/640,384 bytes under the full executed INI. Four unexecuted modes
retain 65 development IDs/64 slots, with ID64 alone in its independent tail.
Artifact archive/relocation inspection passes; no policy score, learning or
runtime binary provenance is certified. No GPU job, dataset, system change or
push. Both GPU holds and numerical/memory/evaluator/calibration/selection/
frontier inference gates remain. A byte/layout match is not model acceptance.

## Prepared — arbitrary candidates with native references 20261006

[Comparison contract](CANDIDATE_BASELINE_COMPARISON.md) ·
[Retained evidence](results/candidate-baselines-preparation-20261006/README.md).
Opt-in panel-v2 adds exact Nature/default, IMPALA and Impoola native selectors
alongside every shared custom architecture. Eighteen normal float32 targets
compile with explicit sm_120; no production game/encoder/learner change.
Actual two custom candidates plus three references across six games/two seeds
prepare 60 jobs, 82 shared suites and 410 drawing bindings at two checkpoints
each (820 evaluations/13,940 episode executions declared). All 310 native
family/config world/raster comparisons pass; 100 Connect4 bindings retain the
separate declared empty-board/identity-seed proof. Native scalar geometry and
mixture checks pass. Host starts aren't policy episodes; live assignment and
model math remain separate acceptance gates.

The v2 inspector reconstructs complete frozen per-game recipes and exact
binary/family/output bindings, rejecting all-family rehashed learner edits,
secondary overrides and baseline relabeling. Twenty new host/config/scalar/
synthetic tests pass (4.626s), as do 20 reporting regressions (3.640s). The new
offline report retains all paired-seed/checkpoint/drawing curves, missing cells,
declines and separate Pong bounds without partial-seed frontiers. The actual
prior 12/82/24 smoke passes the current reader and original artifact verifier.

Artifact-only archive/relocation verification passes with 18 build receipts,
41 incomplete conditions and 820 missing observations. The first verifier's
manual four-model count expectation failed and is retained; expected counts
were corrected to the actual five models without changing the prepared panel
or archive. Declared 3,932,160 decisions, short caps and budgets are uncalibrated
plumbing examples, not a learning campaign. Zero new jobs/evaluations executed;
no GPU query, model/library load, dataset, system change or push. Numerical,
memory/reload/evaluator/calibration/selection/inference gates and both GPU holds
remain. No quality superiority or cross-game adaptive optimizer is established.

## Completed — cross-game candidate smoke 20261006

[Protocol](CROSS_GAME_CNN_SWEEP.md). User explicitly authorizes bounded local
5060 training/inference; no encoder-5/5090/full campaign or push implied.
Eight host/config/scalar tests and the existing 20 reporting regressions pass.
Six fresh normal float32 native builds compile with explicit sm_120; owned
sources/compiler/environment/commands/binary identities are retained. Initial
build and revised-auditor build are preserved separately; no native production
code was changed by this task. System/toolkit/dependency closure is incomplete.

Actual host preparation: two encoder-4 candidates, H128/L1, six games, 41 fixed
development suites, 12 mixed-training jobs/82 evaluation targets at one 65,536
decision checkpoint each. Native scalar checks verify positive matched learner
geometry and deterministic full-catalog assignments. Seventeen episode starts,
16 slots, evaluation seed 67173, appearance seed 35173; per-game recipes fixed.
Short Pong/Breakout caps are uncalibrated plumbing controls. Live vector assignment
isn't independently accepted. Graph repeat/eager adds 24 drawing-0 evaluations.

Initial paths: `build/candidate-panel/native-20261006-v2/registry.json`,
`build/candidate-panel/smoke-20261006/plan.json` (retained failed attempt).
Don't call prepared starts policy episodes or smoke scores a frontier result.

The first smoke stopped at native parsing (`missing section [sweep]`), with
zero completed training jobs/evaluations. Preserve that original packet/process/
failure. Preparation now retains required `[sweep]` defaults while removing only
adaptive dimensions; a regression checks their presence. Use fresh v3 builds
and a fresh v2 smoke packet, never overwrite/relabel the failed attempt.

Actual fresh `native-20261006-v3` / `smoke-20261006-v2` execution and strict
offline audit now pass **12/12 training jobs, 82/82 all-drawing evaluations,
24/24 graph-repeat/eager exact-byte checks**. All 1,394 drawing episodes plus
408 repeat/eager episodes complete at the fixed 17/16 quota/batch. Successful
campaign window: 89.623248519s; native training processes total 9.177963895s.
No long run remains active. Reports preserve native metrics plus separately
labeled process SPS and all scores/bounds; the 65K-step results select no winner.

Final nine host/config/scalar tests also prepare the 18-CNN-dimension native
discovery recipe for all six games and verify every per-game non-budget train
setting stays unchanged. The existing 20 reporting regressions pass. Actual
old native `sweep.jfu9hfqr` import preserves every completed trial: 12 candidates,
72 planned jobs/492 drawing targets, zero new executions. Native PROTEIN remains
single-game; no cross-game adaptive objective is implemented.

[Retained report/source/receipts](results/candidate-panel-20261006/README.md)
bind the full raw artifact under ignored `build/hardware-artifacts/` by SHA256.
No production encoder/game/learner/system dependency changed, no dataset was
generated, no push occurred. Numerical/calibration/held-out/publication gates
remain; the completed plumbing smoke isn't a new Pareto or SOTA result.

## Prepared — native full-policy metadata 20261006

[Contract](POLICY_CHECKPOINT_PREFLIGHT.md) ·
[Receipts](results/policy-metadata-20261006/README.md).
Three isolated targets compile with explicit `sm_120`, unchanged CUDA/NCCL.
Actual `build_arch`/`weights_create` register encoder, decoder and MinGRU
parameters without CUDA data allocation/init or computation. Seven host tests
pass (initial 9.739s, final 9.738s), with 144 game/model/H64/128/256/L1/2 cases
matching independent scalar C counts and layouts. All 175 native host calls
(152 accepted/23 rejected), 30 scalar count configurations/120 rows, action
headers and source/command/compiler/binary identities are retained.
The existing Flappy quality checkpoint has 160,096 parameters/640,384 bytes
under its actual merged metrics INI; the incomplete default file alone is
rejected. No weights are loaded into a model, no numerical test/policy/game/GPU
executes. Stat-only size checking intentionally cannot detect equal-size NaNs
or prove recipe/architecture identity. Launchers/production kernels are unchanged;
vendor/system/link closure is incomplete. Both holds and broader gates, no push.

## Prepared — interactive multi-game reporting 20261006

[Contract](PIXEL_FRONTIER_REPORTING.md) ·
[Retained viewer receipts](results/pixel-frontiers-viewer-20261006/README.md).
The offline reporter now embeds audited JSON and display code into `curves.html`:
all four CNNs, condition/seed/time/decision controls, every chronological curve,
missing checkpoint gaps and model coverage. Mean time-front markers use stored
flags only for complete conditions; hiding models does not recompute the front.
Pong lower/upper censoring curves stay separate, without midpoints or intervals
presented as confidence. Twenty Python artifact/synthetic checks and ten Node
display/minimal-DOM checks pass. These do not certify browser layout, native
model math or learning. Current actual multi-game ledger stays at 24 jobs/328
planned evaluations/zero observations, 41 incomplete conditions. Kinvert says
six environments are enough for now. Native policy/learner code and old archives
remain unchanged; both GPU holds, no GPU query/model/dataset/pretraining/push.
The related report/catalog/binding/transfer suite passes all 45 checks in
21.758 seconds. Actual preparation/reporting confirms 328 missing cells and
zero observations; the original reporting archive remains hash-intact.

## Prepared — paired numerical protocol v2 adds common learner batch 20261006

[Contract](DENSE_ALIAS_ACCEPTANCE.md) ·
[Receipts](results/dense-alias-acceptance-20261006-v2/README.md).
The same two compiled native libraries now have a v2 external test declaration
including B2048: quality/Nature × H64/128/256 × B1/3/64/2048 × three states,
72 fixtures/four workers/four modes/1,152 planned native harness calls, zero
executed. Native code, tolerances and production training are unchanged by
this expansion. Actual preparation/inspection and 39 host/synthetic tests pass
(1.377 seconds); original 54-case v1 source/packet/archive remains unchanged
and verifies. Owned native/tool closure is captured; vendor/system/link paths
are recorded but not fully copied/hashed, so no full toolchain certificate.
No library/model/oracle load/execution, GPU query, training, game dataset or
pretraining occurred. Larger batches, general callbacks/encoder5, full-policy
concurrency/reload, timing/learning/statistics remain gates; both holds, no push.

## Prepared — paired dense-copy numerical supervisor 20261006

[Contract](DENSE_ALIAS_ACCEPTANCE.md) ·
[Receipts](results/dense-alias-acceptance-20261006/README.md).
Two fresh baseline/candidate test libraries compile at explicit sm_120, with
native variant markers and device-parameter exports. Explicit-output builds
freeze source/compiler/environment/command/binary receipts and refuse overwrite;
legacy default build paths retain their behavior. Actual preparation/inspection
passes 54 fixed quality/Nature fixture declarations across H64/128/256 and
B1/3/64, three input states, four independent workers and four modes: 864
planned test-harness calls, zero executed. The equal-projection H64 branch is
explicitly included. All parameters/gradients and independent references would
be retained with original tolerances; oracle computation exists only in the
scheduled native-GPU worker, never a standalone CPU model test. Offline auditing
checks seeded fixtures/finite arrays/parameter mutation/signed-zero byte
differences and exact native-process commands/cwd/terminal clocks/completion.
First 38 and final 39 host/synthetic tests pass (final 1.673 seconds), with no
model/library load, GPU query, training, dataset or pretraining. Test processes/
queries/arrays are mocked/synthetic, not numerical evidence. All GPU paths,
general Flex/compact/encoder5, larger batches, whole-policy/concurrency/reload
and timing/learning remain open gates; both holds remain, no push or Pareto point.

## Prepared — isolated dense-input copy candidate 20261006

[Candidate and gates](DENSE_PATCH_ALIAS_CANDIDATE.md) ·
[Receipts](results/dense-patch-alias-20261006/README.md).
Profiler/test-only CUDA callbacks replace identity dense im2col with local
views of retained activations in quality and Nature, preserving owned buffers,
parameters, GEMM wrappers and original gradient gathers. IMPALA/Impoola remain
unchanged controls. Six profiler targets and two isolated test libraries compile
at explicit sm_120; 48 paired host descriptors match all registration fields.
Both variants pass 48 independent scalar-registration, ten invalid-input and
nine host-hash checks, and each prepares a complete 64-case timing packet.
Initial independent audits failed with an older-schema local calculator;
failures/partial artifacts are retained, then current C source was recompiled.
Seven scalar/build-guard plus 17 host/synthetic supervisor tests pass. Candidate
metadata is explicit in host and future runtime receipts, with mismatch rejection.
No library was loaded and no GPU query/model execution/training/dataset ran.
Owned-buffer saving is zero, timing/SPS/score are unmeasured. Direct paired GPU
math, equal-projection H64 branch, full-policy concurrency/reload and timing
remain gates; production encoder/learner unchanged, both holds remain, no push.

## Prepared — offline multi-game frontier ledger/reporting 20261006

[Contract](PIXEL_FRONTIER_REPORTING.md) ·
[Receipts](results/pixel-frontiers-20261006/README.md).
An audited binding packet now allocates null-valued per-job training receipts
and every declared checkpoint/target result. Offline reporting audits exact
training/resolved/secondary configs, binary/weights, strict native post-rename
monotonic checkpoint/process clocks, every assigned-ID CSV and completion summary.
Hardware/toolchain mixing and overlapping timed training are refused; native
SPS remains explicitly unavailable. All game/appearance/seed/checkpoint cells,
failures and declines stay separate. Complete paired means are equally weighted;
incomplete conditions have null frontier flags. Pong retains deterministic
censoring bounds and no game-score averaging/dominance inference is enabled.
The related 42-test suite passes in 33.772 seconds; final 17 focused tests pass
in 5.503 seconds, including ten graph/eager prepared configurations from the
five pending adapters with mocked metadata/spawns and synthetic outcomes/weights.
The earlier one-error report retains a Snake fixture's invalid `2048.0` CSV
integer; the fixture alone was corrected. Actual six-game/41-drawing preparation
retains 24 training jobs and 328 checkpoint evaluations, zero observations and
all 328 explicitly missing. No GPU query, policy/model, training, dataset or
pretraining ran; both holds and acceptance/calibration/inference gates remain.
No native kernel/training path changed, no new Pareto claim, no push.

## Prepared — Flappy geometric variations and frozen catalogs 20261006

[Contract](FLAPPY_GEOMETRIC_ROBUSTNESS.md) ·
[Receipts](results/flappycnn/geometric-preparation-20261006/README.md).
Rounded bird/outlined pipes/both extend Flappy to seven IDs and current six-game
inventory to 41 drawings. Game semantics and tensor shape are unchanged;
default mixing keeps the old four IDs, explicit expanded mixing uses seven.
Environment/raster/parity/repeatability/ASan/UBSan pass, including an independent
21-frame geometric oracle. Old four pixel streams match their previous hashes
over 49,152 frames each. All four CNNs and original state compile; 107 host/
configuration/native-manifest checks pass in 57.908 seconds. Earlier three
old-count expectation failures and 21 missing-process-library launch errors
are retained beside the final report. Captured-header catalog auditing keeps
four actual historical packets readable unchanged. Actual mixed preparation
retains 24 jobs, 41 fixed-target suites, 164 bindings and 124 non-Connect4 native
world/raster comparisons; two plumbing checkpoints mean 328 planned evaluations,
zero completed. Source/build snapshots precede subsequent test expectation
edits and retain their exact ledgers. No neural model, GPU query/inference,
training, dataset or pretraining executed. Both holds and acceptance/learning/
inference gates remain; this is not a robustness/Pareto/SOTA result. No push.

## Prepared — deterministic mixed training and fixed-ID bindings 20261006

[Contract](MIXED_PIXEL_TRAINING.md) · [Receipts](results/pixel-mixed-20261006/README.md).
Panel-v4 now exposes persistent native per-slot drawing mixtures shared across
every CNN, with explicit seed/catalog/count receipts and native scalar learner
validation. Pong uses all seven IDs; historical default mixing remains five.
Actual six-game/two-seed preparation passes 48 planned jobs, 76 fixed-target
suites of 1,000 assigned starts, 304 target bindings and 224 native per-family
world/raster checks. Every checkpoint targets all drawings; matched mixed mode
is rejected. Two plumbing checkpoints mean 608 planned evaluations, zero
completed. The full 60-test host/config/scalar suite passes (28.270 seconds),
including independent unsigned/rejection arithmetic under UBSan and twelve
native invalid-input rejections. Older packets inspect unchanged. Mixture
counts are unequal and shared between both learner seeds; live vector/reload
is explicitly unvalidated. No production policy/game/learner code changed,
and no GPU, neural model, training, dataset or pretraining executed. Both holds
and runtime/math/memory/calibration/inference/selection gates remain open;
this is no learned robustness, frontier or SOTA result. No push made.

## Prepared — same-policy cross-drawing bindings 20261006

[Contract](PIXEL_APPEARANCE_TRANSFER.md) ·
[Durable receipts](results/pixel-transfer-bindings-20261006/README.md).
Explicit all-drawing evaluation now binds the same original training job and
complete checkpoint list to every drawing of its task. Native host preparation
passes six games, 24 drawing-0 jobs, 38 suites of 1,000 assigned starts, 152
target bindings and 112 per-family world/raster comparisons. Two checkpoint
steps per job give 304 planned evaluations, zero completed. Fifty host/config/
scalar/binding tests pass (22.648 seconds), including seven new transfer checks;
untouched binding-v1 and per-game-budget packets still inspect successfully.
No policy, GPU query, training, dataset or pretraining ran. Both holds and
runtime/math/memory/reload/calibration/inference/selection gates remain open.
All targets of a checkpoint share one training-time receipt and are correlated;
they are not additional training seeds. This is allocation/starting-state
evidence, not a learned transfer score or Pareto/SOTA claim. No push made.

## Prepared — six-game task budgets and complete checkpoint binding 20261006

[Contract](TASK_BUDGETS.md) ·
[Durable source/config/start receipts](results/task-budgets-20261006/README.md).
Fixed-panel preparation now accepts per-game decisions/checkpoint cadence with
identical settings across all encoders/drawings/seeds within that game. v3 has
resolved/default/override receipts and native scalar geometry for every task;
old v1/v2 archives still inspect without modification. 43 host/config/scalar
tests pass (18.465 seconds), including 304 two-seed cells, strict integer types,
partial/final cadence, CLI failures and GPU-free checkpoint binding.
Actual six-game/38-drawing/152-job preparation passes 38 assigned-start suites
and 112 native per-job comparisons; Connect4's declared empty-board manifest
stays separate. 5,836 checkpoint bindings and 10,496,245,760 training decisions
are **declared only**, using one development seed and uncalibrated recipes/
resource scales. Do not launch this large design as a confirmation campaign.
No policy, GPU query, dataset, training, SPS or score was produced. All runtime/
learning/inference gates and both GPU holds remain.

## Prepared — learner memory restriction and shared smaller batch 20261006

[Findings](LEARNER_MEMORY_PREFLIGHT.md) ·
[Receipts/table](results/learner-memory-20261006/README.md).
Three new native metadata targets compile. Host `--describe` alone now allows
batches through 65,536; GPU `--run` remains capped at 2,048/on hold. 72 actual
registrations match C counts, ten native rejection and nine scalar hash checks
pass; current parameter counts match 52 retained rows/Flappy's separate anchor.
Stock Flappy IMPALA/Impoola encoder training tensors alone require
16.950/16.894 GiB, ruling out the unchanged recipe on nominal 8 GiB hardware.
This isn't an observed OOM or intrinsic architecture-memory claim. A stock
overlay differing only in shared minibatch 2,048 now prepares twelve drawing-0
jobs across all four frozen CNNs and three paired seeds. 19.923M decisions,
19 checkpoint steps, 9,728 updates versus stock 1,216; zero trained.
35 learner/config/scalar tests pass (14.337 seconds), sixteen host/synthetic
profiler tests pass (0.969 seconds). Full fit/math/evaluation/learning and both
GPU holds remain. The corrected offline table preserves its initial missing-
protocol-key failure; historical panels/receipts aren't rewritten.

## Prepared — explicit shared Flappy learner recipes 20261006

[Contract](SHARED_LEARNER_RECIPES.md) ·
[Retained evidence](results/shared-learner-recipes-20261006/README.md).
Fixed-panel preparation now accepts per-game numeric learner overlays with
native scalar preflight, frozen/repository build ledgers and full recipe closure.
Four CNNs/H128-L1 remain unchanged and randomly initialized. Two Flappy panels
each contain 48 planned jobs: four drawings, three paired development seeds,
19,922,944 decisions and 19 checkpoint steps per job. Stock-learner arithmetic
is 1,216 optimizer updates versus small-batch 9,728; this is two shared regimes,
not identical regimes between candidates. All encoders within each candidate
receive identical settings. 34 host/config/scalar tests pass (14.117 seconds).
Two twelve-suite preparations and 96 per-job native starting manifests pass;
12,000 paired start rows are byte-identical across recipes. These are starting
states, not played episodes. **Zero jobs trained; no new SPS or score.** Both
GPU holds and evaluator/math/memory/learning/inference gates remain. Earlier
prototype packets and all historical learning evidence are preserved.

## Prepared — native learner geometry and calibration 20261006

[Design and findings](LEARNING_RECIPE_CALIBRATION.md) ·
[Durable receipts](results/learner-geometry-20261006/README.md).
Native C scalar arithmetic uses the existing INI parser; no CPU policy/model,
GPU query, optimizer, training or dataset ran. A source/config/compiler audit
covers 166 current configuration cases, including all 152 paired preparation
jobs with identical positive effective updates and exact budgets. Stock Flappy
executed INIs reproduce 19.923M decisions, 152 rollouts, 19 checkpoints and
1,216 optimizer updates; the small-batch template would use eight times more
updates at that decision count. Float32 replay truncation can mean zero updates;
partial passes visit a deterministic row prefix. Ten host arithmetic/UBSan
tests pass (3.921 seconds), including a 304-job two-seed fixture and independent
hand-calculated control counts. No production game/learner/encoder changed.
The plan preserves shared settings within every encoder comparison and
separates recipe effects, prior censored Pong results and future calibration.
Both GPU holds, evaluator/numerical/learning/inference gates remain; no new
SPS, score, frontier improvement or SOTA follows from arithmetic checks.

## Prepared — six-task paired evaluation bindings 20261006

[Design and commands](PIXEL_EVALUATION_BINDINGS.md) ·
[Durable receipts](results/pixel-evaluation-binding-20261006/README.md).
No GPU, policy, training, dataset or production-kernel change. Complete
captured-recipe audits and binding tests pass (14 tests, 7.240 seconds);
the full two-seed configuration fixture covers 304 fixed-model cells.
Actual host preparation binds 152 planned jobs across six tasks/38 drawings
to 1,000-ID development suites, retaining all planned checkpoint steps and
model/build/config identities. The 112 non-Connect4 job manifests match
canonical native world/raster starts; cross-drawing worlds/RNG/levels match.
Connect4 uses its declared empty-board identity manifest, explicitly without
a new native host spawn/raster proof. Offline inspection/executable-byte checks
pass. Caps and common recipes remain uncalibrated; pending evaluator GPU
acceptance/inference/replication gates and both GPU holds remain. No learning
score, SPS, frontier improvement or SOTA follows from preparation.

Later configuration follow-up: the native loader's secondary game INI could
override an otherwise audited default. Two old-auditor regression cases now
fail as expected; current 15-test suite passes (8.015 seconds). Stricter current
inspection also accepts the untouched actual six-task packet, whose second
INIs are already empty. [Separate receipts](results/pixel-evaluation-binding-audit-20261006/README.md)
retain the gap/fix without rewriting initial tools, tests or native manifests.

## Prepared — exclusive shared GEMM supervisor 20261006

`research/workspace_supervisor.py` wraps the unchanged native 96-case packet.
GPU-free preparation/inspection freeze native source/compiler/dependency/tool
receipts. Scheduled execution implements the shared GPU reservation, hardware/
busy/query checks, bounded native process groups, strict live command/cwd/PID/
monotonic-clock checks, immutable inputs/tools and earlier completed outputs.
All six groups of sixteen cases must preserve raw input/reference/actual
SHA256 across schedules/modes/workspaces/process repetitions and one stable
runtime/driver/cuBLAS identity. Failure/timeout evidence remains without a
successful aggregate; no subset launch, automatic retry or overwrite mode.

Thirteen host/control-flow tests pass (33.710 seconds). The first pass exposed
JSON tuple/list roundtrip incompatibility in aggregate audit; it is fixed by
using lists explicitly, with the original failure retained. Tests include
synthetic full-panel comparisons, live cwd versus archive relocation,
earlier-result/plan/binary tampering, query/reservation/deadline/clock/command
failures and current/relocated offline auditing. Full-panel array auditing
uses explicit doubles; the earlier ten preparation/export tests remain its
separate schema/numerical-consistency coverage. One plain host child verifies
process-group timeout cleanup; no CPU model or matrix product ran.

The actual final supervised plan and GPU-free inspection pass. No GPU query,
native GEMM, inference, training or dataset generation occurred. Production/
native code is unchanged; no rebuild was needed. All actual GPU runtime/math/
full-model and cross-game learning/frontier gates remain open, with both holds
preserved. [Contract](SHARED_GEMM_ACCEPTANCE.md) and
[retained regression/test/source/preparation evidence](results/workspace-supervisor-20261006/README.md).

## Prepared — shared GEMM main/dW workspace acceptance 20261006

An isolated native float32 test compiles actual `puf_mm`, asynchronous
`puf_mm_tn` and `puf_mm_nn` paths with native event fork/join. Two disjoint
buffer lanes exercise queued side-stream gradients versus main-stream work.
Serial/overlap submission and eager/graph modes compare default workspace
behavior against reapplying each handle's distinct aligned 32 MiB workspace
after stream selection. Only this translation unit intercepts API calls;
production helpers/encoders remain unchanged.

Six representative odd/projection/quality/Nature/IMPALA/learner GEMM shapes
have independent GPU scalar-double references using exact dyadic input
arithmetic. Warmup and three replays must match references numerically and
repeat raw bytes; all inputs remain unchanged. These CUDA paths are compiled,
**not executed**. Exact dyadics don't certify general FP32 accuracy, full
encoder gradients, actual kernel overlap or a systems speed advantage.

Ten host/synthetic checks pass in 7.103 seconds. They execute native metadata/
early rejection only, verify shape/exactness/96-case configuration, provenance/
dependency/input closure, relocation and fabricated export failure cases.
The final 96-case packet freezes source/compiler/tooling/binary/dependency
receipts without GPU discovery or policy execution. Export auditing does no
CPU matrix product and retains false certification flags. A complete exclusive
runtime/process-group/paired-SHA supervisor is still to be implemented;
do not launch bare native execution.

No GPU query, training, inference or dataset generation occurred; both holds
and encoder-5 gates remain. This changes no observed frontier. Initial compile
failure and final source/build/host receipts are preserved separately.
[Contract](SHARED_GEMM_ACCEPTANCE.md) and
[durable evidence](results/workspace-acceptance-20261006/README.md).

## Prepared — shared five-game evaluator acceptance 20261006

`research/eval_acceptance.py` freezes existing checkpoint cases for Flappy,
Pong, Breakout, Snake and Maze: graph/repeat/eager plus an independent partial
tail at the original batch size. Every assigned ID/start/RNG/action receipt
must match; complete effective INIs agree after relocating output paths only.
Source/input/output hashes and native clocks/parameter bytes are retained.
Each game retains its native supervised reservation/timeout/failure handling.
Connect4's accepted checker and encoder-5 math gates remain separate.

Fourteen host/synthetic tests pass (35.417 seconds). Actual host manifests run,
but policy launch/GPU discovery are forbidden during preparation. Tests reject
same-score action drift, rehashed world/vectorization/learner drift, altered
inputs/quotas/paths/preparation flags, partial failures and tampered outputs.
Unchanged archived evidence audits after relocation without rewriting INIs.
Synthetic weights/receipts are fixtures, not neural execution or learned scores.

Actual completed Flappy quality/state final weights and executed full INIs
have eight prepared modes, seed 56119, 65 IDs/64 slots, independent tail ID 64
at the same slot count. Weight-byte totals are 640,384/100,608; shape/math and
GPU reset/reload/quota remain unvalidated. Preparation and relocated inspection
pass with execution/math/frontier flags false. No reference-family Flappy or
other-game policy checkpoint was substituted or created.

No GPU query, inference, training or dataset generation occurred. Both holds
remain; this does not change the observed learning frontier or establish
multi-task superiority. [Contract](EVAL_ACCEPTANCE.md) and
[durable source/logs/prepared inputs](results/eval-acceptance-20261006/README.md).

## Prepared — Flappy checkpoint supervision and configuration integrity 20261006

Three preserved pre-fix regressions fail against the older Flappy loader:
changed source INI, changed manifest INI and rehashed physics drift were accepted.
New configuration receipt 2 hashes both INIs and reconstructs expected settings
from frozen source/suite. Legacy suites remain readable with original source
hash and derived-manifest checks, explicitly labeled as missing an original
manifest hash. Historical files/scores aren't rewritten or upgraded.

`run --prepare-only` preserves every architecture/core/world/native-cap setting,
copies immutable inputs/tools and verifies actual native starts without GPU
queries. Empty/malformed/nonfinite weights, reservation/query failures, changed
effective/copied/source files/tools, bad clocks/quotas/summary/parameter totals
retain failed receipts. Offline `audit` checks all assigned IDs; floating means
use order-independent compensated summation and runtime certification stays
false. Policy inference remains native C/CUDA; no game/learner/kernel changed.

Twenty-one host/synthetic tests pass on old and fresh native targets; four
normal float32 builds compile with source/hash receipts. Four original 1,000-ID
suites satisfy stricter legacy closure. Maze's 14 and panel's two checks pass.
Using archived-executed INIs (byte-identical to local completed-pilot logs),
the actual 19.923M-decision quality H128/L1 and stock state H64/L2 final weights
have fresh GPU-free 65/64 acceptance packets. Four 1,000-ID quality drawing
suites plus the paired state suite are also prepared. Weights match historical
parameter byte counts (160,096 quality / 25,152 state); no neural shape/math
execution or new score is claimed. Drawings 1–3 would test transfer of weights
trained on drawing 0, not separate scratch learning on those drawings.

No training, inference, GPU query or dataset generation occurred. Both holds
and encoder-5 gates remain. Future first acceptance can reuse those weights;
reference CNN Flappy checkpoints are still missing. Fresh six-task packets
capture the current wrapper source, not the earlier Maze-only snapshots.
[Contract](../ocean/flappycnn/DETERMINISTIC_EVAL.md) and
[retained failure/build/preparation evidence](results/flappycnn/exact-launcher-20261006/README.md).

## Prepared — Maze deterministic evaluation and checkpoint supervisor 20261006

The dedicated native adapter now assigns the same original level and RNG start
by episode identity across all policy/core sizes. Evaluation ends at the first
goal or native area timeout; every assigned episode contributes once. A goal
exactly at timeout retains two native log entries but counts as one success.
Stock training, original Maze and CNN math are unchanged.

Four normal float32 targets compile. Independent CPU game/counter/reference/
sanitizer checks pass for all six drawings, including dirty/reset/cyclic/prefix
allocation, invalid actions and goal-at-timeout. Fourteen native-host/synthetic
supervision tests pass: 24 pixel family/core and one state preparation, exact
quota/summary/parameter/clock, immutable inputs, reservation/query failures and
retained partial timeouts. Host suites retain the same 1,000 assigned starts
per drawing across three builds/H512-L4/state/independent tails, using the
original full 8192-level table and an explicit 1024-level panel.

`run --prepare-only` freezes matching training graph/core, checkpoint byte
preflight, suite, tooling and evaluation INI without querying a GPU or executing
a policy. A held-out-prefix INI may expand only its evaluation table; the source
training INI is preserved. This is a verified configuration declaration, not
verified checkpoint training/tuning exclusion. All reports retain false GPU/
held-out certification flags. Synthetic weights/receipts are not learned policies.

Both GPU holds and encoder-5 gates remain. No Maze learning/SPS or new frontier
claim exists. Current six-task/38-condition preparations use gate
`maze-exact-compiled-gpu-validation-pending`; earlier environment packets and
unimplemented gates remain historical. [Instructions](../ocean/mazecnn/DETERMINISTIC_EVAL.md)
and [durable receipts](results/mazecnn/exact-preparation-20261006/README.md).

## Prepared — native Maze pixels and six-task panel 20261006

MazeCNN adds navigation/local vision with six direct-buffer drawings and
reuses all existing CNN kernels. Original Maze header/INI, generation/RNG,
movement/reward/timeout, both terminal level draws and goal-at-cap duplicate
logging are preserved. Unsafe configuration inputs are rejected before casts/
allocations. Core changes only extend pixel encoder/encoder-5 sweep allowlists.

CPU game/raster/reference/BFS/worker-order/ASan/UBSan tests pass: 49,152 supplied
decisions per panel, all six fixed and three mixed appearances, no policy.
Four normal sm_120 float32 targets compile; 22 sweep/config and two six-task
panel tests pass. Fresh packets preserve 24 default / 152 all-appearance jobs,
18 builds per seed, common fixed quality graph and within-condition learner/
budget/seed settings. All 304 two-seed fixture cells are checked. Three native
PROTEIN trial configs are prepared only. Source/config/binary/dependency hashes,
raw traces, first fixture failure and passing reruns are
[archived](results/mazecnn/environment-20261006/README.md).

No GPU query or neural execution, no dataset generation and no new learning/
SPS/frontier result. Both GPU holds and encoder-5 gates remain. Dedicated Maze
evaluation/held-out level allocation is unimplemented; preparation explicitly
marks that gate. Next Maze work is exact single-episode terminal capture and
policy-independent level/start allocation, followed by scheduled GPU acceptance.

## Prepared — Pong texture and five-task appearance panel 20261006

Two new native reversible checker/contour drawings bring Pong to seven presets
and the five-task inventory to 32 conditions. Legacy fixed images and mixed
five-ID assignments are preserved; expanded mixing requires catalog 1 and is
reported separately. No policy/backend/physics/system change or GPU execution.
CPU world/raster/reference/sanitizer checks pass for all drawings/catalogs;
four normal float32 builds compile. Fourteen exact host/glue tests pass on old
and new binaries; 1,000 assigned starts per drawing match across three pixel
build families, H512/L4, state worlds and independent tails. Old five CSVs
are byte-identical. Twenty-one sweep/sidecar and two panel tests pass.
Fresh default/all-appearance preparation packets contain 20/128 jobs and
15 builds each. Source/config hashes match their snapshots and this worktree;
neither packet was executed. Their small budgets remain plumbing only.

No neural validation, SPS, learning curve or new Pareto claim exists. Both GPU
holds remain. Initial missing-NCCL-loader failures and corrected host reruns
are preserved in [the source/trace/log archive](results/pongcnn/texture-preparation-20261006/README.md).

## Built — supervised native encoder profiler preparation 20261006

New native receipt exports retain input/upstream/weight/output-gradient bytes
outside event timing. Host hashing returns before CUDA/encoder construction.
Three updated targets compile. Registration audit: 36 shape comparisons, seven
rejections and nine empty/known-answer/block-tail file hashes pass. Actual
GPU-free preparation freezes 64 cases (four fixed encoders × two phases ×
eager/graph × default/reassigned workspace × two independent repetitions).

Sixteen host/synthetic tests pass: immutable sources/binaries/configs, old-interface
rejection, all receipt shapes/modes/counters/durations, missing/mismatched panels,
lock/query/partial failures and a real host-only timeout/process-group cleanup.
Fixed and retained a Python stdlib-shadowing regression by naming the runner
`encoder_profile.py`. Native policy execution remains C/CUDA; Python is temporary
external preparation/audit supervision. No CPU/GPU neural policy, GPU query,
CUDA timing or learning result was produced by these checks. Scheduled runtime,
independent model math, concurrent trainer behavior and learning remain pending.
No production source/system/dependency changed; both GPU holds remain.
[Protocol](NATIVE_ENCODER_PROFILER.md) ·
[Receipts](results/encoder-profiler-supervisor-20261006/README.md).

## Built — native encoder profiler groundwork 20261006

Three standalone CUDA targets (default quality/Nature, IMPALA and Impoola)
compile from actual production callbacks. Host-only metadata matches 36
parameter/payload counts against independent C arithmetic over four families,
H64/128/256 and B1/64/2048. Seven invalid config/family cases are rejected. This
is registration arithmetic, not a CPU neural policy. First-build header/API
alias errors are retained separately; corrected builds/checks pass.

Scheduled code includes fixed synthetic inputs/native weight initialization,
CUDA-event samples, eager/graph bytes, parameter-gradient finiteness, unchanged
weights and isolated stream/workspace API hooks. GPU paths remain unexecuted.
Supervision, independent math and concurrent trainer acceptance are pending;
no new timing, SPS, score or cross-task dominance evidence. Production helpers
and both GPU holds are unchanged.
[Interface and gates](NATIVE_ENCODER_PROFILER.md).
[Retained build/registration receipts](results/encoder-profiler-20261006/README.md).

## Audited — native baseline compute/backend groundwork 20261006

Inspected actual Nature/Flex/IMPALA/Impoola constructors, patch/GEMM/backward/
scratch paths and shared cuBLAS math/stream setup. No Python policy/convolution
loop runs in any family. A C configuration-arithmetic utility emits fixed-graph
encoder/core/head parameters, GEMM MACs/calls and activation/scratch tensor
payloads. Audit matches all 52 existing Connect4 parameter receipts, matching
native encoder/core source hashes, and Flappy's retained quality parameter count.
Projection-equals-core and invalid-input cases are checked without a policy.

At H128/L1, estimated encoder forward GEMM MACs/sample are quality 187,184,
Nature 647,168, IMPALA 11,493,120 and Impoola 11,374,336. These explain structural
work differences; they aren't runtime timings or complete-network FLOPs.
Tensor counts exclude allocator padding, weights/grads, core/optimizer/replay/
input targets and vendor workspace, and aren't measured peak VRAM.

The cuBLAS API contract reveals a shared issue to investigate: `cublasSetStream`
before GEMM resets the explicit workspace assigned only at handle creation.
This affects all families. No nondeterministic outcome or speed impact was
observed here, and existing accepted repeatability receipts aren't invalidated.
Prepare native stream/workspace/profile acceptance before new timed panels.

No CPU/GPU neural policy, training, checkpoint evaluation or profiler executed;
no system/dependency/production source changed. Counts remain separate from
measured frontier points. Both GPU holds and encoder-5 gate remain.
[Audit and next gates](BASELINE_BACKEND_AUDIT.md) ·
[Retained source/count receipts](results/backend-audit-20261006/README.md).

## Built — supervised Snake checkpoint launcher 20261006

External `exact_eval.py run` and GPU-free `--prepare-only` now wrap the compiled
native exact adapter. Full trained policy/core settings are retained; the
suite fixes evaluation drawing, world/horizon and explicit identity allocation.
Finite float32 byte preflight and native spawn comparison precede any GPU
queries. Original/copied/effective inputs and launcher tooling are hashed.
Scheduled execution uses the shared reservation/idle query and bounded native
process helper, records monotonic evaluation time, and requires complete
quota/length/food/end-reason/parameter summary agreement before reporting.
Failures/timeouts retain partial rows/logs/status without successful aggregates.

Fourteen host/glue tests pass: eight earlier suite/receipt tests plus six
launcher checks. Includes 24 numeric/compiled pixel family/core preparations,
matched local-state H256/L2, malformed bytes, busy/failed/empty queries,
reservation conflicts, incomplete counts/summaries, changed copied config/
suite/tooling and missing/backward clocks. Synthetic completions/timeouts and
finite dummy weights are test data, not trained policy/GPU evidence.

No CPU/GPU neural policy, training, checkpoint evaluation, W&B upload, SPS or
frontier point was produced. The native adapter and all game/learner/CNN code
are unchanged. Both GPU holds and encoder-5's separate gate remain. Scheduled
runtime acceptance and common learning/horizon calibration still precede
matched multi-task frontier evidence. See
[instructions](../ocean/snakecnn/DETERMINISTIC_EVAL.md) and
[separate launcher receipts](results/snakecnn/exact-launcher-20261006/README.md).

## Built — native Snake exact assigned-episode groundwork 20261005

The evaluation-only Snake adapter compiles for default/IMPALA/Impoola and
matched local-state SnakeBench. Shared game/CNN/training kernels are unchanged;
core changes are guarded adapter include/host-mode/dispatch integration.
Explicit episode IDs determine world and native action RNG independent of
drawings/encoder/core size. Starting terminal=1 requests production recurrent
reset. Assigned slots stop after first death or finite game horizon;
replacements/padding never contribute. Every assigned ID must end once.

Independent CPU game/counter fixtures pass 96 trajectories plus dirty reset,
UINT32 boundary identities, death at horizon, rewarded/zero-reward horizon,
growth saturation, rounded-return accounting, invalid actions/counters, all
drawings/state, repeat and ASan/UBSan. Eight host/synthetic CSV audit tests pass:
quotas, order invariance, tampered identities/world/config closure, six drawings/
build families/H512-L4, independent tail and invalid settings before output.
Two panel and twenty shared sweep configuration checks pass. Synthetic
receipts are not model scores.

No CPU/GPU neural model executed, no training/evaluation/SPS/frontier point
was produced, and neither GPU hold was lifted. GPU reset/reload/math/repeatability
and supervised launch remain pending, followed by learning/horizon calibration
and paired full curves. The earlier world archive remains historical; current
native/source/test/start receipts are
[retained separately](results/snakecnn/exact-preparation-20261005/README.md).

## Built — native Snake local episodic pixel/state groundwork 20261005

Added separately named SnakeCNN and matching SnakeBench local-state control
under `snake-local-episodic-v1`; original stock Snake/config remain unchanged.
One shared native game uses local portable uint32 RNG, clean board/ring
resets, four stock-style movement actions, neck reversal, occupied-tail
collision and capped live growth. Death and finite game horizon emit terminals
and preserve ending rewards/counters through automatic reset. This changes
stock population/RNG/reset/termination semantics and is labeled accordingly.

Both interfaces observe the same centered 11x11 crop; pixels are direct
float32 1x36x44 with six square/disk/gap/palette/reflection presets. No global
food/length/time/RNG channel enters the policy image. Game arrays allocate
only at initialization. Metrics are ending length, clipped length/120, food,
return and duration plus death/horizon counts, not wins.

Independent array/deque/reset/spawn reference checks pass for 49,152 decisions
per panel over three board/food/ring regimes. All six fixed/three mixed pixel
panels match the state control's complete game traces. Explicit ring wrap,
neck/growth-limit, occupied-tail/wall death at horizon, rewarded horizon,
dirty resets, hidden-food isolation, literal inverse-coordinate raster,
overwrite/guards, global-RNG/worker-order independence, repeat and ASan/UBSan
checks pass. Four shared native float32 sm_120 targets compile; untouched
stock Snake also compiles. No CNN/training numerical kernel changed.

Two panel and twenty shared sweep/config/sidecar checks pass. Initial new
Python fixtures incorrectly assumed the kernel recipe's projection and omitted
a fake sweep completion line; those fixture failures are retained separately
from the corrected passing run. Three short native PROTEIN trial configs were
prepared, not executed. Fixed-model preparations now cover five tasks/thirty
drawings, record rules identities, and retain one frozen quality graph with
matched per-task learners/seeds/budgets/cadence. Earlier four-task preparations
remain historical; don't overwrite their source closures.

No CPU or GPU policy executed, no training/evaluation/SPS/frontier result was
collected, and no W&B data uploaded. Snake exact evaluation, GPU train/reload/
reset/numerical acceptance and common learner/horizon calibration remain gates.
Both GPU holds and separate encoder-5 requirements remain unchanged.
[Protocol](../ocean/snakecnn/PROTOCOL.md) ·
[Instructions](../ocean/snakecnn/README.md) ·
[Retained evidence](results/snakecnn/environment-20261005/README.md).

## Built — supervised Pong checkpoint launcher 20261005

Added external `exact_eval.py run` and GPU-free `--prepare-only` around the
compiled native exact-match adapter. Keeps the matching encoder/core/full INI,
fixed suite/drawing and actual native start manifests. Checks finite checkpoint
bytes before GPU queries, reserves the shared measurement lock, rejects busy or
failed GPU queries, bounds the native process group and retains monotonic
evaluation timing. Exact quota, completion/point/win totals, parameter byte
count and immutable original/copied/effective/tooling hashes must all agree.
Failures/timeouts preserve partial evidence without successful aggregates.

Thirteen host/glue tests pass: the prior seven suite/audit checks plus six
launcher checks, including 24 pixel family/core preparations and original state.
Failure fixtures cover malformed weights, lock/query conflicts, changed copied
inputs, corrupt/missing/duplicate summaries/rows and missing/backwards clocks.
Reports recompute censoring bounds from every assigned match; an intentionally
incorrect auxiliary native floating summary cannot change the audited score.
Test hardware/processes/weights/scores are synthetic; no neural model executes
on CPU or GPU. No new SPS, learning score or Pareto point was measured.

Actual 1,000-ID host starts were also retained for every Pong drawing, across
all three pixel build families and matching original state worlds, plus an
independent final-ID replay. All four native Pong targets and three other pixel
task regressions compile; original CUDA-environment Breakout still compiles.
Original environment/training and CNN math are unchanged. Runtime acceptance,
cap calibration and replicated comparative inference remain pending. Both
machine holds and encoder-5 gates remain; no automatic GPU launch is implied.
[Protocol](../ocean/pongcnn/DETERMINISTIC_EVAL.md) ·
[Native/host evidence](results/pongcnn/exact-preparation-20261005/README.md) ·
[Launcher evidence](results/pongcnn/exact-launcher-20261005/README.md).

## Built — exact Pong match/censoring groundwork 20261005

Native adapter and GPU-free suite/audit tooling now retain whole-match
decisions and both ending point totals while using unchanged original
`puf_step`. The native rally counter resets after every point; ending logs
remain checked as final-rally length rather than mislabeled match duration.
Natural completion at the decision cap takes precedence; administrative caps
add no fake terminal/loss and retain partial net points plus deterministic
possible-final-point-fraction bounds. Zero-point caps contribute [0,1]. These
are censoring bounds, not confidence intervals. No complete-only primary mean.

Four float32 sm_120 targets compile (state/default/IMPALA/Impoola). CPU
environment/pixel fixtures preserve original physics and independently check
initial reset RNG, dirty starts, both scoring sides/terminals, whole-match
versus rally lengths, zero/nonzero-point caps and terminal-at-cap. First-to-1
through first-to-21 capped bounds match enumeration of all possible terminal
scores. All five drawings, 96 stock-reference action trajectories, repeat and
ASan/UBSan pass. Seven actual native-host/synthetic-audit tests pass, including
cross-family/drawing/state-world manifests, independent last identity, frozen
configuration/hash controls and exact/corrupt/capped quota accounting.

CPU-only evaluation guards also preserve original CUDA Breakout compilation;
the earlier unguarded include referenced CPU-only fields in that backend.
No CNN/training kernel or original environment was changed. No policy executed
on CPU or GPU; no new score, SPS, timing frontier or W&B result was collected.
Supervised Pong launcher, GPU reset/reload/repeat/eager/quota acceptance,
cap/common-recipe calibration and publication inference remain pending.
Earlier preparations/results are preserved; new source-pinned panels remain
unexecuted. Both machine holds and encoder-5 gates remain.
[Instructions](../ocean/pongcnn/DETERMINISTIC_EVAL.md) ·
[Retained evidence](results/pongcnn/exact-preparation-20261005/README.md).

## Built — supervised Breakout checkpoint launcher 20261005

Added `exact_eval.py run` and GPU-free `--prepare-only` to the compiled native
Breakout adapter. Preserves the training encoder/core, evaluates a fixed drawing
from an immutable suite, regenerates actual native host starts, and retains
original/effective/copy/tooling hashes. Shared reservation/idle-query checks
precede scheduled native policy execution; process-group timeout and monotonic
evaluation timing reuse the accepted measurement helper. Partial/failed runs
retain evidence and never produce a successful aggregate/report. Native summary,
exact quota, checkpoint byte count and input hashes must match before reporting.

Twelve host/glue tests pass; preparation fixtures cover 24 pixel family/core
settings plus original state, with no GPU queries or policy construction.
Failure fixtures cover weights, busy/query/lock conflicts, missing/duplicate
rows, copied input changes, summary/parameter mismatch and missing/backwards
clocks; timeout fixtures retain partial CSVs/time. Ten Flappy and eight Connect4
audit regressions pass. Synthetic process/report fixtures are temporary tests,
not training or policy-evaluation measurements. No CUDA kernel/source changed
and no GPU job, W&B upload, score/SPS or frontier point was collected.

GPU numerical/reset/reload/repeat/eager acceptance and cap calibration remain
pending. Prepare-only is not a weight-shape/math certification. Native compiled
source provenance remains separate from launcher snapshots. New multi-task
source preparations are regenerated rather than overwriting earlier receipts.
[Instructions](../ocean/breakoutcnn/DETERMINISTIC_EVAL.md) ·
[Retained tests/source](results/breakoutcnn/exact-launcher-20261005/README.md).

## Built — deterministic Breakout evaluation groundwork 20261005

Added native exact adapter and host suite/receipt auditor, preserving original
game/training behavior. Evaluation stops at the first native terminal before
frame-skip continuation into a replacement game; the ending reward/log belongs
to the assigned episode. A separate physics-frame cap retains partial score
without generating a fake loss or terminal. Natural terminal on the cap frame
takes precedence. Report every assigned episode under the bounded-score
estimand, including capped outcomes.

Four float32 sm_120 targets compile (state/default/IMPALA/Impoola), as do
Connect4/Flappy integration regressions. CPU environment/pixel-only fixtures
check independent reset/launch RNG, final-loss and max-score terminal, partial
frame-skip cap, tampered metrics, no finished-episode reuse, five drawings,
96 stock-reference action trajectories, repeat and ASan/UBSan. Six native-host/
synthetic-audit tests cover manifests across families/drawings, independent tail
identity, exact quotas and corrupt/overflow/incomplete receipts. A first host
test invocation omitted the process-local NCCL runtime helper and failed before
loading binaries; that log is preserved alongside the passing configured run.

No GPU policy execution, learning score, SPS, W&B upload or Pareto claim. The
supervised launcher, GPU recurrent-reset/quota/reload/repeat/eager acceptance,
cap calibration and common learning recipe remain open gates. Source-pinned
16/96-job multi-task preparations were regenerated without execution; earlier
source states remain archived separately. Both machine holds remain.
[Protocol](../ocean/breakoutcnn/DETERMINISTIC_EVAL.md) ·
[Retained evidence](results/breakoutcnn/exact-preparation-20261005/README.md).

## Built — deterministic Flappy evaluation groundwork 20261005

Native exact episode adapter, separate host-only spawn-manifest mode and audit
runner implemented. State/default/IMPALA/Impoola float32 targets compile with
existing sm_120 toolkit/dependencies. Default/IMPALA/Impoola produced identical
1,000-episode initial image/world/RNG manifests; original state produced the
same worlds/RNG with its distinct observation hashes. Independent reset/pipe
RNG, native cap/crash/pass logging and dirty prior-state fixtures pass with
existing state/pixel parity and ASan/UBSan. Ten host suite/audit tests check
all assigned IDs, input corruption, cap scoring, core preservation, independent
tail starts and retained failures. Synthetic process outcomes test reporting
only; they are not neural results.

No GPU policy ran, no training score/SPS/learning frontier was measured. Native
GPU recurrent-reset/quota/reload/repeat/eager checks are pending. Old Flappy
pooled-v1 scores stay separate. Multi-task preparation now marks Flappy as
compiled-but-GPU-unqualified and remains nonlaunchable for confirmation.
[Protocol/instructions](../ocean/flappycnn/DETERMINISTIC_EVAL.md) ·
[Retained evidence](results/flappycnn/exact-preparation-20261005/README.md).

## Prepared — multi-task robustness and BreakoutCNN 20261005

Added direct-buffer BreakoutCNN with unchanged original CPU Breakout physics,
five reversible drawing IDs, shared CNNs and no copied CUDA kernels. Original
Breakout remains unchanged. All five fixed/three mixed appearance panels match
stock state/RNG/rewards/logs/terminals across 18,432 decisions per panel;
literal pixels, launch/brick/life/terminal fixtures, repeat and ASan/UBSan pass.
Default (quality/Nature dispatch), IMPALA and Impoola sm_120 float32 native
targets compile; no GPU training/evaluation or throughput/quality measured.

Fixed-model preparation now covers Connect4/Pong/Flappy/Breakout, all 24
existing drawing conditions, one frozen quality graph and H128/L1 core,
matched per-task learner/seeds/budget/cadence and source/config/build commands.
The local v2 default/all-appearance directories contain 16/96 unexecuted
plumbing jobs. Two panel tests and eighteen config/sidecar tests pass; a
three-trial Breakout architecture canary is prepared only. Flappy exact adapter,
Pong whole-match/Breakout terminal capture, GPU runtime, fair baseline profiling
and publication uncertainty gates remain open. This is software preparation,
not a measured Pareto result. [Plan](MULTI_ENV_ROBUSTNESS.md) ·
[Breakout validation receipts](results/breakoutcnn/environment-20261005/README.md).

## Completed — paired development replication 20261005

Ten native training jobs and all 510 checkpoint evaluations completed;
independent archive audit verifies 510 finite/correct-sized checkpoints and
510,000 exact assigned episode outcomes. No failures or missing cells, no
sampled GPU contention flags. Fixed quality/Nature, common H128/L1 learner/core,
float32, corrected Connect4, appearance 0, seeds 53101–53105, 13.312M decisions
and 51 checkpoints per job. Native monotonic completion receipts define time.

| Model | Final mean wins | Mean train process seconds | Mean process SPS | Mean native SPS |
|---|---:|---:|---:|---:|
| Quality Flex | 76.94% | 183.382 | 72,712 | 72,891 |
| Nature | 71.36% | 202.153 | 65,866 | 66,014 |

Ours has a 5.58-point higher final mean and 9.29% lower mean process time;
it wins four of five paired seeds, with differences -23.0, +13.4, +3.4, +20.8,
+13.3 percentage points. Seed variability is substantial. Ours contributes
38/39 observed mean frontier points; Nature's point is early at 0.08% wins.
All seed/checkpoint curves and declines remain. Candidate bands are disabled;
this development result does not establish certified dominance/SOTA. Training
processes totaled 32m 7.7s; evaluation/build time is separate. No seed extension
or new runs occurred during review; the GPU reservation has ended.

[Independent report](results/connect4cnn/replication-20261005/completed/independent-analysis/REPORT.md)
· [Full curves](results/connect4cnn/replication-20261005/completed/independent-analysis/curves.html)
· [Audit](results/connect4cnn/replication-20261005/completed/audit.json)
· [Methods and limitations](CONNECT4_DEVELOPMENT_REPLICATION.md).

## October 5, 2026 — whole exact-suite frontier, timing gate and fresh replication

[All 52 existing corrected-rules checkpoints](results/connect4cnn/exact-frontier-20261005/analysis.audited/REPORT.md)
were evaluated against the same frozen seed-51005 suite, exactly 1,000 games
each. No missing evaluations; 52,000 episode records/hashes pass independent
audit. The four final scores are quality 90.2%, Nature 77.0%, IMPALA 97.0%,
Impoola 81.1%. Original training times remain approximate file times and old
pooled-v1 scores are retained. All chronological curves are present; one-seed
frontier observations do not establish a replicated advantage.

[Timing/recovery GPU canary](results/connect4cnn/timing-canary-20261005/README.md)
passed 16 short on/off control jobs and 128 byte-identical paired checkpoint
files. Median receipt-on/off duration ratios: quality 1.01869, Nature 1.02659;
short noisy samples, not precise overhead bounds. A deliberately interrupted
17th training fixture preserved its first complete checkpoint and native
startup config, then completed the assigned 65-game evaluation while remaining
a failed training run. Optional atomic `resolved.ini` persistence and strict
partial-prefix audit support were added; partial-run SPS stays unavailable.

**Launched, not completed:** local `replication-20261005`, quality versus Nature,
five paired fresh seeds 53101–53105, common H128/L1 learner/core, float32,
corrected rules and appearance 0. Ten jobs × 13.312M decisions, all 51
checkpoints × 1,000 fixed-suite seed-253101 games. Native monotonic checkpoint
timing, separate process/native SPS, weight/config/source/episode receipts and
contention checks. Candidate confidence bands disabled; no adaptive extension
or selective retries. An independent GPU-free archive worker is scheduled.

[Methods and launch instructions](CONNECT4_DEVELOPMENT_REPLICATION.md) ·
[Frozen running protocol](results/connect4cnn/replication-20261005/protocol.json).
The future `completed/` report and `STATUS.json` in that archive determine
completion; do not treat this launch entry as final learning evidence.

## October 5, 2026 — dedicated deterministic checkpoint evaluation

Native exact evaluation now accepts arbitrary supported saved policy shapes via
their matching binary/resolved INI, with a frozen episode manifest shared by all
policies. A real GPU test exposed and fixed graph capture on the legacy NULL
stream; the evaluator now reuses the production actor stream. Training and CNN
kernels are unchanged by this fix. Fixed suites reject slot-based mixed rendering.

[Retained GPU acceptance](results/connect4cnn/deterministic-eval-20261005/README.md)
contains 32 passing evaluations across eight configurations: 1,000 games each
in two graph runs and one eager run, plus 40 final-wave episodes replayed alone.
Every assigned game is counted exactly once, with identical per-episode rows
across repeat/eager/independent-tail comparisons. State H128/H256 and six pixel
fixtures cover differing architectures and parameter counts. One-episode
unsigned-boundary/appearance-3 evaluation and nonfinite/oversized-checkpoint
rejection also pass. Eight configuration/receipt tests and twelve existing
claim regression tests pass; encoder-5 GPU verification remains pending.

For the four final corrected-rules 5060 pilot checkpoints only, seed-51005 exact
evaluation gives quality 902/1,000 wins, Nature 770, IMPALA 970, Impoola 811.
These are fresh **development evaluation** results of existing single-seed
models, not new training or a replicated learning/frontier claim. Other fixtures
used historical unequal training and are compatibility checks only. No new
training SPS measurement. Historical pooled-v1 scores remain unchanged.

[Usage and limitations](../ocean/connect4cnn/DETERMINISTIC_EVAL.md).

## October 5, 2026 — inherited Connect4 draw rule corrected (no GPU)

Both original state and pixel Connect4 used an incorrect full-board literal that
also produced two-piece draws. The new full-board regression fails before the
fix; corrected independent board fixtures, all ten representations, state/pixel
parity, repeatability and ASan/UBSan pass. Twelve artifact/configuration checks
pass. [Report and retained evidence](CONNECT4_DRAW_CORRECTION.md) document the
23 one-decision draws found in each old 4,096-transition native test trace.
New measurement manifests label `connect4-full-board-draw-v2`; historical learning
curves remain legacy-rule results. No CNN training/evaluation, SPS/VRAM/quality
measurement or GPU work occurred; ranking impact is unknown. Do not pool scores
across this rule change or automatically rerun previous campaigns.

## September 26, 2026 — original stock Flappy state control completed

Kinvert requested original vanilla PufferLib Flappy training. `stock-pilot.tgKBYrNs` uses unchanged original state observations and stock H64/L2, native float32 on the idle 5060, stock learner/vector/game settings, train seed 73, 20M requested decisions and the pixel pilot's recording/evaluation controls. No pixel run repeated. [Comparison, checks and receipts](results/flappycnn/stock-pilot.tgKBYrNs/REPORT.md), [paired table CSV](results/flappycnn/stock-pilot.tgKBYrNs/comparison.csv).

State completed **19,922,944 decisions in 9.73 process seconds**, **2,047,579 process SPS / 2,264,975 native average SPS**. Final evaluation: **48.533836 mean pipes**, clipped perf **0.947180**, 266 completed episodes (256 requested), 12.36 process seconds. Parameters 25,152; last saved VRAM 1.853 GiB. All 19 checkpoint hashes/counts/finiteness, final steps, finite metrics and executed-stock-config checks pass. Complete resolved learner/vector/selfplay settings match the pixel pilot. Training was launched detached, checked for completion when needed, and not actively monitored.

State is **5.7677× faster by process SPS** than the already measured pixel quality model (56.12s / 355,006 SPS / 52.252918 pipes / perf 0.975875). The pixel score difference is descriptive for one seed; state H64/L2 versus pixel H128/L1 and different observation information prevent isolating pixel-encoder overhead. Pooled-v1 evaluation counts differ. This is training SPS, not standalone environment-step throughput. No multi-seed superiority or Pareto claim; no tuning/sweep was launched.

## September 26, 2026 — 5060 FlappyCNN stock-learner pilot completed

Outside-sandbox execution is now available. RTX 5060 preflight passed (no competing compute process; desktop Xwayland only). Launched `stock-pilot.DN7C2JF6` in detached tmux `flappy-stock-20260926-2050` with the [authorized stock-learner recipe](FLAPPY_STOCK_PILOT.md): quality encoder/H128-L1, seed 73, float32, fixed appearance 0, 20M requested decisions. Native build/source-hash checks passed and training startup is verified beyond 2.49M decisions. Native dashboard at startup reported roughly 369K SPS and 5GB VRAM; these are provisional samples, not final timing or quality evidence. Logs/configs/checkpoints are in `build/flappycnn/stock-pilot.DN7C2JF6`; launcher log is `build/flappycnn/stock-pilot-gpu-launch.log`. Separate final evaluation and completion/checkpoint audit remain pending. The earlier blocked attempts remain preserved.

Completion and audit: **19,922,944 decisions in 56.12 process seconds; 355,006 process SPS / 361,619 native average SPS**. Final evaluation: **52.252918 mean pipes passed**, clipped perf **0.975875**, 257 completed episodes (256 requested), 14.06 process seconds. Parameters 160,096; last saved VRAM 4.976 GiB. All 19 expected checkpoints have matching recorded hashes, expected float32 counts and finite weights. Executed configs preserve stock learner/vectorization/original game values; selected network replaces stock H64/L2 with H128/L1. Native agent steps and finite metric arrays pass inspection. [Durable report and receipts](results/flappycnn/stock-pilot.DN7C2JF6/REPORT.md) supersede the provisional startup figures above. Successful single-seed learning/reload pilot only: no Nature/baseline comparison, repeatability test or final exact-episode protocol. Native perf is clipped pipes/20, not wins. No training/sweep was repeated after completion.

## September 26, 2026 — stock-learner FlappyCNN pilot blocked before training

Kinvert authorized a 5060 pilot of the existing quality network with otherwise stock Flappy settings and requested execution outside the sandbox. [Recipe and native launcher](FLAPPY_STOCK_PILOT.md): quality encoder ID 4 with H128/L1 core, stock Flappy vectorization/physics/learner, stock requested 20M decisions (19,922,944 after native rollout rounding), random initialization, default seed 73, fixed appearance 0. The H128/L1 network replaces stock Flappy H64/L2; no learner tuning or `compare.ini` is used. Intermediate checkpoints and separate final evaluation are provided.

`stock-pilot.HcBs3fcB` compiled in float32 and passed source hashes. Configuration inspection confirms all stock learner/vector keys and original environment values are preserved. Its GPU preflight failed, leaving status `blocked_gpu`; **no training or evaluation started**. Direct WSL NVML queries report `GPU access blocked by the operating system`. Current tool policy is `approval_policy=never` and does not permit outside-sandbox execution. [Blocked-attempt receipts](results/flappycnn/stock-pilot.HcBs3fcB/README.md) retain source/config/build provenance. No SPS, quality, weights, VRAM or training duration was measured. The runner is ready for a device-accessible session and rejects competing compute processes.

User-requested retry `stock-pilot.40z1KsJe` also compiled and passed source hashes, then stopped at GPU preflight with status `blocked_gpu` (launcher exit 3). A fresh NVML query again reported the OS access block. No training/evaluation started and no settings were changed. [Retry receipts](results/flappycnn/stock-pilot.HcBs3fcB/retry-40z1KsJe/) preserve the new attempt independently of the original.

## September 26, 2026 — FlappyCNN environment and build preparation (no training)

Added a third native pixel development task: [FlappyCNN](../ocean/flappycnn/README.md), selected after inspecting Flappy, Breakout, Snake, Maze, Memory and LightsOut ([candidate assessment](NATIVE_PIXEL_ENV_CANDIDATES.md)). The original Flappy game remains unchanged. Observation generation clears/fills a 1×36×44 float32 buffer in memory, without a graphics context, allocation or game RNG consumption. Four fixed or deterministically assigned appearances share the original physics/rewards/reset behavior. Pixels omit velocity and offscreen pipe state exposed by the state baseline.

49,152 transitions, explicit event/raster fixtures, fixed/mixed appearance parity, repeatability and ASan/UBSan pass; these are CPU environment checks, not model validation. Fifteen configuration/sidecar tests pass. A five-model native canary provides matched state/Flex/Nature/IMPALA/Impoola configs, 65,536 decisions, four checkpoints, separately seeded final evaluation and source/config/binary receipts. Architecture-search preparation now supports FlappyCNN with native optimization unchanged. All five native targets compile and source-hash checks pass (`canary.fn3qdTXH`, status `built`); old and expanded search recipes prepare successfully. [Durable validation evidence](results/flappycnn/environment-20260926/README.md) preserves checks/configs/build hashes and an earlier failed source check (`canary.WxHbxOCH`) during reporting edits. No failed receipt was accepted as validation.

No GPU training, search, learning result or dataset generation. GPU execution/reload and the canary's checkpoint/metric reporter remain unvalidated in the managed shell. SPS, train time, parameter counts and quality are **unmeasured** for this task. Short canary settings are untuned and are not a Pareto comparison. Retain the native capped-episode semantics and pooled-v1 evaluator limitation before subsequent learning experiments.

## September 26, 2026 — expanded encoder build and dataset design (no training)

Encoder 5 adds four configurable stages, dilation, 0–2 residual repetitions, adaptive spatial readout and five fixed/learned activation choices. Native Connect4CNN/PongCNN float32 builds and the GPU test-library compilation pass. Thirteen existing configuration/sidecar regression tests pass, along with Python glue syntax and prepare-only recipes at depths 1/2/3/4; active search dimensions are 10/17/24/31. No model executed on CPU or GPU, no training/sweep launched, no dataset generated. WSL NVML explicitly reports `GPU access blocked by the operating system` in the current managed shell; GPU math, older shared-kernel regressions, optimizer, seeded train/reload, learning and timing checks remain pending. SPS, return, VRAM and train seconds are therefore **unmeasured** for encoder 5. See [expanded grammar](FLEX2_CNN_SWEEP.md) and [pixel-label design](PIXEL_PRETRAINING_DATASET_PLAN.md). Historical measurements below are not results of this source state.

## 2026-09-15 — appearance.xRxsamIZ: seeded appearance reproducibility

[Report/table](results/connect4cnn/appearance.xRxsamIZ/REPORT.md), [CSV](results/connect4cnn/appearance.xRxsamIZ/results.csv). RTX 5060, CUDA 12.8, float32, one GPU; source snapshot/hash receipts identify the dirty measured tree on base `9b829e07`. All four fixed encoders on Connect4CNN and PongCNN, 16,384 decisions, mode 1 / appearance seed 12345 / 64 slots. Training seed 56173, evaluation action seed 66173. Sixteen short jobs and sixteen pooled-v1 evaluations passed; both saved checkpoints per paired run are byte-identical across one/two CPU workers, and evaluation records match. All 32 checkpoints have expected parameter count and finite weights. Per-job process time/SPS, native raw logs, configs, commands and assignment CSVs retained. These startup-dominated timings and near-zero scores are validation only. No encoder math/game-rule changes. All fixed/mixed CPU pixel/parity/ASan/UBSan tests passed; 25 reporting/configuration/runtime regression tests passed. Exact evaluator quotas and Pong long-match/censoring fixes are not implemented by this canary.

## 2026-09-15 — Protocol correction: full frontier, common learner

User requires full observed Pareto curves and identical training regimes. [Current protocol](CONNECT4_CLAIM_PROTOCOL.md) and [design JSON](connect4_claim_design.json) supersede the narrow time-point proposal below: all four fixed encoders complete 13.312M decisions under one shared recipe, with proposed 51-checkpoint cadence. No short wall cap or separate per-model hyperparameter selection. Retain every seed/checkpoint and failures; calibrate simultaneous time/score inference and freeze training-seed replication before launch. Extra evaluation seeds measure evaluation noise, not independent training variability. Old snapshot/power calculations remain historical development artifacts. No new training or measured result.

## 2026-09-15 — Focused Connect4 claim design and CPU validation (design superseded)

[Design and execution gates](CONNECT4_CLAIM_PROTOCOL.md), [machine-readable settings](connect4_claim_design.json), and [development-only analysis](results/connect4cnn/claim-design-20260915/REPORT.md). Target: quality at 60s versus Nature/IMPALA/Impoola at 30/45/60s, with nine adjusted paired contrasts. Historical 5060 and one-seed 5090 data remain separate; selection takes latest available checkpoints, preserves declines and never imputes missing early checkpoints as zero. Four CPU selection safeguards passed. Native match-bound/terminal checks passed original/pixel parity, repeats and sanitizers across all ten representations. Proposed next stages: the same six LR/gamma recipes on three development seeds per family, then one frozen recipe per family on 40 fresh paired seeds. Power sensitivity is approximate and assumes effect/variance; no statistical victory is guaranteed. Native deadline publication, exact episode allocation and baseline review remain open gates. No GPU training or final-data collection launched.

## 2026-09-15 — 5090 evaluation/profile archive received and checked

[Findings and evidence](PONG_EVALUATION_AUDIT_RESULTS.md): 764 package hashes, 57 raw attempt receipts and 90 included source hashes verified locally. Twelve completed old/new pairs match exactly; all eight timeout pairs time out again. Progress shows 14.64–18.22M advancing decisions but only 0–13 matches around 25s: sparse completion, not a hard stall. No consistent telemetry speedup. Remote profiling reports residual-backend shared GEMMs 61–64%, patch materialization 19–22% of summed kernel durations, with caller attribution still incomplete. Original 40/312/8 counts and missing outcomes remain unchanged. Full trace databases/checkpoint bytes remain remote. Evaluation v2 is a proposal only; no new runs, patch application or push during intake.

## 2026-09-15 — Evaluation telemetry fix and portable Pong launchers

Removed per-rollout resource queries from `trainer_eval_log`; standalone evaluation refreshes them only for periodic/final dashboards. Added flushed five-second `CUDA_EVAL_PROGRESS` diagnostics without changing the final result format, scoring, RNG or match rules. Pong canary/sweep use PATH-first GPU discovery with WSL fallback and reject failed status queries. Ten focused CPU tests and Bash syntax checks passed. [GPU regression receipts](results/pongcnn/eval-audit-20260915/README.md): two old/patched saved-checkpoint pairs return identical final metrics/counts, unchanged checkpoint hashes; intentional seven-second timeout retains progress and no success. Initial missing-final-dashboard-field failure is preserved. Timings varied; no speedup established. No new training ran. [Self-contained 5090 follow-up](PONG_EVALUATION_AUDIT_HANDOFF.md) limits evaluation diagnosis and reference profiling to a 45-minute GPU campaign, preserving the completed study.

## 2026-09-15 — RTX 5090 Pong replication reported

[Full reported table and limitations](PONG_5090_RESULTS.md): 40/40 training jobs, 312/320 evaluations, eight fixed-cap timeouts; all 320 checkpoint hash/finiteness checks reportedly passed. Source `1d9e9e0dca88706c7a8495644c8dc584a4446f11`. Total 3h13m2s: training 1h22m4s, evaluation 1h50m11s. Ours had 7–9% lower mean training wall than Nature but no demonstrated quality advantage; Nature was much better under recipe B, and Impoola reached >=95% in 5/5 seeds under A. Three final row means remain unresolved due to missing evaluations. Raw archive is not yet on G240; full-frontier/source/config review is pending. Do not present the supplied final table as a full-frontier audit or rerun censored cases selectively. No new local training was launched.

## RTX 5090 result received and receipt audit passed

[Results and provenance](HARDWARE_5090_RESULTS.md): four fixed models, 52 evaluations, zero failures at revision `9b829e07`, float32, seed 173, 13.312M decisions each. Ours 71.78% / 77.819 s; Nature 66.11% / 83.790 s; IMPALA 96.49% / 461.455 s; Impoola 82.77% / 476.149 s. Total training-process time 18m19.214s. Ours took 7.13% less time than Nature and scored +5.67 pp in this one seed. The received archive audit verified 42 source files against Git, all 52 CSV rows against raw evaluation logs, and matching non-encoder settings for all four jobs. Runtime helper inspected without executing it. Numerical/repeatability/sanitizer checks were reported passed remotely, not rerun here; binary/checkpoint arrays are absent from the transfer. The [complete observed time frontier](results/connect4cnn/compare.vtew3n12/analysis/REPORT.md) contains ours at low cost (72.14% at 53.960s), Impoola in the middle (83.76% at 138.215s), and IMPALA at high score (96.88% at 357.207s); Nature contributes no points this seed. Step frontier contains IMPALA/Impoola only. All observations, including declines, survive. Same-revision cross-host and fresh-seed confirmation remain pending; no new training was launched.

## 2026-09-14 — Pong discovery recovered after Nature evaluation timeout

Status check found `hypers.jwyApeXS` stopped: ours and Nature each completed 24 training trials, taking 437.21 / 614.65 seconds of total native sweep wall. Ours completed 24 evaluation/upload jobs; Nature had 23 successful evaluations and one timeout at 180.04 s (`sweep_1789436514815_0004`, exit 124). Strict reporting aborted before Nature upload and the remaining two families. Preserve this failure; do not infer a score from the incomplete evaluation dashboard or silently drop it from comparisons.

Launched bounded recovery in tmux `pong-cnn3-resume-20260914`, log `build/pongcnn/cnn3-hypers-resume.log`. Upload Nature's existing results, then run only IMPALA/Impoola with the original compiled binaries, configs, seeds, budgets and one-hour native search cap each. Original Nature timeout is not retried; completed training is not repeated. Recovery receipts live in the campaign's `recovery-eval-timeout/`. Reporting now records attempted evaluation failures without aborting unrelated full-search families; correctness checks remain strict, and final campaign status retains failures. [Protocol and recovery details](PONG_HYPER_SWEEP.md).

## 2026-09-14 — Pong GPU validation and frozen-model hyperparameter discovery

[Protocol and interpretation](PONG_HYPER_SWEEP.md). Pixel encoders all reuse the existing native kernels and H128/L1 core. Pong's three-action head gives total parameters ours 160,224, Nature 138,016, IMPALA 269,984, Impoola 151,200. `perf` is fraction of points won, not match win rate. Native Pong is not ALE Pong.

- [Five-model GPU canary](results/pongcnn/canary.w5tqV8nE/results.txt): state/ours/Nature/IMPALA/Impoola train, save/reload and evaluation passed; all short-budget evaluation point fractions were zero.
- [Quality-model learning pilot](results/pongcnn/pilot.p0ydqKv4/eval.txt): 13,312,000 decisions, **61.82 seconds whole-process training**, final point fraction **0**, score **−21**, 256 evaluation matches. Same common learner (.001 LR, replay 1), fixed quality encoder/core, train seed 9173, eval seed 29173. [Full native metric history, including SPS/uptime](results/pongcnn/pilot.p0ydqKv4/metrics/pongcnn/pilot.ini). This negative result motivated hyperparameter discovery; the budget is not proven adequate for Pong learning.
- [Initial sweep audit failure](results/pongcnn/hypers.l6li6thO/launch.txt): two native ours trials completed; reporting rejected quoted `None` versus native unquoted `None`. Normalization fixed, original failure retained; subsequent diagnostic report regenerated with the corrected checker.
- [Corrected online canary](results/pongcnn/hypers.hF7MhfRo/status.txt): two native trials per fixed pixel model, eight saved-checkpoint evaluations and eight W&B uploads passed. One architecture per family verified, finite arrays/parameter counts and fixed settings audited. Four new CPU safeguards and twelve existing sidecar/config tests passed.

Full development campaign **`hypers.jwyApeXS`** launched in tmux `pong-cnn3-hypers-20260914`; log `build/pongcnn/cnn3-hypers-launch.log`. Native PROTEIN searches nine training/budget dimensions in isolated configs, with 2.097M–13.312M requested decisions, at most 24 completed trials / 3,600 native sweep seconds per family, serial ours/Nature/IMPALA/Impoola. GPU had no compute process before launch. Source/config snapshots, hashes, dirty-tree patch, binary/build receipts, complete native training curves and evaluation receipts are retained under the campaign. All final checkpoints get 256-match development evaluation; W&B [kinvert-k/cnn3](https://wandb.ai/kinvert-k/cnn3) uploads between families outside search timing. Startup verified; results pending. No active monitoring. Actual tuning resources may differ under equal ceilings. Fresh-seed statistical confirmation and a defensible Nature advantage remain open.

## 2026-09-14 — Frozen confirmation full-frontier analysis

Audited completed `confirm.ol9tcj5k`: all 30 jobs, 390 checkpoint evaluations and W&B uploads passed. Verified every checkpoint hash/parameter count/finiteness, 38 source hashes, binary hashes, full seed/checkpoint coverage, paired evaluation seeds and matched non-encoder configs. No GPU compute process appeared in pre-job snapshots; snapshots were not continuous, and brief CPU-only development/tests occurred during the campaign. No new training or evaluation was launched for this analysis. [Interpretation](CONFIRMATION_RESULTS.md) · [Interactive whole-frontier chart](results/connect4cnn/confirm.ol9tcj5k/analysis/frontier.html) · [Complete numerical report](results/connect4cnn/confirm.ol9tcj5k/analysis/REPORT.md).

Final five-seed means: quality 79.66% / 144.47 s / 92,382 native SPS; small 78.01% / 143.28 s / 93,134 SPS; fast 71.66% / 147.12 s / 90,687 SPS; Nature 73.35% / 158.71 s / 84,066 SPS; IMPALA 97.45% / 1,231.81 s / 10,811 SPS; Impoola 80.09% / 1,229.68 s / 10,830 SPS. Wall values here are complete training-process means. The complete mean wall-time frontier contains ours at low cost and IMPALA at high performance; Nature/Impoola contribute no mean-frontier points. The mean step-count frontier consists entirely of IMPALA checkpoints. Quality's final paired advantage over Nature is +6.31 pp, pointwise percentile-bootstrap interval −0.07 to +13.00 pp, based on all 3,125 ordered five-seed resamples. This does not establish statistical dominance. Only 1/5 quality seeds reached 90% at any measured checkpoint, versus 5/5 IMPALA seeds.

All 390 observations, 78 checkpoint means, per-seed/per-model/combined frontiers on time and steps, paired differences and exploratory threshold non-achievement are archived. Graph/JavaScript checks passed 48 view combinations and empty selection; no automated real-browser rendering test was available. Native training/source/checkpoint arrays remain in the local campaign directory; small receipts and analysis are archived. Paper-plan item 1 is complete; cross-game generalization, equally tuned references and native delivery remain open.

## 2026-09-14 — Connect4 representation selector and native canary

Added `env.representation` IDs 0–9 with original ID 0 unchanged: piece-size/gap variations, raster disks, X/O glyphs and centered smaller boards down to one pixel per location. All retain the original game and 1×36×44 tensor, so there is no encoder FLOP reduction from smaller occupied regions. Masks are prepared once; native C observation generation consumes no RNG. Invalid IDs fail at initialization. The human viewer is still the standard board view. [Exact preset/config mapping](../ocean/connect4cnn/REPRESENTATIONS.md).

Validation: all ten presets passed independent pixel fixtures, dirty-buffer/reset checks, 4,096 original-state transition comparisons each, repeated traces and ASan/UBSan. Twelve sweep/config/report tests and two existing confirmation config tests passed. The native PROTEIN canary [sweep.wstneiqe](results/connect4cnn/sweep.wstneiqe/REPORT.md) completed all three trials, 98,304 total decisions, 2.720293 seconds sweep-process wall excluding compilation. Selected IDs 0/2/4 were correctly saved in native INIs, CSV and disabled-sidecar payloads; all checkpoints had 160,736 finite float32 parameters and the same architecture hash. Canary native cost/SPS were respectively 0.52 s/63,015, 0.49 s/66,873 and 0.48 s/68,267, from rounded PROTEIN costs; these startup-heavy 32K-step measurements are not speed or learning conclusions. W&B disabled; no learning-scale representation sweep launched. Source/config/binary hashes, raw logs and small test receipts are archived with the run.

The new representation-only recipe locks CNN/core/learner/budget and sweeps the integer appearance choice. Existing architecture recipes may opt into the same sweep section. Reports retain appearance separately and calculate Pareto flags within each representation; PROTEIN itself still optimizes the joint objective. Equal representation coverage and paired seeds are necessary before calling an architecture robust; selecting only an easy appearance is insufficient.

## 2026-09-14 — PongCNN CPU validation and GPU canary preparation

Added native PongCNN while the Connect4 frozen confirmation continued. The copied native Pong physics/opponent/rewards are unchanged; observations are direct C-generated float32 `[1,36,44]` pixels. Shared encoder factory selection adds PongCNN through two existing conditional branches, without kernel edits. CPU validation passed 49,152 complete state-hash/reward/terminal trace entries against native Pong across eight seeds, frame skips 1/3/8 and both action modes; repeated pixel traces matched. ASan/UBSan, image corner/clipping/score fixtures, reset/reward checks and buffer guards passed. Command: `bash ocean/pongcnn/tests/run_all.sh`; [small validation evidence](results/pongcnn/environment-20260914/README.md).

Prepared `build/pongcnn/canary.NH9AYFJK` using `bash ocean/pongcnn/canary.sh --prepare-only`. All five isolated effective configs matched, with active Flex construction keys present and environment values identical to original Pong. This preparation contains no native trainer build or GPU work. The future fixed canary tests state, existing Flex quality shape, Nature, IMPALA and Impoola at 65,536 decisions each, hidden-128/one-layer core and the same untuned learner recipe. Source/config/checkpoint/build receipts and native SPS/uptime will be saved automatically when it is actually run; no SPS, learned performance, or GPU correctness result exists yet. GPU validation is deferred until confirmation finishes. See [PongCNN README](../ocean/pongcnn/README.md) for exact controls and native commands.

Pong metric caution: `perf` is episode-averaged fraction of points won, not match wins. State Pong supplies velocity; single-frame pixels require temporal inference. Original state score components use integer division and remain zero during play, whereas the new visible score bars expose current score. Preserve and disclose these differences. This task is native PufferLib Pong, not Atari/ALE, and is now a development environment rather than a held-out test.

## 2026-09-14 — Complete CNN2 frontier and reference comparison

Analyzed completed `sweep._u86vi03`: 128 trials, 40 active architectures, 1,355,720,704 decisions, 14,950.940 s native sweep wall. All 128 final arrays finite, binary hash verified, and all fixed learner/core/vec/env settings matched the nine archived Nature/IMPALA/Impoola runs. No worker failures. Evaluated all 128 final checkpoints plus 12 earlier checkpoints of the training-selected winner using evaluation seed 10073 and 1,024 requested games. No new training. Exact commands, actual games, checkpoint hashes and raw native evaluation outputs are archived.

[Full analysis](CNN2_RESULTS.md) · [Interactive Pareto frontiers](results/connect4cnn/sweep._u86vi03/analysis/frontier.html) · [All observations](results/connect4cnn/sweep._u86vi03/analysis/observations.csv) · [Evaluation receipts](results/connect4cnn/sweep._u86vi03/analysis/evaluations.csv) · [Sweep report](results/connect4cnn/sweep._u86vi03/REPORT.md).

Final Flex winner `brave-comet-51`: 92.00% held-out wins, 12,662,784 decisions, 160,736 parameters, 137.37 native seconds / 92,180 native SPS; approximate launch-to-checkpoint wall 137.57 s. Its earlier 8.192M checkpoint achieved 90.53% at 90.56 wall seconds, and 10.24M checkpoint 92.05% at 112.11 s. These are checkpoints within a longer annealing schedule, not independently short-budget runs. All Flex training uses one seed; references have three. Current observations favor Flex's middle score/time region, while IMPALA retains the roughly 98–99% region. Impoola has no combined frontier points in the matched-seed view. The report distinguishes final-trial frontiers from the optional winner learning curve and provides both time and step axes.

Search limitations: 120 single-stage trials; all eight deeper trials only 2.19–2.79M steps. Twenty-one repeated active configuration/budget/seed trials added no new coverage and produced byte-identical checkpoints within groups. Last-window training scores varied despite identical weights; fixed held-out evaluation avoids that reporting-window variation. These are not independently optimized reference-family frontiers or multi-seed confirmation of Flex superiority.

## 2026-09-13 — cnn2 timestep range extended to 2.10M–13.28M

User requested roughly 2M–13M searchable timesteps. Updated `sweep_flex.ini` to 2,097,152–13,279,232 requested decisions and initial budget 13,279,232, retaining all eighteen architecture dimensions and 128 trials. Raised the predicted suggestion-cost ceiling from 300 to 600 seconds to accommodate longer trials. Native PROTEIN still selects budgets; the upper-budget first observation does not guarantee later proposals will cover that end of the range. Actual steps follow native rounding, and learning-rate annealing depends on each trial's total budget. Ten existing CPU configuration/tooling tests passed; no native implementation changed.

Previous campaign `sweep.3la4j4n2` finished naturally before replacement: 128 trials, 41 active architectures, 2,507.759 seconds native sweep wall, no failed workers, successful sidecar completion. Its artifacts and W&B runs remain intact. Short-budget results provide throughput observations but weak learning evidence; no new checkpoint or held-out validation was performed for this completion check.

Replacement campaign `build/connect4cnn/sweep._u86vi03`, tmux session `cnn2-budget-20260913`, launch log `build/connect4cnn/cnn2-budget-launch-20260913.log`, W&B [kinvert-k/cnn2](https://wandb.ai/kinvert-k/cnn2), group `sweep._u86vi03`. GPU was idle before launch. Same float32 common learner/core, seed 73, concurrent online sidecar, `NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1`, and 43,200-second whole-sweep safety deadline. Exact source/config snapshots are saved per campaign. Startup verified detached execution; results pending. No active monitoring.

## 2026-09-13 — Flexible CNN discovery launched in cnn2

Campaign `build/connect4cnn/sweep.3la4j4n2`, detached tmux session `cnn2-flex-20260913`, launcher log `build/connect4cnn/cnn2-launch-20260913.log`. W&B: [kinvert-k/cnn2](https://wandb.ai/kinvert-k/cnn2), group `sweep.3la4j4n2`. User selected this project for the next sweep. GPU inspection immediately before launch showed the RTX 5060 idle, with 503 MiB display memory.

Recipe `ocean/connect4cnn/sweep_flex.ini`: 128 sequential native PROTEIN trials, encoder 4, eighteen architecture dimensions and requested budgets 829,952–6,639,616 decisions; initial budget 3,319,808. Common learner and hidden-128 single-layer recurrent core remain fixed, seed 73, float32. Predicted suggestion-cost ceiling is 300 s, not a hard per-trial limit. Whole native sweep deadline is 43,200 s (12 hours), a safety cap rather than a runtime estimate. Launch uses `NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1`, `--wandb online --project cnn2 --entity kinvert-k`. Source is the current uncommitted implementation on `9ed6bc2a`; exact source/config snapshots and hashes are captured in the campaign.

One startup check confirmed live detached execution, five completed native trials, and successful online uploads with friendly names (including `happy-cat-5`). Native SPS, uptime, agent steps, performance, score, and losses are logged by the existing concurrent sidecar. Leave training detached without active monitoring; reports and completion receipts are generated automatically. This is architecture discovery only. Fixed longer-budget Nature controls and held-out, multiple-seed finalist evaluation remain necessary before claiming a learning/time advantage. Results are pending.

## 2026-09-13 — Expanded INI architecture controls

Added encoder ID 4 with independent stage channels/kernels/strides, optional residual convolutions, none/max/average pooling, flatten/GAP readout, and projection widths 16–128. Existing IDs and kernel paths retain their meanings. All families now allow any nonempty subset of legal sweep dimensions; omitted sections hold their configured values fixed, including optional fixed training budgets. The full recipe offers 18 shape dimensions plus budget and defaults to 128 trials. No learning-scale 128-trial run was launched. See [control documentation](FLEX_CNN_SWEEP.md).

Validation: 78 targeted flexible configurations passed independent float64 forward/all-parameter-gradient checks, eager/graph repeatability, and rollout parity. Tests cover mixed kernels and stages, pooling/skip gradients, tied max-pool values, tiny maps, and boundary layouts; seven nonblank configurations also passed per-array finite differences. Nature's existing regression suite and all 54 compact cases passed. Ten CPU tooling tests passed, including one-knob selection, fixed budgets, invalid fixed-value rejection, and fingerprints ignoring inactive stages.

[Full flexible canary](results/connect4cnn/sweep.jfu9hfqr/REPORT.md): 12 trials, 12 active architectures, **7.532 s** native sweep-process wall. [One-knob/fixed-budget canary](results/connect4cnn/sweep.jd06qqgp/REPORT.md): four distinct first kernels, all at exactly 32,768 decisions, **3.221 s** wall. No failed workers. All 16 final checkpoint arrays were finite and their parameter counts matched independent architecture calculations. SDK logging was disabled for these canaries; native metrics and complete architecture JSON were saved locally.

The largest-workspace allowed layout (three stages, channels 32 throughout, kernels 8/5/5, strides 4/1/1, all skips enabled, no pooling, flatten and projection 128) trained twice at 16,384 steps using the common 2,048-decision training batch on the 8 GB RTX 5060. Both final checkpoints were byte-identical: SHA256 `42fb1d331d863026967aaf6eb42fd069ebfd5371e783136746abca0760c78939`. Its 536,896-parameter checkpoint reloaded and completed 292 evaluation games (128 requested), with zero wins at this plumbing budget. Memory fitness and repeatability are established for the common recipe; these runs do not establish learning quality or speed rankings. Validation receipts are archived with the full canary.

## 2026-09-13 — Compact versus Nature discovery results

Completed [Nature sweep](results/connect4cnn/sweep.4h5iffsm/REPORT.md): 12 trials, 191.242 s native sweep-process wall. Completed [compact sweep](results/connect4cnn/sweep.od0_6e75/REPORT.md): 24 trials, 10 shapes, 919.912 s wall. All 36 final checkpoint arrays were finite; resolved non-budget learner, recurrent core, vectorization, environment, and selfplay settings matched across all trials. Both runners finished successfully, including online sidecars. Small evidence is archived with each report.

| Run | Family / shape | Actual decisions | Final training wins | Native cost | Native average SPS |
|---|---|---:|---:|---:|---:|
| bright-tree-1 | Nature | 3,317,760 | 0.11% | 43.18 s | 76,836 |
| bright-badger-1 | Compact C8 / depth1 / stride4 / projection32 | 3,317,760 | 0.33% | 38.03 s | 87,241 |
| gentle-cedar-18 | Same compact shape | 6,074,368 | 9.70% | 68.48 s | 88,703 |
| merry-maple-23 | Same compact shape | 6,639,616 | 22.76% | 72.66 s | 91,379 |

The matched initial-budget pair shows **13.54% higher native average SPS** and **11.93% shorter native training time** for compact. The selected shape has 75,432 total parameters versus Nature's 138,528. Compact's roughly 91K longer-run SPS is about ten times the previous high-return residual model's 9K SPS, but its best final training win rate here was only 22.76%. Speed improved; a learning/time advantage over Nature is unestablished.

The control sweep's actual budgets were 868,352–3,317,760 decisions, mostly near 1M. Compact's extended to 6,639,616. Nature therefore has no same-budget observation for the promising longer compact runs, even though both recipes allowed the same range. PROTEIN's adaptive choices do not ensure adequate baseline coverage. Fixed longer-budget Nature controls are needed. These final training rates also cannot be compared directly with the earlier 78.71% held-out Nature result at 13.3M steps. Repeated compact trials used the same seed, not independent seed confirmation. Next: fixed-budget Nature/compact runs with matched evaluation and several seeds before claiming a learning-quality winner.

## 2026-09-13 — Compact family and Nature control validation

Learning-scale pair launched after validation in tmux session `cnn-small-20260913`; parent campaign `build/connect4cnn/fast-search.o61nFcx4`, launcher log `build/connect4cnn/small-launch-20260913.log`. The `nature.log` and subsequent `compact.log` identify child campaign directories/W&B groups. Nature runs 12 budget trials, then compact runs 24 shape/budget trials, serially. Shared 829,952–6,639,616 requested decisions, seed 73, learner/core, float32, 25 history points, checkpoint interval 500, predicted-cost ceiling 300 s, and four-hour hard deadline per campaign. GPU was idle (0% utilization, 503 MiB display memory) immediately before launch. Startup confirmed a live native sweep/worker. Results are pending; no active monitoring. Exact source/config snapshots are saved per child campaign. Local research index refreshed to 13,761 chunks before launch.

Added encoder ID 2 (exact existing adapted Nature) and 3 (compact valid-strided convolutions), keeping IDs 0/1. Compact varies channels 8/16/32, depth 1/2/3, initial stride 2/4, and projection 32/64/128: 54 configurations. Nature kernels are shared, with dynamic layer count; no new kernel algorithm or system CUDA changes. Recurrent width remains 128. See [the plan](COMPACT_CNN_SWEEP.md).

All 54 configurations passed independent float64 forward/all-parameter-gradient comparison, repeated eager/graph checks, and rollout parity. Six small configurations also passed finite differences across every parameter array. Nature's existing reference/regression checks passed. The Nature-equivalent compact shape matched Nature bit-for-bit; an old-binary Nature training run matched the new path's 32,768-decision checkpoint exactly (SHA256 `c7ef5d77ac701f5a1be0f24c567ea4810fbb20a8c3ac850ba65138baccf258c4`). Eight CPU tooling tests passed, including matching Nature/compact learner and budget settings, shape-specific dimensions, and inactive-field handling.

[Nature canary](results/connect4cnn/sweep.9sl0k5ll/REPORT.md): 12 completed trials, 7.932 s native sweep-process wall. [Compact canary](results/connect4cnn/sweep.lvfycy5l/REPORT.md): 12 completed trials, 9 shapes, 8.033 s wall. No failed workers. All compact final checkpoint arrays were finite. Largest sampled compact checkpoint (380,064 parameters) reloaded and completed 288 evaluation games (128 requested), with zero wins at its 16,384-decision plumbing budget. These timings/scores are not learning or speed rankings. The canaries preceded removal of inactive inherited CNN keys from effective INIs; the active shapes and kernels were unchanged by that reporting cleanup.

## 2026-09-13 — Discovery sweep completed

[Archived report](results/connect4cnn/sweep.co1g6diu/REPORT.md): all 24 trials completed, no failed native workers, **11,611.189 s (3 h 13 m 31 s)** total sweep wall. Eight distinct architectures, 13 model-guided proposals, and 127,352,832 actual training decisions. All 24 final checkpoint arrays were finite. W&B API verification found exactly 24 finished runs in group `sweep.co1g6diu`, all with native `SPS`, `uptime`, `agent_steps`, and `env/perf` summaries.

Notable points on the observed training-score/time frontier:

| W&B run | Channels / blocks / GAP | Decisions | Final training wins | Native cost | Native average SPS |
|---|---|---:|---:|---:|---:|
| golden-tree-1 | 8 / 0 / 0 | 6,639,616 | 14.83% | 165.30 s | 40,167 |
| swift-owl-16 | 16 / 1 / 0 | 5,588,992 | 62.00% | 349.62 s | 15,986 |
| cosmic-fox-17 | 16 / 1 / 0 | 6,356,992 | 69.00% | 397.20 s | 16,005 |
| clever-river-24 | 32 / 1 / 0 | 5,595,136 | 96.59% | 619.05 s | 9,038 |
| lucky-maple-21 | 32 / 1 / 0 | 5,785,600 | 100.00% | 641.40 s | 9,020 |

PROTEIN concentrated later proposals on channels 32, one residual block per stage, and flatten readout (518,016 parameters). The fastest observed 100% final training point was `lucky-maple-21`. This is a final training measurement against the existing opponent, not a held-out estimate or a solved-game claim. Discovery used one seed and variable budgets (including budget-dependent learning-rate schedules). Repeated seeds and held-out evaluations are required before ranking finalists or comparing with reference results. Suggested confirmation candidates are the best high-return 32-channel configuration and the faster 16-channel configuration.

The launcher emitted the expected error for the original sidecar that was deliberately replaced during training. Native completion is successful; the replacement sidecar completed all uploads. Both the launcher receipt and replacement sidecar log are archived, preserving the distinction. No new training or evaluation was launched during this completion check.

## 2026-09-13 — First learning-scale custom-CNN discovery sweep (launched)

Campaign: `build/connect4cnn/sweep.co1g6diu`; launch log: `build/connect4cnn/discovery-launch.nqVXJ7.log`; persistent tmux session: `cnn-discovery-20260913`. Source baseline `9ed6bc2a` plus recipe/concurrent-sidecar changes, with exact source/config hashes and snapshots saved in the campaign. No `src/` or CUDA-stack changes for this launch.

Native PROTEIN: 24 sequential trials on the idle RTX 5060, float32, seed 73, the same 18 custom-CNN shapes and fixed hidden-128/one-layer core and learner. Requested budget range 3,319,808–13,279,232 decisions; initial candidate 6,639,616. Predicted suggestion-cost ceiling 3,600 s; whole-sweep hard deadline 43,200 s. Checkpoints every 1,000 updates plus final. `OPENBLAS_NUM_THREADS=1` bounds numerical-library CPU threads. Trial SPS, native cost, training wins, actual timesteps, parameters, and provenance are saved automatically. This is discovery; held-out evaluation and multi-seed confirmation remain pending.

W&B project: https://wandb.ai/kinvert-k/puffer-cnn, run group `sweep.co1g6diu`. Online CPU sidecar uploads completed trials during training; possible CPU/I/O timing overhead is part of this protocol. The 12 finalized canary trials were successfully uploaded first. Five CPU tooling tests passed, followed by a fresh two-trial concurrent-online canary (`sweep.effs260q`, 2.625 s native sweep wall, no worker failures). Startup inspection confirmed one launcher, one sidecar, and one native sweep/worker pair. No long-run performance results are claimed yet. After completion, archive small reports/configs/metrics/provenance using the existing results convention.

Latest infrastructure milestone: the September 13 [custom-CNN native PROTEIN canary](results/connect4cnn/sweep.7s4ovb72/REPORT.md) completed 12 trials, 8 architectures, and offline W&B logging. Canary training scores are excluded from the held-out baseline table below.

September 13 presentation correction during discovery: sidecar schema 2 uses friendly adjective/noun/trial names and the original native metric names (`SPS`, `uptime`, `agent_steps`, `env/perf`, `env/score`, losses), plotting against agent steps. Existing online runs are updated in place with the same W&B IDs. Native numeric history and the five-point downsampling are unchanged. Six CPU tests passed; W&B API readback confirmed corrected names, summaries, and history on the two logging-canary runs. The original discovery sidecar PID 3542239 was terminated and replaced in tmux session `cnn-wandb-20260913`; native training was not stopped or modified. Replacement logging is in `build/connect4cnn/sweep.co1g6diu/sidecar-v2.log`. The original launcher will report its old sidecar's nonzero exit after training completes; assess native `finished.json`/reports and the replacement sidecar log separately. This expected logger-replacement message is not a failed training trial.

Persistent results across code and configuration changes. The comparison runner appends a dated entry after each comparison; keep earlier results, including failures. Each entry links the exact protocol, source/binary hashes, recipe, hardware record, raw logs, and checkpoints. Add interpretation below an entry rather than rewriting its measured numbers.

## Metric definitions

- **Perf / win rate:** independent checkpoint evaluation against the unchanged scripted Connect4 opponent. The board remains 7 columns × 6 rows.
- **Process SPS:** training agent decisions divided by measured training-process wall time, including startup and checkpoint writes. Compilation and later evaluation are excluded.
- **Native average SPS:** training agent decisions divided by the trainer's final logged uptime. Its timing boundary differs from process wall time.
- **Native last SPS:** final logged SPS sample/bin; not an overall average or guaranteed steady-state rate.
- **VRAM last GB:** last native logged GPU-memory reading, not a measured peak or exclusively encoder memory.
- Retain training budget, seeds, float precision, parameters, core/learner settings, device, and concurrent-load context. Compare speed alongside learning; a faster policy with lower win rate is not automatically better.

## Current measured baselines (2026-09-12 local time)

**IMPALA/Impoola extension completed (2026-09-13 local time):** all six trials and 24 checkpoint evaluations passed under the same common recipe. Both variants use original-SAME pooling and the same 16/32/32 backbone; GAP is the readout change. See the environment README for reference adaptations and numerical validation.

Arithmetic means across training seeds 73/74/75, evaluated with separate paired seeds 10073/10074/10075. All runs use float32 on G240's RTX 5060, and the same game/opponent and 64-agent evaluation setup. Stock, Nature, state/tiny, and IMPALA/Impoola ran in separate invocations, not one interleaved speed experiment.

| Policy | Training recipe | Actual decisions per seed | Parameters | Mean wins | Mean process SPS | Mean train wall s |
|---|---|---:|---:|---:|---:|---:|
| Stock state | Stock training config | 13,238,272 | 209,408 | 98.87% | 306,403 | 43.414 |
| State | Modified common recipe | 13,279,232 | 55,552 | 18.51% | 108,831 | 122.017 |
| Tiny CNN | Modified common recipe | 13,279,232 | 151,680 | 64.04% | 97,733 | 135.875 |
| Adapted Nature CNN | Modified common recipe | 13,279,232 | 138,528 | 78.71% | 80,840 | 164.279 |
| Adapted IMPALA CNN | Modified common recipe | 13,279,232 | 270,496 | 99.19% | 10,800 | 1,229.612 |
| Adapted Impoola CNN | Modified common recipe | 13,279,232 | 151,712 | 69.49% | 10,833 | 1,225.833 |

Reports: [stock state](results/connect4cnn/compare.9egh6y44/REPORT.md), [state/tiny CNN](results/connect4cnn/compare.we7qgdcg/REPORT.md), [Nature](results/connect4cnn/compare.2s__8t8l/REPORT.md), [IMPALA/Impoola](results/connect4cnn/compare.l6d5sbk2/REPORT.md). Stock has by far the lowest measured training time, with a different learner/network/batching configuration. IMPALA reached the highest mean win rate, but its small numerical edge over stock does not establish a reliable advantage. Nature outperformed tiny CNN in each paired seed under the common recipe, while processing about 17.3% fewer decisions per second. These are adapted encoders, not complete reproductions of the reference agents. No pixel policy has yet been tested with the stock training recipe.

## Policy identities and hyperparameters for compare.we7qgdcg

**The 18.51% state result is PufferLib's original Connect4 environment and default network architecture with a modified configuration. It is not a measurement of untouched stock PufferLib tuning.** Both tested policies use the same modified learner/core recipe. Stock training settings were subsequently measured separately in `compare.9egh6y44`: **98.87% mean wins**, using float32 and common evaluation. See the dated results below.

| Property | `state` — modified-config baseline | `tiny_cnn` — custom pixel encoder |
|---|---|---|
| Input | Original 42 board values | 1×36×44 grayscale; each board cell is 6×6 pixels, with side padding |
| Encoder | Default linear projection, 42 → 128 | Conv4×4, stride 4, 1 → 8 channels → ReLU → flatten 792 → linear 128; no biases |
| Recurrent core | Existing PufferLib MinGRU, hidden 128, one layer | Same |
| Trainer and action/value heads | Existing PufferLib implementation | Same |
| Total parameters | 55,552 | 151,680 |
| Mean final evaluation win rate | 18.51% | 64.04% |
| Mean process SPS | 108,831 | 97,733 |

Tiny CNN is the custom encoder implemented in [connect4cnn.cu](../ocean/connect4cnn/connect4cnn.cu), not Nature, IMPALA, or Impoola. Both environments preserve the same 7-column × 6-row game, seven actions, scripted opponent, rewards, and termination rules.

The following values describe this completed experiment, not every future invocation. Stock values come from the run's saved upstream [Connect4 config](results/connect4cnn/compare.we7qgdcg/source/config/connect4.ini) plus [default config](results/connect4cnn/compare.we7qgdcg/source/config/default.ini), at revision `89414204ce85`. Tested values come from the [saved common recipe](results/connect4cnn/compare.we7qgdcg/recipe.ini), command overrides, and resolved run INIs.

| Setting | Tested state | Tested tiny CNN | Stock configuration |
|---|---:|---:|---:|
| Training decisions per seed | 13,279,232 | 13,279,232 | 13,272,299 |
| Core hidden size | 128 | 128 | 256 |
| Core layers | 1 | 1 | 1 |
| Initial learning rate | 0.001 | 0.001 | 0.00847027 |
| Replay ratio | 1 | 1 | 3.16619 |
| Parallel agents | 64 | 64 | 4,096 |
| Vector buffers | 1 | 1 | 8 |
| Environment threads | 2 | 2 | 2 |
| Minibatch size | 2,048 | 2,048 | 8,192 |
| Rollout horizon | 32 | 32 | 32 |
| Async execution | 0 | 0 | 1 |
| Gamma | 0.8 | 0.8 | 0.8 |
| GAE lambda | 0.962627 | 0.962627 | 0.962627 |
| PPO clip coefficient | 0.511829 | 0.511829 | 0.511829 |
| Value loss coefficient | 5 | 5 | 5 |
| Value clip coefficient | 1.99178 | 1.99178 | 1.99178 |
| Maximum gradient norm | 0.552251 | 0.552251 | 0.552251 |
| Entropy coefficient | 0.0000324222 | 0.0000324222 | 0.0000324222 |
| Optimizer momentum | 0.878636 | 0.878636 | 0.878636 |

Both tests also use learning-rate annealing to zero, entropy annealing disabled, CUDA graphs enabled, one GPU, and float32 builds. `min_ent_coef_ratio` is overridden from 0.1 to 0 but is inactive with entropy annealing disabled. Checkpoint interval changes from 500 updates to 1,621, yielding four checkpoints at 3,319,808-decision intervals. Output directories/run IDs are isolated per trial. Training sets `eval_episodes=0`; separate evaluation processes request 1,024 games per checkpoint instead of the stock evaluation default of 10,000, with actual batched counts retained in the report.

Both policies use training seeds 73/74/75 and corresponding separate evaluation seeds 10073/10074/10075. All six resolved training configurations were checked to agree after excluding environment names, seeds, and artifact paths. Equal seed labels do not imply identical weight initializations or trajectories across different architectures.

**What the result establishes:** the custom CNN achieved higher evaluation win rates under this shared modified recipe. Direct state provides the game information, but the network still must learn useful features. The CNN adds nonlinear feature extraction and more parameters; either may contribute, and this experiment does not isolate their effects. It does not establish that pixels are intrinsically better, that the CNN beats stock-tuned PufferLib, or that either policy has converged. The subsequent stock-config run achieves much higher win rates than either earlier policy.

## Earlier workflow checks (2026-09-12)

- Tiny CNN native checks and repeated 65,536-step training produced identical finite checkpoints. See [environment verification](../ocean/connect4cnn/README.md).
- [Matched smoke comparison](results/connect4cnn/compare.16ohu06z/REPORT.md): state and tiny CNN, seeds 73/74, 65,536 steps each, two checkpoint evaluations each. All final win rates were zero. This was infrastructure validation; these short process times are not a meaningful speed ranking. SPS was not yet a dedicated report column; original native timing samples remain in each run's resolved INI.
- Earlier `compare.otwm9q9m` was a reporting diagnostic: its selected native score metric was win rate. Its report is marked superseded; do not read that score column as episode return.

## Planned first learning-budget comparison

Keep the same common recipe and both architectures; increase training to **13,279,232 decisions per run**, the first multiple of rollout-batch × four checkpoints above the stock config's 13,272,299 decisions. Use three training seeds (73/74/75), separate paired evaluation seeds (10073/10074/10075), and 1,024 requested evaluation games per checkpoint. This changes only the budget from the smoke comparison. The shared recipe still differs from the stock Connect4 tuning; learning failure in both variants would require recipe investigation before judging CNN quality.

## 2026-09-13T04:51:13+00:00 — compare.we7qgdcg

Change/purpose: First full-budget comparison; unchanged common recipe and architectures, increased from 65,536-step smoke budget

Revision `89414204ce85`; recipe SHA256 `6cdd8042d7c848923d119aa65062adfe4c52a8cb5bf85ce4ab8e503b5a6b34cf`. [Report](results/connect4cnn/compare.we7qgdcg/REPORT.md) · [CSV](results/connect4cnn/compare.we7qgdcg/results.csv) · [Source/build hashes](results/connect4cnn/compare.we7qgdcg/protocol.json) · [GPU](results/connect4cnn/compare.we7qgdcg/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 73 | 13,279,232 | 21.88% | -0.5426 | 55,552 | 121.865 | 108,967 | 109,330 | 116,265 | 1.316 |
| tiny_cnn | 73 | 13,279,232 | 65.91% | 0.3286 | 151,680 | 135.649 | 97,894 | 98,136 | 99,146 | 1.365 |
| tiny_cnn | 74 | 13,279,232 | 66.89% | 0.3541 | 151,680 | 135.361 | 98,102 | 98,318 | 102,663 | 1.365 |
| state | 74 | 13,279,232 | 18.56% | -0.6057 | 55,552 | 121.714 | 109,102 | 109,354 | 113,224 | 1.316 |
| state | 75 | 13,279,232 | 15.09% | -0.6762 | 55,552 | 122.474 | 108,425 | 108,718 | 114,764 | 1.316 |
| tiny_cnn | 75 | 13,279,232 | 59.32% | 0.2082 | 151,680 | 136.614 | 97,203 | 97,448 | 101,845 | 1.365 |

Matched learner/core recipe; state and CNN parameter counts differ. See the report for checkpoint curves and actual evaluation counts.

Interpretation (2026-09-12 local time): all six training jobs and 24 checkpoint evaluations passed. Hardware was G240's RTX 5060 (8 GB), float32, with serial trials; the GPU was idle before launch. Arithmetic means across the three training seeds: state **18.51% wins / 108,831 process SPS / 122.017 s**, tiny CNN **64.04% wins / 97,733 process SPS / 135.875 s**. The CNN achieved higher win rates in each paired seed, with about 10.2% lower throughput and 11.4% more training wall time. This establishes useful learning at this budget; it does not establish convergence or isolate architecture from parameter count. The common recipe is not the untouched stock tuning. Nature, IMPALA, and Impoola still need implementation and measurement.

## 2026-09-13T05:09:12+00:00 — compare.9egh6y44

Change/purpose: Stock Connect4 training configuration, float32; same held-out evaluation protocol as the modified-recipe baselines

Revision `89414204ce85`; recipe SHA256 `536a53ff54dd958d85e9384cfe6ecbd7031aefc5f2d8d9244d48f756e0965af6`. [Report](results/connect4cnn/compare.9egh6y44/REPORT.md) · [CSV](results/connect4cnn/compare.9egh6y44/results.csv) · [Source/build hashes](results/connect4cnn/compare.9egh6y44/protocol.json) · [GPU](results/connect4cnn/compare.9egh6y44/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock_state | 73 | 13,238,272 | 98.51% | 0.9703 | 209,408 | 41.913 | 315,852 | 320,983 | 551,820 | 2.429 |
| stock_state | 74 | 13,238,272 | 99.06% | 0.9820 | 209,408 | 40.634 | 325,793 | 330,723 | 599,676 | 2.429 |
| stock_state | 75 | 13,238,272 | 99.04% | 0.9809 | 209,408 | 47.695 | 277,564 | 281,410 | 627,027 | 2.429 |

Stock training configuration in float32; common 64-agent evaluation. Training hypers differ from the earlier state/tiny-CNN comparison.
See the report for checkpoint curves and actual evaluation counts.

Interpretation: all three runs and 12 evaluations passed. Mean wins **98.87%**, mean process SPS **306,403**, mean training wall **43.414 seconds**. Stock hidden size 256 gives 209,408 parameters. All stock train/vec/policy/env/selfplay settings and async mode were checked against the resolved INIs. Training budget 13,272,299 truncates to 13,238,272 actual decisions in 101 rollout batches; checkpoint controls yield evaluations at 26/52/78/101 batches. Separate evaluation uses the earlier 64-agent, one-buffer, two-thread synchronous setup and seeds 10073/10074/10075. This is stock **training configuration in float32**, not a default-precision claim. The result shows that the smaller shared recipe substantially weakened the state baseline; the tiny CNN did not outperform stock-config PufferLib. Parameter count, optimization settings, and vectorization all differ.

## 2026-09-13T05:21:22+00:00 — compare.z10zc3xf

Change/purpose: Nature encoder smoke: validated forward/backward; first native training and checkpoint evaluation

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.z10zc3xf/REPORT.md) · [CSV](results/connect4cnn/compare.z10zc3xf/results.csv) · [Source/build hashes](results/connect4cnn/compare.z10zc3xf/protocol.json) · [GPU](results/connect4cnn/compare.z10zc3xf/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 65,536 | 0.35% | -0.9366 | 138,528 | 1.317 | 49,762 | 67,315 | 76,667 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T05:22:02+00:00 — compare.ltxw6alr

Change/purpose: Nature same-seed repeat to verify byte-identical training checkpoints

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.ltxw6alr/REPORT.md) · [CSV](results/connect4cnn/compare.ltxw6alr/results.csv) · [Source/build hashes](results/connect4cnn/compare.ltxw6alr/protocol.json) · [GPU](results/connect4cnn/compare.ltxw6alr/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 65,536 | 0.35% | -0.9366 | 138,528 | 1.317 | 49,759 | 68,001 | 77,442 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T05:35:01+00:00 — compare.2s__8t8l

Change/purpose: First adapted Nature CNN at the same 13.28M-decision common recipe as state and tiny CNN; stock training configuration recorded separately

Revision `89414204ce85`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.2s__8t8l/REPORT.md) · [CSV](results/connect4cnn/compare.2s__8t8l/results.csv) · [Source/build hashes](results/connect4cnn/compare.2s__8t8l/protocol.json) · [GPU](results/connect4cnn/compare.2s__8t8l/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nature_cnn | 73 | 13,279,232 | 77.10% | 0.5420 | 138,528 | 162.552 | 81,692 | 81,900 | 82,193 | 1.552 |
| nature_cnn | 74 | 13,279,232 | 88.33% | 0.7686 | 138,528 | 164.117 | 80,913 | 81,076 | 83,865 | 1.552 |
| nature_cnn | 75 | 13,279,232 | 70.68% | 0.4457 | 138,528 | 166.168 | 79,915 | 80,068 | 81,722 | 1.552 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T06:11:19+00:00 — compare.ij5_xpo7

Change/purpose: IMPALA/Impoola training smoke after numerical validation; original SAME pooling, base widths, common recipe

Revision `43850f172637`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.ij5_xpo7/REPORT.md) · [CSV](results/connect4cnn/compare.ij5_xpo7/results.csv) · [Source/build hashes](results/connect4cnn/compare.ij5_xpo7/protocol.json) · [GPU](results/connect4cnn/compare.ij5_xpo7/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 65,536 | 0.00% | -0.9377 | 270,496 | 6.679 | 9,812 | 10,513 | 10,744 | 3.503 |
| impoola_cnn | 73 | 65,536 | 0.00% | -0.9491 | 151,712 | 6.633 | 9,880 | 10,531 | 10,682 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T06:12:52+00:00 — compare.sqngvlom

Change/purpose: IMPALA/Impoola identical-seed smoke repeat for training determinism

Revision `43850f172637`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.sqngvlom/REPORT.md) · [CSV](results/connect4cnn/compare.sqngvlom/results.csv) · [Source/build hashes](results/connect4cnn/compare.sqngvlom/protocol.json) · [GPU](results/connect4cnn/compare.sqngvlom/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 65,536 | 0.00% | -0.9377 | 270,496 | 6.630 | 9,885 | 10,510 | 10,741 | 3.503 |
| impoola_cnn | 73 | 65,536 | 0.00% | -0.9491 | 151,712 | 6.580 | 9,960 | 10,605 | 10,758 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-13T08:16:45+00:00 — compare.l6d5sbk2

Change/purpose: First IMPALA and Impoola full-budget common-recipe comparison; shared original-SAME backbone, flatten versus GAP, base widths 16/32/32

Revision `43850f172637`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/compare.l6d5sbk2/REPORT.md) · [CSV](results/connect4cnn/compare.l6d5sbk2/results.csv) · [Source/build hashes](results/connect4cnn/compare.l6d5sbk2/protocol.json) · [GPU](results/connect4cnn/compare.l6d5sbk2/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| impala_cnn | 73 | 13,279,232 | 98.99% | 0.9798 | 270,496 | 1227.228 | 10,821 | 10,824 | 10,734 | 3.503 |
| impoola_cnn | 73 | 13,279,232 | 61.75% | 0.2350 | 151,712 | 1224.202 | 10,847 | 10,851 | 10,892 | 3.496 |
| impoola_cnn | 74 | 13,279,232 | 73.59% | 0.4718 | 151,712 | 1228.583 | 10,809 | 10,812 | 10,814 | 3.496 |
| impala_cnn | 74 | 13,279,232 | 99.01% | 0.9801 | 270,496 | 1232.505 | 10,774 | 10,778 | 10,744 | 3.503 |
| impala_cnn | 75 | 13,279,232 | 99.57% | 0.9914 | 270,496 | 1229.103 | 10,804 | 10,808 | 10,835 | 3.503 |
| impoola_cnn | 75 | 13,279,232 | 73.14% | 0.4628 | 151,712 | 1224.713 | 10,843 | 10,846 | 10,721 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

Interpretation: all six training jobs and 24 held-out evaluations passed; all saved checkpoints were finite. Resolved train/vec/policy/env/selfplay settings matched across trials and against the completed Nature baseline. IMPALA averaged **99.19% wins / 10,800 process SPS / 1,229.612 s**, with seed win rates 98.99/99.01/99.57%. Impoola averaged **69.49% / 10,833 SPS / 1,225.833 s**, with 61.75/73.59/73.14%. IMPALA exceeded Impoola in each paired seed; GAP's parameter reduction brought little throughput improvement in this implementation. IMPALA's shared-recipe training time was about 7.5 times Nature's. These timings combine architecture cost and our initial native implementation overhead; individual kernels have not been profiled. The stock state baseline achieved 98.87% in 43.414 s with a different recipe. No generalization or SOTA claim follows from this Connect4 comparison.

## 2026-09-13 — Native PROTEIN custom-CNN canaries

Change: optional INI-to-encoder construction settings, numeric encoder ID 1, and a bounded residual CNN family. Same Connect4CNN pixels, hidden-128 single-layer MinGRU, and common learner settings. PROTEIN varies channels (8/16/32), residual blocks per stage (0/1/2), flatten/GAP, and training budget. Source baseline `ed4b7655` plus archived source hashes. No reference-family search or randomized environments yet.

Reports: [first canary](results/connect4cnn/sweep.r3qepq1g/REPORT.md), [finalized canary](results/connect4cnn/sweep.7s4ovb72/REPORT.md). Each completed 12 native trials covering 8 distinct architectures, including one model-guided proposal (`gp_obs=11`). Finalized sweep wall was **19.409 seconds**, excluding compilation and the subsequent sidecar. Actual decisions ranged from **14,336 to 32,768**; native integer/batch truncation, including floating-point rounding at a log-budget lower bound, is retained in the reported actual counts. Parameters ranged from **55,920 to 684,224**. First-run observed native average SPS ranged roughly **5,689–33,437**; these very short timings include substantial fixed overhead and are not a speed ranking. Use each report's exact trial results.

All completed checkpoint arrays were finite and had the expected parameter counts; all effective non-swept learner/core settings matched. The first 11 checkpoint hashes matched across repeats. Native float32 numerical and eager/graph/rollout checks passed for all 18 legal shapes; finite differences also passed for the smallest-width variants. Default tiny-encoder training matched the old binary exactly (SHA256 `d418cf69c822e2b04bda3ff03deb42a717bff1e4f0117a09ea52e429dd4046d5`). Vanilla state Connect4 and the largest sampled CNN both passed checkpoint reload/evaluation.

Both canaries generated 12 offline W&B runs. Re-ingestion of the finalized canary left that count unchanged. Sidecar JSON preserves final PROTEIN observations separately from binned INI histories, plus architecture/checkpoint hashes. Reported final score/cost use native stdout rounding (four score decimals, two time decimals); SPS divides actual steps by that cost. Training win rates at these canary budgets were near zero and establish no learning advantage.

Temporary Python preparation/reporting glue is permitted only for this first end-to-end proof; native tooling is the delivery target. The search, worker scheduler, CNN, and training are already C/CUDA. Online W&B, BF16, larger-budget performance, other architecture families, and other environments remain unvalidated in this workflow.

## 2026-09-14T18:34:07+00:00 — confirm-canary.hdq_fms8

Change/purpose: Confirmation CANARY; plumbing only

Revision `9ed6bc2a192e`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/confirm-canary.hdq_fms8/REPORT.md) · [CSV](results/connect4cnn/confirm-canary.hdq_fms8/results.csv) · [Source/build hashes](results/connect4cnn/confirm-canary.hdq_fms8/protocol.json) · [GPU](results/connect4cnn/confirm-canary.hdq_fms8/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 9173 | — | FAILED | — | — | — | — | — | — | — |
| nature_cnn | 9173 | 65,536 | 0.00% | -0.9549 | 138,528 | 1.267 | 51,723 | 71,619 | 80,814 | 1.552 |
| flex_fast | 9173 | — | FAILED | — | — | — | — | — | — | — |
| impala_cnn | 9173 | 65,536 | 0.00% | -0.9615 | 270,496 | 6.629 | 9,887 | 10,565 | 10,750 | 3.503 |
| flex_small | 9173 | — | FAILED | — | — | — | — | — | — | — |
| impoola_cnn | 9173 | 65,536 | 0.00% | -0.9593 | 151,712 | 6.579 | 9,961 | 10,580 | 10,778 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-14T18:58:40+00:00 — confirm-canary._0y3shor

All six native jobs, 24 checkpoint evaluations and six online W&B uploads passed. The preceding attempt failed for Flex because newer architecture INI keys were absent; isolated per-job configs fixed it without native source changes. Both attempts are archived. CPU frozen-selection/history tests and sidecar payload checks passed; canary performance is not a learning benchmark.

After validation, the authorized full confirmation was queued in tmux `cnn2-confirm-20260914`; launcher log: `build/connect4cnn/confirmation-launch-20260914.txt`. Startup verification found GoldenEye evaluation PID 937816 using the GPU, so `confirm_when_idle.sh` waits up to one hour before starting. No other process was interrupted. Planned 30 jobs × 13,312,000 decisions, 390 checkpoint evaluations; fixed [protocol](CONFIRMATION_PROTOCOL.md), online project `kinvert-k/cnn2`. The generated `confirm.*` group/path will appear in the launcher log. This is a launch record, not confirmation completion; preserve any later contention/failures when analyzing timing.

Change/purpose: Confirmation CANARY; plumbing only

Revision `9ed6bc2a192e`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/confirm-canary._0y3shor/REPORT.md) · [CSV](results/connect4cnn/confirm-canary._0y3shor/results.csv) · [Source/build hashes](results/connect4cnn/confirm-canary._0y3shor/protocol.json) · [GPU](results/connect4cnn/confirm-canary._0y3shor/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 9173 | 65,536 | 0.00% | -0.9599 | 160,736 | 1.267 | 51,720 | 73,646 | 85,645 | 1.431 |
| nature_cnn | 9173 | 65,536 | 0.00% | -0.9549 | 138,528 | 1.267 | 51,736 | 69,956 | 81,777 | 1.552 |
| flex_fast | 9173 | 65,536 | 0.00% | -0.9571 | 261,856 | 1.267 | 51,729 | 70,717 | 83,343 | 1.459 |
| impala_cnn | 9173 | 65,536 | 0.00% | -0.9615 | 270,496 | 6.680 | 9,811 | 10,488 | 10,715 | 3.503 |
| flex_small | 9173 | 65,536 | 0.00% | -0.9539 | 109,768 | 1.267 | 51,735 | 72,901 | 83,689 | 1.418 |
| impoola_cnn | 9173 | 65,536 | 0.00% | -0.9593 | 151,712 | 6.630 | 9,885 | 10,517 | 10,712 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-09-14T23:26:40+00:00 — confirm.ol9tcj5k

Change/purpose: Frozen five-seed Connect4CNN architecture confirmation v1

Revision `9ed6bc2a192e`; recipe SHA256 `5dc685bea68d293cbce8a2cc367584dab252d2eb5d33d344d5dcb8dfd8bbad4b`. [Report](results/connect4cnn/confirm.ol9tcj5k/REPORT.md) · [CSV](results/connect4cnn/confirm.ol9tcj5k/results.csv) · [Source/build hashes](results/connect4cnn/confirm.ol9tcj5k/protocol.json) · [GPU](results/connect4cnn/confirm.ol9tcj5k/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 173 | 13,312,000 | 66.73% | 0.3346 | 160,736 | 145.820 | 91,290 | 91,513 | 94,263 | 1.431 |
| nature_cnn | 173 | 13,312,000 | 60.89% | 0.2408 | 138,528 | 159.652 | 83,382 | 83,564 | 83,733 | 1.552 |
| flex_fast | 173 | 13,312,000 | 65.35% | 0.3104 | 261,856 | 148.894 | 89,406 | 89,594 | 91,831 | 1.459 |
| impala_cnn | 173 | 13,312,000 | 96.30% | 0.9260 | 270,496 | 1227.800 | 10,842 | 10,846 | 10,794 | 3.503 |
| flex_small | 173 | 13,312,000 | 87.09% | 0.7417 | 109,768 | 146.226 | 91,037 | 91,247 | 94,278 | 1.418 |
| impoola_cnn | 173 | 13,312,000 | 78.50% | 0.5700 | 151,712 | 1228.714 | 10,834 | 10,838 | 10,826 | 3.496 |
| nature_cnn | 174 | 13,312,000 | 85.96% | 0.7202 | 138,528 | 158.501 | 83,987 | 84,180 | 85,108 | 1.552 |
| flex_fast | 174 | 13,312,000 | 58.20% | 0.1649 | 261,856 | 146.831 | 90,662 | 90,844 | 93,577 | 1.459 |
| impala_cnn | 174 | 13,312,000 | 97.69% | 0.9537 | 270,496 | 1230.243 | 10,821 | 10,824 | 10,838 | 3.503 |
| flex_small | 174 | 13,312,000 | 67.98% | 0.3596 | 109,768 | 142.466 | 93,440 | 93,662 | 97,504 | 1.418 |
| impoola_cnn | 174 | 13,312,000 | 95.82% | 0.9165 | 151,712 | 1236.616 | 10,765 | 10,771 | 10,760 | 3.496 |
| flex_quality | 174 | 13,312,000 | 90.82% | 0.8172 | 160,736 | 141.965 | 93,770 | 94,000 | 96,612 | 1.431 |
| flex_fast | 175 | 13,312,000 | 78.39% | 0.5677 | 261,856 | 146.427 | 90,912 | 91,122 | 90,837 | 1.459 |
| impala_cnn | 175 | 13,312,000 | 97.21% | 0.9442 | 270,496 | 1235.913 | 10,771 | 10,775 | 10,795 | 3.503 |
| flex_small | 175 | 13,312,000 | 87.27% | 0.7454 | 109,768 | 142.930 | 93,136 | 93,360 | 96,411 | 1.418 |
| impoola_cnn | 175 | 13,312,000 | 73.05% | 0.4610 | 151,712 | 1224.228 | 10,874 | 10,878 | 10,929 | 3.496 |
| flex_quality | 175 | 13,312,000 | 82.29% | 0.6459 | 160,736 | 143.821 | 92,559 | 92,787 | 95,148 | 1.431 |
| nature_cnn | 175 | 13,312,000 | 74.67% | 0.4934 | 138,528 | 158.265 | 84,112 | 84,296 | 85,142 | 1.552 |
| impala_cnn | 176 | 13,312,000 | 99.18% | 0.9837 | 270,496 | 1233.977 | 10,788 | 10,792 | 10,823 | 3.503 |
| flex_small | 176 | 13,312,000 | 89.25% | 0.7851 | 109,768 | 143.017 | 93,080 | 93,293 | 95,366 | 1.418 |
| impoola_cnn | 176 | 13,312,000 | 77.54% | 0.5508 | 151,712 | 1230.621 | 10,817 | 10,821 | 10,828 | 3.496 |
| flex_quality | 176 | 13,312,000 | 87.58% | 0.7515 | 160,736 | 144.569 | 92,080 | 92,322 | 95,937 | 1.431 |
| nature_cnn | 176 | 13,312,000 | 69.81% | 0.3962 | 138,528 | 158.051 | 84,226 | 84,407 | 86,887 | 1.552 |
| flex_fast | 176 | 13,312,000 | 74.57% | 0.4914 | 261,856 | 146.334 | 90,970 | 91,197 | 94,300 | 1.459 |
| flex_small | 177 | 13,312,000 | 58.45% | 0.1690 | 109,768 | 141.761 | 93,904 | 94,110 | 98,694 | 1.418 |
| impoola_cnn | 177 | 13,312,000 | 75.56% | 0.5111 | 151,712 | 1228.198 | 10,839 | 10,842 | 10,849 | 3.496 |
| flex_quality | 177 | 13,312,000 | 70.91% | 0.4181 | 160,736 | 146.168 | 91,073 | 91,291 | 92,496 | 1.431 |
| nature_cnn | 177 | 13,312,000 | 75.43% | 0.5216 | 138,528 | 159.056 | 83,694 | 83,885 | 84,296 | 1.552 |
| flex_fast | 177 | 13,312,000 | 81.80% | 0.6361 | 261,856 | 147.123 | 90,482 | 90,681 | 94,826 | 1.459 |
| impala_cnn | 177 | 13,312,000 | 96.87% | 0.9374 | 270,496 | 1231.117 | 10,813 | 10,817 | 10,817 | 3.503 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-10-05T22:43:20+00:00 — compare.qq5_i44p

Change/purpose: Hardware CANARY: four-model build/train/reload plumbing only; exclude from speed claims

Revision `a6fae230efbf`; recipe SHA256 `d5833065dda8eda2035e70797d6aec84c9b377546e30bc003ce736a12424ae6e`. [Report](../build/connect4cnn/compare.qq5_i44p/REPORT.md) · [CSV](../build/connect4cnn/compare.qq5_i44p/results.csv) · [Source/build hashes](../build/connect4cnn/compare.qq5_i44p/protocol.json) · [GPU](../build/connect4cnn/compare.qq5_i44p/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 9173 | 65,536 | 0.00% | -1.0000 | 160,736 | 1.969 | 33,283 | 59,373 | 73,627 | 1.431 |
| nature_cnn | 9173 | 65,536 | 0.00% | -1.0000 | 138,528 | 1.617 | 40,536 | 52,814 | 62,074 | 1.552 |
| impala_cnn | 9173 | 65,536 | 0.00% | -1.0000 | 270,496 | 6.976 | 9,395 | 10,085 | 10,249 | 3.503 |
| impoola_cnn | 9173 | 65,536 | 0.00% | -1.0000 | 151,712 | 7.077 | 9,261 | 9,965 | 10,385 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-10-05T23:33:51+00:00 — compare.a2lt6qft

Change/purpose: Fixed four-model hardware comparison; seed 173; same-revision cross-host timing, not a new sweep

Revision `a6fae230efbf`; recipe SHA256 `d5833065dda8eda2035e70797d6aec84c9b377546e30bc003ce736a12424ae6e`. [Report](../build/connect4cnn/compare.a2lt6qft/REPORT.md) · [CSV](../build/connect4cnn/compare.a2lt6qft/results.csv) · [Source/build hashes](../build/connect4cnn/compare.a2lt6qft/protocol.json) · [GPU](../build/connect4cnn/compare.a2lt6qft/gpu.txt)

| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flex_quality | 173 | 13,312,000 | 91.96% | 0.8393 | 160,736 | 179.246 | 74,267 | 74,508 | 75,526 | 1.431 |
| nature_cnn | 173 | 13,312,000 | 76.19% | 0.5237 | 138,528 | 196.779 | 67,650 | 67,815 | 68,734 | 1.552 |
| impala_cnn | 173 | 13,312,000 | 97.73% | 0.9593 | 270,496 | 1293.430 | 10,292 | 10,296 | 10,484 | 3.503 |
| impoola_cnn | 173 | 13,312,000 | 76.98% | 0.5562 | 151,712 | 1281.174 | 10,390 | 10,395 | 10,556 | 3.496 |

Matched learner/core recipe; state and CNN parameter counts differ.
See the report for checkpoint curves and actual evaluation counts.

## 2026-10-06 — quality versus Nature six-game RTX 5060 smoke

Explicit short-local-run authorization. [Full protocol and process-SPS table](NATURE_MULTIGAME_5060_SMOKE.md)
and [audited evidence](results/nature-multigame-5060-20261006/README.md).
All 12 training jobs, 82 fixed-drawing evaluations, 24 exact-byte repeat/eager
checks and 12 repeated full-policy layouts pass. Six games/all 41 drawings,
one paired seed 58173, 65,536 decisions/job; matched within-game learners.
Campaign 93.243s; training processes 9.679s. G240 RTX 5060/driver 591.86/CUDA
compiler 12.8.93; process costs include startup/checkpoint writes. Original
native SPS arrays are retained separately. Tiny-budget performance remains
poor; no frontier/quality selection or independent math certification.
No IMPALA/Impoola training, 5090 launch, dependency change or push.

## 2026-10-06 — saved-candidate evaluator comparisons on RTX 5060

[Protocol](CANDIDATE_EVALUATOR_ACCEPTANCE.md) ·
[Retained evidence](results/candidate-evaluator-5060-20261006/README.md).
Ten ours/Nature cases (Pong/Flappy/Breakout/Snake/Maze, drawing 0), 40 native
graph/repeat/eager/independent-tail processes and 520 assigned episode
executions pass exact episode/action-row comparisons and offline audit.
The final wave's ID 16 is rerun alone at the original 16-slot neural batch.
Actual existing 65,536-decision checkpoints/full INIs from seed 58173 are
preserved, with native full-policy byte/layout checks before evaluation.
Inference process time totals 27.950s; first-to-last native interval 44.326s.
Every hardware receipt identifies G240 RTX 5060/driver 591.86, distinct from
5090 evidence. No training or throughput/quality/frontier claim. The failed
initial host export/binding lookup and absent-cases preparation log are retained;
corrected nine-test export uses fresh output paths. Separate real 62-case
all-drawing export is host-only, with no additional tail checks executed.
Other drawings/batches/models/math/memory/reload/learning/selection stay open.

## 2026-10-06 — isolated float64 CUDA oracle versus native encoders

[Protocol and error table](GPU_ENCODER_NUMERICAL_SMOKE.md) ·
[Actual evidence](results/gpu-encoder-smoke-5060-20261006/README.md).
Seven host guards pass. Actual RTX 5060 execution passes 18 quality/Nature
H128/B1/3/64 cases, two independent workers, 144 eager/graph harness calls,
all output/encoder-parameter gradients and exact device-parameter/process
bytes, two dyadic CUDA self-tests and 28 selected CUDA finite differences.
Offline full-array/process/perturbation/hash audit passes. Worker times
0.865086/0.814964s; first-launch to last-completion 1.779923s, not kernel/training
timing. Independent reference math runs entirely in CUDA float64; NumPy only
generates/inspects arrays and compares scalars/bytes. No CPU CNN/Torch/new
dependency/production change. Original paired alias worker's NumPy reference
is clarified, not executed or relabeled. B2048/other widths/general/flex2/
IMPALA/Impoola/full-policy/concurrency/reload/learning/frontier remain separate.

## 2026-10-06 — learner-batch encoder numerical failure and ReLU diagnostic

[Protocol and next-check design](GPU_ENCODER_LEARNER_BATCH.md) ·
[Retained evidence](results/gpu-encoder-learner-5060-20261006/README.md).
Nine host/configuration tests pass; actual v2 preparation/inspection succeeds.
The actual 5060 worker fails quality H256/B2048/nonblank after 33 successful
cases. Planned: 72 cases/two workers/576 calls/80 selected probes. Executed:
34 started cases/133 harness calls/16 selected probes, one failed worker
lasting 2.116921885s. Nature/second worker were not reached. All 26 failing
gradients are first-conv channel 12; fixed tolerances and failure are preserved.

A separate actual CUDA diagnostic uses unchanged libraries and original first-
conv weights with two identity readouts. Across that channel's 202,752
activations, one native-zero/reference-positive ReLU mismatch occurs at image
1723, row 5, column 10 (+8.847564459e-9 in float64). The observed bias-gradient
delta times its 7x7 image patch matches weight-gradient deltas with residual
at most 1.358785075e-6. Supports a rounding explanation; probe downstream
weights differ, so no acceptance or universal backward correctness is implied.
No native/reference/production code, tolerance, seed, system dependency,
training run, 5090 job or push changed. Prior successful v1 remains untouched.
Next independent layer-forward/float32-branch backward protocol is planned,
not implemented; full learner/math/memory/learning/frontier gates stay open.

## 2026-10-06 — authorized short-time RTX 5060 mini allocation

[Frozen protocol](MINI_SHORT_BUDGET_5060.md). Kinvert explicitly requested small
5060 runs at the short-time end, avoiding long accuracy training. New bounded
mini mode preserves old smoke/development behavior and permits only frozen
H128 quality plus Nature. All41 host/configuration/reporting checks pass with
the process-local runtime helper; initial missing-loader-path test failures
are retained separately. No production CNN/learner/system dependency changed.

Fresh 18-family native build registry and 24 native full-policy declarations
prepare successfully; all328 drawing/checkpoint evaluations are assigned before
launch. Seeds59173/59174, 524,288 decisions/job, checkpoints262,144/524,288,
17 exact episodes/16 slots, same per-game learners/full41-drawing training
mixtures. Pong512-decision/Breakout2048-frame caps retain their usual bounds/
partial outcomes. Campaign deadline600s, per-process20s. Final drawing0 gets
48 repeat/eager checks; IMPALA/Impoola binaries compile but aren't executed.
Results and failures must be audited after completion, never inferred from
these planned counts. No 5090 launch, long sweep, new architecture or push.

### Completed outcome — short-time mini on RTX 5060

[Audited report, complete curves and SPS table](results/mini-short-5060-20261006/README.md).
All 24 training jobs/48 checkpoints/328 evaluations/48 repeat-eager checks/
24 repeated layouts/6,392 assigned episode executions pass; zero missing cells.
Campaign 379.587885s, training processes 88.709230s, evaluation processes
201.206052s. Per-model training averages range 2.59–6.55s across games.
Ours has 9.18–22.23% higher process SPS, with no steady-state speed claim.
Connect4/Pong/Flappy remain zero on all drawings; other metrics are small and
mixed, not replicated Pareto dominance. Every checkpoint/drawing/seed remains.

Supplemental raw native metrics expose three inconsistent uptime values
exceeding entire process duration (seed 59173: Connect4/ours, Pong/Nature,
Maze/Nature). Captured `wall_clock()` uses CLOCK_REALTIME; cause isn't established.
Their native-average-SPS comparison is unqualified. Raw arrays remain unchanged;
monotonic process/checkpoint timing supplies these tables/curves. No timer,
production policy/learner, system stack, 5090 job or push changed. Completed
allocation is not scheduled to repeat; wider learning/math/inference gates remain.
