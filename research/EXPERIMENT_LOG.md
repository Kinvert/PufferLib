# CNN experiment history

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
