# Next realistic-image campaign

Implementation specification; the overnight run is **not ready to launch**.
Use fresh history. The45 tiny-image observations remain historical evidence.

## Frozen comparison

Six games/all41 drawings, direct full-size RGB frames, H128/L1. Nature and every
custom CNN receive identical pixels, per-game learners, vector settings, replay,
budgets, checkpoints and seed lists. Different aspect ratios are intentional.
No hidden resize to36x44. Grayscale support is separate from RGB qualification.
Our local Nature reference is the native32/64/64 valid-convolution stack with
8/s4,4/s2,3/s1 kernels and a flatten-to-common-H128 readout. Declare that head
adaptation; do not call the whole recurrent policy an exact original DQN model.
On larger frames its flattened head grows substantially. Retain its parameter,
memory and timing receipts, and distinguish an architecture advantage in this
protocol from an Atari-wide claim or an unmeasured optimized-Nature variant.

Development training replicas:7/42/69; episode seeds7007/7008/7009. Reserve
confirmation training replicas104729/130363/155921 and final episode
seeds420/1337/9001 before selecting a graph. Appearance/optimizer seeds also
stay fixed. **Never sweep seeds.** Confirmation cannot feed PROTEIN, learner
selection or architecture revisions. Existing seed7 plumbing is development.
These lists must also be enforced by the remaining campaign implementation.

Different episode seeds are not unseen games, styles or Maze levels. Maze's
deterministic table remains shared unless a new prefix/table contract is added.
Do not call the41 seen drawings a held-out visual benchmark.

Calibrate substantial per-game budgets on the new inputs with eight
rollout-aligned checkpoints including the final one. Old tiny-image SPS and
512-decision plumbing do not establish those budgets. Require useful learning
in each game's paired development controls before search eligibility. Retain
unsuccessful calibration recipes/cost; do not drop a failed game from selection.

## Curve and coverage objective to implement

The v3 plumbing loop still uses a final-checkpoint average. This replacement
needs its own objective version; never relabel previous feedback.

1. Freeze an audited Nature calibration panel. Its eight monotonic checkpoint
   times become anchors for each game/training replica. Freeze finite per-game
   score offset/meaningful target from development calibration, not search wins.
2. At each anchor use the candidate's latest chronological **actually
   affordable** checkpoint. No best-so-far return, interpolation, extrapolated
   learning, best seed or selected drawing. Preserve declines. If none is
   affordable, use a declared measured untrained control, not invented zero.
3. Normalize lower/upper scores to[0,1] with frozen offset/target. Average
   drawings equally inside a replica, replicas equally inside a game. Keep
   Pong's conservative lower bound separate from its upper bound.
4. Across six game means use a balanced geometric mean at each anchor:
   `(product(0.05 + game_score) ** (1/6)) - 0.05`. It is bounded[0,1], permits
   feedback when a game is zero and penalizes weak games more than an arithmetic
   mean. Freeze0.05 before search. Also report the arithmetic mean, every game
   and versus-Nature win/tie/loss coverage; the scalar is not a statistical test.
5. Average eight balanced values for native PROTEIN quality. Return actual
   final monotonic training seconds, charged once per job, as cost. Preserve
   full game/drawing/seed/checkpoint curves independently of the scalar.

Incomplete panels produce no observation. Retain duplicates, failures and every
checkpoint. Final claims require fresh paired confirmation and score/time
comparisons; this selection rule alone proves neither generalization nor SOTA.

## Calibration and stopping

Calibrate the5090's actual new-image workloads on that hardware; the5060 is a
separate collection. Record startup/checkpoint time, sustained training time,
process/native SPS, numerical-check/evaluation/audit/report time and measured
VRAM. Native policy layout/storage/index rejection must precede CUDA validation.
A storage floor is not a measured peak-memory certificate.

Reserve an entire next all-game panel, numerical checks, audit, final native
feedback replay and reporting before admitting a proposal. Estimate
conservatively from actual5090 calibration and the search envelope. Stop between
complete panels when another cannot fit inside approximately12h. Record
`budget_stop` separately from failure. Unexpected hard-deadline exhaustion keeps
partial/failure evidence; never submit a truncated mean. Numerical workers now
share the outer deadline; that is not yet complete-panel admission logic.

## Evidence affecting calibration

The5060 eight-checkpoint packet completed six512-decision trainings,48 finite
checkpoints,280/328 drawing evaluations and10 repeat/eager checks. Its first
Maze evaluation failed at30s. That packet remains failed. A separate final-Maze
probe completes the same five assignments/native horizons in36.637–36.839s
across graph/repeat/eager, with exactly matching CSV rows. Do not shorten the
horizon, exclude the unfinished episode or report the failed packet as complete.

Captured Flappy output reports about63% of profiled work in Copy for the tiny
four-actor full-RGB run. This is a bottleneck lead, not sustained bandwidth
evidence. `src/pufferl.cu` uploads float32 CPU observations each actor step and
transposes full rollout observations. The host frames are already pinned with
`cudaHostAllocPortable`; Copy combines observation uploads and action downloads.
Profile those paths before assuming CNN kernels dominate. Byte-frame/normalization
or upload changes require
explicit contracts and matched baseline acceptance, not silent quantization.

`image_fast_spatial.ini` and `image_fast_spatial_deep.ini` are unexecuted targeted
controls. Coarse spatial readouts with pooling preserve position that GAP can
discard. Compare them under controlled native recipes before wide search.

Final handoff: executable setup/build/canary/calibration/run/report commands,
uncalibrated-input rejection, reporting reserve, no5090 coding required. Commit
and push verified delivery only toKinvert/PufferLib:cnn-research, never official
upstream or as the eventual clean PR. No long campaign runs onG240.
