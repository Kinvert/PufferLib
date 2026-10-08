# Temporary native cross-game PROTEIN feedback

October 7 delivery update: Kinvert authorizes this temporary research code on
his fork's `cnn-research` branch. Read
[NEXT_5090_FEEDBACK.md](../NEXT_5090_FEEDBACK.md) for the two-command handoff.
Do not push to PufferAI or treat it as the eventual upstream PR. Later implement
the selected architecture separately with minimal, clean normal PufferLib code.

Status October 7: **the bounded 5090 GPU feedback canary completed**; read
[the received evidence](results/feedback-canary-5090-20261007/README.md).
All three trials / 18 jobs / 123 drawing evaluations / 36 repeat-eager checks
pass the remote audit, with local artifact checks. All proposals round to C16,
with explicit duplicates. No long search or restart of completed allocations
is scheduled. No learning, numerical acceptance, speed or Pareto win follows
from this plumbing canary. Earlier preparation-only statements remain historical.

## Actual loop

```text
Frozen CNN search space + fixed learner recipe for each game
                              |
                              v
     Native PROTEIN replays previous suggestions/observations
                  and proposes ONE CNN architecture
                              |
              optimizer process exits completely
                              |
                              v
     Fresh random weights: train that same architecture on
      Connect4 / Pong / Flappy / Breakout / Snake / Maze
        (paired seeds, fixed per-game learners, mixed drawings)
                              |
                              v
     Exact deterministic evaluation on ALL 41 fixed drawings
                 + offline native receipt audit
                              |
                              v
     Normalize each game's final-checkpoint scores separately
        Average drawings/seeds within each game, then games
       Cost = sum training seconds once per game/training seed
                              |
                              v
        Append score + cost + normalized CNN coordinates
                  to the native feedback ledger
                              |
                              +------> NEXT PROTEIN PROPOSAL
```

`src/protein_feedback.cu` directly calls existing native
`protein_sweep_create`, `protein_sweep_suggest` and `protein_sweep_observe`.
Two include/dispatch hooks in `src/pufferl.cu` exist only when built with
`PUFFER_RESEARCH_PROTEIN_FEEDBACK`. Normal builds omit that mode. Production
CNN kernels, learner and standard single-game sweep implementation are unchanged
by this prototype. Other pre-existing research changes remain separate.

`research/cross_game_feedback.py` is external preparation, supervision, scalar
reduction and audit glue. Native CUDA PROTEIN chooses the coordinates; Python
implements no optimizer, trainer or CPU neural reference. No dataset/pretraining;
each game/seed starts with fresh random weights. The existing generic candidate
panel remains nonadaptive; this **new outer supervisor** supplies feedback.
Earlier panels/archives do not acquire new capabilities retroactively.

## Objective, costs and fixed learners

Declare each game's fixed `offset` and `target` before searching. Normalize using
`clip((exact_score-offset)/(target-offset), 0, 1)`. Average drawings/seeds within
each game, then weight all six games equally. Ten Connect4 drawings therefore
do not outweigh five Breakout drawings. Pong uses the conservative lower bound;
retain its upper bound/censored episodes separately, not as confidence intervals.

Tiny recipe anchors: Connect4/Pong/Maze 0→1, Flappy 0→100, Breakout 0→20,
Snake ending length 1→20. **These are declared development choices, not calibrated
targets or a scientifically justified general ranking.** Calibrate learners,
anchors and caps before a serious search, freeze them, and reserve independent
confirmation seeds/tasks. Saturation or poor anchors can conceal differences.

Use every game's final declared checkpoint, never its best checkpoint/seed.
Retain all earlier checkpoints and declines. Incomplete/nonfinite/duplicate/
unassigned/reversed score cells cannot supply successful feedback. Failure stops
the campaign with partial evidence. Automatic failure feedback/recovery is not
implemented; do not invent partial means or zero performance for a timeout.

Cost sums native post-rename monotonic checkpoint training seconds once per job,
not once per drawing. Keep process SPS, startup/checkpoint overhead, optimizer
overhead and evaluation time separate. Do not substitute earlier inconsistent
native realtime uptime. The aggregate objective selects development candidates;
it does not replace full per-game/per-drawing curves or establish a frontier.

Learners are fixed per game across candidates. Optional
`prepare --learner-recipe ENV=INI` overlays use the existing strict panel contract
and are frozen for every trial. That contract forbids world/core/seed/budget
overrides. Encoder4/H128/L1 remain fixed; encoder5 retains its separate gates.
Deterministic per-slot mixed training assignments share one appearance seed/
catalog, and exact evaluation uses all fixed drawings with paired episode IDs,
world RNG and batch sizes. Raw rewards from different games are never averaged.

October 7 opt-in development preparation adds `[search] learner_mode=per_game`
and optional `[task.ENV]` sections with `steps` / `checkpoint_steps`. Unspecified
games use the root budget. Numeric `--learner-recipe ENV=INI` overlays can then
choose rollout/minibatch geometry per game, held fixed across every suggestion.
Native scalar preflight rejects rounded decision budgets, misaligned checkpoint
cadence and zero optimizer updates before a proposal. Feedback-v2 / panel-v3
freeze and inspect these proofs offline; legacy contracts are unchanged.

Depth can be swept from a depth-one default through three stages using all 18
CNN coordinates. Audits bind inactive coordinates too; effective-graph duplicates
are retained. Native random-warmup controls remain in `[protein]`. These controls
enable a broader search; they do not qualify its graphs or calibrate its learners.
`bash research/run_feedback_5090.sh prepare-development` exercises GPU-free
preparation using explicitly uncalibrated examples. It schedules no GPU work and
cannot be passed to the launcher's old bounded `run` command. Read
[the next experiment](NEXT_FEEDBACK_EXPERIMENT.md) before planning a campaign.

## Replay and retained evidence

Native optimizer processes reserve the shared hardware lock and reject busy
GPUs. Each exits before training/inference, avoiding a resident GP context that
would contaminate memory/idle checks/timing. A fresh process reconstructs the
previous native sequence from its frozen ledger and refuses changed float32
normalized proposals before observing them. GPU replay passes the three-trial
5090 canary; larger histories and other hardware remain untested.

Replay repeats earlier GP work: overhead grows with trial count and must be
measured before thousands of trials. Standard PROTEIN merges sufficiently close
normalized observations, so stored success count can be smaller than completed
observation calls. Architectural duplicates remain explicitly recorded, never
discarded or counted as independent seeds. A final extra **unused** proposal
replays the last feedback; it has no associated training job.

Keep each proposal's ledger/JSON/command/clocks/log and each panel's full configs,
owned source/build identities, checkpoint bytes/times, exact episode CSVs,
registrations and audits. Offline `audit` re-audits native panels, recomputes
feedback, verifies fixed learners across trials, and checks ledger/proposal chains
and nonoverlapping optimizer/panel clocks. It executes no GPU query or policy.
Use fresh output paths. Packets bind local paths; remote agents regenerate local
builds/packets. Owned-source closure does not certify all vendor/system/link inputs.

## Commands and bounded allocation

Existing venv and process-local toolkit/NCCL only; no system changes. Build and
prepare are GPU-free. Replace these placeholders with fresh paths.

```bash
source ocean/connect4cnn/runtime_env.sh
NVCC_ARCH=sm_120 bash research/build_protein_feedback.sh build/cross-game-feedback/FRESH_OPTIMIZER
.venv/bin/python research/candidate_panel.py build --out build/cross-game-feedback/FRESH_GAMES
NVCC_ARCH=sm_120 bash research/build_policy_metadata.sh build/cross-game-feedback/FRESH_METADATA

.venv/bin/python research/cross_game_feedback.py prepare \
  --recipe research/recipes/cross_game_feedback_smoke.ini \
  --registry build/cross-game-feedback/FRESH_GAMES/registry.json \
  --optimizer-build build/cross-game-feedback/FRESH_OPTIMIZER \
  --policy-metadata build/cross-game-feedback/FRESH_METADATA \
  --out build/cross-game-feedback/FRESH_PACKET
.venv/bin/python research/cross_game_feedback.py inspect --out build/cross-game-feedback/FRESH_PACKET

# Reference only: do not execute under current GPU holds.
.venv/bin/python research/cross_game_feedback.py run --mode smoke --allow-gpu \
  --plan build/cross-game-feedback/FRESH_PACKET/plan.json

# After execution, offline only:
.venv/bin/python research/cross_game_feedback.py audit \
  --plan build/cross-game-feedback/FRESH_PACKET/plan.json \
  --out build/cross-game-feedback/FRESH_REVIEW
```

Declared smoke: three trials, six games, one seed, 65,536 decisions/job, one
checkpoint, 17 episodes/16 slots per drawing. If complete: **18 training jobs,
123 drawing evaluations, plus 36 repeat/eager checks, 2,703 assigned episodes**.
That allocation completed on the 5090. Campaign ≤600s, with launcher-specific
training/evaluation/proposal caps. Vary only C8/C16, depth1/K7/S4/projection64, so three
trials necessarily repeat an architecture. This is plumbing, not useful learning
or broad search. `num_random_samples=0` exercises GP feedback immediately after
the first panel; ordinary native discovery defaults to10.

`--mode canary5090` targets only the separate 5090, shares the tiny allocation
limits, and requires drawing-zero graph/repeat/eager checks for each job.
The launcher caps execution at 600 seconds. Larger `--mode development` requires
a separately scheduled campaign. Broader learner-batch numerical failure remains unresolved; successful
H128 small-batch tests do not qualify every newly proposed architecture.

Build/preparation/test receipts:
[current evidence](results/cross-game-feedback-20261006/README.md).
