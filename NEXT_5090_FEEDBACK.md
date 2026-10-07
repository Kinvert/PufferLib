# First task on the 5090: native cross-game feedback canary

Pull Kinvert's `cnn-research` branch, then use the two commands below. No code
editing is needed on the 5090. This is temporary research, not the upstream PR.
Do not push to PufferAI/PufferLib. This handoff supersedes earlier no-delivery /
5090-hold wording **only for this small canary**. No long search is scheduled.

```bash
cd ~/Git/ml/cnn-5090
git pull --ff-only origin cnn-research
bash research/run_feedback_5090.sh prepare
```

`prepare` prints the fresh directory and exact `run` command. It compiles the
native PROTEIN bridge, six normal float32 game targets, and three host metadata
tools, then freezes and inspects their inputs. It performs host/configuration,
scalar game/registration and malformed-input checks; **no GPU query, neural
computation or training**. Builds may take several minutes. On failure report
the fresh directory and failing log; do not patch code, install global packages,
change CUDA, or launch another campaign.

The launcher reuses `build/connect4cnn/runtime-5090.sh` if present. Otherwise
export `CUDA_HOME` and `NCCL_ROOT` pointing at your **existing** toolkit/NCCL,
then run prepare again into a new directory. It sources the normal process-local
runtime helper, uses explicit `sm_120`, and never installs CUDA or Torch.
An existing Python 3.12 `.venv` must already import NumPy. If no venv exists,
prepare uses `uv venv --python 3.12 .venv` and installs only NumPy 2.5.3 there.
Do not replace an incompatible existing venv silently; report it.

When the 5090 is free, execute the **printed** command, for example:

```bash
bash research/run_feedback_5090.sh run build/cross-game-feedback/5090-TIMESTAMP
```

That command runs the bounded allocation, audits it offline, and creates an
archive plus SHA256 under `build/hardware-artifacts/`. Busy/wrong hardware is
rejected using the shared reservation and idle checks. It never waits for another
job or retries an allocation. Do not actively poll. Preserve failures and report
the log/error if it exits nonzero. Existing G240 results are not 5090 evidence.

## What this allocation does

```text
native PROTEIN suggestion -> optimizer exits
            |
            v
same CNN independently trains on all SIX games
fixed per-game learner settings, paired seeds, mixed pixel drawings
            |
            v
exact deterministic evaluation of all 41 drawings + receipt audit
            |
            v
equal-game normalized quality + once-per-job monotonic training cost
            |
            +-> native PROTEIN observes feedback and suggests again
```

Three trials; one training seed; 65,536 decisions/game; encoder4/H128/L1.
Only first-convolution width varies, C8 or C16. Architecture duplicates remain
in the record. Expect **18 training jobs, 123 drawing evaluations and 36
repeat/eager checks**, 2,703 assigned evaluation episodes in total. Each final
checkpoint is tested on every drawing; drawing-zero graph/repeat/eager CSVs
must match exactly. A fourth **unused** suggestion proves the last observation
was replayed; it does not train another model. The GP has zero random warmup.

The whole execution has a 600-second deadline, excluding builds/preparation.
Training processes cap at 10 seconds, evaluations at 20, proposals at 30.
These are deadlines, not completion-time predictions. Success requires every
trial, every assigned episode, all byte checks, and the final feedback audit.
No partial means, best-checkpoint selection, or failure-as-zero score.

## Return the result

Report the source commit, GPU/toolkit identity, exit status, archive path/SHA,
and `review/analysis.json`. State whether all three trials completed and the
final native proposal replayed three observations with a nonzero GP count.
Keep per-game scores, Pong censoring bounds, training/process/optimizer/evaluation
time separate. The archive includes raw outcomes, checkpoints and build receipts;
retain it even on failure. Copying the terminal summary alone loses evidence.

This canary establishes whether the feedback plumbing runs on the 5090. Its
short training, one seed, uncalibrated anchors/caps and two architectures cannot
establish learning superiority, a Pareto winner, generalization or SOTA.
Independent math for searched shapes/learner batches, learner/anchor calibration,
meaningful budgets and held-out confirmation remain gates before a real search.
See [the protocol](research/CROSS_GAME_PROTEIN_FEEDBACK.md), including the retained
local learner-batch numerical failure. Do not waive it or expand this canary.
