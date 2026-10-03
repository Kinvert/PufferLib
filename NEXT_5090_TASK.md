# Current 5090 task: prepare first, validate before searching

October 2, 2026. This is the current entry point for the agent on the **separate
RTX 5090 machine**, normally in `~/Git/ml/cnn-5090`. G240 is the development
machine with an RTX 5060; its filesystem, runtime helper and build outputs are
not available here. Handoffs travel through commits on Kinvert's fork, not chat
context shared between agents.

## Scope and first action

**Current scope is preparation only.** Kinvert previously deferred GPU work;
publishing this handoff does not lift that hold. Start with the GPU-free checks
and preparation below. If Kinvert separately schedules GPU validation, the
first GPU task is encoder-5 mathematical verification, followed by regressions
and bounded train/reload checks. Do not jump directly to a sweep or rerun the
historical hardware, Pong, confirmation or appearance campaigns.

This file takes precedence over the old launch sequence in `START_HERE_5090.md`.
Read `AGENTS.md` and the linked contracts before acting. A later explicit user
instruction can change the scope without requiring another permission round.

## Update the correct checkout safely

```bash
cd ~/Git/ml/cnn-5090
git status --short
git remote -v
git fetch origin cnn-research
git log --oneline --left-right HEAD...origin/cnn-research
```

Require `origin` to be `Kinvert/PufferLib`. If the tree is clean and strictly
behind, use `git merge --ff-only origin/cnn-research`. Otherwise preserve local
commits, uncommitted work and artifacts, and reconcile on a separate branch or
worktree. Never reset or force-push to erase divergence. Never push to official
`PufferAI/PufferLib`. Record the actual revision and diff used for validation.

The checkout must include this file, `research/FLEX2_VERIFICATION.md`,
`ocean/connect4cnn/tests/flex2_reference.py`, `research/CLAIM_PIPELINE.md`,
`ocean/connect4cnn/claim.py` and `ocean/connect4cnn/exact_eval.cu`.

## Execute now: configuration and artifact preparation

Reuse this clone's existing Python 3.12 `.venv`. If it is absent, follow the
venv setup in `START_HERE_5090.md` using `uv venv`; only the pinned NumPy is needed
for these checks. Do not install Torch, change CUDA/NCCL, or modify another venv.
The artifact checks compile a tiny native configuration/RNG probe with `cc`;
they do not execute a neural model. Node is optional for the chart check; record
its absence instead of installing system software.

Run from the repository root in Bash:

For preparation, invoke the external configuration script directly: the Bash
sweep launcher also requires NCCL paths even when no build or GPU is needed.

```bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
mkdir -p build/connect4cnn
handoff_dir=$(mktemp -d build/connect4cnn/handoff-preflight.XXXXXXXX)
git rev-parse HEAD > "$handoff_dir/commit.txt"
git status --short > "$handoff_dir/status.txt"
git diff HEAD > "$handoff_dir/working.diff"
.venv/bin/python -B ocean/connect4cnn/tests/test_sweep_tools.py \
  2>&1 | tee "$handoff_dir/sweep-tools.log"
.venv/bin/python -B ocean/connect4cnn/tests/test_claim_tools.py \
  2>&1 | tee "$handoff_dir/claim-tools.log"
if command -v node >/dev/null 2>&1; then
  node research/tests/test_claim_chart.cjs 2>&1 | tee "$handoff_dir/chart.log"
else
  echo 'SKIPPED: Node unavailable; chart check remains pending.' > "$handoff_dir/chart.log"
fi
.venv/bin/python ocean/connect4cnn/sweep.py \
  --recipe ocean/connect4cnn/sweep_flex2_fast.ini \
  --max-runs 3 --canary --prepare-only --wandb disabled \
  2>&1 | tee "$handoff_dir/fast-preparation.log"
.venv/bin/python ocean/connect4cnn/sweep.py \
  --recipe ocean/connect4cnn/sweep_flex2_stage2.ini \
  --max-runs 3 --canary --prepare-only --wandb disabled \
  2>&1 | tee "$handoff_dir/stage2-preparation.log"
.venv/bin/python ocean/connect4cnn/claim.py prepare --canary \
  --out "$handoff_dir/measurement-canary" \
  2>&1 | tee "$handoff_dir/measurement-preparation.log"
printf 'Handoff evidence: %s\n' "$handoff_dir"
```

A nonzero command exit stops this stage; retain its log. Inspect the printed
campaign paths and their `protocol.json`/INIs. Both architecture preparations
must select encoder 5, H128/L1, appearance 0 and a 32,768-decision canary budget;
the fast/stage2 panels have depths 1/2 and 7/8 search dimensions respectively.
The separate measurement preparation must contain eight planned jobs: four
locked models × two seeds, 65,536 decisions and two checkpoints per job. It uses
the existing encoder-4 quality model, not encoder 5. Preparation is not training.

## First scheduled GPU job

When GPU validation is scheduled, source **this machine's** existing
`build/connect4cnn/runtime-5090.sh`, set `NVCC_ARCH=sm_120`, and verify the 5090
is idle. Follow the fresh-directory command block in
[FLEX2_VERIFICATION.md](research/FLEX2_VERIFICATION.md). Rebuild the test library
locally; do not reuse G240 binaries.

1. Require the encoder-5 report to say `passed`, with all **69 operator checks
   and 14 complete encoder fixtures**, forward/all-parameter gradients, sampled
   finite differences and eager/graph/actor repeatability passing.
2. Run the Nature/Flex regressions in
   [NEXT_5090_FLEX2_SWEEP.md](NEXT_5090_FLEX2_SWEEP.md), section 3.
3. Then run the bounded three-trial canaries and separate same-seed
   training/save/reload checks described there. Reload must use the selected
   trial's resolved architecture. That last step is still a manual integration
   task, not an already verified one-command runner.

Preserve every failure and investigate it; do not weaken tolerances, remove
fixtures or change precision to manufacture a pass. Successful math checks do
not validate optimizer behavior, full training determinism or scientific timing.
No long search is implied by a successful canary or by the 64-trial recipes.

The locked-model exact-evaluation/timing canary is a separate task documented in
[CLAIM_PIPELINE.md](research/CLAIM_PIPELINE.md). It has compiled locally but its
GPU runtime behavior is unvalidated. Full confirmation launch is unavailable.
The experimental frontier confidence bands failed preliminary coverage checks
and must not certify dominance or determine a paper claim.

## Benchmark groundwork after preparation

[Benchmark priorities](research/BENCHMARKS_AND_SWEEPS.md#next-benchmark-priorities-october-2-2026)
recommend original Procgen as the next external suite. Its native feasibility
probe is a future engineering task, not an implemented command here: pin the
original source, inspect/build its `gym3.libenv` C interface using local
dependencies, step a seeded environment, and verify the 64×64 RGB buffer and
reset/terminal semantics. Record blockers instead of changing system packages.
CPU environment stepping is acceptable; CPU neural-model substitutes are not.
RGB/general-resolution encoder support needs separate validation. Native
Breakout is an optional small extension, not equivalent to ALE Breakout.

## Return a small status report

Report the actual commit/diff, host, checks passed/failed/skipped, exact evidence
paths, prepared campaign counts, and the first remaining gate. Say explicitly
whether any GPU command ran. Keep large artifacts/checkpoints on the 5090;
do not require an SCP transfer for this preparation report. No learning, speed,
Pareto dominance or SOTA result follows from preparation or compilation.

## Local preparation rehearsal

On G240, October 2, the publication preflight passed 16 sweep-configuration
checks, 11 configuration/artifact checks and 12 chart views plus empty data.
Both three-trial architecture preparations and the eight-job measurement
preparation completed without GPU execution. Local receipts are in
`build/connect4cnn/handoff-preflight.dbnSuVdi/`, with architecture preparations
`sweep.9dsifqt2` and `sweep.p_vzjpd_`; these are G240-local diagnostics, not remote
inputs or model results. Earlier native compilation evidence is retained in
`research/results/connect4cnn/claim-groundwork-20261002/`. The 5090 runtime and
encoder-5 numerical suite still need their first execution on this source.
