# 5090: realistic-image CNN discovery

All code/configuration is supplied by G240. Do not edit sources, install CUDA/
Torch/NCCL, modify learners/seeds, or substitute the old tiny-image sweep.
Use only Kinvert/PufferLib:cnn-research. Preserve every prior allocation.

## First execute the fixed calibration

Follow `NEXT_5090_IMAGE_CALIBRATION.md` for an isolated checkout, the existing
uv Python3.12/runtime/lock setup, fresh builds and the explicit fixed-model
calibration. Its prepared directory becomes `IMAGE_CALIBRATION_DIR` below.
Do not pull new code into a checkout with a running or retained frozen campaign.
A completed compatible older calibration may be imported from its absolute
path; the new setup replays its original artifacts without changing them.

Continue only after `execution/RESULT.json` says `calibration-complete` and
every game passes its recorded learning gate. No actual full-image5090
calibration receipts were available on G240 when this code was prepared.
The wrapper independently refuses missing/failed/unlearned calibration; its
successful build or our local smoke does not establish useful learning.

The discovery uses all six games/all41 drawings, direct varied RGB frames,
H128/L1, the actual calibrated per-game learners/budgets, seeds7/42/69 and
episode seeds7007/7008/7009. Appearance17 and optimizer83 are fixed. Initial
plus eight trained checkpoints stay matched. Confirmation seeds remain unused.
No learner or seed is a search coordinate.

## Prepare and run the measured feedback pilot

Set `IMAGE_CALIBRATION_DIR` to that completed directory. In the new delivered
checkout with its existing venv/runtime/lock:

```bash
IMAGE_PILOT_DIR="build/image-pilot-5090-$(date -u +%Y%m%dT%H%M%SZ)"
bash research/run_image_discovery_5090.sh prepare-pilot \
    "$IMAGE_CALIBRATION_DIR" "$IMAGE_PILOT_DIR" \
    > "$IMAGE_PILOT_DIR.prepare.log" 2>&1
bash research/run_image_discovery_5090.sh inspect "$IMAGE_PILOT_DIR"

# Only when the5090 is idle. Run once; do not restart a failed allocation.
bash research/run_image_discovery_5090.sh run "$IMAGE_PILOT_DIR" \
    > "$IMAGE_PILOT_DIR.run.log" 2>&1
```

Preparation performs no GPU query or policy execution. It builds the native
PROTEIN bridge, imports the eligible calibration and freezes an18-coordinate
CNN-only envelope. The template exposes depth1–3, channels8/16/32, kernel sizes,
first strides1/2/4/8, later strides1/2/4, pooling, residual blocks, projection and
flatten/GAP readout. All native constraints remain: support does not mean every
combination fits memory or passes numerical checks.

The explicit pilot runs three complete native PROTEIN observations, preserving
duplicates and exact all-game feedback. Fresh independent CUDA checks precede
each candidate. It has a four-hour execution/report window and no partial-game
observation. An invalid graph, storage/index rejection, CUDA/math failure or
timeout remains a failure; no automatic retry or invented penalty is submitted.
The broad envelope still needs this actual hardware-specific pilot.

The5090 memory policy uses a24GiB **storage-floor admission** budget, rather
than the5060's4GiB guard. Every native proposal has a retained nvidia-smi
receipt for the exact GPU UUID, total/free MiB and at least4GiB free headroom
above the configured budget. This requires at least28GiB free, not an exact
32GiB reported total. It does not certify peak VRAM: native workspaces and
unexpected allocation pressure can still fail honestly. No CPU CNN runs.

Start with `$IMAGE_PILOT_DIR/RUN.json`. A successful pilot also produces
`runtime-review/runtime-calibration.json` and the complete report/plots at
`runtime-review/review/REPORT.md`. The measured reserve includes whole panels,
numerical checks, evaluation/audit, native final replay and full reporting.
If the pilot fails, retain it and send G240 the error/raw prefix. Do not prepare
a sweep from it or remove a game/drawing/seed to manufacture a pass.

## Prepare the approximately overnight sweep

Only after that pilot completes successfully:

The launcher requires `RUN.json` to bind all three native observations, successful
final reporting and a completed four-hour window. A failed/unfinished launcher
cannot seed the sweep even if some runtime files were written.

```bash
IMAGE_SWEEP_DIR="build/image-sweep-5090-$(date -u +%Y%m%dT%H%M%SZ)"
bash research/run_image_discovery_5090.sh prepare-sweep \
    "$IMAGE_CALIBRATION_DIR" "$IMAGE_PILOT_DIR" "$IMAGE_SWEEP_DIR" \
    > "$IMAGE_SWEEP_DIR.prepare.log" 2>&1
bash research/run_image_discovery_5090.sh inspect "$IMAGE_SWEEP_DIR"

# Explicitly scheduled when idle; do not launch from preparation automatically.
bash research/run_image_discovery_5090.sh run "$IMAGE_SWEEP_DIR" \
    > "$IMAGE_SWEEP_DIR.run.log" 2>&1
```

The new allocation imports **all three actual pilot observations** into native
PROTEIN and begins with the next proposal. It does not retrain their policies
or reset the GP. Original pilot allocations/receipts stay immutable and are
replayed during the final audit. Up to128 additional observations fit a12-hour
window; the measured complete-panel/report reserve decides admission before
each proposal. Reporting reserves include the imported observations and grow
with history. `budget_stop` is clean stopping, not failure. Empirical reserves
are not hard guarantees for unseen architectures; actual elapsed time/overrun
is recorded, and unexpected failures preserve all evidence.

Read `RUN.json`, `feedback/execution/result.json` and `report/REPORT.md`.
The report includes every imported/new trial, model/checkpoint/drawing/seed,
process SPS, unchanged native metric bins and actual monotonic training costs.
Missing cells, declines, duplicates and failures stay visible. Initial-policy
cost is measured, not invented zero. These are development curves, not fresh
confirmation, a statistically established frontier gain or a SOTA claim.

Both explicit run commands automatically package compact summaries under
`build/hardware-artifacts/` with SHA256. Raw numerical arrays, weights and
binaries stay on5090; keep them for independent audit. Packaging can be repeated
without executing a policy:

```bash
bash research/run_image_discovery_5090.sh handoff "$IMAGE_SWEEP_DIR"
```

Send G240 the final status, actual elapsed time, observation counts, missing/
failure summary, report path and printed archive filename/SHA256. No source
editing, automatic continuation, seed search or confirmation run is needed.
