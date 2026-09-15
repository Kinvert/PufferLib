# Locked Pong architecture replication on RTX 5090

This implements [the G240 handoff](../../research/PONG_5090_HANDOFF.md). Native PufferLib PongCNN uses three actions and 1x36x44 float32 pixels; its reported point fraction is not match win rate. Native code, encoder kernels, environment and initialization are unchanged.

## Run and resume

```bash
source build/connect4cnn/runtime-5090.sh
bash ocean/pongcnn/confirm.sh --prepare-only
bash ocean/pongcnn/confirm.sh --canary
bash ocean/pongcnn/confirm.sh --full
bash ocean/pongcnn/confirm.sh --resume /absolute/path/to/the/existing/campaign
```

Each new invocation creates a unique `build/pongcnn/confirm-5090.MODE.ID` directory. Preparation uses Bash/AWK, builds and train/eval execution use Bash and native PufferLib, and Python/NumPy only audit/report the saved receipts. Resume executes the captured runner and configs/binaries, checks their hashes, and takes an exclusive campaign lock. A started attempt is never automatically repeated, even if its exit receipt is missing. Pending evaluations of successfully trained checkpoints can continue. A finished campaign is a no-op. Numerical/configuration/native failures stop for investigation; resource caps retain missing/censored observations and allow unrelated jobs to proceed. No scores are invented for failed evaluations.

The eight family/recipe conditions form a cyclic Latin order within paired seed blocks. Five rows give each condition five different positions. Recipe A and B each appear four times per block. Exact order, source snapshots (including untracked files), diff, INIs and configuration hashes are saved before execution. Binary hashes and host/runtime receipts are frozen into `protocol.json` before the first training job. Use the saved protocol, not a later working-tree recipe, to identify a run.

## Frozen protocol

- Families: ours quality, adapted Nature, IMPALA and Impoola; H128/L1 core; parameters 160224, 138016, 269984 and 151200.
- Exact learner values: `recipe_a.ini` and `recipe_b.ini`, crossed with every family. No searched dimensions remain. Native integer conversion of replay ratio yields one update/rollout for 1.91617525 and three for 3.57716227; the floating inputs are retained unchanged.
- Full: training seeds 31001–31005, evaluation seeds 41001–41005, 4,194,304 decisions, 8 checkpoints, 512 requested matches/checkpoint. 40 jobs, 320 evaluations. Train/eval caps 1800/300 seconds. Local prior Pong metric configs are checked for seed reuse before preparation.
- Canary: training/evaluation seeds 51001/61001, 65,536 decisions, 4 checkpoints, 32 requested matches, 120-second caps. 8 jobs and 32 evaluations; scores are not a learning gate.
- Primary: ours versus Nature within each recipe at 95% episode-averaged point fraction. Keep all seeds, first observed crossing brackets, non-achievers, missing observations and subsequent declines. No success-only time means.
- Paired whole-seed bootstrap: 10,000 draws, RNG seed 20260915, pointwise percentile 95% intervals; identical seed-block draws across recipes/models/checkpoints. All observed seed fronts and complete-paired mean curves are reported, with descriptive time-front membership stability. No interpolation or best-seed pooling defines a frontier. Missing final scores retain full-panel [0,1] identification bounds rather than an available-case mean.
- Five seeds estimate reliability and variance; they are not a power guarantee. There is no adaptive seed addition or full-recipe adjustment after outcomes. Development recipe selection used unequal tuning coverage and retained two failed evaluations; Pong has informed learner development.

## Results and evidence

`REPORT.md`, `results.csv`, `jobs.json`, `finished.json`, `audit.json`, per-seed target/frontier data, mean curves with uncertainty and standalone `curves.html` are generated automatically. Every planned checkpoint appears in CSV, including failures with blank scores. Hashes/configs, native metrics, checkpoint timing, process commands/exit codes, and raw evaluation logs are retained. Process SPS includes startup/checkpoint writes; native average SPS uses adjusted trainer time. Build, training-process sum, evaluation sum and campaign elapsed (through the last evaluation and intermediate audits, before final analysis/packaging) have separate fields. Resuming includes downtime in calendar elapsed and records active execution segments separately.

All evals follow uninterrupted training. GPU availability checks precede builds and train/eval execution; they do not reserve the GPU. Launch full runs detached in tmux, verify startup once, and leave them unattended. W&B is optional; the local reporting venv currently lacks its SDK and no installation is needed.

The packager retains report HTML, JSON/CSV/TSV, diffs, native sources and runtime receipts while excluding checkpoint weights, executables and caches. Keep original weights/executables locally. Return the printed archive with:

```bash
scp build/hardware-artifacts/ACTUAL_ARCHIVE.tar.gz claude@g240:/home/claude/cnn/build/hardware-artifacts/
```
