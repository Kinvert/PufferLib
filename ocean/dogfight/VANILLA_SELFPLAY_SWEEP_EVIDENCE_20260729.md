# Vanilla Native Self-Play Sweep Evidence - 2026-07-29

## Purpose

Run PufferLib 5c's unchanged native Protein sweep against Dogfight's official
native self-play implementation, record every resolved trial configuration, and
upload completed trial metrics to W&B project `df42` without modifying
PufferLib core.

This sweep is a candidate generator. Its training metric `env/perf` is not
evidence of curriculum mastery. Candidates must still pass the documented
fixed-seed, mirrored stage matrix before promotion.

## Native 5c Findings

- The supported entry point is the generated native `./puffer` executable.
- Native override values are single tokens. The correct form is
  `sweep.max_runs=1000` or `--sweep.max-runs=1000`.
- The older form `--max-runs 1000` is invalid in native 5c.
- Native Protein ignores Python's `sweep_only` control and otherwise inherits
  all 18 parameter sections from `config/default.ini`.
- Native training writes the complete resolved trial configuration and
  downsampled metrics to `logs/dogfight/<run_id>.ini`.
- Native `./puffer` has no W&B client. It accepts W&B configuration keys but
  does not upload them.
- The obsolete Python `_C` sweep path cannot run this native Dogfight build and
  must not be revived by restoring old PufferLib bindings.

## Dogfight-Local Solution

[`native_sweep.py`](native_sweep.py) performs four local orchestration tasks:

1. Preserve a per-sweep experiment directory and manifest.
2. Stage `config/default.ini` without inherited `[sweep.*]` parameter sections,
   then overlay the current Dogfight INI and exactly the dimensions named by
   `sweep_only`.
3. Run the unchanged native `./puffer sweep dogfight` Protein implementation.
4. Watch completed native trial INI logs and upload at most 31 explicit metric
   series per trial to W&B `df42`, using the native run ID idempotently.

Upload failure is recorded and retried without killing an active native
training process.

The verified Dogfight payload contains 30 explicit logged values. PufferLib's
core `n` value brings the total to 31, which remains below the 32-value limit.

## Checkpoint Storage

The previous `checkpoint_interval = 8` would produce approximately 5,761 files
per 12.08B-step trial. Across 1,000 trials that is approximately 5.76 million
files and 897 GB before filesystem overhead.

The dedicated sweep profile uses:

```ini
[base]
checkpoint_interval = 384
```

At 4,096 agents and horizon 64 this is 100,663,296 agent steps, closely matching
`selfplay.opp_timeout_steps = 100000000`. A full trial retains approximately
121 snapshots, or about 18.8 GB across 1,000 trials at the observed model size.

## Commands

Preview the exact native and tmux commands without writing files:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --dry-run --max-runs 1 --timesteps 8388608
```

Launch the structural canary:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --max-runs 1 --timesteps 8388608
```

The eventual full-cap command is:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh --max-runs 1000
```

No session or experiment arguments are required. The launcher generates both
names locally; they do not define W&B metric names or chart axes.

Resume or reattach to an existing experiment with:

```bash
bash ocean/dogfight/sweep_vanilla_selfplay.sh \
  --screen-existing EXPERIMENT_DIR
```

The dedicated screening profile runs `671,088,640` agent steps, approximately
four to six minutes at measured throughput. It retains the full 134M-step
acquisition school, then compresses progressive spawn difficulty into the
remaining budget. The 1,000-run cap is intended for the user's 5090 machine.
It must not be launched on this g240/5060 validation host.

## TDD Record

The initial focused red test exposed the missing launcher/parser behavior:

```text
3 failed, 1 passed
```

That failure is retained as resolved parser discovery, not a current launch
failure. The completed automated gate is:

```text
Dogfight tests: 85 passed, 3 skipped
trainer build:   succeeded
eval build:      succeeded
```

## Structural Canary Result

The fresh no-name native tmux canary completed on the g240/5060 host:

```text
experiment:        df42-native-sweep-20260729-102730
max_runs:          1
agent steps:       8,388,608
native exit code:  0
W&B project:       df42
uploaded runs:     1
explicit Dogfight: 30
with core n:       31
```

Artifacts:

```text
ocean/dogfight/sweeps/df42-native-sweep-20260729-102730/
```

The experiment preserved the staged config, manifest, complete native trial
INI, initial checkpoint, and final 8,388,608-step checkpoint. This proves the
native Protein, tmux, GPU, checkpoint, and completion-time W&B paths. It does
not prove learning or fixed-stage mastery.

## Determinism and Post-Sweep Screening

The post-sweep fixed evaluator completed normally:

```text
status:                   completed
completed cells:          6
rejected candidates:      1
evaluation failures:      0
stage 0 minimum perf:     0.046875
stage 0 mean perf:        0.0663052
max absolute signed bias: 42.275864
```

The deliberately tiny canary is therefore structurally valid but correctly
rejected as a policy candidate.

Identical repeated canaries produced the same final checkpoint:

```text
SHA-256: fbf5dc6cfd8477382596c315797c7848465c4da3ccc018468802d74164d0b868
```

Their fixed-evaluation metrics were also identical. This verifies deterministic
canary training and screening for the tested seed and configuration.

## Full-Cap Launch Gate

Do not launch `max-runs=1000` directly from this host. The required sequence is:

1. Push the clean experimental branch to the user's PufferLib fork.
2. Clone that branch on the 5090 machine and create its clone-local `.venv`.
3. Run a 16-32 trial calibration sweep on the 5090.
4. Fixed-evaluate the calibration candidates and measure whether native sweep
   score correlates with fixed-stage results.
5. Launch `max-runs=1000` only if that calibration shows that Protein's score
   provides a useful candidate ranking.
