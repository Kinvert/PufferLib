# Dogfight Stock-Core Evaluation Contract Evidence

Date: 2026-07-28

## Scope

This change fixes Dogfight evaluation tooling without editing PufferLib core.
`ocean/dogfight/build_eval.sh` still treats `src/pufferl.cu` as a pristine
input and generates `build/pufferl_dogfight_eval.cu`.

The generated evaluator now:

1. uses the BF16-compatible evaluation horizon of 8;
2. clamps evaluation environments for small requested game counts;
3. forces the same two-environment, two-buffer topology for `eval_bot` and
   `render`;
4. disables frozen banks, self-play, role randomization, domain
   randomization, and extra vertical randomization;
5. allows visible evaluation to terminate after `base.num_games`;
6. prints the actual completed-episode count and final metrics.

`ocean/dogfight/eval_checkpoint.sh` supplies the same checkpoint, seed, fixed
stage, and episode target to both paths.

## Why the old 100 percent result was not trustworthy

Stock 5c chooses at least 1024 evaluation agents when `base.eval_agents` is not
specified. For requests below 1024 games, the old generated evaluator did not
clamp that count. It could therefore stop after the earliest subset of a much
larger population completed.

Dogfight wins often finish earlier than failures or timeouts. Stopping on the
first completions can overrepresent easy wins. Visible evaluation also used a
different buffer topology and did not have a finite numerical stopping rule.

The previous stage-2 headless result of `1.0000` and human estimate near 72
percent were therefore not comparable.

## TDD evidence

The evaluator tests were written first and failed for three expected reasons:

```text
small-eval agent guard was unchanged
headless and visible topology/defaults differed
checkpoint evaluation command wrapper did not exist
```

After implementation:

```text
focused evaluator tests: 4 passed
full Dogfight suite:      46 passed, 3 skipped
native Dogfight build:    passed
Dogfight eval build:      passed
```

Commands:

```bash
source /home/claude/PufferLib/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

./build.sh dogfight
bash ocean/dogfight/build_eval.sh
python -m pytest ocean/dogfight/tests -q
```

## Deterministic training canary

Two stock-core fixed-stage-0 canaries used identical seeds, settings, and
1,048,576 training steps:

```bash
bash ocean/dogfight/train_reproduction.sh \
  no_core_eval_contract_canary_1m_20260728_a 1048576 0

bash ocean/dogfight/train_reproduction.sh \
  no_core_eval_contract_canary_1m_20260728_b 1048576 0
```

Both final checkpoints were bit-identical:

```text
SHA-256 6d18e8b2793e18d80accc797c69fd78e023fa7a6dad0ea86379b62bd65c042e6
```

Both final dashboards also reported:

```text
perf=0.076
score=-0.386
entropy=7.172
```

This is a plumbing and determinism canary, not a learned-policy acceptance
result. The proven fixed-stage-0 profile required roughly 25.7M steps to cross
90 percent in the earlier accepted run.

## Matched stage-2 evaluation

Checkpoint:

```text
checkpoints/dogfight/no_core_continuous_phase2profile_curriculum2_64m_20260728_a/0000000037748736.bin
```

Headless:

```bash
bash ocean/dogfight/eval_checkpoint.sh headless \
  checkpoints/dogfight/no_core_continuous_phase2profile_curriculum2_64m_20260728_a/0000000037748736.bin \
  2 64 42
```

Human-visible:

```bash
DISPLAY=:0 bash ocean/dogfight/eval_checkpoint.sh visible \
  checkpoints/dogfight/no_core_continuous_phase2profile_curriculum2_64m_20260728_a/0000000037748736.bin \
  2 64 42
```

Both completed with exactly:

```text
games=64/64
perf=0.6562
score=0.484
avg_stage=2.000
player_ground=0.000
```

Human flight-quality assessment:

```text
flying okay, but not great
```

This resolves the 100 percent versus roughly 72 percent disagreement. The
checkpoint has moderate stage-2 competence, not mastery.

## Decision

Use `eval_checkpoint.sh` for fixed-stage acceptance gates. Do not use the
stock high-parallelism default for small Dogfight evaluation sets, and do not
use an unbounded render dashboard as the numerical comparison.

No hyperparameter sweep should begin until its objective uses this matched
contract. Flight quality remains a separate human acceptance gate even when
the numeric result passes.
