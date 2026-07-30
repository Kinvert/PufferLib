# Dogfight Native Self-Play: 5090 Handoff

## Contract

Training uses the same native Protein topology as Robocode: one trainable
policy, one frozen historical bank, native checkpoint sampling, and final
candidate-versus-history pool evaluation. No PufferLib core file may differ
from Joseph's 5c branch. The older `native_sweep.py` harness remains available
for controlled benchmarks but is not the production sweep controller.

Protein maximizes final `selfplay/pool_score`. W&B project `df43` receives that
value through the observer-only sidecar as `protein/fitness`. Training
`env/perf`, shaped reward, and `env/score` are diagnostics.

## Clone and build

```bash
git clone <USER_FORK_URL> PufferLib-dogfight5c
cd PufferLib-dogfight5c
git checkout dogfight5c-robocode-stage10
uv sync
source .venv/bin/activate
python -m pytest -q ocean/dogfight/tests
CUDA_HOME=/usr/local/cuda \
  bash ocean/dogfight/build_eval.sh ./puffer
```

`build_eval.sh` compiles the complete native executable from a generated
Dogfight-local copy. The tracked `src/pufferl.cu` stays pristine; the generated
copy changes the stock match horizon from 1 to 8 so final pool evaluation
satisfies the advantage-vector width. It also restores the trained INI after
match evaluation so W&B records the sampled training parameters rather than
temporary evaluation topology such as `frozen_bank_pct=1`.

Do not replace this binary with a clean-core `./build.sh dogfight` binary:
stock 5c currently asserts during final historical-pool evaluation.

## Structural canary

Start the sidecar first:

```bash
python ocean/dogfight/wandb_sidecar.py \
  --wandb --wandb-project df43 \
  --env dogfight --max-runs 2
```

In another terminal:

```bash
./puffer sweep dogfight sweep.max_runs=2
```

Both trials must finish with `selfplay/pool_score`, upload to `df43` with
normal W&B-generated names, and retain checkpoints under
`checkpoints/dogfight/<run_id>/`. W&B updates after each completed trial, not
every epoch. Each completed run publishes 31 downsampled training-history
points; this is curve resolution, not the number of learner updates.

## Local qualification

Before the 5090 handoff, run at least 12 normal native trials:

```bash
python ocean/dogfight/wandb_sidecar.py \
  --wandb --wandb-project df43 \
  --env dogfight --max-runs 12
```

```bash
./puffer sweep dogfight sweep.max_runs=12
```

Rank by `selfplay/pool_score`, then evaluate leading checkpoints with:

```bash
python ocean/dogfight/eval_stage_matrix.py CHECKPOINT \
  --stages 0..10 --seeds 42,314159,271828 --mirrors 0,1 \
  --episodes 128 --json-output RESULT.json
```

Promotion requires outcome mastery and the target-conditioned roll gate.
Policies that command the same aileron direction for targets on both sides are
rejected even when their score or perf is high. A visible `DISPLAY=:0`
evaluation remains the final human flight-quality check.

## 5090 sweep

The checked-in INI contains the full production ranges and a 1000-run cap.
After the two-run 5090 canary, launch the observer and native controller in
separate tmux sessions:

```bash
tmux new-session -d -s df43-sidecar \
  'source .venv/bin/activate && python ocean/dogfight/wandb_sidecar.py --wandb --wandb-project df43 --env dogfight --max-runs 1000'
```

```bash
tmux new-session -d -s df43-sweep \
  'source .venv/bin/activate && ./puffer sweep dogfight'
```

No W&B group, session, experiment ID, or custom run name is supplied. The tmux
names do not affect W&B plotting.

## Production profile

- 4096 agents, 64-step horizon, 64K minibatch, 64x3 policy.
- One frozen bank and 4096-game evaluation against up to eight checkpoints.
- 671,088,640 base steps per trial; Protein may sample up to 805,306,368.
- Progressive spawn completion sampled from 402,653,184 to 805,306,368.
- No one-step acquisition shortcut and no steering-sign shaping.
- Checkpoint interval 384, approximately one snapshot per 100M agent steps.
- At most 31 explicit logged values; native `n` keeps the total at 32 or less.

The favorable-to-hard spawn schedule shapes training opportunities, but the
final Protein score remains outcome-only. Pool score cannot detect shared
flight defects, which is why the mirrored fixed evaluator and visible eval are
mandatory promotion gates.
