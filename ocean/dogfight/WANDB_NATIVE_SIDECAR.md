# Native Protein W&B Sidecar

The sidecar observes completed native Protein trials. It does not control the
sweep, suggest parameters, name W&B runs, or modify PufferLib core.

## Objective

Dogfight follows native Robocode's self-play contract:

- native Protein trains one current policy against current and historical rows;
- final evaluation matches the candidate against up to eight historical
  checkpoints over 4096 games;
- wins score 1, draws score 0.5, and losses score 0;
- the raw mean is logged unchanged as `selfplay/pool_score`;
- candidate flight quality is logged as `selfplay/pool_flight_quality`;
- quality-adjusted outcome credit is logged as `selfplay/pool_fitness` and
  becomes Protein's score.

The sidecar publishes raw `selfplay/pool_score` as
`selfplay/pool_winrate`, and publishes adjusted `selfplay/pool_fitness` as
`protein/fitness`. These values are deliberately different. `env/perf` and
shaped reward metrics are diagnostics, not the sweep objective.

Fitness is computed per episode and then averaged:

```text
episode_fitness =
    raw_candidate_outcome
    * terminal-cause credit
    * candidate_flight_quality
```

Gun wins use terminal-cause credit `1`, opponent ground/OOB wins use `0.25`,
and timeouts use `0`. Because averaging happens after the per-episode
multiplication, aggregate pool fitness is not generally equal to aggregate
pool score multiplied by aggregate flight quality.

Dogfight overrides native history downsampling to 31 points. This gives W&B a
useful post-run learning curve without changing update cadence. The native log
is still written at trial completion, and all three historical pool values are
single final-only measurements.

## Clean-core compatibility build

Stock 5c forces match evaluation to horizon 1 while its advantage kernel
requires a multiple of 4 or 8. Dogfight keeps `src/pufferl.cu` pristine.
`build_eval.sh` copies the source, changes only the generated copy to horizon
8, restores the trained INI after native match evaluation mutates it, keeps
raw pool score separate from Dogfight's adjusted Protein fitness, adds the
fixed evaluator contract, and compiles the complete native binary:

```bash
CUDA_HOME=/usr/local/cuda \
  bash ocean/dogfight/build_eval.sh ./puffer
```

Use that generated `./puffer` for both native training and evaluation.
Restoring the INI affects logging only: W&B sees the actual sampled training
parameters rather than the temporary one-on-one match topology.

## Native sweep with W&B

Start the observer before Protein:

```bash
python ocean/dogfight/wandb_sidecar.py \
  --wandb --wandb-project df43 \
  --env dogfight --max-runs 12
```

Then run stock native Protein:

```bash
./puffer sweep dogfight sweep.max_runs=12
```

For the 5090 cap, change both values from 12 to 1000. Do not pass W&B flags to
`./puffer`; the native binary has no W&B client. W&B receives a completed
trial when its native INI log closes, not live epoch updates.

On first upload, the sidecar supplies neither an ID nor a name, so W&B creates
its normal random identity. Restart state is kept at:

```text
logs/dogfight/.wandb-sidecar-df43.json
```

The sidecar never creates groups, sessions, tags, or a W&B-managed sweep.
Checkpoints remain in `checkpoints/dogfight/<native_run_id>/`.
