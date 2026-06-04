# AGENTS.md

Guidance for autonomous agents working on the Dogfight 5.0 port in this clone.

## Current Workspace

This machine is `g240`. The human is using it as an isolated work box so agents
can work freely without risking the current 4.0 Dogfight branch.

Writable working repo for this effort:

- `/home/claude/dogfight5`
- Branch: `dogfight5-port`
- Base: official PufferLib `5.0`
- Writable remote: `origin git@github-kinvert:Kinvert/PufferLib.git`
- Fetch-only upstream: `upstream https://github.com/PufferAI/PufferLib.git`

This repo is the clean PufferLib 5.0 Dogfight port workspace. It is intended to
produce work that can later be pushed to the Kinvert fork and opened as PRs
against official PufferLib when ready.

Reference repos:

- `/home/claude/dogfight3`
  - PufferLib 3.0-era Dogfight branch.
  - Trains well and is the behavioral reference for self-play, league, reward,
    curriculum, evaluation, observations, and opponent handling.
  - Read-only.
- `/home/claude/dogfight4`
  - Prior PufferLib 4.0 Dogfight port attempt.
  - Structurally useful but trains poorly.
  - Read-only.
- `/home/claude/PufferLib`
  - Current PufferLib 4.0 Dogfight control workspace.
  - Contains useful diagnostics and port work.
  - Treat as a reference/control unless the human explicitly asks to work there.

## Goal

Port Dogfight into clean PufferLib 5.0 in a way that can eventually be cleaned
up and merged upstream, while preserving the behavior that made Dogfight 3.0
train well.

This is not a blind restart. The strategy is:

- Use PufferLib 5.0 as the clean base.
- Bring in Dogfight incrementally, tests first.
- Use `/home/claude/PufferLib` and `/home/claude/dogfight4` for 4.0 structure.
- Use `/home/claude/dogfight3` for behavior and training quality.
- Do not enable 5.0 state memory until a normal Dogfight baseline works.
- Add 5.0 state save/load only after deterministic state roundtrip tests exist.

## Safety Rules

- Do implementation work for the 5.0 port only in `/home/claude/dogfight5`.
- Do not modify `/home/claude/dogfight3` or `/home/claude/dogfight4`.
- Do not use `/home/claude/PufferLib` as the implementation target for this 5.0
  effort unless the human explicitly redirects the work.
- Keep implementation changes inside `ocean/dogfight/` whenever possible.
- Changes outside `ocean/dogfight/` are exceptional and must be minimal:
  - `config/dogfight.ini` is allowed for Dogfight config.
  - `config/default.ini` is allowed only when needed to expose or document 5.0
    trainer defaults relevant to Dogfight.
  - Registration/build glue is allowed only when Dogfight cannot work through
    existing 5.0 extension points.
- Never change tests to make bad behavior look good.
- If a test is wrong because the intended behavior changed, document the reason
  and update it deliberately; do not weaken assertions just to get green output.
- Avoid destructive Git operations. Do not run `git reset --hard`, mass deletes,
  or history rewrites unless the human explicitly asks.

Before pushing, verify:

```bash
git -C /home/claude/dogfight5 remote -v
git -C /home/claude/dogfight5 branch --show-current
ssh -T github-kinvert
```

Expected remote shape:

```text
origin   git@github-kinvert:Kinvert/PufferLib.git (fetch)
origin   git@github-kinvert:Kinvert/PufferLib.git (push)
upstream https://github.com/PufferAI/PufferLib.git (fetch)
upstream DISABLED (push)
```

Expected branch:

```text
dogfight5-port
```

## TDD Loop

Use test-driven development for behavior changes:

1. Write or port the smallest test that proves the behavior.
2. Run it and confirm it fails for the expected reason when practical.
3. Implement the minimum code needed to pass.
4. Re-run the narrow test.
5. Re-run the broader Dogfight suite before committing.

For physics/autopilot changes, prefer deterministic C tests under
`ocean/dogfight/tests/`. For Python league/self-play/eval behavior, port or
write focused Python tests near the relevant logic before wiring it into
training.

Do not silently count skip stubs as coverage. If a test must be deferred, make
the reason explicit in the test file or plan.

## Porting Strategy

Work in small, reviewable increments:

1. Baseline the clean 5.0 repo and environment.
2. Add the smallest Dogfight test inventory first.
3. Add only enough `ocean/dogfight/` code to make the first test compile and
   fail for the expected missing behavior.
4. Port the minimal Dogfight env files needed for build/test.
5. Make `./build.sh dogfight` pass.
6. Make Dogfight regression tests pass.
7. Smoke-test normal 5.0 Dogfight training without self-play or state memory.
8. Port higher-level behavior one feature at a time:
   - observation schemes,
   - reward shaping,
   - curriculum/stage logic,
   - opponent observations,
   - AutoAce/autopilot behavior,
   - self-play pool behavior,
   - league/eval tools.
9. Add 5.0 state-memory compatibility only after the normal baseline works.

Avoid `git cherry-pick` from reference repos. Read the code and history,
identify a discrete behavior, then adapt it manually to 5.0 conventions.

## 5.0 State Memory Rule

PufferLib 5.0 has state/continuation-learning support. Dogfight should not use
it until these conditions are true:

- Normal Dogfight build passes.
- Dogfight regression tests pass.
- A short normal training smoke starts without state memory enabled.
- A `DogfightState` struct exists.
- Deterministic save/load roundtrip tests prove restored states produce the same
  observations, rewards, terminal flags, and death reasons under scripted
  actions.

The state struct must include all rollout-local state that affects the future:
plane state, opponent autopilot state, RNG, tick/max steps, cooldowns, previous
actions, reward history, curriculum variant flags, recovery/self-play flags, and
termination flags. Do not serialize pointers or render/client resources.

Current Dogfight state-memory status:

- `ocean/dogfight/dogfight.h` now has a Dogfight-local mirrored `State`
  payload and capture/restore helpers.
- `ocean/dogfight/binding.c` now wires `puffer_state_refresh` to restore the
  mirrored state and recompute observations.
- `ocean/dogfight/tests/test_state_roundtrip.c` proves a saved state restored
  into a fresh env produces the same scripted future step.
- The default Dogfight train run still leaves state curriculum disabled through
  inherited `train.state_buffer_size = 0` and `train.cl_frac = 0`. State
  curriculum is exposed only as a sweep option in `config/dogfight.ini` via
  `sweep.train.state_buffer_size` and `sweep.train.cl_frac`.

How PufferLib 5.0 state works in working envs:

- The env defines `State`; `src/vecenv.h` aliases it to `PufferState`.
- `src/curriculum.cu` checkpoints by copying `envs[i].state` directly.
- State restore assigns `env->state = saved_state`, then calls
  `puffer_state_refresh(env)` to rebuild observations/action masks.
- Boxoban, G2048, and Craftax keep live rollout data inside `env->state`.

Dogfight currently keeps live rollout data directly on `Dogfight`, not inside
`env->state`, so the current implementation mirrors the live fields into
`env->state` after reset, nonterminal step, and forced test state. Keep extending
roundtrip coverage before relying heavily on state curriculum for training.

## Useful Commands

Use these from the working repo:

```bash
cd /home/claude/dogfight5
```

Environment setup uses `uv`, not system `pip` or `python -m venv`:

```bash
uv venv --python 3.12 .venv
uv sync
source .venv/bin/activate
python --version
```

Do not modify system CUDA, global Python packages, or machine CUDA environment
configuration for this repo. Keep dependencies in `.venv`. If PyTorch needs a
specific CUDA wheel, install it through `uv` inside `.venv` only after checking
the current 5.0 dependency constraints.

Verified build/train commands on `g240`:

```bash
source .venv/bin/activate
./build.sh dogfight
python -m pufferlib.pufferl train dogfight
```

Simplest sweep command:

```bash
source .venv/bin/activate
python -m pufferlib.pufferl sweep dogfight
```

On a single GPU, optionally bound the number of jobs/runs with config-style CLI
overrides such as `--sweep.max-runs 8 --sweep.gpus 1 --train.gpus 1`.

Current hyperparameter search should use short sweeps. The goal is to find
settings that clear early curriculum stages before `200M` steps, not to spend
each trial on long `400M+` validation runs. `config/dogfight.ini` currently
sets `sweep.train.total_timesteps` to `50M-125M` with a `75M` mean. A useful
single-GPU W&B search command is:

```bash
source .venv/bin/activate
python -m pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1 --sweep.max-runs 20 --wandb --wandb-project df39 --wandb-group df39-stage9-short
```

The Dogfight sweep should stay centered on the Dogfight 3-derived trainer
profile until a normal baseline shows early-stage progress: `learning_rate =
0.00045`, `horizon = 64`, `policy.num_layers = 1`, `ent_coef = 0.0024`, and
`clip_coef = 0.11`. Wider or higher-learning-rate probes are only useful after a
short run proves stage movement without action saturation.

Dogfight now exposes the PufferLib 5 state-memory knobs only in the sweep
space: `sweep.train.state_buffer_size` and `sweep.train.cl_frac`. Plain
training remains state-memory-off through inherited defaults. Before trusting a
large W&B state-memory sweep, run a short `max-runs 2` state-buffer smoke and
check that both trials launch, use GPU, keep expected SPS, and emit normal
curriculum metrics.

Keep architecture/topology-like sweep parameters discrete. Dogfight's
`sweep.policy.num_layers` and `sweep.vec.num_buffers` use `int_uniform` so
Protein samples real layer and buffer counts instead of fractional values that
only get coerced later.

`source .venv/bin/activate` is the local Dogfight5 environment. Do not source
CUDA helpers from `/home/claude/dogfight3`, `/home/claude/dogfight4`, or
`/home/claude/PufferLib`. The current venv activation sets
`CUDA_HOME=/usr/local/cuda-12.8`, keeps `.venv/bin` first on `PATH`, adds CUDA
and WSL GPU library paths, and leaves dependencies inside the Dogfight5 venv.

The train command above is the clean Dogfight GPU training path. It has reached
and completed the normal PufferLib TUI run in this clone with GPU active. With
the current Dogfight 3-derived profile, the latest full 400M-step smoke through
plain local venv activation showed about `3.3M-3.6M` SPS, `GPU: 33-40%`, and
`VRAM: 1.6/8G`. For a bounded smoke, wrap it with shell `timeout`; do not add
trainer arguments unless the specific experiment requires them. In particular, do not use
`--train.total-timesteps 262144` or any tiny trainer timestep override as the
smoke path. That changes the experiment and is not how the reference Dogfight
environments run.

Codex sandboxed commands may not see `/dev/dxg`/NVIDIA devices. If training
aborts at `create_pufferl` with `device_count > 0 && "CUDA is not available"`,
rerun the same clean training command with escalated/unsandboxed execution; do
not switch to CPU fallback or add `--cpu`.

Narrow Dogfight regression loop:

```bash
.venv/bin/python -m pytest ocean/dogfight/tests -q
```

Reference inventory:

```bash
find /home/claude/dogfight3/pufferlib/ocean/dogfight -maxdepth 3 -type f | sort
find /home/claude/dogfight4/ocean/dogfight -maxdepth 3 -type f | sort
find /home/claude/PufferLib/ocean/dogfight -maxdepth 3 -type f | sort
```

Build syntax may differ on 5.0. Inspect `./build.sh --help` or the script before
assuming 4.0 flags still apply.

## Current Status

The initial bootstrap milestone has been passed in this clone:

- `uv` virtualenv exists under `.venv`.
- Dogfight build has passed with `./build.sh dogfight`.
- Dogfight regression tests have passed with
  `.venv/bin/python -m pytest ocean/dogfight/tests -q`.
- The clean GPU training command starts through the normal PufferLib TUI with
  GPU active.

The current training work is not complete. The active near-term milestone is a
normal Dogfight baseline that trains through curriculum stages without self-play
or 5.0 state memory. Current config experiments use the config file rather than
trainer CLI timestep overrides.

The current trainer profile is intentionally Dogfight 3-derived and mapped to
PufferLib 5 keys in `config/dogfight.ini`:

- `policy.hidden_size = 128`
- `policy.num_layers = 1` (Dogfight 3 used one recurrent layer)
- `policy.action_init_scale = 0.01` default, with sweep max reaching `1.0`.
  Dogfight 3's config constructed the continuous action head with `0.01`, but
  its LSTM wrapper then reinitialized wrapped policy weights with scale `1.0`.
- `policy.value_init_scale = 1.0`
- `learning_rate = 0.00045`
- `horizon = 64` (Dogfight 3 `bptt_horizon`)
- `gamma = 0.99`
- `gae_lambda = 0.999`
- `ent_coef = 0.0024`
- `clip_coef = 0.11`
- `vf_coef = 2.9`
- `vf_clip_coef = 1.5`
- `max_grad_norm = 1.8`
- `momentum = 0.9896` (Dogfight 3 `adam_beta1`)
- `eps = 1.49e-08`
- `prio_alpha = 0.99`
- `prio_beta0 = 0.99`
- `vtrace_rho_clip = 2.79`
- `vtrace_c_clip = 3.5`
- `replay_ratio = 1.0`
- `env.domain_randomization = 0.05`
- `env.vertical_spawn_prob = 0.02`
- `env.recovery_trigger_prob = 0.067`

Dogfight behavior restored from the 3.0 reference:

- Observation code matches Dogfight 3.
- Flight physics differs only by a macro rename from `K` to
  `INDUCED_DRAG_K`.
- Stage 10 dive attack uses the Dogfight 3 angle window `120-175` degrees.

Dogfight sweep space now includes:

- Sweep metric: `curriculum_mastery_quality`, derived from curriculum target,
  base-stage kill rate, and surface-control saturation. This replaced
  `curriculum_soft_quality` because target progress alone can prefer early
  promotion with poor base-stage kills.
- `vec.total_agents` narrowed to `2048-4096`, default `4096`. A measured
  `8192`-agent run on `g240` was fast but trained worse, finishing around
  `base_stage_kills ~= 0.33`.
- `vec.num_buffers` sampled as `int_uniform`.
- `policy.action_init_scale` centered around `0.01`, but allowed up to `1.0`
  because the Dogfight 3 recurrent wrapper made that effective scale reachable.
- `policy.num_layers` from `1` to `3`, sampled as `int_uniform`.
- Sweep trial length is short: `50M-125M`, centered at `75M`, so bad
  configurations are rejected before wasting `400M` validation-scale runs.
- `train.state_buffer_size` from `512` to `20_000`, centered at `8192`, sampled
  as `int_uniform` so PROTEIN produces whole buffer sizes.
- `train.cl_frac` from `0.05` to `0.8`, centered at `0.35`, so sampled trials
  actually enable state curriculum while staying below the 5.0 `<= 0.9`
  assertion.

Do not replace this with ad hoc high-LR/tiny-run trainer CLI overrides while
debugging training quality. First compare behavior against Dogfight 3 and port
one concrete behavior mismatch at a time.

Latest verification after the state-memory sweep-space update:

- `.venv/bin/python -m pytest ocean/dogfight/tests -q`:
  `57 passed, 1 skipped`.
- `source .venv/bin/activate; ./build.sh dogfight`:
  built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `PYTHONUNBUFFERED=1 .venv/bin/python -m pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1 --sweep.max-runs 2`:
  completed with exit code `0`; the second PROTEIN trial launched with
  nonzero state-memory sweep knobs and active GPU, around `1.8M` SPS and
  `1.9/8G` VRAM.

Latest plain local-venv GPU smoke after the env-default restore:

- Command: `source .venv/bin/activate; python -m pufferlib.pufferl train dogfight`
- Wrapper only: `timeout 180s bash -lc 'source .venv/bin/activate; python -m pufferlib.pufferl train dogfight'`
- Result: completed the configured 400M-step run with exit code 0.
- Throughput: mostly `3.3M-3.6M` SPS.
- GPU: active throughout, approximately `33-40%`, `VRAM: 1.6/8G`.
- Model params: `53.3K` after switching Dogfight default `policy.num_layers` to
  `1`.
- Training quality: still not acceptable. It did not promote; it finished around
  `stage ~= 0.27`, `avg_stage ~= 0.90`, `curriculum_quality ~= 0.028`,
  `base_stage_kills ~= 0.39`, with `kl = 0.000` and `clipfrac = 0.000`. It
  improved some survival/return stats versus earlier smokes but still shows high
  aileron saturation around `0.87`.

That means environment setup and GPU execution are working, but the 5.0 port
still has a Dogfight behavior/training mismatch to find against Dogfight 3.
