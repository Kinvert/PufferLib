# df39 Training Investigation

Date: 2026-06-04

## Constraints

- Do not make the Dogfight curriculum easier than Dogfight3.
- Treat the df36 PufferLib 3.0 run `kinvert-k/df36/8wv5m6ru`, commit
  `171482a9b9889bebaa427ab658c95c0fc391c031`, as the known-trainable source of
  truth for curriculum stages, observations, physics, rewards, reset semantics,
  and self-play behavior.
- Make only the changes needed to adapt that environment to PufferLib 5.0 APIs,
  state handling, build/registration, and trainer interfaces. Do not change
  difficulty levels or flight physics as a training-quality fix.
- Prefer Dogfight-local fixes. Do not change PufferLib core unless Dogfight-local options have been checked and ruled out.

## W&B Findings

df39 local artifacts and W&B agree that the best runs are now on commit `1f1e7a04`:

- `qbktakds`: target `9.9`, mastery stage `10`, soft quality `0.3678`, kill rate `0.0417`, action saturation `0.7194`.
- `dp2u2tx8`: target `9.9`, mastery stage `10`, soft quality `0.4105`, kill rate `0.5667`, action saturation `0.4432`.
- Earlier df39 best before the GELU encoder was `bxmro3re` on `0825ada5`, target `4.9`.

df38 reached target `8.9` on commit `ae750d44` in multiple runs. Representative top runs include `acj17oyc`, `k9wof24s`, `raewv8jo`, `y5aa27z9`, all at target `8.9`.

df36 used different metric names. It logs `environment/stage` and `environment/avg_stage` instead of `env/curriculum_target`. The user-provided df36 W&B run `kinvert-k/df36/8wv5m6ru` points at the PufferLib 3.0 commit `171482a9b9889bebaa427ab658c95c0fc391c031` (`Increase Max Checpoint Promotions`). Top df36 runs reached stage `20` on commit `171482a9`; the best score query found `d8jn9wiv` at stage `20`, score `0.9139`, base-stage kills `0.9569`. High-ELO df36 references also include commit `51514907`.

df40 local W&B snapshot on 2026-06-04, while the sweep was still running:

- Local artifacts showed 62 summarized df40 runs.
- Best target so far: `jguhff7i`, target `10.9`, soft quality `0.386`, mastery quality `0.250`, base-stage kill rate `0.647`, ground rate `0.353`, base-stage action saturation `0.716`.
- Later local artifacts also showed `vg730646`, target `10.9`, soft quality `0.370`, mastery quality `0.370`, base-stage kill rate `1.000`, ground rate `0.000`, base-stage action saturation `0.797`. It used hidden size `64`, 3 layers, `learning_rate ~= 0.0010`, `horizon = 64`, `minibatch_size = 4096`, `replay_ratio ~= 1.03`, `gamma ~= 0.996`, `state_buffer_size = 4447`, and `cl_frac = 0.8`.
- Next best: `hujy34rz`, target `9.9`, soft quality `0.375`, base-stage kill rate `0.625`, ground rate `0.344`, base-stage action saturation `0.630`.
- User-highlighted early run: `6yzvvs4b`, target `8.9`, soft quality `0.330`, base-stage kill rate `1.000`, ground rate `0.000`, base-stage action saturation `0.625`.
- Hidden-size distribution in 78 local df40 summaries: `256` appeared in 24 runs, `128` in 36, `64` in 15, and `32` in 3. The verified df36 hidden size (`256`) is therefore reachable in df40, but the local `256` summaries inspected so far did not climb past target `0.9`.
- Recomputing `curriculum_soft_quality` from `curriculum_target` and base-stage action saturation matches the logged values within about `0.01` for the leading runs. The soft metric is probably not the direct explanation for the stage climb; the current concern is that stage 10-11 runs climb with high ground-crash rate and high control saturation.

df40 local W&B snapshot after the user stopped the sweep:

- Local artifacts showed 165 summarized df40 runs.
- The ceiling was still target `10.9`, with two runs: `jguhff7i` and
  `vg730646`. `jguhff7i` finished with kill rate `0.647`, ground rate `0.353`,
  and action saturation `0.716`; `vg730646` finished with kill rate `1.000`,
  ground rate `0.000`, and action saturation `0.797`. Both remain useful but
  not clean evidence for stage-17 readiness because saturation is still high,
  and one of the two crashes hard at the current base stage.
- All runs that reached target `>= 9.9` used `policy.num_layers = 3`,
  `horizon = 64`, `minibatch_size = 4096`, fixed Dogfight3 stage-9 bank
  (`stage9_bank_deg = 30.0`), and state curriculum enabled. Their `cl_frac`
  ranged from about `0.66` to `0.80`, learning rate from about `0.0010` to
  `0.0041`, gamma from about `0.993` to `0.999`, and effective minibatches per
  rollout from about `55` to `66`.
- Runs at target `>= 8.9` followed the same shape: 3 layers, horizon `64`,
  minibatch `4096`, `cl_frac` usually high, and effective minibatches around
  `47-66`. This supports centering the next short sweep around the measured
  stage climbers rather than drifting back to the plain df36 baseline batch
  shape, but it does not justify changing curriculum difficulty.
- Config follow-up: `config/dogfight.ini` now keeps plain `[train]` on the
  df36-derived baseline, but recenters the Dogfight sweep space for the next
  short W&B search around the stopped-df40 high-stage cluster: `3` layers,
  `vec.num_buffers = 4`, `horizon = 64`, `minibatch_size = 4096`,
  `learning_rate = 0.0025`, `ent_coef = 0.012`, `replay_ratio = 0.95`,
  `gamma = 0.996`, `clip_coef = 0.06`, `vf_coef = 4.6`,
  `max_grad_norm = 3.4`, `vtrace_rho_clip = 0.1`,
  `vtrace_c_clip = 2.5`, `prio_alpha = 0.4`, `prio_beta0 = 0.82`,
  `state_buffer_size = 4096`, and `cl_frac = 0.77`. This is a sweep-space
  change only; curriculum difficulty and Dogfight physics remain unchanged.
- Local max-runs-2 smoke after the recenter completed with exit code `0` under
  the local Dogfight5 venv and GPU. It validates launch, GPU use, and sampled
  state-curriculum knobs, but it does not validate training quality. The first
  run was the plain short baseline (`128` hidden, `1` layer, no state
  curriculum), ended at target `0.9`, about `1.7M` SPS, `1.65G` VRAM, final
  base-stage kill/ground/timeout rates `0.312/0.208/0.480`, action saturation
  `0.300`, and soft quality `0.0426`. The second sampled run used `256`
  hidden, `2` layers, `2048` agents, `state_buffer_size = 3829`, `cl_frac =
  0.684`, `learning_rate = 0.00357`, `ent_coef = 0.0265`, `gamma = 0.99973`,
  and `horizon = 64`; it ended at target `0.9`, about `0.86M` SPS, `1.37G`
  VRAM, nonzero update signal (`kl ~= 0.0026`, `clipfrac ~= 0.0265`), final
  base-stage kill/ground/timeout rates `0.434/0.205/0.361`, but severe action
  saturation `0.996` and soft quality only `0.0251`. Next W&B df41 should be
  treated as a real search, not proof that the recentered space is solved.
- Config follow-up: raised the Dogfight sweep lower bound for
  `policy.action_init_scale` from `0.003` to `0.2`. This is not an env or
  curriculum change. It keeps the df36 effective action-head init (`1.0`) and
  the stopped-df40 high-stage cluster reachable while avoiding the latest
  sampled low-init smoke corner (`action_init_scale ~= 0.0062`), which ended at
  target `0.9` with severe control saturation.
- Post-tightening local max-runs-2 smoke also completed with exit code `0`.
  The sampler output now shows the lower action-init random sample around
  `0.669`, not the old `0.003-0.006` corner. The sampled run used `256`
  hidden, `3` layers, `4096` agents, `vec.num_buffers = 4`, `horizon = 128`,
  `action_init_scale = 0.251`, `learning_rate = 0.00614`,
  `state_buffer_size = 3017`, and `cl_frac = 0.605`. It still ended at target
  `0.9`, but the failure mode changed: final base-stage kill/ground/timeout
  rates were about `0.634/0.000/0.366`, action saturation was `0.409`, and soft
  quality was `0.0395`. This supports keeping the action-init lower bound
  tightened. It does not justify treating the current sweep space as solved.
- Later local df40 logs after the tighter action-init bound produced additional
  target `6.9-9.9` runs. They still share the same broad shape: `3` layers,
  horizon `64`, minibatch `4096`, high action init, and state curriculum on.
  `h5qvaso2` reached target `8.9` with kill rate near `1.0`, ground rate `0.0`,
  saturation about `0.721`, `learning_rate ~= 0.0070`, `kl ~= 0.0099`, and
  `clipfrac ~= 0.233`; this is a real update signal, but not stage-17 evidence.
  `kwvprdrt` reached target `9.9` with kill rate about `0.848`, ground rate
  about `0.152`, saturation about `0.520`, and mastery quality about `0.344`.
  `pqqg89wn` reached target `9.9` and soft quality `0.417`, but its final
  base-stage window had reset to `0` episodes after promotion, so the final
  base-stage kill/ground/timeout rates are not meaningful and mastery quality is
  `0.0`. Treat soft quality as a useful search proxy, not proof that the final
  policy is healthy at the new target.
- Config follow-up: widened only `sweep.train.learning_rate.max` from `0.0065`
  to `0.01` for df41. The sweep center remains `0.0025`; the change just keeps
  the latest clean target-8.9 df40 climber reachable. No Dogfight curriculum,
  reward, observation, physics, or PufferLib core behavior changed.
- Follow-up local log scan after the LR max widening found `202` local df40
  summaries and no local df41 summaries yet. Target counts were: `0.9` in `155`
  runs, `1.9` in `21`, `3.9` in `2`, `5.9` in `4`, `6.9` in `9`, `7.9` in
  `1`, `8.9` in `4`, `9.9` in `4`, and `10.9` in `2`. Newer df40 logs
  (`m11zq8iu`, `xonrmjka`, `50jh4fkb`) stayed at target `0.9`; they do not
  justify moving the df41 center.
- High-target df40 runs remain covered by the current df41 space. The strongest
  common traits are still `policy.num_layers = 3`, `horizon = 64`,
  `minibatch_size = 4096`, high action-init scale, and state curriculum on.
  Entropy often sits near the upper half of the current range, but the mixed
  results at `ent_coef ~= 0.02` do not justify another config-space move before
  a real df41 sweep.

Core-change audit after the user asked to avoid PufferLib core changes:

- No PufferLib core files were edited during this audit pass.
- Current dirty core diffs should remain suspect until reviewed explicitly.
- `pufferlib/torch_pufferl.py` has an env-local model component resolver. That
  only supports the `--slowly --torch.network LSTM` diagnostic path with
  Dogfight-local `DogfightEncoder`/`DogfightDecoder`; normal native training
  ignores it. This is optional diagnostic glue, not required for normal df41
  sweeps.
- `src/vecenv.h` and `src/bindings.cu` have a `MY_GLOBAL_STEP` hook. This maps
  to df36 behavior: exact Dogfight3 `train_dual_selfplay.py` called
  `binding.vec_set_global_step(...)`, and exact Dogfight3 `binding.c` wrote
  `env->global_step` for every env so shaping reward decay could use trainer
  steps. Current Dogfight5 needs an equivalent path for reward-shaping parity,
  but the implementation is still core opt-in glue and should be reviewed
  before upstream.
- `src/pufferlib.cu` has native recurrent-state minibatch plumbing. PufferLib 5
  already had `TrainGraph.mb_state` and passed it to `policy_forward_train`, but
  base `HEAD` did not store rollout recurrent state or copy selected state rows
  into `graph.mb_state`. The current diff fills that gap. This is not a
  Dogfight-local change; treat it as a suspected native learner issue to test
  and review, not as an automatic Dogfight-port requirement.

Crash/OOB reward check for that exact df36 commit:

- `171482a9b9889bebaa427ab658c95c0fc391c031:pufferlib/ocean/dogfight/dogfight.h` already has the simplified crash rewards: crasher `-1.0`, survivor `+0.25`.
- Opponent OOB/crash at that commit sets player reward `0.25` and opponent reward `-1.0`.
- Player OOB/crash at that commit sets player reward `-1.0` and opponent reward `0.25`.
- Supersonic at that commit sets both player and opponent rewards to `-1.0`.
- Timeout at that commit sets both player and opponent rewards to `-0.5`.

## Open Metric Anomaly

The df39 runs `dp2u2tx8` and `qbktakds` advanced to target `9.9`/mastery stage `10` despite apparently large signed control-bias metrics such as `base_stage_signed_bias_rudder`. This needs a deeper audit before treating those runs as clean evidence of learning quality.

Decoded local W&B histories on 2026-06-04:

- Both runs reached target `9.9` only after old-stage promotion windows reported high kill rates, usually `0.90-1.00`. Promotion itself only gates on `base_stage_kills/base_stage_eps`; signed bias is diagnostic and was not a promotion gate.
- `dp2u2tx8` promoted `8.90 -> 9.90` at `71.3M` steps with window kill rate `0.995`, ground rate `0.005`, episode length about `747`, and cumulative signed biases `elevator=-408.8`, `aileron=196.3`, `rudder=-361.6`. Normalized per-step, those are about `-0.55`, `0.26`, `-0.48`.
- `qbktakds` promoted `8.90 -> 9.90` at `64.5M` steps with window kill rate `0.990`, ground rate `0.010`, episode length about `641`, and cumulative signed biases `elevator=-269.7`, `aileron=106.4`, `rudder=498.0`. Normalized per-step, those are about `-0.42`, `0.17`, `0.78`.
- `qbktakds` then collapsed by the final row at `68.7M` steps: base-stage kill rate `0.042`, window kill rate `0.033`, ground rate `0.958`, action saturation `0.719`, and normalized per-step biases about `elevator=-0.21`, `aileron=-0.61`, `rudder=0.87`. It reached target `9.9`, but the final policy was not actually healthy at stage 10.
- Root metric issue found: `base_stage_signed_bias_*` was emitted as cumulative signed action per episode averaged over episodes, not mean signed action per tick. This made W&B values look like impossible constant full-control commands. The Dogfight-local telemetry path now normalizes signed-bias logs by episode length before exporting them.

Questions still open:

- Can high signed rudder/aileron bias still coexist with enough stage-specific kills because the stage geometry/reward allows a biased tactic, especially before self-play/anchor eval exists?
- Did the old stage-9 ramp let a biased policy advance in df39 in a way that the Dogfight3 fixed curriculum would not?
- Does `curriculum_soft_quality` over-credit stage advancement relative to true kill/win behavior when action bias is extreme?

Next audit step: after the next sweep uses normalized signed-bias telemetry, compare promotion rows against final rows and look for policies that both advance and retain healthy stage-10 kill/ground rates. Treat `qbktakds`-style "advanced then crashed" runs as weak evidence even if `curriculum_soft_quality` is high.

## Reference Differences Checked

- Curriculum stage enum and `STAGES` table match Dogfight3, including stage 9 bank `30`, stage 17 bank `60`, stage 18 crossing geometry, and stage 20 AutoAce.
- Dogfight5 had an active stage-9 bank ramp through config (`stage9_bank_deg=-1`, `stage9_bank_curriculum=1`). Dogfight3 did not train with that easier stage-9 ramp; stage 9 comes directly from `STAGES[9].bank == 30`.
- Exact df36 source at commit `171482a9b9889bebaa427ab658c95c0fc391c031` defines observation schemes as:
  `OBS_MOMENTUM_GFORCE = 0` with 17 observations, `OBS_PILOT = 1` with 22 observations, and `OBS_OPPONENT_AWARE = 2` with 26 observations.
- Current Dogfight5 defines observation schemes as:
  `OBS_PILOT = 0` with 22 observations and `OBS_OPPONENT_AWARE = 1` with 26 observations. The old 17-wide momentum/g-force scheme is preserved in code but removed from dispatch.
- This means current `config/dogfight.ini` `obs_scheme = 1` trains 26-wide `OBS_OPPONENT_AWARE` in Dogfight5. In exact df36 source, `obs_scheme = 1` meant 22-wide `OBS_PILOT`.
- The previous investigation note claiming Dogfight3 scheme 1 was `OBS_OPPONENT_AWARE`/26-wide was wrong. It was likely caused by comparing against the already-renumbered Dogfight5 code or another non-df36 working tree.
- df36 W&B run config for `kinvert-k/df36/8wv5m6ru` was verified with the W&B API on 2026-06-04: `env.obs_scheme = 2`, `policy.hidden_size = 256`, `train.total_timesteps ~= 955.8M`, summary `environment/stage = 20`, summary `environment/avg_stage = 20`, summary `environment/score = 0.6776`. This means the known-good df36 run used old `OBS_OPPONENT_AWARE` / 26-wide observations, not old `OBS_PILOT` / 22-wide observations.
- Current Dogfight5's `obs_scheme = 1` also means 26-wide `OBS_OPPONENT_AWARE`, so the active df40 observation family appears correct for the known-good df36 run. The dangerous part is numeric compatibility: df36 old scheme `2` and Dogfight5 current scheme `1` both mean 26-wide opponent-aware, while df36 old scheme `1` and Dogfight5 current scheme `1` do not mean the same thing.
- `autoace.h` and `autopilot.h` are byte-identical to exact df36. `flightlib.h` differs only by a macro rename from `K` to `INDUCED_DRAG_K` plus env-local deterministic RNG helpers for state restore; the physics equations and constants still appear parity-safe.
- `test_flightlib_source_compat.py` now guards the full `flightlib.h` source
  against exact df36 after normalizing only those known intentional differences.
  This makes flight physics drift a lower-probability explanation for df40/df41
  stage-climb failures unless a future diff breaks that test.
- Side-stage energy spawn is not a df36 mismatch. Exact df36 already had a
  hardcoded `20%` energy-building branch in `spawn_side` for stages 6-9.
  Current Dogfight5 made that branch configurable as `side_energy_spawn_prob`
  and currently keeps the same default, `0.2`. The added side-variant telemetry
  is diagnostic; the active geometry probability still matches df36.
- Early stage spawn geometry does not currently look like a df36 drift either.
  `test_stage_spawn_source_compat.py` now compares `spawn_tail_chase`,
  `spawn_head_on`, `spawn_crossing`, `spawn_vertical`, `spawn_gentle_turns`,
  `spawn_offset`, and `spawn_angled` against exact df36 after normalizing only
  `dogfight_rndf(env, ...)` back to `rndf(...)` and ignoring
  `low_altitude_variant` telemetry. The same test also compares `spawn_side`
  after normalizing the documented 5.0 config/telemetry wrappers back to df36's
  hardcoded side-energy probability and `cfg->bank` branch. This covers stages
  0-9 at source level; the existing C geometry tests still cover sampled stage
  8-17 safety and stage-9 bank defaults.
- Stage 10-17 spawn geometry does not currently look like a df36 drift. A
  normalized source comparison against exact df36 commit
  `171482a9b9889bebaa427ab658c95c0fc391c031` found the full `STAGES` table
  equal, and found `spawn_dive_attack`, `spawn_zoom_attack`, `spawn_rear`,
  `spawn_full_predictable`, `spawn_full_random`, `spawn_medium_turns`, and
  `spawn_hard_maneuvering` equal after normalizing the Dogfight5
  `dogfight_rndf(env, ...)` helper back to df36's `rndf(...)` and ignoring
  comments/whitespace. Existing C coverage checks stage 8-17 safety/caps and
  stage 10-12 vertical altitude relationships, but it is not a full
  deterministic df36 spawn fixture for every sampled value. If stage 10-17
  geometry becomes suspect again, add numeric parity fixtures before changing
  difficulty. Current evidence points away from stage-geometry drift as the
  reason df40 target 10-11 policies show high ground/crash or saturation.
  `test_stage_spawn_source_compat.py` now guards this comparison directly. It
  is source-level rather than reset-by-reset numeric because df36 used global
  `rand()` while Dogfight5 intentionally uses `env->rng`/`dogfight_rndf` for
  deterministic state-memory restore.
- Reward-shaping decay now has explicit df36-style C coverage.
  `test_reward_shaping_decay_reference.c` reuses the scripted nonterminal state
  and checks three `global_step` positions: before the decay window, midway
  through it, and at completion. It verifies that only `r_closing` and `r_aim`
  anneal while control-rate, altitude, player-energy, and energy-advantage terms
  remain unchanged. This reduces risk that the 5.0 `global_step` adapter is
  silently changing reward semantics before the stage-climb search.
- Continuous action handling has a location difference, but not a tanh/squash
  difference. Exact df36 samples from an unsquashed Normal, stores the raw
  sampled action and raw logprob in rollout buffers, then clips the NumPy action
  to the env action-space before `vecenv.step`. Its C `c_step` does not clamp
  before `step_plane_with_params` because the trainer already did that.
- Current Dogfight5/PufferLib 5 also uses an unsquashed Normal. The PyTorch
  decoder returns `torch.distributions.Normal`, the PyTorch logprob path uses
  `logits.log_prob(raw_action).sum(1)`, and the native CUDA sampler writes raw
  Normal samples/logprobs into rollout buffers. The Dogfight-local adapter then
  clamps `env->actions[i]` at the top of `c_step`, before debug output, physics,
  action telemetry, and reward terms.
- A separate continuous-distribution difference was found on 2026-06-04 but
  was not changed: exact df36 clamps continuous `decoder_logstd` with
  `.clamp(min=-20, max=2)` before constructing `torch.distributions.Normal`.
  Current PufferLib 5 native CUDA reads raw `logstd` in `sample_logits`,
  `ppo_continuous_head`, and the continuous PPO gradient path. The Dogfight
  PyTorch diagnostic `DogfightDecoder` already restores the df36 clamp for
  `--slowly`, but normal native Dogfight training still uses the unclamped
  native CUDA path.
- Interpretation: Dogfight5 moved the Box clipping from the df36 Python trainer
  boundary into Dogfight C because the PufferLib 5 native path feeds env action
  buffers directly. This should preserve bounded physics inputs while keeping
  PPO logprobs on raw unsquashed samples. The current high action-saturation
  metrics are therefore more likely a learned-policy/training symptom than a
  hidden tanh/squashed-action mismatch. The native `logstd` clamp difference is
  still a candidate trainer mismatch, but fixing it would touch PufferLib core
  continuous-action code and should be handled as a separate, tested decision
  only after Dogfight-local options are exhausted.
- df36's `train_dual_selfplay.py` does not use dual/opponent-perspective
  experience during the early curriculum climb by default. It initializes
  `use_dual_selfplay = False`, calls the standard trainer path while false, and
  only activates dual self-play after three checks at
  `selfplay_min_stage - 0.1` with the default `selfplay_min_stage = 20`.
  Therefore Dogfight5's current one-agent native stage-climb baseline is not an
  obvious explanation for failing stages 0-17.
- Exact df36 updates C-side `global_step` before Dogfight steps so reward
  shaping decay uses training progress. The regular Dogfight env path used
  `vec_set_global_step(..., tick * num_agents)`, and the dual path used
  `vec_set_global_step(..., trainer.global_step)`. Dogfight5 native rollout was
  incrementing `pufferl.global_step` after each rollout but did not pass the
  current value into Dogfight envs before `c_step`. That meant `env->global_step`
  could remain at init/restored values in native training, keeping aim/closing
  shaping decay stuck before the configured `100M-150M` window.

## Changes Made From This Investigation

- Current `config/dogfight.ini` default is `obs_scheme = 1`, which maps to 26-wide `OBS_OPPONENT_AWARE` in Dogfight5. This matches the observation family used by verified df36 run `8wv5m6ru` (`env.obs_scheme = 2` in old df36 numbering), but it does not preserve numeric scheme IDs from df36.
- Set `config/dogfight.ini` `stage9_bank_deg = 30.0` and `stage9_bank_curriculum = 0` so active training uses Dogfight3 stage-9 difficulty.
- Native Dogfight binding currently uses `OBS_SIZE 26` and does not clamp to 22-wide `OBS_PILOT`. That supports the verified df36 26-wide observation family, but the numeric scheme-ID remap remains a metadata/checkpoint compatibility risk.
- Added `obs_compat.py` and `test_obs_scheme_compat.py` so the df36 old scheme `2` -> Dogfight5 current scheme `1` mapping is explicit and covered by tests.
- Added `action_contract.py` and `test_continuous_action_contract.py` so the
  df36/PufferLib 5 continuous-action clipping-location difference is explicit:
  raw unsquashed Normal samples/logprobs for PPO, and Dogfight-local clamping
  before physics.
- Added a guarded PufferLib 5 native global-step adapter for Dogfight. Dogfight
  defines `MY_GLOBAL_STEP` and `my_set_global_step`; `src/vecenv.h` provides a
  no-op default for other envs and propagates per-rollout/per-horizon global
  step before `net_callback` and `c_step`; `src/bindings.cu` seeds the rollout
  from `pufferl.global_step` before `static_vec_omp_step`.
- Updated `test_observation_padding.c` to prove scheme 0 writes only its declared 22 values instead of clearing/writing a 26-wide tail.
- Set Dogfight's omitted-key stage-9 defaults to Dogfight3 behavior (`30.0` bank, no substep ramp). Explicit `stage9_bank_deg = -1` still keeps the diagnostic ramp reachable, but it is no longer the default in C init, native binding fallback, or Dogfight-specific pufferl curriculum setup.
- Set Dogfight's direct native `policy.action_init_scale` default to `1.0`. Dogfight3's policy constructor used `0.01`, but `DogfightRecurrent` immediately reinitialized wrapped policy weights with orthogonal gain `1.0`, making `1.0` the effective df36 action-head scale.
- Moved Dogfight reset/spawn/domain-randomization draws from process-global `rand()`/`rndf()` to `env->rng`, matching the PufferLib 5 state-memory pattern used by Boxoban/G2048. The added terminal roundtrip test first failed because restored pre-kill states reset into different future geometry, then passed after the local RNG fix.
- Normalized Dogfight signed-bias telemetry by episode length before logging it, so `avg_signed_bias`, `base_stage_signed_bias_*`, and side-variant signed-bias diagnostics are mean per-step action biases rather than cumulative per-episode sums.

## Not Changed

- Did not change the `STAGES` curriculum table or make any curriculum step easier.
- Did not change Dogfight physics constants or integration equations in `flightlib.h`; only the reset-time random source was made env-local.
- Did not change action scaling, continuous-action distribution code, or optimizer code.
- Did not change the action path while documenting the df36-vs-5.0 clipping
  location difference.
- Did not change PufferLib core for the newly found native continuous `logstd`
  clamp difference; it is documented as an open candidate mismatch only.
- Did not change `side_energy_spawn_prob`; exact df36 already used the same
  `20%` side-stage energy-spawn probability.
- Did not change curriculum promotion logic; signed bias remains diagnostic and does not gate stage advancement.
- Did not change general PufferLib behavior for envs that do not opt in to the
  global-step hook; the new vecenv hook defaults to no-op unless an env defines
  `MY_GLOBAL_STEP`.
- Did not enable self-play or anchor-rating optimization for df39/df40 stage-climb sweeps.

## Current Culprit Ranking

1. Observation numeric compatibility: verified df36 `8wv5m6ru` used old `obs_scheme=2`, 26-wide `OBS_OPPONENT_AWARE`. Current Dogfight5 uses `obs_scheme=1` for the same 26-wide observation family. This probably is not the df40 training blocker, but it is a compatibility trap for checkpoints, metadata, and future comparisons unless documented/tested explicitly.
2. Policy width/hyper mismatch: verified df36 `8wv5m6ru` used `policy.hidden_size = 256`; current Dogfight5 plain baseline uses `128`, while df40 sweeps can sample `256`. Local df40 summaries include 24 hidden-size-256 runs, none beyond target `0.9` so far, while stage-climbers are `64`/`128`. Hidden size alone is not the obvious blocker, but exact df36-style width plus df40's best native hyperparameters has not been isolated.
3. Update-budget mapping: exact df36's PyTorch trainer used
   `update_epochs = 4` and computed `total_minibatches = update_epochs *
   batch_size / minibatch_size`. PufferLib 5 native uses
   `total_minibatches = replay_ratio * batch_size / minibatch_size`. With the
   current plain Dogfight5 `minibatch_size = 65536`, `replay_ratio = 1.0` gives
   a lower plain update budget than the old df36 `update_epochs = 4` path if
   batch/minibatch shapes are otherwise comparable. However, df40's best short
   stage climbers used `minibatch_size = 4096`, making effective minibatches per
   rollout much higher even with `replay_ratio ~= 0.5-1.0`. This is a real
   mapping issue for plain baseline parity, but not a simple "increase replay"
   explanation for the current df40 top runs.
4. Stage-9 easing: df39 target `9.9` happened while stage 9 was easier than Dogfight3. Treat those results as not directly comparable to df36 until the fixed 30-degree stage-9 path is swept. The active config and omitted-key defaults now both use 30 degrees.
5. Native policy architecture: commit `1f1e7a04` restored Dogfight3-style `Linear + bias + GELU` encoder behavior and df39 improved after it. Keep this under scrutiny because it lives in `src/ocean.cu`, but it uses the existing custom encoder extension point.
6. State-memory reset RNG: before the env-local RNG fix, a restored preterminal state could produce a different post-terminal reset because reset/spawn used process-global `rand()`. This is now fixed for the stage-climb reset/spawn path and covered by `test_state_roundtrip.c`.
7. Continuous action tanh/squash mismatch is now lower risk: df36 and Dogfight5 both logprob raw unsquashed Normal samples and clip only before physics. `test_continuous_action_contract.py` now covers the source-level contract: unsquashed Normal decoder, no PyTorch tanh action transform, and Dogfight C clamp before `step_plane_with_params`. It does not replace a full GPU rollout/update integration audit if the action path becomes suspect again.
8. Global-step shaping decay adapter: this was a real PufferLib 5 native
   adapter gap and is now fixed/tested. It matters most after `100M` steps,
   where df36 decayed aim/closing shaping but Dogfight5 native could leave it
   stuck at full strength.
9. Native continuous `logstd` clamp: df36 clamps `decoder_logstd` to `[-20, 2]`
   before Normal sampling/logprob. Current native CUDA uses raw `logstd` in
   sampling, PPO logprob, entropy, and `grad_logstd`. This is not Dogfight
   local and was not changed. If pursued, first add focused coverage proving
   the df36 clamp semantics and only then decide whether a core native
   continuous-action fix is justified.
10. Remaining likely areas: exact native encoder/recurrent weight initialization vs Dogfight3 recurrent wrapper, broader observation parity, step/reset semantics across multi-reset traces, reward terms under broader scripted coverage, and physics parity. AutoAce stage-20 tactical randomness still needs a separate state-memory audit if sweeps target stage 20. No curriculum step changes should be used as a fix.

Observation parity update:

- Added a Dogfight3 numeric fixture for a scripted `OBS_OPPONENT_AWARE` state.
- `test_observation_scheme1_reference.c` confirms Dogfight5's 26-wide scheme1 observation math matches that Dogfight3 fixture.
- This reduces observation-math drift risk for the restored scheme1 path, but it does not prove reward, step/reset, action distribution, or training quality.

Step/reward parity update:

- Added a Dogfight3 numeric fixture for one nonterminal scripted `c_step` with explicit player and opponent actions.
- `test_scripted_step_reference.c` confirms Dogfight5 matches Dogfight3 for next player/opponent plane state, reward components, terminal/death flags, and next scheme1 observations in that fixture.
- This reduces risk in the nonterminal step/reward path. Basic kill/OOB/timeout/opponent-kill terminal transitions are now covered separately.

Scripted trace parity update:

- Added a numeric fixture from the exact df36 PufferLib 3.0 commit `171482a9b9889bebaa427ab658c95c0fc391c031` for a 12-step nonterminal scripted trace with explicit player and opponent actions.
- `test_scripted_trace_reference.c` confirms Dogfight5 matches df36 for every per-step reward, final player/opponent physics state, reward components and accumulators, previous-action state, and final scheme1 observations.
- This reduces risk in longer nonterminal physics/reward drift. Longer traces across reset boundaries still need coverage before treating the port as behavior-complete.

Terminal kill parity update:

- Added a Dogfight3 numeric fixture for a scripted player kill terminal step.
- `test_terminal_kill_reference.c` confirms Dogfight5 matches Dogfight3 for the player kill reward, zero-sum opponent reward, terminal flag, reset-visible death/winner bookkeeping, and kill/log counters.
- This rules out the basic player-kill terminal reward/log/reset path as the current training-quality mismatch. Basic OOB, timeout, and opponent-kill terminal paths are now covered separately; longer reset traces still need coverage.

Terminal OOB parity update:

- Added a numeric fixture from the exact df36 PufferLib 3.0 commit `171482a9b9889bebaa427ab658c95c0fc391c031` for scripted player-crash and opponent-crash terminal steps.
- `test_terminal_oob_reference.c` confirms Dogfight5 matches df36 for crasher/survivor rewards, zero-sum opponent rewards, terminal flag, reset-visible OOB bookkeeping, ground-hit counters, base-stage counters, and score/perf logs.
- This rules out the basic player/opponent OOB terminal reward/log/reset path as the current training-quality mismatch. Basic timeout and opponent-kill terminal paths are now covered separately; longer reset traces still need coverage.

Terminal timeout parity update:

- Added a numeric fixture from the exact df36 PufferLib 3.0 commit `171482a9b9889bebaa427ab658c95c0fc391c031` for a scripted max-step timeout.
- `test_terminal_timeout_reference.c` confirms Dogfight5 matches df36 for timeout rewards (`-0.5` for both sides), terminal flag, reset-visible timeout bookkeeping, base-stage counters, and score/perf logs.
- This rules out the basic timeout terminal reward/log/reset path as the current training-quality mismatch. Longer reset traces still need coverage.

Terminal opponent-kill parity update:

- Added a numeric fixture from the exact df36 PufferLib 3.0 commit `171482a9b9889bebaa427ab658c95c0fc391c031` for a scripted self-play opponent kill.
- `test_terminal_opponent_kill_reference.c` confirms Dogfight5 matches df36 for opponent-kill rewards (`-1.0` player, `+1.0` opponent), terminal flag, reset-visible winner bookkeeping, self-play opponent kill counters, clean-fight counters, base-stage counters, and score/perf logs.
- This rules out the basic self-play opponent-kill terminal reward/log/reset path as the current training-quality mismatch. Longer multi-reset traces still need coverage.

Continuous-action contract update:

- Added a Dogfight-local action contract note/helper for the df36/PufferLib 5
  clipping-location difference.
- `test_continuous_action_contract.py` confirms the current policy decoder uses
  `torch.distributions.Normal`, the PyTorch continuous logprob path uses raw
  unsquashed actions, and Dogfight clamps env actions before player physics.
- This rules out a simple hidden tanh/squashed-action mismatch as the current
  training-quality explanation. It does not prove every native CUDA PPO update
  detail matches df36; that remains a deeper trainer audit only if needed.
- Follow-up `logstd` clamp audit on 2026-06-04: df36 PyTorch clamps
  `decoder_logstd` to `[-20, 2]`, while current Dogfight5 native CUDA,
  dogfight4, and `/home/claude/PufferLib` native/PyTorch 4-era paths use raw
  `logstd`. Dogfight has 5 continuous actions, so the df36 upper clamp implies
  a maximum Normal entropy of about `17.095` if all action dimensions hit
  `logstd = 2`.
- Local artifact scan: all 953 `logs/dogfight/*.json` summaries with
  `loss/entropy` were checked. Eight exceeded the df36 upper-clamp entropy
  proxy; seven of those never passed target `0.9`, and one reached target
  `1.9`. No run at target `>= 3.9` exceeded the proxy cap. The highest
  stage-climb runs stayed below it: `vg730646` target `10.9` max entropy
  `13.235` (implied mean `logstd ~= 1.23`), `jguhff7i` target `10.9` max
  entropy `12.333` (implied mean `logstd ~= 1.05`), `6yzvvs4b` target `8.9`
  max entropy `10.540`, `dp2u2tx8` target `9.9` max entropy `8.025`, and
  `qbktakds` target `9.9` max entropy `9.766`.
- Interpretation: the native `logstd` clamp is a real df36-vs-5.0 learner
  difference, but current local evidence does not point to it as the blocker
  for the best df40 stage climbers. Do not change PufferLib core for this on
  theory alone. If it becomes worth testing, first add explicit `logstd`
  telemetry or a focused continuous-action unit test, then evaluate a narrow
  core patch separately.

Config/update-budget audit update:

- Exact df36 config at commit `171482a9b9889bebaa427ab658c95c0fc391c031`
  uses the same checked-in Dogfight environment reward/default values that are
  currently in `config/dogfight.ini`: `reward_aim_scale = 0.001695`,
  `reward_closing_scale = 0.0001`, `penalty_neg_g = 0.035`,
  `control_rate_penalty = 0.002`, `low_altitude_threshold = 1200`,
  `low_altitude_penalty = 0.005`, `speed_min = 50`,
  `shaping_decay_start = 100000000`, `shaping_decay_end = 150000000`,
  `energy_gain_scale = 0.001`, `energy_loss_scale = 0.0005`,
  `energy_advantage_scale = 0.004`, `domain_randomization = 0.05`,
  `vertical_spawn_prob = 0.02`, and the recovery settings
  `recovery_enabled = 1`, `recovery_altitude_threshold = 500`,
  `recovery_trigger_prob = 0.067`, `recovery_speed_threshold = 70`,
  `recovery_bank_deg = 60`. This lowers the likelihood that a simple
  Dogfight-local reward/default drift explains the current stage-climb ceiling.
- Exact df36 config at commit `171482a9b9889bebaa427ab658c95c0fc391c031`
  uses `[train] update_epochs = 4`, `minibatch_size = 65536`,
  `max_minibatch_size = 65536`, and `bptt_horizon = 64`. The old trainer
  computes `total_minibatches = update_epochs * batch_size / minibatch_size`.
- Current PufferLib 5 native and PyTorch trainers compute
  `total_minibatches = replay_ratio * batch_size / minibatch_size`. Therefore
  `replay_ratio` is the closest 5.0 knob to old `update_epochs`, but the
  effective update budget also depends strongly on `minibatch_size`, `horizon`,
  and `total_agents`.
- Current plain Dogfight5 config uses `total_agents = 4096`, `horizon = 64`,
  `minibatch_size = 65536`, and `replay_ratio = 1.0`, which gives `4`
  minibatches per native rollout. If compared to the same batch shape with old
  `update_epochs = 4`, the df36-style update budget would be `16` minibatches.
  This is a plain-baseline parity risk.
- The df40 sweep evidence is different from the plain baseline because the
  stage climbers use much smaller minibatches. Local artifact scan of 953
  `logs/dogfight/*.json` summaries showed the best target/soft runs clustered
  around `48-66` effective minibatches per rollout: `jguhff7i` target `10.9`
  used about `59.9`, `vg730646` target `10.9` used about `65.6`, `dp2u2tx8`
  target `9.9` used `64.0`, `hujy34rz` target `9.9` used about `64.8`, and
  `6yzvvs4b` target `8.9` used about `47.2`.
- In that same local scan, replay-ratio bins above `1.25` did not produce the
  current top stage climbers. Effective minibatches above `128` had max target
  `0.9`, while `48-80` contained the current target `10.9` runs. So the
  current evidence argues for preserving/centering the short-sweep update
  budget near the measured climbers, not blindly increasing `replay_ratio` to
  the df36 numeric `update_epochs` value.

Global-step adapter update:

- Added `test_global_step_adapter.py` after confirming it failed because
  Dogfight native training had no `MY_GLOBAL_STEP` hook.
- Added a guarded PufferLib 5 vecenv hook that propagates `pufferl.global_step`
  into Dogfight before each native rollout step, including before state
  checkpoint capture in `net_callback`.
- This restores df36's C-side reward-shaping decay contract for native Dogfight
  training. It is unlikely to explain failures before `100M` steps, but it is a
  real mismatch for longer runs and stage-climb validation.
- Post-adapter smoke note: a bounded plain local-venv native run
  (`timeout 180s bash -lc 'source .venv/bin/activate; python -m
  pufferlib.pufferl train dogfight'`) hit the wrapper timeout with exit code
  `124`, not a trainer crash. It reached at least `215.2M` steps with GPU active
  and about `1.6/8G` VRAM, so it crossed the `100M-150M` shaping-decay window
  that previously lacked df36-style global-step propagation.
- That smoke did not show acceptable training quality. The final visible TUI row
  was still around `stage = 0.559`, `avg_stage = 0.897`,
  `curriculum_quality = 0.024`, `base_stage_kills = 0.261`,
  `base_stage_timeouts = 0.636`, and `base_stage_action_saturation = 0.409`;
  `kl` and `clipfrac` were still effectively zero. Treat the global-step
  adapter as a necessary 5.0 compatibility fix, not as evidence that the current
  port should now train.

Native learner audit update:

- Exact df36 did not train Dogfight through the current PufferLib 5 native CUDA
  policy stack. Its config named `policy_name = DogfightPolicy` and `rnn_name =
  DogfightRecurrent`; those classes live in `pufferlib/ocean/torch.py` at the
  df36 commit. `DogfightPolicy` inherits the old default PyTorch policy, and
  `DogfightRecurrent` wraps it with `pufferlib.models.LSTMWrapper` using
  `input_size = hidden_size` and `hidden_size = hidden_size`.
- The df36 default policy path was `Linear(obs -> hidden) + GELU`, continuous
  `Normal(mean, exp(logstd))`, value head, and a real PyTorch LSTM wrapper for
  recurrence. The wrapper reinitialized the wrapped policy weights
  orthogonally with gain `1.0` and zeroed biases before constructing the LSTM.
  The LSTM weights themselves came from PyTorch's LSTM defaults.
- Current normal Dogfight5 training resolves to the native C/CUDA backend unless
  `--slowly` is passed. In native `src/pufferlib.cu`, `build_policy` hard-wires
  the recurrent `Network` to `mingru_forward`, `mingru_forward_train`, and
  `mingru_backward`; there is no native config switch to use the PyTorch LSTM
  wrapper. The PyTorch backend has `torch.network = MinGRU` by default too,
  although it can instantiate `pufferlib.models.LSTM` if explicitly configured.
- Current Dogfight5 native initialization also differs from df36. The Dogfight
  custom encoder, native decoder, and native MinGRU all call `puf_kaiming_init`
  (`Uniform(-gain/sqrt(fan_in), gain/sqrt(fan_in))`). That is not the same as
  df36's PyTorch orthogonal initialization of the wrapped policy layers, even
  when the numeric `action_init_scale` and `value_init_scale` are set to `1.0`.
- Current native training also uses the CUDA `muon_step` implementation, while
  df36 used the Python trainer's optimizer path (`optimizer = muon`, implemented
  through HeavyBall/ForeachMuon in the exact df36 code). The hyperparameters can
  share names and values while still producing different optimizer dynamics.
- Native recurrent-state training mismatch found on 2026-06-04: before the
  fix, PufferLib 5 native rollout eval maintained per-buffer recurrent state,
  and native `policy_forward_train` accepted a minibatch state tensor, but
  `RolloutBuf` had no state tensor and `select_copy` never populated
  `graph.mb_state`. That meant training segments used zero or stale recurrent
  state rather than the rollout state at the start of the sampled horizon row.
  Exact df36's LSTMWrapper path passed an LSTM state dict into training forward,
  and the PufferLib 5 native API clearly intended `mb_state` to carry this
  information. This is a trainer-plumbing fix, not a Dogfight curriculum or
  physics change.
- Native recurrent-state fix: `src/pufferlib.cu` now stores a layer-major
  `RolloutBuf.states` tensor shaped `(num_layers, agents, hidden_size)`,
  captures each bank's initial recurrent state at rollout timestep zero before
  `policy_forward` mutates the persistent buffer state, copies that state tensor
  into the transposed training rollout buffer, and has `select_copy` copy the
  selected agent's state row into `graph.mb_state` before
  `policy_forward_train`.
- Interpretation: the environment/physics/reward path now has substantial df36
  parity coverage, but the learner is still not behavior-equivalent to df36.
  The post-adapter smoke's near-zero `kl`/`clipfrac` and rising control
  saturation should be investigated as native policy/update/optimizer behavior
  before changing Dogfight curriculum or physics. A short PyTorch-backend LSTM
  diagnostic could isolate env behavior from native MinGRU/Muon behavior, but it
  should be treated as a diagnostic only, not as the normal sweep path.
- Existing 5.0 config/backend options checked: `pufferlib.pufferl` exposes
  `--slowly` to choose the PyTorch backend, and `pufferlib/torch_pufferl.py`
  instantiates `args['torch']['network']`, `args['torch']['encoder']`, and
  `args['torch']['decoder']`. Since `pufferlib.models` has `LSTM`, a diagnostic
  command can request `--slowly --torch.network LSTM` without editing core. The
  normal native backend ignores that torch-network setting and still builds the
  CUDA `Network` as MinGRU in `src/pufferlib.cu`.
- Current/reference 5.0-style configs checked in this clone, `/home/claude/PufferLib`,
  and `/home/claude/dogfight4` do not show an environment using `network = LSTM`
  as its normal path. Defaults are MinGRU; `double_pendulum.ini` uses `MLP` for
  the PyTorch network setting. That means a Dogfight LSTM diagnostic would be a
  targeted comparison against df36, not a common 5.0 env precedent.
- Current Dogfight5 has no Dogfight-local PyTorch policy file equivalent to the
  df36 `DogfightPolicy`/`DogfightRecurrent` names. The current Python backend
  path would use generic `DefaultEncoder + DefaultDecoder + LSTM` unless a
  Dogfight-local PyTorch policy/encoder wrapper is added later.
- Added `learner_compat.py` and `test_learner_compat.py` as data-only/source
  coverage for this distinction. The tests confirm exact df36 used
  `DogfightPolicy` + `DogfightRecurrent`/`LSTMWrapper`, current normal native
  Dogfight5 builds CUDA MinGRU, and the only existing LSTM route is the
  `--slowly --torch.network LSTM` diagnostic path.
- Added a Dogfight-local PyTorch diagnostic encoder in `ocean/dogfight/torch.py`
  and a small `pufferlib/torch_pufferl.py` resolver hook so the PyTorch backend
  can resolve `DogfightEncoder` from `ocean.dogfight.torch` before falling back
  to generic `pufferlib.models`. `config/dogfight.ini` now overrides only
  `[torch] encoder = DogfightEncoder`; default `network = MinGRU` and
  `decoder = DefaultDecoder` still come from `config/default.ini` unless a
  diagnostic command overrides them. This does not change the normal native
  backend, which still builds CUDA MinGRU.
- `DogfightEncoder` matches the df36 observation encoder shape for the diagnostic
  path: `Linear(obs -> hidden) + GELU` with orthogonal weight initialization and
  zero bias. A closer df36-style diagnostic run can now use `--slowly
  --torch.network LSTM` without also losing the Dogfight3 encoder activation.
- Added `DogfightDecoder` for the same diagnostic path. This avoids another
  df36 mismatch in current `pufferlib.models.DefaultDecoder`: in PufferLib 5,
  `action_init_scale = 1.0` and `value_init_scale = 1.0` leave PyTorch default
  decoder initialization in place, while df36's `LSTMWrapper` explicitly
  orthogonalized wrapped policy weights with gain `1.0` and zeroed biases.
  `DogfightDecoder` always orthogonalizes the continuous action mean and value
  layers with the configured scales, zeroes biases, and clamps continuous
  `logstd` to the df36 `[-20, 2]` range. `config/dogfight.ini` now overrides
  only the PyTorch diagnostic decoder with `[torch] decoder =
  DogfightDecoder`; native training still ignores `[torch]` and still builds
  the CUDA policy stack.
- Diagnostic result on 2026-06-04: after rebuilding Dogfight with `./build.sh
  dogfight --float`, the bounded command `timeout 180s bash -lc 'source
  .venv/bin/activate; python -m pufferlib.pufferl train dogfight --slowly
  --torch.network LSTM'` launched under the local venv with GPU active. It hit
  the wrapper timeout with exit code `124`, not a trainer crash. The final
  visible TUI row was around `60.0M` steps, `136.3K` params, `GPU = 94%`,
  `VRAM = 3.0/8G`, `SPS = 277.5K`, `stage = 0.000`, `avg_stage = 0.000`,
  `base_stage_kills = 0.236`, `base_stage_ground = 0.181`,
  `base_stage_timeouts = 0.583`, aggregate base-stage action saturation around
  `0.310`, and `approx_kl`/`clipfrac` still `0.000`.
- Interpretation of the diagnostic: swapping from native MinGRU to the current
  PufferLib 5 PyTorch `LSTM` path is not enough by itself to recreate df36
  early-stage learning. The diagnostic is also not exact df36: current 5.0
  composes `Policy(encoder, decoder, network)`, uses `pufferlib.models.LSTM`
  as the recurrent block, and trains through the current `torch_pufferl.py`
  update path. Exact df36 used `LSTMWrapper(policy)` around the whole old
  `DogfightPolicy` class and the older Python trainer/optimizer path. This
  lowers the chance that native MinGRU alone is the blocker, but it does not
  clear current learner/update dynamics.
- Current dirty config note: `config/dogfight.ini` has
  `policy.action_init_scale = 1.0` and `value_init_scale = 1.0`. Older status
  text that says the plain Dogfight5 baseline still uses `action_init_scale =
  0.01` is stale for this working tree. After the diagnostic decoder update,
  the PyTorch `--slowly --torch.network LSTM` comparison no longer loses df36's
  scale-one orthogonal action/value init through `DefaultDecoder`.

## Next Tests

- Run a short df41 sweep from the stopped-df40-centered search space. The
  useful signal is whether it can reproduce target `9.9-10.9` quickly and then
  push toward target `17-18` without high base-stage ground rate or severe
  surface-control saturation.
- If df41 still favors `64`/`128`, run a controlled short follow-up around the
  best stage-climb hypers with only hidden size varied (`64`, `128`, `256`)
  before treating hidden size as settled.
- The short `max-runs 2` state-buffer sweep smoke from the recentered space now
  launches and exits cleanly with GPU. The remaining risk is training quality:
  the sampled state-curriculum run saturated controls badly and did not promote.
- The next df41 W&B sweep should use the tightened action-init lower bound
  (`0.2`) so the search spends less time on no-shot low-init samples.
- The post-tightening smoke did not promote and used `horizon = 128`, but do
  not eliminate `128` solely from this one run. Older local logs include two
  target `9.9` runs at `horizon = 128`; the stronger current evidence is still
  just that target `10.9` has only appeared at `horizon = 64`.
- The normal native training smoke after the global-step adapter has now crossed
  `100M`/`200M` with GPU active but without promotion. Next work should inspect
  why `kl`/`clipfrac` remain near zero and why control saturation rises as the
  policy approaches or passes the shaping-decay window.
- After the recurrent-state minibatch fix, the normal native smoke again crossed
  `100M` with GPU active and no crash, but still did not promote. Next work
  should continue on learner/update parity, especially why `kl` and `clipfrac`
  remain effectively zero even when recurrent state is now plumbed into native
  training.
- Audit native learner differences before more Dogfight environment changes:
  MinGRU vs df36 LSTMWrapper, native Kaiming-style init vs df36 orthogonal
  policy-layer init, and CUDA Muon vs df36 HeavyBall/ForeachMuon. Do this as
  measurement/diagnostics first; do not change PufferLib core until Dogfight-
  local and config-level options are exhausted.
- If running the learner diagnostic, keep it clearly separate from the normal
  sweep path: use local venv activation, GPU enabled, `--slowly --torch.network
  LSTM`, no `--cpu`, and no tiny timestep override. The useful comparison is
  whether the same Dogfight env climbs better under a closer df36-style PyTorch
  LSTM learner than under native MinGRU, not whether `--slowly` is fast enough
  for production sweeps.
- PyTorch-backend diagnostics require the extension to be built with
  `./build.sh dogfight --float`. After the LSTM diagnostic and diagnostic
  decoder update, the local binary was rebuilt with plain `./build.sh dogfight`
  so normal native sweeps are back on the default Dogfight build.
- Add additional parity tests against Dogfight3 for scheme 1 observations under broader scripted states.
- Add longer multi-reset parity traces against Dogfight3.
- Add additional physics/step parity probes for neutral traces and broader scripted trace coverage.
- Audit `flightlib.h` differences before changing physics.

## Verification

- `git diff --check`: clean.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py ocean/dogfight/tests/test_curriculum_progress.py -q`: `12 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `60 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_port_build.py::test_dogfight_gpu_vec_creates_and_resets -q`: skipped in this environment.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py ocean/dogfight/tests/test_native_policy_architecture.py ocean/dogfight/tests/test_port_build.py::test_dogfight_training_backend_builds -q`: `9 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `61 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_policy_init.py -q`: `7 passed`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k state_roundtrip`: failed before the env-local RNG fix because restored terminal futures diverged in observations/plane state, then passed after the fix.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `6 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `61 passed, 1 skipped`.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so`.
- `python -m pufferlib.pufferl train dogfight --train.gpus 1 --train.total-timesteps 5000000`: native trainer completed, TUI showed GPU usage and normal Dogfight curriculum metrics.
- Red TDD check for scheme1 restore failed before implementation because native binding was `OBS_SIZE 22` and config was `obs_scheme = 0`; after the Dogfight-local fix, the same narrow check passed with `2 passed`.
- `python -m pytest ocean/dogfight/tests/test_port_config.py ocean/dogfight/tests/test_native_policy_architecture.py ocean/dogfight/tests/test_policy_init.py -q`: `19 passed`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `6 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `61 passed, 1 skipped`.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so` after native `OBS_SIZE 26` was restored.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k observation_scheme1_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `7 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `62 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k scripted_step_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `8 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `63 passed, 1 skipped`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k scripted_trace_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `13 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `68 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k terminal_kill_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `9 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `64 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k terminal_opponent_kill_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `12 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `67 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k terminal_oob_reference`: first failed because the new parity test file was missing, then passed after adding the fixture test.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `10 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `65 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_obs_scheme_compat.py ocean/dogfight/tests/test_port_config.py ocean/dogfight/tests/test_native_policy_architecture.py -q`: `14 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `70 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_continuous_action_contract.py -q`: first failed with `2 failed` because `action_contract.py` was missing, then passed after adding the Dogfight-local contract helper.
- `python -m pytest ocean/dogfight/tests/test_continuous_action_contract.py -q`: `2 passed`.
- `python -m pytest ocean/dogfight/tests/test_continuous_action_contract.py ocean/dogfight/tests/test_obs_scheme_compat.py ocean/dogfight/tests/test_port_config.py -q`: `14 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `72 passed, 1 skipped, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_global_step_adapter.py -q`: first failed because Dogfight native had no `MY_GLOBAL_STEP` hook, then passed after adding the guarded adapter.
- `source .venv/bin/activate && ./build.sh dogfight`: built `pufferlib/_C.cpython-312-x86_64-linux-gnu.so` after the global-step adapter.
- `python -m pytest ocean/dogfight/tests/test_global_step_adapter.py ocean/dogfight/tests/test_port_build.py::test_dogfight_training_backend_builds -q`: `2 passed`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q`: `13 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `73 passed, 1 skipped, 2 warnings`.
- `timeout 180s bash -lc 'source .venv/bin/activate; python -m pufferlib.pufferl train dogfight'`: exited `124` from the timeout wrapper after reaching at least `215.2M` steps. GPU stayed active, VRAM was about `1.6/8G`, and the final visible metrics still showed no promotion (`stage ~= 0.56`, `avg_stage ~= 0.90`).
- `python -m pytest ocean/dogfight/tests/test_learner_compat.py -q`: first failed with `3 failed` because `learner_compat.py` was missing, then passed after adding the data-only compatibility helper.
- `python -m pytest ocean/dogfight/tests/test_learner_compat.py -q`: `3 passed`.
- `python -m pytest ocean/dogfight/tests/test_native_policy_architecture.py ocean/dogfight/tests/test_policy_init.py ocean/dogfight/tests/test_port_config.py -q`: `19 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests/test_torch_diagnostic_policy.py -q`: first failed because Dogfight had no `[torch] encoder` override and `torch_pufferl.py` had no env-local component resolver, then passed after adding `DogfightEncoder` and the resolver hook.
- `python -m pytest ocean/dogfight/tests/test_torch_diagnostic_policy.py ocean/dogfight/tests/test_learner_compat.py ocean/dogfight/tests/test_native_policy_architecture.py ocean/dogfight/tests/test_policy_init.py ocean/dogfight/tests/test_port_config.py -q`: `24 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `78 passed, 1 skipped, 2 warnings`.
- `source .venv/bin/activate && ./build.sh dogfight --float`: built
  `pufferlib/_C.cpython-312-x86_64-linux-gnu.so` for the PyTorch backend
  diagnostic.
- `timeout 180s bash -lc 'source .venv/bin/activate; python -m
  pufferlib.pufferl train dogfight --slowly --torch.network LSTM'`: exited
  `124` from the timeout wrapper after reaching a final visible row around
  `60.0M` steps. GPU was active, VRAM was about `3.0/8G`, and the visible
  curriculum metrics still showed no promotion (`stage = 0.000`, `avg_stage =
  0.000`).
- `python -m pytest ocean/dogfight/tests/test_torch_diagnostic_policy.py -q`:
  first failed with missing `[torch] decoder` and missing `DogfightDecoder`,
  then passed after adding the Dogfight-local diagnostic decoder.
- `python -m pytest ocean/dogfight/tests/test_torch_diagnostic_policy.py
  ocean/dogfight/tests/test_learner_compat.py
  ocean/dogfight/tests/test_policy_init.py
  ocean/dogfight/tests/test_port_config.py
  ocean/dogfight/tests/test_native_policy_architecture.py -q`: `25 passed, 2
  warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `79 passed, 1 skipped, 2
  warnings`.
- `source .venv/bin/activate && ./build.sh dogfight`: rebuilt the default
  native Dogfight backend after the float-only PyTorch diagnostic.
- `python -m pytest
  ocean/dogfight/tests/test_port_build.py::test_dogfight_training_backend_builds
  -q`: `1 passed` against the rebuilt default native backend.
- `python -m pytest ocean/dogfight/tests/test_native_recurrent_state_contract.py
  -q`: first failed because native rollouts had no recurrent-state tensor, then
  passed after adding native rollout-state capture and minibatch state copy.
- `source .venv/bin/activate && ./build.sh dogfight`: rebuilt the default
  native Dogfight backend after the recurrent-state minibatch fix.
- `python -m pytest ocean/dogfight/tests/test_native_recurrent_state_contract.py
  ocean/dogfight/tests/test_port_build.py::test_dogfight_training_backend_builds
  -q`: `2 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `80 passed, 1 skipped, 2
  warnings`.
- `timeout 90s bash -lc 'source .venv/bin/activate && python -m
  pufferlib.pufferl train dogfight'`: sandboxed run failed because CUDA was not
  visible, then the same command was rerun unsandboxed. The unsandboxed run used
  GPU, stayed around `1.6/8G` VRAM, reached a final visible row around `153.4M`
  steps, and exited `124` from the shell timeout rather than a trainer crash.
  Final visible curriculum stats were still not acceptable: around `stage =
  0.657`, `avg_stage = 0.893`, `base_stage_kills = 0.264`,
  `base_stage_timeouts = 0.629`, aggregate base-stage action saturation around
  `0.359`, and `kl`/`clipfrac = 0.000`.
- `python -m pytest ocean/dogfight/tests/test_stage_spawn_source_compat.py -q`:
  `2 passed`. This guards exact df36 compatibility for the full `STAGES` table
  and stage 10-17 spawner source after normalizing only the intentional
  `rndf(...)` -> `dogfight_rndf(env, ...)` helper rename.
- `python -m pytest ocean/dogfight/tests/test_stage_spawn_source_compat.py
  ocean/dogfight/tests/test_c_regressions.py::test_dogfight_c_regression -q -k
  'stage_spawn_source_compat or curriculum_stage_geometry'`: `3 passed, 12
  deselected`.
- `python -m pytest
  ocean/dogfight/tests/test_port_config.py::test_dogfight_sweep_space_centers_native_short_stage_climb_profile
  ocean/dogfight/tests/test_port_config.py::test_dogfight_sweep_can_search_state_curriculum_knobs
  ocean/dogfight/tests/test_port_config.py::test_dogfight_state_curriculum_sweep_space_roundtrips_through_protein
  ocean/dogfight/tests/test_sweep_metrics.py::test_dogfight_sweep_hyperparameters_honor_configured_means
  ocean/dogfight/tests/test_sweep_metrics.py::test_protein_first_suggestion_uses_configured_search_center
  ocean/dogfight/tests/test_sweep_metrics.py::test_dogfight_sweep_samples_discrete_topology_as_integers
  -q`: first failed against the old sweep center, then passed after recentering
  the Dogfight-local sweep space (`6 passed, 2 warnings`).
- `python -m pytest ocean/dogfight/tests/test_port_config.py
  ocean/dogfight/tests/test_sweep_metrics.py -q`: `24 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `83 passed, 1 skipped,
  2 warnings`.
- `timeout 300s bash -lc 'source .venv/bin/activate && python -m
  pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1
  --sweep.max-runs 2'`: completed with exit code `0` when run unsandboxed for
  GPU access. It produced local logs `logs/dogfight/1780614612769.json`
  (baseline short run) and `logs/dogfight/1780614644488.json` (sampled
  state-curriculum run). GPU was active; no `--cpu` or tiny timestep override
  was used.
- `python -m pytest ocean/dogfight/tests/test_policy_init.py -q`: first failed
  after adding the low-action-init guard because the config still allowed
  `policy.action_init_scale = 0.003`; passed after raising the Dogfight sweep
  lower bound to `0.2` (`8 passed`).
- `python -m pytest ocean/dogfight/tests/test_port_config.py
  ocean/dogfight/tests/test_sweep_metrics.py -q`: `24 passed, 2 warnings`.
- Post-tightening smoke with the same command
  `timeout 300s bash -lc 'source .venv/bin/activate && python -m
  pufferlib.pufferl sweep dogfight --sweep.gpus 1 --train.gpus 1
  --sweep.max-runs 2'`: completed with exit code `0`. The relevant local logs
  are `logs/dogfight/1780615251554.json` (baseline short run) and
  `logs/dogfight/1780615288647.json` (sampled state-curriculum run). GPU was
  active; no `--cpu` or tiny timestep override was used.
- `python -m pytest
  ocean/dogfight/tests/test_port_config.py::test_dogfight_sweep_learning_rate_keeps_recent_clean_stage8_climber_reachable
  -q`: first failed because `sweep.train.learning_rate.max = 0.0065` excluded
  the clean df40 target-8.9 climber `h5qvaso2`; passed after widening the
  Dogfight sweep LR max to `0.01`.
- `python -m pytest
  ocean/dogfight/tests/test_port_config.py::test_dogfight_sweep_learning_rate_keeps_recent_clean_stage8_climber_reachable
  ocean/dogfight/tests/test_port_config.py::test_dogfight_sweep_space_centers_native_short_stage_climb_profile
  ocean/dogfight/tests/test_sweep_metrics.py::test_dogfight_sweep_hyperparameters_honor_configured_means
  ocean/dogfight/tests/test_sweep_metrics.py::test_protein_first_suggestion_uses_configured_search_center
  -q`: `4 passed, 2 warnings`.
- `python -m pytest ocean/dogfight/tests -q`: `84 passed, 1 skipped, 2
  warnings`.
- `python -m pytest ocean/dogfight/tests/test_flightlib_source_compat.py -q`:
  `1 passed`. This guards full `flightlib.h` parity against exact df36 after
  normalizing only the `K` macro rename and the env-local state-RNG helper.
- `python -m pytest ocean/dogfight/tests -q`: `85 passed, 1 skipped, 2
  warnings`.
- `python -m pytest ocean/dogfight/tests/test_stage_spawn_source_compat.py -q`:
  first failed with the stricter side-spawn comparison because Dogfight5 exposes
  df36's hardcoded side-energy probability and stage-9 bank through
  config/telemetry fields; after making only those allowances explicit, it
  passed with `4 passed`.
- `python -m pytest ocean/dogfight/tests -q`: `87 passed, 1 skipped, 2
  warnings`.
- `python -m pytest ocean/dogfight/tests/test_c_regressions.py -q -k
  reward_shaping_decay_reference`: `1 passed, 13 deselected`.
- `python -m pytest ocean/dogfight/tests -q`: `88 passed, 1 skipped, 2
  warnings`.
