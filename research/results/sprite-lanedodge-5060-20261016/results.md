# Sprite Lane Dodge — 5060 state vs pixel comparison (2026-10-16)

Environment: `ocean/sprite_lanedodge` (new, rules `sprite-lanedodge-v1`).
20x15 grid (640x480, 32px cells), 4 lanes (x=7/9/11/13), player row 13.
Hazards/coins spawn at row 0 in a random lane (hazard_rate / coin_rate per-tick
percentages) and advance one row per tick; collision at row 13.
3 actions (wait/left/right). 16 ticks/sec, max_steps=400.
Perf (fixed per episode, reward-independent): `0.5*survival + 0.5*coin_frac`,
survival = tick/max_steps on death, 1.0 on clean timeout.
State obs: 9 floats (lane/3, nearest hazard distance and nearest coin distance
per lane, each in [0,1]). Pixel obs: full-res 640x480 RGB (921,600 floats),
Nature CNN (SPRITE_CNN_RESEARCH encoder).

Hardware: RTX 5060 8GB (WSL G240), CUDA 12.8, sm_120, float32.
Binary: `build/sprite_lanedodge/puffer` (built from `src/` + `ocean/sprite_lanedodge/`;
source hashes in `build/lanedodge-tests-F1/source.sha256`).
Seed: fixed 7 (policy/learner). Eval worlds: 2027 / 9001, 64 episodes each,
`--headless --base.eval_agents=1 --train.horizon=1 --train.minibatch_size=1 --sweep.metric=perf`.

**Reading order.** The current canonical recipe is the fixed-code h1c40
allocation described in "## Fixed-code retrain" below; the sections between
this note and that one (h20c40 state/pixel selections, the pre-fix rate
trail, the WSL dentry note) are pre-fix history, preserved for provenance.
2027 and 9001 are development worlds (budget and rate selection), not
untouched holdouts. Difficulty freeze: the current INI (h1c40, max_steps
400) is the v1 benchmark recipe; later confirmation needs new evaluation
worlds and the predetermined independent training replicas.

Canonical task rates (tuned to reach the state target; all runs preserved):
hazard_rate=20, coin_rate=40, max_steps=400. Tuning trail (state, seed 7, 84M):
hazard45/coin25 (original INI default): 0.820923 / 0.793350 on 2027/9001.
hazard30/coin25: 0.868923 / 0.850152.
hazard30/coin40: 0.864800 / 0.890751.
hazard20/coin40: 0.969907 / 0.962675  (selected).

## State run (selected; pre-fix h20c40, historical)

- run_id `ld-state-h20c40-84M-s7`, observation_mode=0, A256/H64/MB2048 (H128-L1 core).
- `./build/sprite_lanedodge/puffer train --base.run_id=ld-state-h20c40-84M-s7 --base.seed=7 --env.observation_mode=0 --vec.total_agents=256 --vec.num_buffers=1 --vec.num_threads=4 --train.horizon=64 --train.minibatch_size=2048 --train.total_timesteps=83886080 --base.checkpoint_interval=64 --base.eval_episodes=0`
- 83,886,080 decisions in 83.037s wall (monotonic) => 1,010,224 process SPS.
- Trajectory on world 2027 (64-episode eval): 8.4M=0.888439, 24.1M=0.873810,
  56.6M=0.971424, 84M=0.969907 (final selected).
- Final checkpoint `checkpoints/sprite_lanedodge/ld-state-h20c40-84M-s7/0000000083886080.bin` (203,264 bytes).
- Eval 2027: perf 0.969907. Eval 9001: perf 0.962675. Mean 0.966291.

State scaling notes (same seed, serial, all preserved):
- 16.7M (INI default, original rates): tail perf ~0.67.
- 84M original rates: 84.444s, tail ~0.82; holdout 0.820923/0.793350.
- 280M original rates: 277.974s (~1.0M SPS); holdout 0.769275/0.672179 — longer
  same-world training overfit the training world; 84M was the better point.

## Pixel run (selected point; pre-fix h20c40, historical)

- run_id `ld-pixel-300s-s7` (+ deterministic reruns), observation_mode=1, A4/H64/MB64
  (H128-L1 core, Nature CNN via `SPRITE_CNN_RESEARCH` encoder, 34,990,880 params).
  A4 chosen from OOM mapping on the 8GB card with full-res 921,600-float obs:
  A8/H64/MB64 ~8.4GB > 7.4GB free (OOM in probe), A4/H64/MB64 ~6.4GB fits.
- `./build/sprite_lanedodge/puffer train --base.run_id=ld-pixel-300s-s7 --base.seed=7 --env.observation_mode=1 --vec.total_agents=4 --vec.num_buffers=1 --vec.num_threads=2 --train.horizon=64 --train.minibatch_size=64 --base.checkpoint_interval=55 --base.eval_episodes=0` under `timeout 300` (SIGTERM at the budget; epoch-completed checkpoints retained).
- LR anneals over the INI default `total_timesteps=16777216` (16.7M); the 300s
  allocation is a timeout on that budget, not a budget change.
- Checkpoint interval 55 epochs (256 decisions/epoch => 14,080 decisions/checkpoint, ~30s apart).
- Timestamp-only selection: last completed checkpoint `000000000140800.bin`
  (140,800 decisions). Selection never uses score.
- Process SPS: 140,800 decisions / 300.104s (canonical rerun r9 wall) = 469.2 SPS.
  Dashboard SPS windowed 386.7-479.0.
- Eval 2027: perf 0.612785. Eval 9001: perf 0.572703. Mean 0.592744 (target ~50%).
- Weights receipt: sha256 `559ec81977d5c79c1c4b3b65418d9b9ca3f0839c2c4f8618c78df4a176d379bb`
  for `000000000140800.bin`, byte-identical across the old binary (runs 2/3) and
  the mirror-writing rebuild (run r9) => cross-build deterministic.

## State repeat check (determinism)

- `ld-state-h20c40-84M-s7-repeat`: same recipe/seed on the mirror-writing rebuild.
- All 80 checkpoints byte-identical to `ld-state-h20c40-84M-s7` (0 diffs), wall
  82.803s => 1,013,080 process SPS.

## WSL dentry flap and the mirror fix

The 140MB pixel checkpoint entries under `checkpoints/sprite_lanedodge/` became
intermittently unresolvable (readdir lists them with correct metadata, but
stat/open return ENOENT for minutes at a time; 200KB state checkpoints and
freshly written files are stable). Deterministic same-seed reruns re-emit
identical weights, and a single env-gated mirror write in `puf_save_weights`
(`src/pufferl.cu`, opt-in `PUFFER_WEIGHTS_MIRROR=PATH`, no default-behavior
change) copies each checkpoint from the in-memory buffer to a fresh path while
its dentry is hot; evals then run from the mirror. The canonical receipt
`build/ld-pixel-selected-v2/000000000140800.bin` carries sha256 above.
## Commands (cwd /home/claude/cnn; `source ocean/connect4cnn/runtime_env.sh` first)

- State train (pre-fix; task rates hazard=20/coin=40 were the INI defaults at
  the time — the current defaults are h1c40, see the fixed-code section):
  the command quoted in the State run section; repeat uses
  `--base.run_id=ld-state-h20c40-84M-s7-repeat`.
- Pixel train (canonical 300s allocation): the command quoted in the Pixel
  section under `timeout 300`; run r9 additionally set
  `PUFFER_WEIGHTS_MIRROR=build/ld-pixel-selected-v2/000000000140800.bin`.
- Eval (both modes): `./build/sprite_lanedodge/puffer eval WEIGHTS --headless --base.seed=7 --base.eval_agents=1 --base.eval_episodes=64 --vec.num_buffers=1 --vec.num_threads=1 --train.horizon=1 --train.minibatch_size=1 --env.seed=2027|9001 --env.observation_mode=0|1 --sweep.metric=perf`
  -> machine line `CUDA_EVAL env=sprite_lanedodge score=<perf> perf=<perf> games=64 params=<n>`.
- Host tests: `bash ocean/sprite_lanedodge/tests/run.sh build/lanedodge-tests-F4`
  (ASan+UBSan, 16 sections, exit 0; fresh on the final tree).
- Resolved INIs: `inis/` (state runs and pixel r7/r8; the timeout-terminated
  pixel runs 1-6/r9 never reached the log write at pufferl.cu:3393, so their
  resolved config is the INI defaults plus the quoted CLI flags).
- Source provenance: `source.sha256-pre-mirror` (captured 00:58, before the
  00:59 `src/ocean.cu` sprite-block sync, and before the opt-in
  `PUFFER_WEIGHTS_MIRROR` edit to `src/pufferl.cu`), `source.sha256-final`
  (current tree). Cross-build determinism: state 80/80 checkpoints and the
  pixel 140,800-ckpt sha256 identical between binaries.

## Fixed-code retrain (2026-10-16, reviewer findings applied)

The reviewer log's 01:06–01:08 entry left three findings; all are fixed in the
retained source (see `research/LANE_DODGE_REVIEW_LOG.md`, 03:45 entry):

1. **Coin consume-once.** The arrival branch now clears the entity on both
   rows (previously the source row 12 was left set, so a coin was
   re-collected every tick while the player waited — a reward pump that
   inflated `coin_frac` past 1, clamped, and could make `coins` exceed
   `coins_spawned`). All pre-fix numbers above (including the 0.97/0.96
   state holdouts) are historical artifacts of that pump; they are preserved,
   not pooled, with fixed-code numbers.
2. **Textures/composition.** Ground fills carry a deterministic
   per-art-pixel shade (keyed to the art column so the rep-2 mirror test
   holds; no RNG or game-state consumption), and entity art composites over
   the shaded road instead of the grass default.
3. **run.sh evidence.** The runner writes `exit-status.txt`, exits nonzero on
   failure, and hashes the test source and itself into `source.sha256`.

A negative control confirms the regression test is live: with the consume
fix temporarily reverted the suite aborts at the new assertion
(`Assertion '!c.terminal && c.env.coins == 1' failed`); with the fix, the
suite passes (ASan+UBSan, 16 sections).

### Rate re-tuning on the fixed game (state, 84M, seed 7, matched-rate eval)

The pre-fix trail was tuned under the pump metric. On the fixed game the
matched-rate points (2027 / 9001) are:

| rates | 2027 | 9001 |
|---|---:|---:|
| h20c40 (old default) | 0.566444 | 0.520280 |
| h10c40 | 0.620085 | 0.636902 |
| h5c40 | 0.748790 | 0.760344 |
| h3c40 | 0.789218 | 0.755992 |
| h2c40 | 0.800054 | 0.791936 |
| h1c40 (selected) | 0.823194 | 0.836414 |
| h0c40 (ceiling, no hazards in play) | 0.875549 | 0.878228 |
| h10c50 | 0.657469 | 0.646509 |
| h5c60 | 0.702065 | 0.686079 |
| h15c50 | 0.557065 | 0.564603 |
| h1c60 | 0.763853 | 0.756665 |

The pre-fix trail (h45c25/h30c25/h30c40/h20c40) is also preserved above and
remains a pump-metric artifact. Selection: `hazard_rate=1, coin_rate=40` is
now the INI default — the closest-to-target rate that keeps hazards in play
(~4 per episode). `h0c40` reaches 0.88 but degenerates the task to pure coin
collection and is recorded as the ceiling, not the canonical task. The ~90%
state target is approached (0.82/0.84) but not reached at the canonical
rates under the corrected coin semantics; pre-fix 0.97 was not a fair
comparison.

### State canonical on fixed code (h1c40, 84M, seed 7)

- run_id `ld-state-h1c40-84M-s7` (INI defaults h1c40, A256/H64/MB2048,
  83,886,080 decisions): wall 83.325s => 1,006,745 process SPS.
- Repeat `ld-state-h1c40-84M-s7-repeat`: wall 82.907s => 1,011,809 process
  SPS; all 80 checkpoints byte-identical (0 `cmp` diffs). A probe-run
  cross-check (`ld-probe3-h1c40-84M-s7`) also matches 80/80.
- Final checkpoint `checkpoints/sprite_lanedodge/ld-state-h1c40-84M-s7/
  0000000083886080.bin` (203,264 bytes), sha256
  `221721ac0929301c74e6537e7ecd3772d8ce735f2eefdfbf3080957e30550059`.
- Eval 2027: perf 0.823194. Eval 9001: perf 0.836414. Mean 0.829804.
- Eval determinism: world-2027 eval run twice, identical `CUDA_EVAL` line
  both times.

### Pixel on fixed code (h1c40, 300s allocation, seed 7)

- run_id `ld-pixel-h1c40-300s-s7`, observation_mode=1, A4/H64/MB64, Nature
  CNN (34,990,880 params), INI-default h1c40 rates, `timeout 300` on the
  INI default `total_timesteps=16777216` (LR anneals over 16.7M, not over
  the truncated allocation), checkpoint interval 55 epochs.
- `PUFFER_WEIGHTS_MIRROR=build/ld-pixel-h1c40/selected.bin`
  `timeout 300 ./build/sprite_lanedodge/puffer train --base.run_id=ld-pixel-h1c40-300s-s7
  --base.seed=7 --env.observation_mode=1 --vec.total_agents=4
  --vec.num_buffers=1 --vec.num_threads=2 --train.horizon=64
  --train.minibatch_size=64 --base.checkpoint_interval=55
  --base.eval_episodes=0`
- Wall 300.092s; last completed checkpoint `0000000000140800.bin`
  (140,800 decisions; timestamp-only selection, no score peeking) =>
  140,800 / 300.092 = 469.2 process SPS.
- Mirror sha256 `48adf1325214f7c89c032f434e8c9812422bfefb2bebb7d0ab728857dad07bdd`
  equals the run checkpoint sha256 (fresh-path receipt, dentry-flap safe).
- Eval 2027: perf 0.407720. Eval 9001: perf 0.427634. Mean 0.417677.

### Fixed-code state-vs-pixel summary (canonical h1c40, seed 7)

| | state | pixel |
|---|---:|---:|
| decisions | 83,886,080 | 140,800 (300s allocation) |
| wall | 83.325s | 300.092s |
| process SPS | 1,006,745 | 469.2 |
| eval 2027 | 0.823194 | 0.407720 |
| eval 9001 | 0.836414 | 0.427634 |
| mean | 0.829804 | 0.417677 |

State is ~2,145x faster in process SPS and reaches 0.83/0.84 vs 0.41/0.43
perf at the canonical rates under the corrected consume-once semantics.
Targets: state ~90% is approached (0.83; the h0c40 ceiling is 0.88 but
removes the dodge element), pixel ~50% is approached from below (0.42).
Pre-fix pixel numbers (0.612785/0.572703 at h20c40) are pump-metric
artifacts and are not comparable with these.

## Fixed-code provenance

- Header `sprite_lanedodge.h` `03760741…` (consume-once + shaded/composited
  tiles), `sprites.h` `9f489e38…`, tests `test_environment.c`
  `08d0f4b4…`, `run.sh` `a0c88f12…`, `config/sprite_lanedodge.ini` now h1c40
  (sha256 `57ea5d84…`; the pre-fix h20c40 INI was `144b57b9…`). The 11-file
  `source.sha256-final` was re-captured from `build/lanedodge-tests-F4`
  after the INI change.
- Host suite: `build/lanedodge-tests-F2` / `F3` pass (ASan+UBSan, 16
  sections); negative control `build/lanedodge-tests-nofix` fails exactly at
  the new assertion when the consume fix is reverted.
- Native binary rebuilt from the fixed sources
  (`build/sprite_lanedodge/puffer`); state and pixel runs above used it.
- Reviewer log: `research/LANE_DODGE_REVIEW_LOG.md` (entries 00:35, 00:38,
  00:49–00:51, 01:06–01:08, 03:45 fix entry, 04:08 h1c40 retrain entry).
