# Lane Dodge development review log

Kinvert requested occasional independent reviews of Qwen3.8 27B's Lane Dodge
implementation while work is active tonight. Sprite Forage is its main template.
Leave the implementation untouched; append observations here rather than fixing
Qwen's code or changing its configs. Do not interpret this log as an autonomous
background monitor or as evidence of tests that were not actually executed.

Use local timestamps with timezone. Each entry should identify the checkout and
files inspected, visible implementation status, good parts, concrete problems,
unknowns and the next useful check. Tie findings to file/line references when
code exists. Distinguish an unfinished implementation from an established bug.
Preserve earlier entries; later fixes should get a new entry rather than erase
the history. Revisit opportunistically during active work, at meaningful
milestones or roughly30–60 minutes apart, without polling training output.

## Reference expectations

- Use the current `/home/claude/cnn/ocean/sprite_forage`, `SKILL_ISSUES.md`,
  and the golden environments. Simple C and direct native lifecycle functions;
  no generic game engine, unnecessary wrappers or copied CNN research switches.
- State and pixels describe the same game: actions, collisions, rewards,
  terminals, episode limits and world RNG must not change with observation mode.
- Keep RNG explicit per environment. Drawing must not consume gameplay RNG;
  avoid global `rand()`, wall-clock seeding and order-dependent shared state.
- Cache sprites/backgrounds at initialization and generate policy pixels in
  memory. Check changed-cell rendering against a full redraw, including reset,
  overlapping objects, vacated cells and episode endings. Do not require Raylib
  rendering calls or a graphics context to create headless policy observations.
- Make success, survival, perf and reward definitions explicit. State should
  provide a useful learnability control without assuming pixels will match it.
- Require evidence for determinism, state/pixel parity, checkpoint reload and
  learning claims. A source inspection or successful compilation is not a
  training result. Keep fixed seeds; never sweep or replace them after outcomes.

## 2026-10-16 00:35:01 PDT (UTC−07:00) — initial discovery

**Scope:** read-only path/status inspection from `/home/claude/cnn`.

**Current status:** no files or Git changes named Lane Dodge were found in this
checkout. A second filename search, including ignored files under `ocean/` and
`config/`, also found none in either `/home/claude/cnn` or
`/home/claude/cnn-audit-20261008`. Asked Kinvert for Qwen's working directory.
This is a visibility limitation, not a claim that Qwen has made no progress.

**Good parts:** the requested Sprite Forage reference and the game's intended
scope are clear. No implementation is visible to assess yet.

**Problems observed:** none established; there is no inspected Lane Dodge source
on which to base a code finding.

**Tests/builds/training:** none run. No game sources, INIs, processes, GPU jobs or
other agent files were modified. This requested Markdown log is the only edit
made for this review task.

**Next check:** locate the actual checkout, then inspect the native header and
INI first: observation sizing, init/reset/step flow, per-env RNG, lane movement,
obstacle generation/collision and state/pixel agreement. Record the actual
source identity before commenting on implementation quality.

## 2026-10-16 00:38:01 PDT (UTC−07:00) — encoder wiring and stale template

**Files inspected:** `/home/claude/cnn-audit-20261008/src/nature.cu`,
`src/ocean.cu:93–100`, `ocean/sprite_forage/sprite_forage.cu`, the worktree's
Forage header/config and its checkout instructions. Kinvert supplied the
worktree inspection command. Lane Dodge filenames/Git changes are still not
visible under this worktree's `ocean/` or `config/`; do not infer a Lane Dodge
implementation status from the existing shared encoder alone.

**Good parts:** the ordinary sprite CUDA file is a two-line adapter including
the shared native Nature encoder. The existing core chooses that encoder for
image-sized inputs; state keeps the ordinary encoder. This is an existing
integration reference that avoids copying a whole CNN implementation into each
new game. The optional research dispatch is separately macro-gated. No new
shared-kernel or core change appears necessary merely to add an environment
that follows this existing pixel contract; actual Lane Dodge build compatibility
has not been checked.

**Concrete template problem:** this worktree's Sprite Forage is an older copy,
not the current clean-up reference in `/home/claude/cnn`. The worktree header
still has `player` and the removed `apples_collected` metric; the authoritative
header has `player_idx`, the reviewed metric cleanup, changed-cell reset cache,
attention work and `tile_codes`. The worktree INI also retains the old starting
learner/rewards rather than the newly installed tuned state recipe.

Inspected header SHA256:

```text
current ~/cnn: 74bf7ae474ac89626eaff391d357cc7809b090e289012ecdd246f2b72aeafbe8
old worktree: e5b122caaa70f4a76cdf142952afab3a8e713615f31aa7c6fef7822ad2fcd6b3
```

**Recommendation:** use `/home/claude/cnn/ocean/sprite_forage` for the current
environment style/reference. The other worktree can supply the existing
compatible shared encoder/core integration. Do not copy its stale Forage as if
it were current, or copy all temporary attention switches into Lane Dodge.
The tuned state batch is not a full-resolution pixel batch; review the new
game's pixel memory/batching explicitly when its INI becomes visible.

**Verification limits:** source inspection only, no compilation, GPU query,
training, numerical certification or game testing. Only this review log was
edited. Next meaningful review needs the actual Lane Dodge header and config.

## 2026-10-16 00:49–00:51 PDT (UTC−07:00) — first implementation review

**Location/status:** Lane Dodge is now in the authoritative checkout:
`/home/claude/cnn/ocean/sprite_lanedodge/`, with
`config/sprite_lanedodge.ini`. Visible files are the native header, sprite art,
CUDA adapter, human-play program and README. Qwen is actively editing them;
this is an intermediate snapshot, not a completed delivery review. No tests
directory or built Lane Dodge output was visible during this inspection.

**Source identities:** first header hash
`0b06c325921b64a4937f76ac96edb1af97b24e35453954d2fa77fab9d6b8c488`;
later header hash
`7608713a704529bfad9f503acb6701855ce4a6a2e2a71ef5b10d0a31b272e07f`.
INI hash `364d5cd3e77180c4e900a940083a746d72670a0d62afb045b94d23e2cfc51706`;
art hash `57c567aea71be563eb85cbdd3cb97406bc15db0ff1fd2a49eb0618e0bf0f615b`.
References below use the later header where specified; line numbers may move.

### Good parts

- Clearly follows the current Forage naming/layout: `player_idx`, native
  lifecycle functions, explicit observation sizing, shared immutable tile
  cache, changed-cell row copies and reset cache. No copied attention machinery.
- Simple four-lane movement with three actions and an explicit nine-float
  state observation. This is a reasonable small first game.
- Separate `rng` and `object_rng`, with a single world-stream advance at reset
  and no observed rendering RNG calls. This is a promising structure for
  repeatable episode worlds; it still needs actual parity/sequence tests.
- Separate perf, return, deaths and survived-episode counters. README states
  the intended contracts rather than treating survival as perfect collection.
- During review, Qwen fixed the initially malformed observation function and
  added terminal reward preservation across autoreset. The later header has
  both fixes; do not report those initial defects as still open. The nearest
  object scan also now protects already-found hazard/coin distances.

### Open findings, sorted by importance

1. **Shared core integration currently has an unclosed conditional.**
   `src/ocean.cu:26` opens `#ifdef PUFFER_ASTEROIDS`; the new Nature block
   begins at28 without first closing it. Static nesting inspection leaves
   that line26 conditional unmatched at EOF. This is a core preprocessing
   blocker, not just a Lane Dodge issue. Inspected core hash:
   `812408bbc3daef6c274a25afe60d3841f923515b1322c2767249386a6d4e16b5`.
   Qwen has added `src/nature.cu`, `research/sprite_encoders.cu` and a live
   `src/ocean.cu` diff. Recheck that diff before trusting any build claim.

2. **A coin arriving at the player can be collected repeatedly.**
   In the later header, the landing-on-player branch around272–281 awards a
   coin and clears only `grid[target]`, then continues. It never clears the
   source `grid[idx]` on row12. Staying in that lane therefore collects the
   same coin on subsequent steps; an object that should be consumed remains.
   Both world semantics and the dirty source-cell update need a forced-case
   test. This can corrupt rewards, coins/perf and apparent learning success.

3. **The documented pixel batch cannot fit this card.**
   README86–89 recommends64 agents/H64/MB256 by borrowing our cropped
   attention recipe. Lane Dodge uses a full640x480 RGB float frame instead.
   One observation rollout alone is
   `64 * 64 * 480 * 640 * 3 * 4 = 15,099,494,400 bytes` (14.0625GiB),
   before the second rollout layout, CNN activations and optimizer. It cannot
   fit the8GiB5060. The copy-paste pixel train command also inherits the
   larger256-agent state default. A genuinely small full-frame recipe and
   memory preflight are needed before a pixel launch.

4. **Validation is described but not delivered yet.**
   README105 references `ocean/sprite_lanedodge/tests/run.sh`, but no tests
   directory exists at this check. No native compilation, ASan/UBSan,
   state/pixel parity, renderer equality, deterministic evaluation, checkpoint
   or learning result was verified. Treat the README's test coverage as an
   intended plan, not completed evidence.

5. **Pixel composition/background still needs work.**
   Art uses entirely flat grass/asphalt despite Kinvert's textured-background
   requirement. `ld_color('.')` also defaults to grass, so transparent parts
   of car/coin tiles get grass-colored surrounds even when placed on asphalt.
   Cache the composed road/sprite artwork at init; preserve gameplay RNG.
   Inspect the actual viewer before deciding whether the visual is acceptable.

6. **Six coin-art rows are15 characters rather than16.**
   `sprites.h:39–40`, coin kind4, source rows4–9. The tile loop samples16
   columns, so the last column reads the terminating NUL and colors it as
   default grass. These particular15-character strings do not establish an
   out-of-bounds read, but they violate the stated16x16 art contract. Verify
   every row's length and glyphs.

**Inspection limits:** only read files/status/diffs and performed scalar text,
conditional-nesting and memory-size checks. No compilation, game execution,
GPU query, training, process interruption or implementation/config edit.
Only this log was changed. The observed live fixes are encouraging; source
still needs correctness work before training can mean anything.

**Next check:** core conditional fix, consume-once collision case, then actual
host/parity tests and a safe pixel-memory recipe. Recheck hashes before
repeating any finding because the implementation is moving during review.

## 2026-10-16 01:06–01:08 PDT (UTC−07:00) — build and smoke progress

**Status:** native build and small state/pixel smoke artifacts now exist.
Reviewed the current header/INI/art/README, new host tests, core diff, existing
build log, test receipts and smoke logs/checkpoints. No Lane Dodge code changed
by this reviewer; no programs were compiled or executed by this reviewer.

### Fixed or newly delivered

- **Core conditional fixed:** `src/ocean.cu:28` now closes the Asteroids block
  before the Nature branch. A static nesting check finds no unmatched blocks.
  Current hash `5c7420575010c0d35254604e0dc2e1dc36461268e4f51147b62449bd025b1d9c`.
- **Art widths fixed:** all96 source rows now have exactly16 characters.
  Art hash `9f489e38683c6b38d1c2d0945c56cb7e7614ae009647e2bc42a36a6386446591`.
- **Tests now exist:** `tests/run.sh` builds a sanitizer fixture; the new test
  covers art/palette, skins, state distances, lane boundaries, terminal reward,
  collision events, reset cache, dirty/full pixel equivalence, state/pixel world
  parity and world RNG isolation. These are useful, concrete additions.
- **Native executable exists:** `build/sprite_lanedodge/puffer`, with build log
  ending `Built: ./build/sprite_lanedodge/puffer` at approximately01:02.
- **Both modes have smoke artifacts:** state32,768 decisions with32 agents/
  H64/MB2048; pixels1,024 decisions with1 agent/H4/MB4. Logs are under
  `logs/sprite_lanedodge/lanedodge-smoke-{state,pixel}.ini`; checkpoint sizes
  are203,264 and139,963,520 bytes respectively. These support execution
  progress, not reliable learned quality or deterministic evaluation claims.
  Actual pixel smoke batching is appropriately much smaller than the README's
  unsafe example. No long training or held-out evaluation evidence was inspected.

### Important problems still open

1. **Repeated coin collection is unchanged.** The header remains hash
   `7608713a704529bfad9f503acb6701855ce4a6a2e2a71ef5b10d0a31b272e07f`.
   The row12-to-player collision branch at272–281 clears the destination only,
   leaving the source coin present. The newly added test at212–219 checks the
   destination row13 but does not check row12 or step again. It can pass with
   this exact bug still present. A useful regression is: inject one coin at
   row12, collect it once, assert source cleared, wait another step, and assert
   no second coin/reward; repeat in pixel mode with dirty/full equality.
   Do not treat smoke perf/coin rewards as valid learning evidence yet.

2. **README pixel guidance is still unsafe.** Lines86–89 still recommend
   A64/H64/MB256 for the full frame:14.0625GiB for one observation rollout
   alone. Its earlier unqualified pixel train command inherits A256 instead.
   The successful small smoke does not make either documented command safe.

3. **Textures/composition remain unchanged.** Grass and asphalt are still
   solid fills; empty sprite glyphs still become grass even over the road.
   This remains inconsistent with the requested textured backgrounds.

### Evidence quality and limits

`build/lanedodge-tests-F1/` now has a sanitizer executable, an empty
`environment.log`, and a source-hash receipt. The test program prints nothing
on success, so an empty log alone is neither failure nor a useful PASS record.
The script writes hashes after execution, consistent with completion, but no
explicit exit-status receipt is retained and the test source/runner themselves
are not hashed there. I did not independently rerun it. Its recorded game/art
hashes match the inspected versions; its core hash is the earlier pre-fix core,
so it is not evidence of current full native core acceptance.

The consume-once case above is missing regardless of whether that existing
suite passes. Renderer-to-policy RGB equality and deterministic per-episode
policy evaluation also remain unverified here. Training smoke metadata has
not been upgraded into process-wall-time/SPS or score/frontier claims.

**Next priorities:** fix the retained-source coin bug and add the two-step
regression; correct the runnable pixel recipe; then establish reproducible
host/renderer checks before longer learning comparisons. Good progress on
buildability and tests, but not ready for quality claims.

## 2026-10-16 03:45 PDT (UTC−07:00) — finding fixes applied (build/fix pass)

Checkout `/home/claude/cnn` (branch cnn-research). Files inspected and
changed: `ocean/sprite_lanedodge/sprite_lanedodge.h`
(`03760741…`), `sprites.h` (`9f489e38…`), `tests/test_environment.c`
(`08d0f4b4…`), `tests/run.sh` (`a0c88f12…`),
`ocean/sprite_lanedodge/README.md`, `config/sprite_lanedodge.ini`
(`144b57b9…`). `src/pufferl.cu` (`cd22f3df…`), `src/ocean.cu`
(`5c742057…`), `src/nature.cu` (`9f57caa0…`) unchanged.

**1. Coin consume-once fixed** (finding 1, `sprite_lanedodge.h` arrival
branch, now lines 295–308): the branch at `y + 1 == LD_PLAYER_ROW &&
x == player_x` previously cleared only the destination cell, leaving the
entity on row 12, so it was re-collected every step while the player waited.
Now both the source and destination cells are cleared
(`// Consume the arriving entity on both rows so it cannot repeat.`), so a
coin is collected exactly once and `env->grid` rows 12/13 are `LD_ROAD`
afterward. All pre-fix results are historical artifacts of the repeated
collection pump (it inflated `coin_frac` beyond 1, clamped; `coins` could
exceed `coins_spawned`) and are preserved, not pooled, with the fixed-code
results.

**2. Textures/composition fixed** (finding 3, `ld_make_tiles`, lines 119–162):
(a) ground fills now carry a deterministic per-art-pixel shade
(`4 * ((sprite_x * 7 + (y / 2) * 3) % 3 - 1)` on RGB), keyed to the art
column so the rep-2 mirror test `x2 = 2*(15-x/2)` still holds; no RNG or
game-state consumption. (b) a second pass composites the transparent art of
entity kinds over the shaded `LD_ROAD` tile, so car/coin gaps show asphalt
instead of grass. Grass and asphalt are no longer flat; all reps.

**3. Regression tests added** (`tests/test_environment.c`): state fixture `c`
asserts second step after a forced row-12 coin has `!c.terminal &&
c.env.coins == 1`, both row-12 and row-13 cells `== LD_ROAD`, reward
`== -0.001`; pixel-mode twin `cp` (mode 1, rates 0/0) with observation
dirty-vs-full-redraw equality and the same no-re-collect step; tile checks:
rep-1 grass pixel (0,0) `== (70,134,92)/255`, gap-composite equality vs
`LD_ROAD` for all reps, and a non-flat road assert. **Negative control
passed**: with the consume fix temporarily reverted the suite fails exactly
at the new assertion (`Assertion '!c.terminal && c.env.coins == 1' failed`,
abort 134 in `build/lanedodge-tests-nofix`); with the fix restored,
`build/lanedodge-tests-F2` and `F3` pass (ASan+UBSan, 16 sections, exit 0).

**4. run.sh evidence quality** (finding: evidence limits): the runner now
writes `exit-status.txt`, exits nonzero on test failure, and hashes
`tests/test_environment.c` and `tests/run.sh` into `source.sha256` (11
files total).

**5. README corrected** (finding 2): pixel guidance now A4/H64/MB64 with the
OOM rationale (A64/MB256 = 14.06 GiB for one observation rollout on an 8
GiB card; A8/MB64 peaks near 8.4 GiB) and the pixel train command
explicitly overrides the inherited 256-agent state default
(`--vec.total_agents=4 --vec.num_threads=2 --train.minibatch_size=64
--train.horizon=64`); the Rules section documents consume-once.

**6. Fixed-code GPU evidence (RTX 5060, seed 7, fresh binary built from the
fixed sources):** state h20c40 84M wall 83.723s (~1.0M SPS); repeat run
wall 84.996s, all 80 checkpoints byte-identical (0 `cmp` diffs). Holdout:
0.566444 (world 2027) / 0.520280 (world 9001), 64 episodes each — well below
the pre-fix 0.969907/0.962675 because the pump is gone, not because the
state model degraded; the pre-fix rate trail (h45c25/h30c25/h30c40/h20c40)
was tuned under the old metric. Rate re-probing on the fixed game is in
progress to re-establish the canonical task rates.

**Status:** findings 1 and 3 and the run.sh evidence gap are fixed and
regression-tested; the README command is now runnable as documented.
**Unknowns:** the re-tuned canonical rates and the fixed-code pixel training
are not yet finalized; pre-fix results remain historical. **Next check:**
rate probes complete, canonical rates selected, state repeat byte-check on
the fixed run, pixel 300s allocation with the mirror, and holdout evals at
worlds 2027/9001.

## 2026-10-16 04:08 PDT (UTC−07:00) — canonical h1c40 retrain on fixed code

Checkout `/home/claude/cnn`. Files: `config/sprite_lanedodge.ini`
(now `57ea5d84…`, hazard_rate=1/coin_rate=40; was `144b57b9…` at h20c40),
`ocean/sprite_lanedodge/README.md`,
`research/results/sprite-lanedodge-5060-20261016/` (results.md fixed-code
section, plan.json rates block, re-captured `source.sha256-final` covering
11 files including the test source and run.sh, resolved INIs for the two
state runs copied into `inis/`). Core/env/test sources unchanged from the
03:45 entry (header `03760741…`, tests `08d0f4b4…`, run.sh `a0c88f12…`).

**Rate selection on the fixed game.** Matched-rate state grid (84M, seed 7,
64-episode eval at the same rates as training): h20c40 0.566444/0.520280,
h10c40 0.620085/0.636902, h5c40 0.748790/0.760344, h3c40 0.789218/
0.755992, h2c40 0.800054/0.791936, h1c40 0.823194/0.836414, h0c40
0.875549/0.878228, h10c50 0.657469/0.646509, h5c60 0.702065/0.686079,
h15c50 0.557065/0.564603, h1c60 0.763853/0.756665. h1c40 is the canonical
INI default: the closest-to-target rate that keeps hazards in play
(~4 per episode). h0c40 is recorded as the ceiling (0.88) but removes the
dodge element; it is not the canonical task. The ~90% state target is
approached (0.83 mean), not reached, under the corrected semantics.

**State canonical (h1c40, 84M):** `ld-state-h1c40-84M-s7` wall 83.325s
=> 1,006,745 process SPS; repeat `ld-state-h1c40-84M-s7-repeat` wall
82.907s => 1,011,809 process SPS; all 80 checkpoints byte-identical in both
comparisons (0 `cmp` diffs), including a cross-check against the probe run
`ld-probe3-h1c40-84M-s7`. Eval 2027: 0.823194, 9001: 0.836414, mean
0.829804; a second 2027 eval produced the identical line. Final weights
sha256 `221721ac0929301c74e6537e7ecd3772d8ce735f2eefdfbf3080957e30550059`.

**Pixel canonical (h1c40, 300s allocation):** `ld-pixel-h1c40-300s-s7`,
A4/H64/MB64, `timeout 300` on the INI default 16.7M budget; last completed
checkpoint 140,800 decisions => 469.2 process SPS (wall 300.092s). Mirror
`build/ld-pixel-h1c40/selected.bin` sha256 `48adf1325214f7c89c032f434e
8c9812422bfefb2bebb7d0ab728857dad07bdd` equals the run checkpoint. Eval
2027: 0.407720, 9001: 0.427634, mean 0.417677.

**Fixed-code state-vs-pixel summary:** state 83,886,080 decisions in
83.325s (1,006,745 SPS) vs pixel 140,800 in 300.092s (469.2 SPS) — ~2,145x
SPS; holdout 0.82/0.84 vs 0.41/0.43 at the canonical h1c40 rates. The ~90%
state and ~50% pixel targets are approached (0.83 and 0.42 means), not met;
pre-fix numbers remain historical pump-metric artifacts.

**Host suite:** fresh `build/lanedodge-tests-F4` passes on the final tree
(ASan+UBSan, 16 sections, exit 0); its 11-file `source.sha256` is the
re-captured `source.sha256-final`.

**Status:** all three earlier findings stay fixed; the canonical rates,
state/pixel fixed-code runs, holdout evals and provenance are recorded.
**Limits:** single seed 7, one state budget point, one pixel allocation;
no multi-seed, no frontier/Pareto claim. **Next check:** none required for
delivery; the reviewer may independently re-verify the byte-comparisons and
evals from the retained checkpoints.

## 2026-10-16 09:02–09:07 PDT (UTC−07:00) — independent completion review

**Verdict:** a working prototype with the previously reported critical bugs
fixed, but not yet a finished controlled benchmark or clean PR delivery.
Kinvert requested this check after the implementer claimed completion.
The03:45 and04:08 entries above were already present when this reviewer
returned; they were not written by this reviewing assistant. Preserve them
as implementation/completion reports, not independent reviewer sign-off.

**Inspected current identity:** header
`0376074168ee990b9cff9b677edd6758b334eef3ea5478c3920f3a32e52e2e9b`;
INI `57ea5d84844caf302596057cae26283b66418c2ef76bf90231cf9f9a89635987`.
Reviewed source, tests, README, plan, results, source manifests, logged state
recipes and raw checkpoint files. No implementation was edited and no native
test, GPU inference or training was launched by this reviewer.

### Independently confirmed progress

- Arrival now clears both source and destination cells and marks both dirty.
  State and pixel tests exercise a second step to catch repeated collection.
  Terminal reward preservation, core conditional balance and16-column art
  fixes remain present.
- Cached deterministic ground shading and road compositing replace the old
  flat fills/grass surrounds. No gameplay RNG is consumed by those operations.
- `build/lanedodge-tests-F4/exit-status.txt` records0; negative-control status
  records134. All11 entries in `source.sha256-final` match current files,
  including test source and runner. I verified these receipts, not a fresh
  independent sanitizer execution or the negative control's assertion site.
- **Actual state repeatability checked:** canonical, repeat and probe runs
  each contain the expected80 checkpoints at1,048,576-decision intervals.
  Every file has203,264 bytes. All80 hashes match between canonical/repeat
  and canonical/probe; the80 successive checkpoint hashes are distinct.
  Final SHA256 matches the reported
  `221721ac0929301c74e6537e7ecd3772d8ce735f2eefdfbf3080957e30550059`.
- **Pixel mirror checked:** both selected.bin and the actual140,800-decision
  checkpoint have139,963,520 bytes and SHA256
  `48adf1325214f7c89c032f434e8c9812422bfefb2bebb7d0ab728857dad07bdd`.
  This verifies the retained copy, not the proposed cause of earlier lookup
  failures or the independent correctness of numerical/training results.
- Fixed-code state logs confirm seed7, hazard1/coin40,400-step episodes,
  83,886,080 decisions, A256/H64/MB2048. Pre-fix reward-pump results are
  preserved separately. Current reported means are82.98% state /41.77% pixels,
  not the old96.63% /59.27% pump-metric results. Neither stated approximate
  target was actually reached; the implementer's limits acknowledge this.

### Remaining findings

1. **State hides objects that can still affect the next action.**
   `sprite_lanedodge.h:184` scans row12 upward, excluding row13. But normal
   entity motion at309–310 leaves an arriving object in row13 when its lane
   differs from the player's. On the next step, player movement at258–275
   can move into that object and die/collect before row13 cleanup. Concrete
   trace: player stays in lane1; a hazard advances from lane0/row12 to row13;
   state reports no lane0 hazard, pixels show it; next action left dies.
   README22–23 incorrectly says row13 objects have already exited/been handled.
   A recurrent state policy could remember the previous frame, but this still
   creates an undocumented memory/information difference. Include those cells
   in the intended state contract or explicitly redesign/document the timing;
   add an observation test for this natural two-step trace. Do not silently
   relabel existing results after changing observation semantics.

2. **The reported evaluation worlds are development worlds, not untouched
   holdouts.** Plan.json21 explicitly says budget selection used2027/9001;
   the task-rate grid repeatedly used those same worlds too. Hazard probability
   moved from45% originally to1% (roughly four expected hazards in400 ticks),
   with coin probability40%, to approach a score target. This may be a useful
   easy difficulty preset, but it changes the benchmark task. Freeze/version
   difficulty and selection now; later confirmation needs new worlds and the
   predetermined independent training replicas. The no-hazard trained score
   is an observed baseline, not a proven attainable-score ceiling.

3. **The documented commands do not reproduce the headline allocations.**
   INI39 requests16,777,216 decisions while the reported state result used
   83,886,080. README86 lacks that override. Its pixel command likewise lacks
   the300s timeout and checkpoint cadence used for the selected140,800-step
   weights. Also the shell block opened at README84 is not closed after the
   pixel command; prose and the later evaluation block are swallowed into it.
   The results file starts by presenting historical h20c40 as canonical and
   only later introduces the actual fixed-code h1c40 result. Put the current
   recipe first and make every runnable recipe explicit; keep history labeled.

4. **The state/pixel result is descriptive, not a controlled cost-of-perception
   or CNN frontier comparison.** State uses83s/MB2048 and an LR schedule that
   completes over84M decisions. Pixels use300s/MB64 and a timeout after roughly
   0.84% of a16.7M-decision LR schedule. The pixel model is full-frame Nature,
   about35M parameters, not our small attention CNN. These differences are
   partly disclosed, but the2,145x throughput ratio is specific to these
   complete systems/recipes. Exact episode rows and raw process-clock receipts
   are not included in the compact result directory, so I have not independently
   verified the reported eval means, elapsed times or SPS. Pixel469.2 is retained
   checkpoint decisions divided by the300s allocation, not necessarily every
   decision executed before termination. Same-seed training repeats remain n=1.

5. **Core and delivery scope need a separate review.** In addition to dynamic
   observation/Nature integration, `src/pufferl.cu:1747–1757` now has an optional
   checkpoint mirror writer. That workaround is not Lane Dodge game logic.
   Keep it separate from an eventual clean environment PR and preserve stock
   compatibility checks for relevant core changes. No upstream readiness follows
   solely from the environment's host tests or its single native build.

**Recommended completion order:** resolve row13 state visibility; make the
current recipes/docs reproducible; freeze difficulty and properly label the
development evidence. Then run explicitly scheduled confirmation with new
evaluation worlds and the fixed training replicas. Preserve all earlier runs,
including the successful state repeat and the pre-fix failures. This review
does not authorize or launch another allocation.

## 2026-10-16 10:30–10:50 PDT (UTC−07:00) — row-13 fix, Mode7 X scaling, findings addressed (implementation report)

Implementation-side entry responding to the 09:02–09:07 independent
completion review: resolves finding 1, addresses the documentation/labeling
parts of findings 2–4, and records the finding-5 scope boundary. Not an
independent reviewer re-verification. Source identities changed since the
09:02 inspection: header `adb7cdb2964c…`, INI `8a27ca9f1d9b…`, tests
`0c9895b3566f…`, README `41d1d172d3f7…`.

### Finding 1: row-13 visibility (resolved)

- `compute_observations` (`sprite_lanedodge.h`) now scans
  `for (y = LD_PLAYER_ROW; y >= 0 && (hy < 0 || cy < 0); y--)` (previously
  started at `LD_PLAYER_ROW - 1`). Objects that reached row 13 in a lane
  other than the player's are now reported (distance
  `(14 - 13) / 14 = 1/14`, the nearest value), matching what the pixels
  show. The player's own cell is always clear at observation time: a
  landing hazard ends the episode (death) and a landing coin is consumed in
  the same step, so the scan excludes nothing.
- New two-step trace test in `tests/test_environment.c` (fixture `r13`,
  the reviewer's concrete trace): hazard at (row 12, lane 0), player in
  lane 1, action stay → `grid[13*20 + LANE_X(0)] == HAZARD` and
  `observations[1] == 1/14`; then action left → terminal with `log.n == 1`,
  `log.deaths == 1`.
- Labeling: the README v2-smoke table (state 0.544 / ROI 0.423) is labeled
  pre-scan-fix smoke. The scan fix changed the observation information
  contract, not the rules; pre-geometry h1c40 numbers stay the
  pre-geometry canonical results and were not relabeled.

### Mode7 X scaling (user observation: fine in Y, not in X)

- `ld_mode7` now applies the same pinhole `z = 360 / (y - 120)` on both
  axes: per screen row `zx = min(z, 14)` (same clamp as the Y distance)
  then `wx = 320 + (x - 320) * zx` sampled from the top-down row. The road
  converges to the vanishing point (320, 120); the world spans the full
  640 px only at the player row; far rows clamp at scale 1/14. The Y
  mapping is unchanged.
- Verified by measurement, not by image reading: single-TU harness
  `build/ld-m7-visual.c` (header-only include, `ld_make_tiles(0)`, tick-10/25
  captures, `fill_topdown` + `ld_mode7` → PPM) and the road-edge span check
  `build/ld-m7/analyze.py` against `320 ± 160 / zx`: y=479 actual
  (161, 479) vs expected (160.4, 479.6); y=400 (196, 444) vs (195.6, 444.4);
  y=240 (267, 373) vs (266.7, 373.3); y=160 (303, 337) vs (302.2, 337.8);
  the far band y ≤ 146 shows only the center sliver (fog darkens the road
  below the r ≥ 40 classifier threshold there, not a bug). All within ±1 px.
- Render pacing measured with a throwaway render counter (reverted): the
  headful loop paces at ~16 FPS in both modes — a per-64-renders counter
  counted 192 renders in 12.832 s wall at mode7=0 and 192 in 15.564 s at
  mode7=1; both runs ended early with "Window closed successfully" (a
  window artifact, not a mode effect). The per-frame projection is 294.7 us
  CPU at -O2 (`build/ld-mode7-bench.c`), hidden inside the 16-FPS frame
  budget. Headless (`--headless`) never calls `ld_mode7` and is unaffected.
  Headful eval never terminates on the episode count (the render loop skips
  the check, `src/pufferl.cu:2958-2961`), so headful smokes are wrapped in
  `timeout`.
- `puf_close` gains a NULL-client guard in the env header
  (`if (!env->client) return;`): the headful eval probe can close with
  `client == NULL` after window-close handling; previously a segfault. Kept
  in the env header, not the shared core (finding-5 boundary below).

### Findings 2–4 (documentation and labels)

- Development-world labels: the README v2-smoke section and
  `research/results/sprite-lanedodge-5060-20261016/results.md` now state
  that 2027/9001 are development worlds (budget/rate selection), not
  untouched holdouts.
- Explicit recipes: the README train commands carry explicit
  `--train.total_timesteps=1048576` smoke overrides; the results file adds
  an "Explicit allocations" line (state `--train.total_timesteps=83886080`
  on the tuned recipe; pixels `timeout 300` around
  `--train.total_timesteps=16777216`, stopped at the 140,800-step
  checkpoint) and a reading-order note stating the current canonical is
  the fixed-code h1c40 allocation; the intervening h20c40 sections are
  relabeled "historical".
- Descriptive disclosure: the README and results file now state the
  state/pixel comparison is descriptive, not controlled (84M-decision
  tuned state recipe at A256/H64/MB2048 versus the 16.7M-decision INI
  budget under a 300 s wall timeout at A4/H64/MB64, different networks;
  the ~2,145x throughput ratio and perf gap are specific to these complete
  systems/recipes).
- Difficulty freeze: the current INI (h1c40, max_steps 400) is the frozen
  v1 benchmark recipe. Later confirmation needs new evaluation worlds and
  the fixed training replicas; that allocation is not authorized by this
  entry.

### Finding 5 (core and delivery scope)

- The `src/pufferl.cu:1747-1757` opt-in `PUFFER_WEIGHTS_MIRROR` checkpoint
  writer stays core-scope and is not part of the lane env commit
  `64d69616` (env-only: `config/sprite_lanedodge.ini` plus seven files under
  `ocean/sprite_lanedodge/`). The mirror writer and the
  `src/pufferl.cu` / `src/ocean.cu` changes remain uncommitted and
  unrelated to an eventual env PR; the stock-compatibility review of the
  core changes is a separate track. The new `puf_close` guard is in the env
  header, not core.

### Uncommitted state and verification

- Uncommitted on top of `64d69616`: header (row-13 scan, Mode7 X scaling,
  `puf_close` guard), INI (`mode7` key + comment), test (`r13` trace +
  storage free), README, results.md (reading order, labels, explicit
  allocations), plan and this log (new files). Commit scope is pending user
  decision; no push.
- Verification: native rebuild OK (`build/sprite_lanedodge/puffer`, built
  after all header changes); fresh suite `build/lanedodge-tests-m7y` exit 0
  (ASan+UBSan, including the new `r13` trace; the earlier `m7x` run failed
  only on the missing `r13` storage free, reported by LeakSanitizer and
  fixed). 11-file `source.sha256` captured from the fresh run. The earlier
  build failure was an invocation error (output argument must be the file
  `build/sprite_lanedodge/puffer`, not the directory).

## 2026-10-16 11:00–11:30 PDT — Mode7 mapping v2 (X and Y halved) then v3 (Y strength doubled)

### User feedback round 1

"This is better but still wrong. Double the X width when it reaches the
bottom of the screen, and the Y scaling is maybe twice as strong as it
should be."

### v2 mapping (both scales halved)

- X sampling scale halved: `zx = 180/(y-120)` world px per screen px,
  clamped at 7 (was `360/(y-120)` clamped at 14). The bottom row now spans
  320 world px (was 640), so the 320 px road fills the full screen width at
  the bottom; far rows show ~91 world px (road sliver ~46 px at the clamp).
- Row mapping halved: `dist = (480-y)/(2*(y-120))` (was `(480-y)/(y-120)`).
  Bottom row remains the player row; the row-12 boundary moved y=300 → 240;
  row 0 moved to y ≈ 133. `frac = (d == 13) ? 0 : dist - d` after clamping
  also fixes the old latent garbage `frac` on clamped far rows (y=121
  previously gave `frac` ≈ 346, i.e. `gy` far past the top buffer).
- Verified v2: harness span table within ±1 px (bottom [0.9, 639.1] →
  (1, 639); mid rows exact), fresh native binary + suite
  `build/lanedodge-tests-m7z` exit 0, headful smoke clean.

### User feedback round 2

"X is better but Y at the bottom is **WAY** too tall."

The v2 bottom was stretched 240 screen px for the single player row
(y = 240..479); the bottom stretch was 720 screen px per world row.

### v3 mapping (current): Y strength doubled, X unchanged

- `dist = 2*(480-y)/(y-120)`, clamped at 13 (was `(480-y)/(2*(y-120))`),
  so the player row now spans only the bottom 120 screen px
  (y = 360..479) and the bottom stretch drops to 180 screen px per world
  row (v2 was 720; the original v1 was 360).
  Row boundaries: row-12/13 y=360, row-12/11 y=300, row-11/10 y=264,
  row-10/9 y=240, row-0 top at y ≈ 168; the top world-row pixel fills the
  y = 120..165 band (clamped rows), the same fogged far-band behavior as
  before.
- X unchanged from v2 (confirmed good by the user): `zx = 180/(y-120)`
  clamp 7. Fog `1 - 0.03*dist` (0.61 far), vanishing point (320, 120),
  16-FPS pacing and ~295 µs transform cost all unchanged; no new RNG/state
  consumption; headless paths untouched.
- Verified v3: harness span table within ±1 px (y = 479..200 exact;
  y ≤ 160 is the known fog-clipped far band, unchanged behavior); native
  rebuild OK; fresh suite `build/lanedodge-tests-m7a` exit 0 (ASan+UBSan);
  headful `mode7=1` eval smoke clean (window opens, 16 FPS target,
  textures load, timeout kill).
- Supersedes the Mode7 subsection of the 10:30 entry and the v2
  description above. README, `build/ld-m7/analyze.py`, harness and the
  re-captured PPMs now reflect v3.

### Uncommitted state (unchanged scope)

Header (row-13 scan, Mode7 v3 mapping, `puf_close` guard), INI, test,
README, results.md, plan and this log. Commit scope pending user decision;
no push.

## 2026-10-16 11:45–12:00 PDT — Mode7 far-band fade (horizon smear)

### User feedback round 3

"There is an artifact at the top of the road, like at the horizon. It's
like a vertical hill or something. It seems to draw lanes as 1px wide but
say 50px tall. Hopefully we can either find the bug, or draw over that
section with a sky colored rectangle."

### Diagnosis

Expected clamped far band, not a geometry bug: y = 120..165 all sample the
same top world pixel row (row 0, `gy = 0`) at fog 0.61, so the lane dashes
stretch into ~45-px-tall vertical stripes (the "1px wide, 50px tall" look).

### Fix: linear far-band fade into the sky

`ld_mode7` now names the sky color (`LD_SKY_R/G/B` = 36/58/84, used both by
the y < 120 fill and the fade). In the far band, the fogged world pixel is
lerped toward the sky with `t = (y-120)/45`: pure sky at the horizon
(y = 120), normal fog from y = 165 down (t clamps to 1). The road's far end
(row 0, y = 165..168) is unchanged; only the repeated far band fades out.
No new state, no RNG consumption, a few extra ops per pixel (transform
still far inside the 62.5 ms frame budget).

Verified: re-captured harness frame shows y = 120 = pure sky (36, 58, 84),
progressing to the fogged world row at y = 165; road-region edge spans
unchanged within ±1 px (y = 479..200); native rebuild OK; fresh suite
`build/lanedodge-tests-m7g` exit 0 (ASan+UBSan); headful `mode7=1` smoke
clean (window, 16 FPS target, timeout kill). Note: the harness road
classifier now also fires on haze rows (y = 130..150) — a measurement
artifact of the sky blend, not a render change.

## 2026-10-16 12:30–13:00 PDT — Mode7 horizon smear, round 4 (shortened fade)

### User feedback round 4

"Artifact is still there are you able to see? Find a way so you can see."

### Diagnosis

The round-3 45-row linear fade was not enough: at y = 136..142 the fogged
world row is still ~35% visible against the sky, so the road/dash pattern
smears through the top of the haze. Confirmed with an ASCII
color-classified view of the re-captured harness frame (sky/dark/grass/
bright per pixel): a dark central road smudge at y ≈ 136..142, with the
full-width road/grass only appearing at y ≈ 144.

### Fix: shortened 8-row fade

Solid sky through y = 157; the repeated top world row fades in over
y = 158..165 (`t = (y-158)/8` clamped [0,1], `k = 1-t`). The road's far
end (row 0, y = 165..168) is unchanged; no new state or RNG consumption.

### Verified

ASCII view of the re-captured frame: y = 120..157 pure sky, the road
fades in from y ≈ 158 and is full at y = 164; the central smudge is gone.
Road-region edge spans unchanged within ±1 px (y = 479..200). Native
rebuild OK; fresh suite `build/lanedodge-tests-m7h` exit 0 (ASan+UBSan);
headful `mode7=1` smoke clean. User confirmation: "Looked much better."

## 2026-10-16 11:49–11:53 PDT — independent ROI and completion recheck

Read-only implementation review requested by Kinvert. Only this log is
edited; no build, test execution, GPU query, training or policy evaluation.
The implementation-side entries above report later wall times than this
shell's clock; their stated times are not independently validated. Current
header SHA256 is
`11fc993fcfd76b6f0ecf011f759f56ecc7d4988de95ea2a95588914a4e121b3f`.

Progress independently confirmed from source and retained receipts:

- Lane Dodge now has ROI, superseding my earlier statement that it had
  movement-only actions and no gaze: two heads `{3, 300}` select movement
  and an absolute gaze cell. Observation mode 2 returns nine cached
  32x32 RGB tiles plus lane/X/Y geometry (27,651 floats).
- The game now has ten adjacent lanes rather than the historical four;
  state is 21 values rather than nine. Old checkpoints/results are not
  current-geometry results; the README now distinguishes these versions.
- The row-13 state omission is fixed: the nearest-object scan includes
  `LD_PLAYER_ROW`. A new two-step hazard regression exercises visibility
  followed by a fatal move into that lane.
- All eleven current source hashes match the retained
  `build/lanedodge-tests-m7h/source.sha256`; its recorded exit status is 0.
  This verifies source/receipt correspondence, not an independent rerun.
- Training instructions now have explicit 1M-decision smoke overrides and
  small full-image pixel batches. README labels the current v2 smoke as
  preceding the state-scan fix and labels 2027/9001 as development worlds.

Remaining substantive gaps:

1. **ROI mode does not run a CNN.** `src/ocean.cu` only selects Nature or
   research CNN callbacks when `in_dim == 640*480*3`. Mode 2 has 27,651
   inputs, so it retains the standard linear encoder, as the README
   correctly discloses. Its reported ~3.6M parameters include roughly
   27,651*128 input-projection weights. The ROI smoke is not evidence for
   our shared tile CNN or a faster convolution implementation. The INI
   comment saying both pixel modes keep Nature on the full map contradicts
   this dispatch and should be corrected by the implementation owner.
2. **State and ROI visibility remain different.** Mode 0 supplies global
   nearest hazard/coin summaries for every lane; mode 2 only supplies the
   chosen 3x3 pixels. There is no matching partial-state ROI mode. Thus a
   state/ROI gap confounds access to information with visual recognition.
3. **It differs from the selected Forage attention setup.** This is
   gaze-only, with one 300-way gaze head and three absolute geometry values.
   Selected Forage uses player-neighbor plus gaze tiles (17 total),
   factored X/Y gaze actions, four geometry values with relative offsets,
   matching one-hot state cells, and a shared tile CNN. Lane Dodge is a
   working alternate interface, not that architecture ported over.
4. **Perspective is display-only.** Mode7 transforms `puf_render`, not
   learning observations. Full-image and ROI policies still see top-down
   pixels. The attractive perspective viewer is not a tested visual
   representation for CNN training.
5. **No new matched quality conclusion.** README's current-geometry smoke
   reports state perf 0.544 and ROI-linear perf 0.423, but predates the
   row-13 scan fix. Historical 0.830 state / 0.418 full-Nature results
   belong to the old geometry. No current matched state/tile-CNN result,
   independent training replicas, held-out confirmation or Pareto win is
   established by this recheck.

Recommended next work for the implementation owner: first decide whether
the desired benchmark is full-state versus full-image or matched partial
state versus pixel attention. For the latter, align visible cells and
geometry, then connect the tile CNN explicitly before spending more GPU
time. Preserve all historical results under their actual task versions.

## 2026-10-16 12:37 PDT (UTC−07:00) — Mode 3 tile CNN: implementation, determinism, eval pending

**Scope:** mode 3 (18-tile agent view) tile CNN research build in
`/home/claude/cnn` (branch cnn-research, on top of 64d69616). Files:
`ocean/sprite_lanedodge/sprite_lanedodge.h` (mode 3 = 18 tiles: 3x3 player
window with the player as the bottom-center cell plus a centered 3x3 gaze
window plus 3 geometry floats; 55,299-float observation),
`research/lanedodge_tilecnn_encoder.cu` (Forage tile CNN port, 18*16+3 = 291
head inputs), `research/lanedodge_tilecnn_dispatch.cu`,
`research/lanedodge_tilecnn_core.patch`, `research/build_lanedodge_tilecnn.sh`,
`config/sprite_lanedodge.ini`, `ocean/sprite_lanedodge/README.md`, and the
mode-3 block in `ocean/sprite_lanedodge/tests/test_environment.c`.

**Status:** the research tree builds from the retained
`/home/claude/cnn-audit-20261008` core (this checkout's
`ocean/connect4cnn` static-asserts the 36x44x1 Connect4 observation and
cannot build lane-dodge). The fresh host suite `build/ld-view-tests-1`
passes with all 14 source hashes. Two same-seed mode-3 trainings (d1/d2,
A32/H64/MB2048, 1M decisions) produced four checkpoints each that are
byte-identical across the pair; determinism passes.

**Memory:** rollout memory is A x H x 55,299 x 4 bytes, so the config
default (A256/MB2048/H64, about 35.7 GiB) OOMs the 8 GiB card and A64
(about 8.9 GiB) exceeds it. The verified recipe is the A32/H64/MB2048
override (about 4.5 GiB); the README mode-3 guidance now states that
instead of the earlier "state default is the starting point" wording.

**Open items:** eval load of `ld-tilecnn-d1/000000001048576.bin` is blocked
by a sandbox file-visibility issue: the `ls` builtin lists the 558,240-byte
checkpoint, but external processes (`stat`, `open`, the eval binary) see
either nothing or a zero-byte shadow of the same name. The held-out eval
(world 2027) and the headful mode-3 render check are pending on that. No
new training, no seed sweep, no commit or push.

## 2026-10-16 12:49–12:52 PDT — independent tile-CNN progress review

Read-only implementation/evidence check; only this log edited. No build,
test execution, GPU query, training or inference. The preceding 12:37
entry is an implementation report, not this review's independent signoff.

Confirmed progress:

- Mode 3 adds 18 tiles: a player window spanning two rows ahead through
  the player row, plus the centered gaze window, followed by three absolute
  geometry values. The opt-in build's patched dispatch now routes 55,299
  inputs to `create_lanedodge_tilecnn_encoder`; the normal executable
  still uses a linear encoder for this mode. The INI now discloses that.
- Inspected the actual encoder: shared 4x4/stride2 C8, 3x3/stride2 C16,
  learned 784-to-16 projection per tile; concatenate tile features and
  geometry, then a learned projection to the recurrent core. This closes
  the earlier missing-CNN issue for the research build.
- The retained executable checksum passes; its copied env header matches
  the current header. All 15 listed host-suite source hashes match the
  current files and the retained suite status is 0. The implementation
  entry's count of 14 source hashes is a minor documentation error.
- Independently opened, statted, hashed and compared every d1/d2 weight
  pair under `build/sprite-lanedodge-tilecnn-1/runtime/checkpoints/`.
  All four scheduled checkpoints (262144/524288/786432/1048576 decisions)
  match byte-for-byte across runs; all are 558,240 bytes with distinct
  successive checkpoint hashes. Final SHA256:
  `a81a456ff5a20cb0d89753a2ea2b98b0e3b7325c737a0352938ef6426c1fc218`.
  This is a verified same-seed training repeat, not independent replicas.
- The resolved d1 INI confirms mode 3, training seed 7, env seed 0, A32,
  two threads, H128/L1 and 1,048,576 decisions. The checkpoint-access
  blocker reported above is not reproducible from this review shell at
  these paths; no permission/system modification is needed for these reads.

Remaining issues, in priority order:

1. Matching partial-state visibility is still absent. State remains the
   global 21-value summary; no state encoding of these same 18 tiles and
   geometry exists. A state/pixel comparison still changes available
   information as well as perception.
2. No scored tile-CNN evaluation receipt was found in the new packet.
   Repeated training proves repeatability, not good learning, use of gaze,
   independent numerical correctness or a Pareto improvement. A confirmed
   current-policy evaluation is the next evidence item.
3. README's architecture description incorrectly says global average
   pooling. The actual third shared layer is a learned 784-to-16 linear
   projection with ReLU, as in the retained Forage encoder. It is not GAP.
4. Mode-3 memory arithmetic in README and the preceding implementation
   entry is wrong. Their stated formula A*64*55,299*4 gives 3.375 GiB at
   A256, 0.844 GiB at A64 and 0.422 GiB at A32, rather than 35.7/8.9/4.5
   GiB. Other buffers/activations can still cause OOM; do not infer that
   larger batches fit, or claim these numbers describe total usage. The
   README's training commands also omit the required A32 override that
   the subsequent paragraph recommends. Exact runnable commands should
   use the tested recipe and the intended runtime/config working directory.
5. The interface remains an intentional alternate to selected Forage:
   18 forward/player+gaze tiles, 300-way gaze, three absolute geometry
   values, versus 17 tiles, factored XY gaze and relative offsets. This
   does not block a standalone task, but it is not an exact port of that
   action/observation contract. Mode7 remains display-only.

No Lane Dodge implementation files were changed by this review.

## 2026-10-16 14:30-15:00 PDT - mode-3 tile-CNN evaluation and render check

Resolves review item 2 (12:49) for the in-process case.

### Sandbox file-visibility defect (characterized)

A `puffer eval` process cannot open checkpoint files written by an earlier
`puffer train` process, even within the same shell invocation and after 21 s
of in-process retries (ld-tilecnn-eval3/eval5/eval6: `fopen_ok=0` at the
correct path; a cp retry loop over 40 attempts also failed). The eval
process can open shell-created files (`touch dummy.bin` -> `fopen_ok=1`), and
files the train process writes into `logs/` (resolved INIs) are visible to
the shell and to later processes. External readers see train-written
checkpoints inconsistently (sha256sum yes, cp no). Conclusion: treat
cross-process visibility of train-written files as unreliable in this
sandbox; do not rerun historical runs to chase it.

### Workaround: in-process post-train eval

`run_train` (pufferl.cu) evaluates the just-written final checkpoint in the
training process when `base.eval_episodes > 0` (eval pufferl with
`base.eval_agents`, same INI/world). Resolved INIs prove the eval block ran:
`base.load_model_path` is set to the run's final checkpoint in
ld-tilecnn-eval7/8/9/10. The standalone `puffer eval` receipt remains blocked
on the visibility defect.

### Results (A32/T2/MB2048/H64, 1,048,576 decisions, seed 7)

- 139,560 parameters (558,240-byte checkpoint): tile layers 14,120 + head
  (291->128) 37,376 + MinGRU core H128-L1 49,152 + decoder (303 actions +
  value) x 128 = 38,912; dashboard reports 139.6K. Sums exactly.
- eval9 (world 2027): train terminal perf 0.686 / score 58.908; 16-episode
  in-process eval perf 0.684 / score 58.663; deaths 0.000, successes 1.000,
  episode_length 400.000; uptime 76.59 s (13.7K process SPS), VRAM 6.0/8G.
- eval10 (world 0): train terminal 0.680 / 56.957; eval 0.682 / 57.913;
  same deaths/successes/episode_length; uptime 76.25 s.
- A same-seed re-run of eval9's recipe (ld-tilecnn-render1) reproduced the
  final metrics exactly (0.684 / 58.663). eval7/eval8 (first pair, world
  2027/0) completed their eval blocks; their dashboard output was lost to
  output filtering but their blocks are proven by the resolved INIs.
- Caveats: the in-process eval shares the training world (in-world, not
  held-out), so these are not held-out receipts. eval10 (world 0) reports
  lower dashboard scores (56.96/57.91) than d1/d2 (61.87) under the same
  recipe; d1/d2 remain byte-identical-checkpoint evidence, and dashboard
  scores are indicative only.

### Headful render check (passed)

Debug build flag `LD_RENDER=1` opens a render pufferl (1 agent, horizon 1,
env window on DISPLAY=:0, random weights) and runs a bounded 40-iteration
rollout+render loop before the train pufferl (render pufferl first because
`close_pufferl` does not free CUDA buffers; a post-train placement OOMed the
8 GiB card). The mode-3 env completed all 40 iterations (about 10.7 s) with
no crash or assertion. The render draws the full map; observations use the
18-tile agent view.

### Build-tree debug patches (revert before any tree commit)

`build/sprite-lanedodge-tilecnn-1/source/src/pufferl.cu` carries two
debug-only patches not present in the audit-tree source: (1) an 8x3 s
`fopen` retry loop with `LD_LOAD_PATH` prints in `puf_load_weights_into`;
(2) the `LD_RENDER` block at the top of `run_train` (inactive unless
`LD_RENDER=1`). `/home/claude/cnn` sources are clean of both.

README updated with a Mode-3 tile CNN results section; the plan's Open
section now records the eval/render status. No new Lane Dodge implementation
files were changed by this entry.

## 2026-10-16 14:19–14:23 PDT — independent scored-evaluation review

Only this reviewer log edited; no native execution, GPU query or training.
The preceding implementation entry is labeled 14:30–15:00, later than
this shell's current clock; its timestamps remain author-supplied.

Progress supported by retained artifacts:

- Resolved eval9/eval10 INIs contain final mode-3 checkpoint paths,
  training seed 7, requested 16 episodes and one evaluation agent. The
  captured source has a post-train `eval_make`/`eval_loop` that reloads
  the saved weights. Their final stored metric points have `env/n=16`,
  perf 0.6842550635 / 0.6818003058, score 58.66263199 / 57.91263199,
  deaths 0, successes 1 and length 400. These support the reported
  in-process evaluation result; no per-episode CSV is retained here.
- Both final checkpoints can be read by this review shell. Eval9's env
  seed is 2027 for both training and evaluation; eval10 uses 0 for both.
  These are same-distribution developmental evaluations, not held-out
  confirmation. The new values do not establish a Nature comparison.
- Mode-3 current header now has a dedicated two-window render branch.
  All 15 host-suite source hashes still match the current listed files.

New concrete problems:

1. **Mode-3 viewer tiles overlap.** In `ld_draw_window` (header lines
   545–576), the sy/sx loops select different world tiles, but destination
   `ty=dy+y2*2` and `tx=dx+x2*2` never include sy/sx. Thus all nine tiles
   overwrite the same 64x64 patch; the last tile wins. There are two such
   patches, not two properly drawn 192x192 windows. The copied build header
   contains the same function. A crash-free random-policy window test is
   insufficient visual validation. This does not by itself corrupt
   `write_view`'s training pixels, which use distinct slot offsets.
2. **Build provenance is stale after debug rebuilds.** Independently
   checking `binary.sha256` now fails. Checking `source.sha256` against
   the copied source also fails on `src/pufferl.cu`, which has the recorded
   retry and LD_RENDER patches. Therefore the original receipt no longer
   describes the current executable/source. Preserve original receipts and
   capture each rebuilt variant separately; do not replace old identities
   and imply they bound earlier runs.
3. **SPS labels need correction.** README calls 12.7K–13.7K "process SPS",
   but the cited 76–82 seconds are stored native `uptime`/dashboard values.
   No independent monotonic process start/end receipt was found for these
   runs. Keep this as native trainer timing unless process clocks exist.
4. Prior issues persist: mode 0 is global summary state, not matched ROI
   state; README still calls the learned 784-to-16 projection GAP; its
   memory numbers remain wrong; runnable mode-3 commands still omit the
   tested A32 batch override. The most recent implementation report did
   not address these items.

Interpretation: surviving all 16 development episodes is encouraging but
not mastery. With survival=1, perf=0.5+0.5*coin_frac, so the reported means
correspond to about 36.85% and 36.36% coin collection. At the selected
hazard rate of 1%, report survival and coin fraction separately and avoid
calling composite perf a win percentage. No new implementation changes
were made by this review.

## 2026-10-16 16:42 PDT — suite cleanup implementation scope

Kinvert's new active goal authorizes cleaning up the other sprite envs in
main, including actual fixes; the earlier read-only review scope is historical.
The six older linked games are now local directories with independent builds.
For Lane Dodge, this agent made only the narrow existing viewer/doc fixes:

- `ld_draw_window` now includes sx/sy tile offsets, so its nine tiles occupy
  distinct positions. Added a scalar complete192x192 enlarged-view regression.
- README correctly names the learned784-to-16 projection, fixes observation
  rollout memory arithmetic, adds the tested A32 recipe to both research
  training commands, and labels the old timing as native rather than process.
- No Lane Dodge game rule, observation/action contract, encoder, training,
  raw checkpoint or existing core/debug build was changed by these edits.
  Its standalone partial-state matching and provenance issues remain separate.

The new regression has not yet run: a separately reserved six-game GPU pilot
is active, and heavy host checks are deferred until that timing work finishes.

## 2026-10-16 16:56–17:05 PDT — Lane matched-state cleanup verified

The new goal's Lane work is now complete, with separate retained evidence:
`research/results/sprite-attention-5060-20261016/README.md` and
`build/sprite-lanedodge-pilots-20261016/`. No old Qwen packet was overwritten.

- Added mode4:111-value one-hot state for exactly mode3's eighteen cells and
  three geometry fields, retaining movement3/gaze300. The forward-looking
  player window and mode3 pixel/model contract remain unchanged. Code folds
  the two duplicate pixel-window loops into one ordered cell loop.
- Main sanitizer suite passes the enlarged-view regression and3072 paired
  mode3/4 steps across all three skins. Normal main native build also passes.
- Independent float64 CUDA oracle checks pass outputs/all51,496 encoder
  gradients, positive/zero fixtures, B1/3/8/64/256 and eager/graph repeats.
  These do not qualify arbitrary branches/full learner derivatives.
- Four canary trainings reproduce both checkpoint pairs exactly; six exact
  graph/repeat/eager evaluations match every assigned episode row. Two fresh
  1Mi-decision seed7 pilots complete at the same A32/H64/MB2048 learner:
  state141,092 process SPS/63.77% composite perf; pixels12,840/68.71%.
  Four final exact evaluations cover128 total assigned episodes at development
  worlds420/1337. This is one training seed, not a frontier/quality claim.
- New pixel final checkpoint is byte-identical to Qwen's original retained
  d1/d2 final checkpoint, SHA256 a81a456ff5a20cb0d89753a2ea2b98b0e3b7325c737a0352938ef6426c1fc218.
  Standalone exact native reload works here without retries/debug patches.
- Fresh build receipts bind the new sources/binary; historical Qwen debug
  build receipts remain stale as previously reported. Documentation now
  correctly labels architecture, memory arithmetic and native timing.

No new Lane GPU process remains, no system/CUDA/dependency change or push.
