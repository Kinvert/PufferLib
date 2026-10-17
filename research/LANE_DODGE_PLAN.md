# Lane Dodge Plan (living document)

Governing plan for the Lane Dodge goal-mode task (2026-10-16). The reviewer
log is `research/LANE_DODGE_REVIEW_LOG.md` (append-only, reviewer-owned;
preserve all entries). This plan is fixed/updated as work proceeds.

## Goal

New "Lane Dodge" sprite env in `ocean/sprite_lanedodge` (template
`ocean/sprite_forage`):

- state ~90% perf, pixel CNN ~50% perf, each <= 5 min wall-clock training on
  RTX 5060.
- State vs pixel SPS + perf reported comparably.
- Dead-simple readable C; determinism is key; bulletproof benchmark env for
  future pixel-NN pareto fronts.
- No commit/push; no CUDA/torch/system changes; short serial GPU runs (the
  5060 is shared).

## Constraints

- `/home/claude/cnn` is the authoritative checkout (branch cnn-research);
  `ocean/sprite_lanedodge` is a real directory.
- Build: `source ocean/connect4cnn/runtime_env.sh && NVCC_ARCH=sm_120 bash
  build.sh sprite_lanedodge build/sprite_lanedodge/puffer --float`.
- Canonical rates: `hazard_rate=1, coin_rate=40` in
  `config/sprite_lanedodge.ini` — re-tuned on the consume-once coin
  semantics (the old h20c40 default was tuned under the repeated-collection
  pump metric).
- Canonical budgets: state 84M decisions (83,886,080, ~84s wall); pixel 300s
  allocation under `timeout 300` on the INI default 16.7M (LR anneals over
  16.7M, not over the truncated allocation).
- Pixel recipe A4/H64/MB64 (A64/MB256 needs 14.06 GiB for one rollout;
  A8/MB64 peaks near 8.4 GiB on the 8 GB card).
- Eval: worlds 2027/9001, 64 episodes, seed 7, rates matched to the
  training run, `--sweep.metric=perf`.
- Preserve every run; never pool pre-fix (pump-metric) numbers with fixed
  code. No seed sweeps.
- SKILL_ISSUES.md style (Joseph's guide): no forward declarations,
  headers-as-source, no bare `{}` blocks, 4-space indent, multi-line
  loops/conditionals.

## Done (with evidence)

1. Env implementation, encoder wiring (src/nature.cu +
   research/sprite_encoders.cu + src/ocean.cu sprite block), native build,
   host suite (16 sections, ASan+UBSan).
2. Reviewer findings fixed (log 03:45 entry): consume-once coin (arrival
   branch clears both rows), deterministic per-art-pixel shading + road
   compositing in `ld_make_tiles` (shade keyed to art column so the rep-2
   mirror test holds; no RNG consumption), `run.sh` evidence
   (`exit-status.txt`, nonzero exit on failure, 11-file `source.sha256`).
   Negative control: reverting the coin fix aborts exactly at the new
   assertion.
3. README pixel guidance corrected: A4/H64/MB64 with OOM rationale; the
   pixel train command explicitly overrides the inherited 256-agent state
   default; Rules section documents consume-once; results table split into
   fixed-code and historical pre-fix tables.
4. Rate re-tuning on the fixed game (state 84M, seed 7, matched-rate eval):
   h20c40 0.543 mean, h10c40 0.628, h5c40 0.755, h3c40 0.773, h2c40 0.796,
   h1c40 0.830 (selected), h0c40 0.877 (ceiling; removes the dodge element),
   h10c50 0.652, h5c60 0.694, h15c50 0.561, h1c60 0.760. INI default now
   h1c40.
5. Fixed-code canonical state h1c40: `ld-state-h1c40-84M-s7` wall 83.325s
   (1,006,745 SPS), repeat wall 82.907s, all 80 checkpoints byte-identical
   (0 cmp diffs; probe-run cross-check too), evals 0.823194/0.836414,
   eval double-run identical, weight sha `221721ac…`.
6. Fixed-code pixel h1c40: `ld-pixel-h1c40-300s-s7` wall 300.092s, last
   completed checkpoint 140,800 decisions (469.2 SPS), mirror
   `build/ld-pixel-h1c40/selected.bin` sha `48adf132…` (== run checkpoint),
   evals 0.407720/0.427634.
7. Docs: README updated, `research/results/sprite-lanedodge-5060-20261016/`
   results.md fixed-code section + plan.json rates block; state-run resolved
   INIs copied to `inis/`.

## Remaining (in order)

All complete as of 2026-10-16 04:12 PDT:

1. [x] Re-ran the host suite fresh (`build/lanedodge-tests-F4`, exit 0) and
   re-captured `source.sha256-final` (11 files, current h1c40 INI
   `57ea5d84…`, header `03760741…`, tests `08d0f4b4…`, run.sh `a0c88f12…`).
2. [x] Appended the reviewer-log 04:08 entry (h1c40 rate grid, state/pixel
   canonical runs, SPS + perf, provenance, limits). Five prior entries
   preserved untouched.
3. [x] Marked the three GPU-validation todos done (exact strings).
4. [x] Final report in-chat (2026-10-16 04:15 PDT); see review-log 04:08 entry.
5. [x] `goal complete` (2026-10-16 04:15 PDT).

6. [x] Reviewer 09:02 finding 1 (row-13 visibility): the observation scan
   now includes row 13 in other lanes (distance 1/14, nearest value); new
   two-step trace test (fixture r13: land in another lane → visible →
   lane change into it dies). The v2-smoke table is labeled pre-scan-fix
   smoke; README state section documents the row-13..0 scan.
7. [x] Mode7 2.5D render as `env.mode7` INI option, tuned through four
   rounds of user feedback (v1: both-axis pinhole `z = 360/(y-120)`; v2:
   both scales halved after "double the X width at the bottom, Y scaling
   twice as strong"; v3 current: Y strength doubled after "Y at the bottom
   WAY too tall"; round 4: shortened the far-band fade after the horizon
   smear was still visible). Current mapping: `dist = 2*(480-y)/(y-120)`
   clamped 13 (player row = bottom 120 screen px, y = 360..479; row-12
   boundary y=300; row 0 at y ≈ 168) and `zx = 180/(y-120)` clamped 7
   (bottom span 320 world px: the road fills the full screen width at the
   bottom). The far band (y = 120..165) is solid sky through y = 157,
   with the repeated top world row fading in over y = 158..165, so it
   does not read as a vertical smear of the lane dashes. Verified by
   measured road-edge spans within ±1 px (`build/ld-m7-visual.c` +
   `build/ld-m7/analyze.py`), fresh suite `build/lanedodge-tests-m7h`
   exit 0. Render pacing 16 FPS in both modes (window target 62.5 ms;
   192 renders/12.832 s vs 192/15.564 s counted), transform ~295 us
   hidden inside the frame budget, headless unaffected; `puf_close`
   NULL-client guard in the env header.
8. [x] Reviewer findings 2–5 addressed: development-world labels (README +
   results.md), explicit runnable recipes (1M smoke override; 84M state /
   300 s + 16.7M-budget pixel allocation), "descriptive, not controlled"
   disclosure, difficulty freeze (current INI h1c40/max_steps 400 = v1
   benchmark recipe); results.md reading-order block + historical section
   labels; finding 5 scope note (the `PUFFER_WEIGHTS_MIRROR` core writer
   stays a separate track from the env commit). Implementation entry
   appended to the review log (10:30 report); fresh suite
   `build/lanedodge-tests-m7y` exit 0; native rebuild OK.

Open (after the 09:02 review, 2026-10-16):

- Commit-scope decision: the row-13 scan fix, Mode7 (header/INI/README),
  the `puf_close` guard, test and results/README label updates, and the
  mode-3 tile-CNN work (header mode 3, config/tests/README, four
  `research/` tile-CNN files) are uncommitted on top of `64d69616`;
  awaiting user decision (default: keep uncommitted, no push).
- Confirmation allocation (new evaluation worlds + fixed training
  replicas 7/42/69/123/173/2027/9001) is not authorized; 2027/9001
  remain development worlds.
- Mode-3 tile CNN (18-tile agent view): implementation + host suite +
  same-seed determinism (d1/d2, four byte-identical checkpoints) are
  complete under `build/sprite-lanedodge-tilecnn-1`. Memory: A256/H64
  OOMs the 8 GiB card; the verified recipe is A32/H64/MB2048 (README
  updated). Evaluation: the standalone `puffer eval` binary cannot open
  checkpoints written by a separate `puffer train` process (sandbox
  file-visibility defect), so 16-episode evaluations use the in-process
  post-train eval (`--base.eval_episodes=16 --base.eval_agents=1`):
  world 2027 (eval9) 0.684/58.663, world 0 (eval10) 0.682/57.913, deaths
  0, successes 1.0, 13.7K process SPS, 139,560 params. These are
  in-world evals (the post-train eval shares the training world), not
  held-out. The headful mode-3 render check passed (debug `LD_RENDER=1`
  path, 40 rollout+render iterations on DISPLAY=:0, no crash).
  Remaining: a held-out `puffer eval` receipt is blocked on the sandbox
  defect; debug-build patches (load retry loop, LD_RENDER block) live in
  the build tree only and need reverting before any tree commit.

## Target status (honest)

- state ~90%: approached, not reached, at rates that keep hazards in play
  (0.83 at h1c40; h0c40 ceiling 0.88 but no hazards in play). The pre-fix
  0.97/0.96 was a pump-metric artifact (preserved, not pooled).
- pixel ~50%: 0.42/0.43 — approached from below at the canonical rate.
