# Six new sprite benchmarks

October16 cleanup and matching local/gaze learning results are complete:
[seven-game report](research/results/sprite-attention-5060-20261016/README.md),
[before/after cleanup](research/results/sprite-cleanup-5060-20261016/README.md).
The original six plus Lane Dodge have actual state/pixel training receipts.
Forage remains the reference environment with its separate retained experiments.

The authoritative implementation is now **/home/claude/cnn**. October16:
the six game directories and `sprite_common` are real local directories,
with their matching INIs copied here. Make edits here. Older worktree copies
and completed experiment packets remain preserved and are not auto-updated.

The active cleanup/local-view plan is
[SPRITE_CLEANUP_PLAN.md](research/SPRITE_CLEANUP_PLAN.md). Compatible frozen
core snapshots may be used for builds, copying current main game sources.

| Game | Current local source |
|---|---|
|Forage reference|[sprite_forage.h](ocean/sprite_forage/sprite_forage.h)|
|Catch|[sprite_catch.h](ocean/sprite_catch/sprite_catch.h)|
|Platform|[sprite_platform.h](ocean/sprite_platform/sprite_platform.h)|
|Maze|[sprite_maze.h](ocean/sprite_maze/sprite_maze.h)|
|Runner|[sprite_runner.h](ocean/sprite_runner/sprite_runner.h)|
|Key|[sprite_key.h](ocean/sprite_key/sprite_key.h)|
|Push|[sprite_push.h](ocean/sprite_push/sprite_push.h)|
|Lane Dodge|[sprite_lanedodge.h](ocean/sprite_lanedodge/sprite_lanedodge.h)|

Start with [current scoped results](/home/claude/cnn-audit-20261008/research/results/sprite-solid-5060-20261015/README.md)
or [native build/train/eval instructions](/home/claude/cnn-audit-20261008/ocean/sprite_common/README.md).
The [first-release results and gallery](/home/claude/cnn-audit-20261008/research/results/sprite-suite-5060-20261015/README.md)
retain their original game versions.

The [native per-episode evaluator checks](/home/claude/cnn-audit-20261008/research/results/sprite-exact-5060-20261015/README.md)
cover all seven sprite games in state/pixel mode and all23 drawing IDs.

The [full-resolution encoder adapter checks](/home/claude/cnn-audit-20261008/research/results/sprite-encoders-5060-20261015/README.md)
cover21 native builds, Nature/Flex CUDA encoder math and Runner checkpoint
compatibility. These are validation checks, not a new training sweep or a
quality/frontier claim. Future comparisons use seven common fixed training
seeds; seeds are never optimized.

Games: sprite_catch, sprite_platform, sprite_maze, sprite_runner, sprite_key,
sprite_push. Every game offers state or640x480 RGB through its INI. New run
instructions will specify the captured core and runtime; historical worktree
commands retain their original scope.
