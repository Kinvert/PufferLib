# Lane Dodge matching partial-state control

October16 suite cleanup. Existing mode3 RGB exposes18 tiles: a3x3 player
window looking two rows ahead through the player row, then the3x3 gaze.
New mode4 exposes one-hot identities for exactly those cells plus identical
lane/gazeX/Y geometry.111 state values versus55,299 RGB values. Both retain
the existing movement3/gaze300 action heads. This intentionally preserves
the earlier road-specific interface, distinct from the six-game suite's
centered windows/factoredX/Y/relative-offset interface.

Mode0 global nearest-object state and all old observation modes remain
available. Mode3 pixel bytes and policy shapes are intended unchanged;
mode4 has a distinct102,272-parameter policy at H128/L1. Mode3 tile-CNN
policy remains139,560 parameters. Never load either state's weights into
the pixel model or interpret old full-state/ROI-linear runs as this pair.

`build_lanedodge_tilecnn.sh` now captures the existing exact per-episode,
all-action-head evaluator under its research macro. This avoids any training
process reload workaround or ad-hoc core debug patches. Old Qwen builds and
checkpoints/receipts remain frozen; new builds need fresh output directories.

Verification sequence: scalar sanitizer tests including full enlarged-view
pixel placement and matched mode3/4 trajectories; compatible fresh research
build; repeated short training/graph-repeat-eager exact evaluation; then
fixed-seed7 matched1Mi-decision state/pixel pilots at the retained A32/H64/
MB2048 learner,32 assigned episodes per development suite420/1337,400-step
cap and16 slots. Record monotonic process cost/SPS separately from native
timers, every final checkpoint identity and all exact episode rows. Keep
all failures, no seed sweep, no automatic retry of native failures.

This is an engineering/learning pilot, not a held-out quality advantage or
frontier claim. Existing numerical scope of the18-tile/three-geometry Lane
encoder remains distinct from the six-game eighteen-tile/four-geometry
encoder checks. No system/CUDA changes or CPU neural computation.
