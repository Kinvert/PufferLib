# Sprite Forage connectivity-check ablation

Requested October16: measure whether reset-time connectivity rejection is
worth keeping. Main environment remains unchanged pending the results.

Run on the local RTX5060, not the RTX5090. Full state layout2, 141 inputs,
H128/L1, 67,968 parameters, 256 agents, horizon64, minibatch2048, four
threads, synchronous float32. Learner and rewards come from the current
`config/sprite_forage.ini`. Both variants use fixed training seed7 and
8,388,608 decisions, with the same LR annealing and four checkpoint times.
No seed or hyperparameter sweep.

The connected variant is the current header. The unchecked variant replaces
placement's rejection loop with one random-empty-cell placement, leaving the
game, observations, policy and learner unchanged. Both compile against the
same captured compatible core used for the earlier state-cleanup comparison.
New snapshots and binaries live under
`build/sprite-forage-connectivity-20261016`; old evidence is not overwritten.
Neither variant enables attention or runs a CNN: this isolates the proposed
map-generator cleanup with the fast state learner first.

Four serial trainings run connected/unchecked/unchecked/connected. Repeats
use the same seed and must match every scheduled checkpoint within a variant;
they are timing/determinism repeats, not independent training replicas.
Whole-process monotonic timings include startup and checkpoint writes.
Native adjusted uptime and dashboard SPS are recorded separately.

The first final policy from each variant is evaluated on **both** generators.
Each condition has 64 exact assigned episodes at each development suite seed
420/1337, 16 slots, cap676: eight evaluations and512 episodes in total.
Policies receive identical initial world/observation hashes and policy RNG
assignments within each generator's evaluation suite. Removing rejection
changes world generation/RNG consumption, so equal seed numbers do not imply
equal worlds across the two generators. Do not pool those as identical maps.
All episodes count, regardless of completion order; final checkpoints are
selected by the fixed decision budget, never by score.

Shared GPU reservation and idle checks precede native runs; no concurrent
heavy CPU work or active TUI monitoring. Timeouts are45s per training,30s per
evaluation,480s for the campaign. Failures and partial receipts remain, with
no automatic retry. These are single-training-seed development results, not
a robust/generalization/Pareto claim.

Runner: `research/forage_connectivity_ablation.py prepare` then `run`, with
the existing runtime helper sourced for execution. Preparation expects the
fresh captured connected/unchecked builds; it refuses an existing plan.
Do not rerun an already started or completed allocation.
