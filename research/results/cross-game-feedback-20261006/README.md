# Native cross-game feedback: compiled and prepared, not GPU-executed

October 6, 2026, G240 checkout. Read
[the contract/instructions](../../CROSS_GAME_PROTEIN_FEEDBACK.md).

Implemented temporary native PROTEIN feedback with a macro-gated core bridge and
external preparation/supervision/scalar audit glue. Same CNN architecture across
all six games, independent random weights per game/seed, fixed per-game learners,
all 41 exact drawing evaluations, normalized equal-game quality and monotonic
training cost charged once per game/seed. No Python optimizer/CPU neural reference.

Actual verification:

- 43 host/scalar/mock tests pass; mocked three-panel adaptive supervision and
  offline audit verify complete feedback, duplicates and changed-history rejection.
- Three native-host tests pass: descriptor, nine malformed ledgers and one
  fractional control rejection (11 native process calls), all before CUDA queries.
- Native research bridge compiles; first compiled build and later tightened
  parser/command-receipt build are both retained.
- Six normal default CNN targets compile and return native host identities.
- Three standalone policy metadata targets and original state Connect4 compile;
  vanilla Connect4's host identity and binary hash are retained separately.
- Actual native descriptor, source/build/metadata/recipe preparation and offline
  inspection pass. Source closure covers owned files, not all vendor/system links.

Native bridge: `build/cross-game-feedback/native-v2-20261006/`.
Normal builds: `build/cross-game-feedback/games-20261006/registry.json`.
Metadata: `build/cross-game-feedback/metadata-20261006/`.
Prepared packet: `build/cross-game-feedback/prepared-20261006/plan.json`.
These are local artifact paths; remote agents regenerate rather than reuse them.

Declared maximum smoke: three trials × six games × one seed, 65,536 decisions/job,
one checkpoint, 17 exact episodes/16 slots per drawing; 18 training jobs,
123 evaluations, 2,091 assigned episodes. **Zero feedback trials/policies/GPU
queries executed.** Three C8/C16 trials necessarily duplicate an architecture.
The tiny anchors/caps are development declarations, not calibrated quality claims.

Initial test failures are retained with final passing logs. They exposed a missing
config option check and an incomplete synthetic fixture, not model failures.
Earlier completed mini/numerical packets are unchanged. Both GPU holds remain;
no long campaign, 5090 job, system/CUDA/dependency change, dataset or push.
Temporary core changes are not the eventual upstream PR. No learning, broad
math/runtime acceptance, deterministic GPU replay or Pareto advantage is claimed.

`receipts/` contains the actual build/test/preparation/inspection logs and native
descriptor. Raw build and prepared source/binary receipts remain at the local
paths above; preserve them and any future failure output unchanged.
