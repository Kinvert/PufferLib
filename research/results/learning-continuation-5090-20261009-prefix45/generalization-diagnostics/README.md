# Generalization and matched-time discovery diagnostics

See the findings and sorted next actions in
[GENERALIZATION_PRIORITIES.md](../../../GENERALIZATION_PRIORITIES.md).

Derived offline from the transported45-observation prefix delivered at0010423d,
using the unchanged retainedb102678e training results. `diagnostics.json` binds
the source point/architecture/recipe files and retains all contribution,
leave-one-game-out and sampling-gap diagnostics. CSVs retain every47-model/six-game
matched-time comparison, coverage count and game contribution.

Select the latest observed checkpoint fitting each game's Nature final mean time,
not the best-scoring checkpoint. No interpolation or extrapolation; missing early
points remain unknown. These are adaptive two-seed descriptive comparisons, not
significance tests, true held-out results or proof that an architecture fails at
every unobserved time. Drawing rows are not independent training replicas.

Four new scalar/artifact tests plus the previous25 host/export/continuation tests
pass. No GPU query/training, CPU CNN, native/learner change, history rewrite or
automatic restart. The failed allocation and excluded partial46 remain unchanged.
