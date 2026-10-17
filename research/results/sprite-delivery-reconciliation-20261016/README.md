# Sprite overnight delivery reconciliation

Merged source revision: `ded075f4`.
Parents: sprite implementation `d4527b4e` and existing Kinvert remote
`ba7a9b164e851925db283baa4c55c46f8d9d1e45`. Both histories are retained.
The primary `/home/claude/cnn` checkout was advanced to this merge; unrelated
profiling-pointer, Pong documentation and optional weight-mirror edits remain
uncommitted. Their original patches/files are preserved locally under
`build/sprite-delivery-local-20261016`.

## Merge decisions

- Kept remote image contracts, generic Nature dimensions, first-stage Flex
  strides 1/2/4/8, image initial-checkpoint support and ForageCNN routing.
- Added the sprite encoder routing alongside that remote routing. Existing
  image and sprite flags remain separate.
- Kept remote IMPALA maximum-over-stages scratch allocation. At packed
  192x96 input this is the same allocation as the earlier sprite formulas:
  294,912 gradient elements and 73,728 skip elements per batch member.
- Retained both sets of research reports and agent instructions. Old source
  hashes and completed GPU packets were not rewritten or restarted.

## Verification of merged source

Executed without GPU discovery or training:

```bash
bash research/run_sprite_overnight.sh prepare build/sprite-delivery-merged-20261016
```

Preparation passed ten supervisor tests, twelve old/packed raster sanitizer
executions, all six native sprite builds, the independent CUDA harness build,
and actual full overnight-plan preparation with seven fixed replicas. The
builder captures committed merged core; the unrelated weight-mirror edit is
excluded. Native PROTEIN host checks passed four tests. Additional sprite/
image compatibility tests ran 39 cases: 35 passed and four optional native
fixtures were skipped. Logs and source/binary hashes are in `receipts/`.

This is publication compatibility evidence, not another learning or CUDA
numerical result. Existing scoped GPU evidence remains in
`../sprite-overnight-5060-20261016/README.md`. The full 5090 run is still
PREPARED, NOT RUN; its launcher performs fresh GPU gates before training.

Read [the 5090 instructions](../../../NEXT_5090_SPRITE_BENCHMARKS.md), pull
Kinvert's `cnn-research`, and prepare a fresh directory on the idle 5090.
