# Compact CNN search and Nature control

The first residual-family discovery sweep found high training returns at about 9,000 SPS. Its family retained large spatial maps and could not choose Nature's early striding. This phase exposes cheaper structures while retaining the same game, pixels, learner, and hidden-128 single-layer recurrent core.

## Architectures

Numeric encoder IDs are fixed per campaign; PROTEIN does not optimize a categorical family ID.

| ID | Family | Search |
|---|---|---|
| 0 | Existing compiled default/reference | Existing comparison tooling |
| 1 | Residual CNN | Original channels/blocks/GAP/budget sweep |
| 2 | Nature | Budget only; exact existing adapted Nature architecture |
| 3 | Compact | Depth, channels, initial stride, projection, and budget |

Compact layers use valid convolutions: first kernel 8 with stride 2 or 4, optional second kernel 4/stride 2, optional third kernel 3/stride 1. Depth is 1/2/3; channels are `[c,2c,2c]` truncated to depth with `c=8/16/32`. Flatten feeds a 32/64/128-unit linear projection; if that differs from the fixed recurrent width, one additional linear layer maps it to 128. Bias and ReLU follow every layer. This gives 54 configurations. The 32-channel, depth-3, stride-4, projection-128 case is exactly Nature, including parameter layout and initialization order.

The initial compact candidate uses channels 8, depth 1, stride 4, projection 32. It requires 65,536 encoder multiply-adds per observation versus Nature's 647,168, excluding activations and the recurrent core. This is an operation count, not a measured speedup. Shallow flattening can make a large projection expensive, so fewer layers do not automatically mean less work.

Both families share the existing native Nature convolution kernels. The extension changes construction and layer count, retaining the same im2col/matrix-multiply implementation, fixed buffers, bias/ReLU, and deterministic backward gathers. It introduces no new CUDA kernel algorithm or system dependency. Kernel profiling and optimization should follow useful candidate measurements; any shared speed improvement must also benefit the Nature control.

## First campaigns

`sweep_nature.ini`: 12 Nature trials. `sweep_compact.ini`: 24 compact trials. Both use seed 73, the same learner/core, a 3,319,808-decision starting budget and a shared 829,952–6,639,616 budget range. Native batch rounding applies. Checkpoints are saved every 500 updates plus final, and histories retain 25 points. The predicted suggestion-cost ceiling is 300 seconds, not a hard worker timeout. Each campaign has a four-hour whole-sweep deadline.

```bash
# Check GPU availability, then launch detached. Do not actively monitor training.
NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1 bash ocean/connect4cnn/sweep_small.sh

# Short plumbing checks for either family, retaining its shape dimensions.
NVCC_ARCH=sm_120 bash ocean/connect4cnn/sweep.sh \
  --recipe ocean/connect4cnn/sweep_compact.ini --canary --max-runs 12 --wandb offline
```

`sweep_small.sh` runs Nature and compact serially on one GPU. Each campaign uses the same isolated native PROTEIN runner and online W&B sidecar. Parent logs are under `build/connect4cnn/fast-search.*`; child logs identify their unique `sweep.*` directories and W&B groups. Native SPS, uptime, steps, perf, scores, and losses retain their usual names. CSVs distinguish inactive dimensions instead of attributing inherited residual settings to Nature or compact policies.

These are discovery campaigns, not equal-total-compute competitions: the 54-shape family receives more trials than the one-shape control. Compare observed return/time frontiers, then confirm candidates with matched budgets, held-out evaluation, and repeated seeds. Short budgets may have no useful learning signal. Success on this enlarged Connect4 board alone does not establish a general pixel-RL improvement.

## Validation

`test_nature.py` checks the original architecture against its independent float64 reference. `test_compact.py` checks all 54 configurations against an independent float64 reference, all parameter gradients, eager/graph repeatability, rollout parity, and selected finite differences. The Nature-equivalent compact configuration must match the Nature path bit-for-bit. CPU tooling tests check family-specific search isolation and matching learner/budget settings.

Validation results and launch identifiers are recorded in `EXPERIMENT_LOG.md` after execution. BF16 is unvalidated; these runs use float32.
