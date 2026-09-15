# Flexible small CNN: INI controls

Encoder ID **4** adds a configurable stage family. IDs 0–3 keep their existing meanings and checkpoint layouts. Architecture construction and all forward/backward operations are native C/CUDA; the temporary launcher only prepares isolated INIs and reports native results.

## Available options

All values are numeric. `_1`, `_2`, and `_3` identify independent stages; only stages up to `cnn_depth` execute.

| Policy key | Allowed values | Effect |
|---|---|---|
| `cnn_depth` | 1, 2, 3 | Number of active stages |
| `cnn_channels_1/2/3` | 8, 16, 32 independently | Output channels of each stage |
| `cnn_kernel_1` | Any integer 1–8 | First convolution kernel size |
| `cnn_kernel_2/3` | Any integer 1–5 | Later convolution kernel sizes |
| `cnn_stride_1` | 4, 8 | Aggressive initial spatial downsampling |
| `cnn_stride_2/3` | 1, 2, 4 independently | Later convolution strides |
| `cnn_residual_1/2/3` | 0, 1 independently | Optional additional SAME 3×3 convolution with identity skip |
| `cnn_pool_1/2/3` | 0 none, 1 max, 2 average | Optional 3×3 stride-2 SAME pooling after the residual |
| `cnn_global_pool` | 0 flatten, 1 global average | Readout before the projection |
| `cnn_projection` | 16, 32, 64, 128 | Projection width; an additional linear layer maps to the fixed 128-unit core when needed |

Each stage is `ReLU(conv + bias)`, optionally followed by `ReLU(x + conv3x3(x) + bias)`, then optional pooling. SAME padding uses ceiling output dimensions and puts any odd extra padding on the trailing edge. All combinations have positive spatial dimensions, including repeated downsampling to 1×1. Average pooling excludes padded positions from its divisor. Max pooling chooses the first maximum in a fixed traversal order. Biases, residuals, and gradients use fixed-order operations; no atomic gradient accumulation was added.

This family's padding differs from the valid convolutions in Nature and compact ID 3. It does not reinterpret old checkpoints. Kernels smaller than strides can leave gaps in image coverage; pooling can discard spatial information. These are real architecture choices, not guaranteed improvements. Initial stride 4/8 and channel bounds keep the family focused on small networks and bound its activation/workspace sizes on this 8 GB GPU. ID 3 still offers initial stride 2.

## Choose what PROTEIN can change

Start from [sweep_flex.ini](../ocean/connect4cnn/sweep_flex.ini). Values in `[policy]` are the initial/fixed values. A corresponding `[sweep.policy.KEY]` section makes that one option searchable. **Delete a sweep section to fix that option.** The runner now accepts any nonempty subset of legal dimensions for any supported family; it no longer requires every shape option to vary.

For example, keep `cnn_depth=1`, all pooling/skips disabled, and sweep only kernel size:

```ini
[sweep.policy.cnn_kernel_1]
distribution = int_uniform
min = 1
max = 8
scale = auto
```

Widths/projection/strides use `uniform_pow2`; depth/kernel/pool/residual options use `int_uniform`. Narrow `min`/`max` to limit a search. To fix training duration too, omit `[sweep.train.total_timesteps]` and set `[train] total_timesteps`. A complete one-knob fixed-budget example lives in [tests/flex_kernel.ini](../ocean/connect4cnn/tests/flex_kernel.ini).

The full recipe exposes **18 architecture dimensions plus training budget**, with a default **128 trials**. Across depths it describes 168,585,984 active parameter configurations before considering training budgets; many behave similarly or poorly. That is search capacity, not validated coverage or evidence that 128 trials can optimize all dimensions. Prefer focused subsets and gradually widen them. PROTEIN still receives inactive-stage proposal coordinates when depth varies; architecture fingerprints and reports omit inactive stages so they do not falsely count as different constructed networks.

```bash
# Short infrastructure validation; does not launch the 128 learning-scale trials.
NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1 bash ocean/connect4cnn/sweep.sh \
  --recipe ocean/connect4cnn/sweep_flex.ini --canary --max-runs 12 --wandb disabled

# Example learning-scale launch after selecting ranges and checking GPU load.
# Run detached; do not actively watch it. Four hours is a campaign hard deadline.
NVCC_ARCH=sm_120 OPENBLAS_NUM_THREADS=1 bash ocean/connect4cnn/sweep.sh \
  --recipe ocean/connect4cnn/sweep_flex.ini --max-runs 128 --timeout 14400 \
  --wandb online --project puffer-cnn --entity kinvert-k
```

No 128-trial training campaign is implied by creating this recipe. Existing reports remain immutable. CSV `architecture_json`, sidecar payloads, W&B configuration, and source/config snapshots preserve every selected active option. Native SPS, uptime, steps, performance, and losses retain the existing metric names.

## Validation scope

The numerical suite covers 78 targeted configurations: individual option values, mixed stage widths/kernels/strides, max/average/GAP gradients, residual gradients, even/odd borders, tied max-pool inputs, tiny 1×1 maps, the largest dense layout, and seeded random combinations. Each compares forward/all parameter gradients with independent float64 NumPy results and verifies eager/graph repeatability plus rollout parity. Seven nonblank configurations also use finite differences across every parameter array. It does not exhaustively test the entire search space. Nature regression and all 54 previous compact cases are retained. Float32 only; BF16 remains unvalidated.

The learning recipe now sweeps 2,097,152–13,279,232 requested timesteps, starting with 13,279,232 to obtain a longer-budget observation first. The predicted suggestion-cost ceiling is 600 seconds; this is not a hard per-trial timeout. The earlier 0.83M–6.64M campaign is retained as a separate experiment. Actual steps follow native batch rounding. Learning-rate annealing depends on each trial's total budget, so a short completed run is not equivalent to an intermediate checkpoint of a longer run.

Read `EXPERIMENT_LOG.md` for executed canaries, checkpoint/reload checks, and memory-bound validation. Baseline learning comparisons still require fixed longer-budget Nature controls and multiple seeds; wider search control does not remove that requirement.
