# Why the full-resolution custom run is slow

Investigation uses the existing completed run and captured native sources;
no additional GPU run or model change is needed for these findings.

## The small-image graph became a large model

The first7x7/s4 C16 convolution produces120x160x16 at640x480. Flattening that
to307,200 features and projecting to64 creates19,660,800 weights. This one
matrix is99.69% of the19,721,472 full-policy parameters, excluding its bias.
The original roughly160K-parameter small-image quality graph therefore does
not remain small when its input is expanded to human-visible resolution.
The architecture should have been checked for this before longer training.

## Tiny minibatches make that matrix especially expensive

The run used8 agents, H8 and MB8:64 decisions per rollout and8 optimizer
updates per rollout. Across221,184 decisions there are27,648 full optimizer
updates. This is a deliberately memory-conservative smoke configuration,
not an efficient established full-resolution training recipe.

Native PufferLib uses Muon. In captured src/algo.cu, muon_step applies five
Newton-Schulz iterations to every matrix parameter. The projection matrix is
64x307,200. Its two large matrix products per iteration imply about25.17
GFLOPs per optimizer update for that matrix alone, counting multiply/add as
two FLOPs, plus copies, reductions and updates. This is theoretical operation
count from the code, not a measured kernel-time attribution. The FLOP total
includes the small Gram-square products and excludes the rest of training.
Batch8 pays this work every8 decisions. No proposal here changes the native
optimizer or silently switches precision to make a comparison look faster.

## Existing timing counters locate the broad costs

The retained metrics INI has five downsampled timing buckets. Ratios of the
sums of those bucket averages give this indicative distribution:

| Native timing component | Approximate share |
|---|---:|
| Training, including forward/backward and Muon | 74.4% |
| Rollout model | 18.6% |
| Rollout copies | 6.25% |
| Environment | 0.15% |

The denominator is perf/train plus perf/rollout. Rollout subcomponents do not
exactly sum to rollout, and these are downsampled buckets, not reconstructed
totals over612 seconds. Training includes both network and optimizer; the
counters do not separate Muon from the other GPU work. Kernel-specific
dominance still requires a short dedicated profile. No measured Muon-time
percentage or unexplained native final-bin slowdown is inferred here.

Dirty-cell drawing speeds the CPU raster stage. The GPU still receives a full
frame and recomputes the CNN; the training gradients and optimizer still touch
the large parameter matrix. The pixel-generation microbenchmark is therefore
not a forecast of end-to-end training SPS. This pipeline is already native
C/CUDA; Python only supervises processes and receipts.

## Priorities before another longer training

1. Keep640x480 input but reduce spatial features inside the encoder before
   flattening. Existing configurable convolution strides/depth/pooling can do
   this. Preserve useful spatial information; global pooling alone can discard
   the player/object locations needed for movement. Do not silently downsample
   observations or remove the intended sprite-detail challenge.
2. Once the model fits comfortably, test larger minibatches and vector sizes
   within5060 VRAM using bounded memory/throughput checks. Larger batches reduce
   full Muon updates per decision but also change optimization; report that
   difference and recheck learning rather than declaring a free speedup.
3. Profile a small representative workload to separate projection, Muon,
   convolution patch materialization, recurrent work and transfers. Use the
   same native optimizer and fixed seed; retain old runs unchanged.

No new configuration is selected or trained by this writeup. Numerical and
full-policy memory/learning qualification must match any newly chosen graph
and batch; existing B1/2/8 checks do not certify arbitrary larger shapes.
Raw arithmetic and timing summary: slowdown-diagnosis.json beside this file.
