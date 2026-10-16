# Sprite Forage pixel training, October 15, RTX 5060

One newly authorized short native pixel training and all four assigned
evaluations completed successfully. The current main environment includes
changed-cell movement/reset copies. It is copied into the isolated build;
the compatible native core is retained from the preceding state timing
packet. Main core sources were not edited. No prior allocation was restarted.

| Measurement | Value |
|---|---:|
| Decisions | 32,768 |
| Native training process seconds | 147.219 |
| Process SPS | 222.58 |
| Native adjusted-average SPS | 223.38 |
| Native final-bin SPS | 224.71 |
| Parameters | 34,991,136 |
| Final training-bin perf | 0% |
| Full train/eval campaign seconds | 167.094 |

The encoder is the built-in **Nature CNN**, randomly initialized: valid
8x8/s4 C32, 4x4/s2 C64, 3x3/s1 C64 convolutions and a flatten/linear H128
projection, followed by the standard H128/L1 policy. Full 640x480 RGB
float32 observations; no state input or downsampling. It is not our Flex
research architecture and not pretrained.

Fixed policy seed 7, training world offset 0, drawing 0, 8 agents, one buffer,
two environment threads, horizon 8, minibatch 8, starting learner LR .001.
Remaining settings/rewards come from the Sprite Forage INI and native defaults;
the exact merged resolved.ini is retained. Seeds are never swept.

## Evaluation

Initial and final policies use the same 16 sequential episodes per development
world offset, one slot, fixed action/policy seed 7, drawing 0, 512-step cap.
This is the legacy native serial aggregate evaluator, not the separate assigned
episode acceptance protocol. Initial policy means random weights, not uniform
random actions. These world offsets have prior development use and are not
new held-out confirmation seeds.

| World offset | Untrained perf | Final perf |
|---|---:|---:|
| 420 | 9.375% | 10.9375% |
| 1337 | 3.125% | 7.8125% |
| Mean over 32 episodes | 6.25% | 9.375% |

This is weak performance after a tiny budget, with one independent training
seed. The small increase does not establish reliable learning or convergence.
No perfect-episode or win-rate claim follows from this collection-based perf.
No architecture/frontier/state-pixel superiority claim is supported.

All four checkpoints (8,192/16,384/24,576/32,768 decisions) have expected length
139,964,544 bytes, finite float32 values and distinct SHA256 hashes. The final
scheduled checkpoint is evaluated without score-based selection. Native
training exits zero; all 64 assigned initial/final evaluation episodes finish.
No independent full-learner gradient/math check is claimed by this pilot.

The earlier state comparison remains 742,328 / 745,912 process SPS with
identical checkpoints and 97.65625% perf. State used 256 agents/H64/MB2048
and a tuned learner versus this 8-agent/H8/MB8 pixel starting recipe. This is
not an apples-to-apples throughput or quality comparison. The preceding
1.76x changed-cell result measures only native game/pixel generation; CNN
forward/backward and full-image upload costs remain in this training run.

## Evidence

Raw packet: `build/sprite-forage-pixel-train-5060-20261015/`. Includes captured
owned sources, binary, runtime INIs, plan, requested commands, GPU-idle queries,
monotonic process times, native metrics, resolved recipe and checkpoint hashes.
The shared benchmark lock is held around native policy work. No active TUI
streaming, competing job launch, automatic retry/extension, system/CUDA/Torch
change, 5090 job, commit or push. Existing CUDA/NCCL/runtime only.

Read `results.json` and `evaluations.json` here for compact copied receipts.
Raw checkpoints remain ignored under build and are not repository additions.
