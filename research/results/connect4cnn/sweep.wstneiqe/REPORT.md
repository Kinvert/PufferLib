# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 2.720 seconds.
Same Connect4 rules and image dimensions; representation, architecture and budget follow the saved config. Learner/core recipe is locked.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Report Pareto flags use rounded final observations within each representation; native PROTEIN still optimizes the joint search objective.

| Run | Family | Representation | Architecture settings | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| happy-panda-1 | connect4-flex-v1 | 0 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 32,768 | 0.02% | 0.52 | 63,015 | 160,736 | False | True |
| bright-maple-2 | connect4-flex-v1 | 2 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 32,768 | 0.02% | 0.49 | 66,873 | 160,736 | False | True |
| happy-bear-3 | connect4-flex-v1 | 4 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 32,768 | 0.04% | 0.48 | 68,267 | 160,736 | False | True |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
