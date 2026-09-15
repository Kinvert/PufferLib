# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 3.221 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | Architecture settings | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| wild-river-1 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=32, residual_1=0, stride_1=4 | 32,768 | 0.09% | 0.52 | 63,015 | 80,296 | False | True |
| sunny-tiger-2 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=2, pool_1=0, projection=32, residual_1=0, stride_1=4 | 32,768 | 0.06% | 0.46 | 71,235 | 79,816 | False | True |
| wild-bear-3 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=4, pool_1=0, projection=32, residual_1=0, stride_1=4 | 32,768 | 0.06% | 0.47 | 69,719 | 79,912 | False | False |
| gentle-owl-4 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=32, residual_1=0, stride_1=4 | 32,768 | 0.04% | 0.48 | 68,267 | 79,984 | False | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
