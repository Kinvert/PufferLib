# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 7.532 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | Architecture settings | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| quiet-tiger-1 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=32, residual_1=0, stride_1=4 | 32,768 | 0.09% | 0.54 | 60,681 | 80,296 | False | True |
| brave-bear-2 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=2, pool_1=2, projection=16, residual_1=0, stride_1=8 | 16,384 | 0.05% | 0.32 | 51,200 | 54,752 | False | False |
| calm-cat-3 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=1, kernel_1=4, kernel_2=2, pool_1=1, pool_2=0, projection=64, residual_1=0, residual_2=0, stride_1=8, stride_2=1 | 18,432 | 0.08% | 0.32 | 57,600 | 60,896 | False | False |
| quiet-tiger-4 | connect4-flex-v1 | channels_1=8, channels_2=16, depth=2, global_pool=0, kernel_1=6, kernel_2=5, pool_1=1, pool_2=2, projection=32, residual_1=1, residual_2=1, stride_1=4, stride_2=2 | 18,432 | 0.00% | 0.36 | 51,200 | 62,896 | False | False |
| gentle-tiger-5 | connect4-flex-v1 | channels_1=8, channels_2=16, depth=2, global_pool=0, kernel_1=5, kernel_2=2, pool_1=1, pool_2=2, projection=64, residual_1=0, residual_2=0, stride_1=4, stride_2=1 | 18,432 | 0.08% | 0.32 | 57,600 | 68,512 | False | False |
| merry-owl-6 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=1, kernel_1=3, kernel_2=3, pool_1=1, pool_2=0, projection=32, residual_1=1, residual_2=1, stride_1=8, stride_2=2 | 16,384 | 0.00% | 0.29 | 56,497 | 62,064 | False | False |
| lucky-bear-7 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=1, pool_1=0, projection=128, residual_1=1, stride_1=8 | 16,384 | 0.00% | 0.28 | 58,514 | 114,096 | False | True |
| bright-tiger-8 | connect4-flex-v1 | channels_1=32, channels_2=8, channels_3=16, depth=3, global_pool=1, kernel_1=7, kernel_2=1, kernel_3=1, pool_1=2, pool_2=1, pool_3=1, projection=32, residual_1=0, residual_2=1, residual_3=1, stride_1=4, stride_2=2, stride_3=1 | 16,384 | 0.00% | 0.31 | 52,852 | 59,856 | False | False |
| golden-maple-9 | connect4-flex-v1 | channels_1=8, channels_2=8, channels_3=16, depth=3, global_pool=0, kernel_1=4, kernel_2=4, kernel_3=3, pool_1=1, pool_2=2, pool_3=1, projection=128, residual_1=1, residual_2=0, residual_3=1, stride_1=8, stride_2=2, stride_3=2 | 18,432 | 0.00% | 0.31 | 59,458 | 57,592 | False | False |
| bright-otter-10 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=1, kernel_1=5, pool_1=1, projection=32, residual_1=0, stride_1=4 | 18,432 | 0.08% | 0.31 | 59,458 | 55,360 | False | True |
| happy-river-11 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=0, kernel_1=8, kernel_2=2, pool_1=2, pool_2=1, projection=64, residual_1=0, residual_2=0, stride_1=4, stride_2=2 | 16,384 | 0.00% | 0.29 | 56,497 | 64,736 | False | False |
| bright-falcon-12 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=32, residual_1=0, stride_1=4 | 18,432 | 0.08% | 0.35 | 52,663 | 80,072 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
