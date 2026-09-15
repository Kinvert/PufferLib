# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 14950.940 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | Architecture settings | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| calm-otter-1 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=32, residual_1=0, stride_1=4 | 13,277,184 | 71.94% | 144.50 | 91,884 | 80,296 | False | False |
| wild-maple-2 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=2, pool_1=2, projection=16, residual_1=0, stride_1=8 | 2,414,592 | 0.07% | 26.90 | 89,762 | 54,752 | False | False |
| clever-cat-3 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=1, kernel_1=4, kernel_2=2, pool_1=1, pool_2=0, projection=64, residual_1=0, residual_2=0, stride_1=8, stride_2=1 | 2,570,240 | 0.15% | 28.90 | 88,936 | 60,896 | False | True |
| lucky-otter-4 | connect4-flex-v1 | channels_1=8, channels_2=16, depth=2, global_pool=0, kernel_1=6, kernel_2=5, pool_1=1, pool_2=2, projection=32, residual_1=1, residual_2=1, stride_1=4, stride_2=2 | 2,553,856 | 0.06% | 35.17 | 72,615 | 62,896 | False | False |
| merry-robin-5 | connect4-flex-v1 | channels_1=8, channels_2=16, depth=2, global_pool=0, kernel_1=5, kernel_2=2, pool_1=1, pool_2=2, projection=64, residual_1=0, residual_2=0, stride_1=4, stride_2=1 | 2,560,000 | 0.11% | 30.52 | 83,879 | 68,512 | False | False |
| wild-maple-6 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=1, kernel_1=3, kernel_2=3, pool_1=1, pool_2=0, projection=32, residual_1=1, residual_2=1, stride_1=8, stride_2=2 | 2,193,408 | 0.05% | 26.28 | 83,463 | 62,064 | False | True |
| lucky-robin-7 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=1, pool_1=0, projection=128, residual_1=1, stride_1=8 | 2,332,672 | 0.13% | 26.38 | 88,426 | 114,096 | False | True |
| swift-maple-8 | connect4-flex-v1 | channels_1=32, channels_2=8, channels_3=16, depth=3, global_pool=1, kernel_1=7, kernel_2=1, kernel_3=1, pool_1=2, pool_2=1, pool_3=1, projection=32, residual_1=0, residual_2=1, residual_3=1, stride_1=4, stride_2=2, stride_3=1 | 2,211,840 | 0.02% | 28.32 | 78,102 | 59,856 | False | False |
| silver-river-9 | connect4-flex-v1 | channels_1=8, channels_2=8, channels_3=16, depth=3, global_pool=0, kernel_1=4, kernel_2=4, kernel_3=3, pool_1=1, pool_2=2, pool_3=1, projection=128, residual_1=1, residual_2=0, residual_3=1, stride_1=8, stride_2=2, stride_3=2 | 2,584,576 | 0.14% | 29.75 | 86,877 | 57,592 | False | False |
| silver-tiger-10 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=1, kernel_1=5, pool_1=1, projection=32, residual_1=0, stride_1=4 | 2,476,032 | 0.07% | 28.62 | 86,514 | 55,360 | False | False |
| swift-fox-11 | connect4-flex-v1 | channels_1=16, channels_2=16, depth=2, global_pool=0, kernel_1=8, kernel_2=2, pool_1=2, pool_2=1, projection=64, residual_1=0, residual_2=0, stride_1=4, stride_2=2 | 2,185,216 | 0.07% | 26.71 | 81,813 | 64,736 | False | False |
| gentle-fox-12 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,017,088 | 73.54% | 137.22 | 94,863 | 109,648 | True | False |
| silver-robin-13 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,954,176 | 69.68% | 126.36 | 94,604 | 109,648 | True | False |
| calm-river-14 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,357,632 | 76.42% | 131.07 | 94,283 | 109,648 | True | False |
| gentle-panda-15 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=6, pool_1=1, projection=64, residual_1=0, stride_1=4 | 8,880,128 | 18.67% | 101.40 | 87,575 | 74,216 | True | False |
| mystic-cedar-16 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,335,680 | 71.72% | 119.00 | 95,258 | 160,528 | True | False |
| brave-badger-17 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 8,437,760 | 43.07% | 96.69 | 87,266 | 160,976 | True | False |
| quiet-tree-18 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=1, projection=32, residual_1=0, stride_1=4 | 13,277,184 | 58.70% | 144.84 | 91,668 | 70,208 | True | False |
| sunny-comet-19 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=16, residual_1=0, stride_1=4 | 4,411,392 | 0.46% | 50.59 | 87,199 | 65,560 | True | True |
| golden-river-20 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 8,321,024 | 56.16% | 93.11 | 89,368 | 160,736 | True | True |
| golden-owl-21 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=1, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,907,648 | 0.08% | 120.95 | 90,183 | 52,768 | True | False |
| cosmic-falcon-22 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 80.43% | 145.06 | 91,529 | 160,736 | True | False |
| cosmic-cat-23 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=1, projection=32, residual_1=0, stride_1=4 | 9,947,136 | 19.22% | 114.22 | 87,088 | 70,832 | True | False |
| merry-maple-24 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=128, residual_1=0, stride_1=4 | 9,701,376 | 59.86% | 108.11 | 89,736 | 254,096 | True | False |
| lucky-falcon-25 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 5,955,584 | 5.90% | 68.06 | 87,505 | 109,768 | True | True |
| golden-tiger-26 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=1, projection=128, residual_1=0, stride_1=4 | 10,053,632 | 32.03% | 111.08 | 90,508 | 112,784 | True | False |
| calm-tree-27 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,024,384 | 81.62% | 121.14 | 91,005 | 160,736 | True | False |
| silver-comet-28 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 81.59% | 145.86 | 91,027 | 160,736 | True | False |
| lucky-badger-29 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 78.82% | 141.13 | 94,078 | 160,528 | True | False |
| merry-cat-30 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,088,448 | 54.94% | 107.74 | 93,637 | 160,528 | True | False |
| golden-panda-31 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,889,792 | 62.57% | 108.94 | 90,782 | 109,544 | True | False |
| cosmic-bear-32 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,975,232 | 60.30% | 122.88 | 89,317 | 160,976 | True | False |
| silver-wolf-33 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,156,032 | 61.71% | 114.57 | 88,645 | 160,976 | True | False |
| happy-bear-34 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=8 | 7,239,680 | 14.66% | 76.63 | 94,476 | 90,320 | True | True |
| sunny-bear-35 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.61% | 146.19 | 90,821 | 160,976 | True | False |
| brave-cat-36 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=1, projection=64, residual_1=0, stride_1=4 | 5,969,920 | 1.30% | 69.95 | 85,346 | 90,080 | True | False |
| gentle-falcon-37 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 78.69% | 145.29 | 91,384 | 160,736 | True | False |
| silver-panda-38 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 79.91% | 141.90 | 93,567 | 160,528 | True | False |
| brave-tiger-39 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 81.44% | 145.05 | 91,535 | 160,736 | True | False |
| swift-comet-40 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.51% | 146.01 | 90,933 | 160,976 | True | False |
| lucky-maple-41 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,432,512 | 64.25% | 117.58 | 88,727 | 160,976 | True | False |
| sunny-maple-42 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.51% | 145.40 | 91,315 | 160,976 | True | False |
| silver-cedar-43 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,439,552 | 85.69% | 135.83 | 91,582 | 109,768 | True | False |
| silver-tiger-44 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,817,536 | 77.39% | 119.07 | 90,850 | 109,768 | True | False |
| lucky-cat-45 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,777,152 | 61.54% | 108.21 | 90,353 | 160,736 | True | False |
| clever-cedar-46 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,154,880 | 83.14% | 134.51 | 90,364 | 160,976 | True | False |
| merry-maple-47 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=1, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 48.26% | 149.39 | 88,876 | 74,440 | True | False |
| wild-tree-48 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,325,440 | 82.31% | 125.95 | 89,920 | 160,976 | True | False |
| cosmic-robin-49 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,035,776 | 66.63% | 101.82 | 88,743 | 160,736 | True | True |
| wild-falcon-50 | connect4-flex-v1 | channels_1=32, channels_2=32, depth=2, global_pool=0, kernel_1=4, kernel_2=2, pool_1=0, pool_2=0, projection=32, residual_1=0, residual_2=1, stride_1=4, stride_2=4 | 2,791,424 | 0.05% | 34.41 | 81,122 | 77,568 | True | False |
| brave-comet-51 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,662,784 | 90.20% | 137.37 | 92,180 | 160,736 | True | True |
| calm-owl-52 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,189,120 | 88.02% | 146.41 | 90,083 | 160,976 | True | False |
| merry-river-53 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,047,808 | 78.62% | 143.00 | 91,243 | 160,976 | True | False |
| quiet-tiger-54 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,702,272 | 79.13% | 131.51 | 88,984 | 160,976 | True | False |
| happy-comet-55 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,093,120 | 42.44% | 104.53 | 86,991 | 160,976 | True | False |
| lucky-robin-56 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,507,712 | 63.89% | 125.33 | 91,819 | 160,736 | True | False |
| golden-panda-57 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,909,696 | 72.84% | 123.01 | 88,690 | 160,976 | True | False |
| quiet-cat-58 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.51% | 146.86 | 90,407 | 160,976 | True | False |
| mystic-robin-59 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,732,416 | 71.11% | 136.03 | 93,600 | 109,648 | True | False |
| clever-otter-60 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,431,360 | 87.30% | 136.22 | 91,259 | 109,768 | True | True |
| calm-otter-61 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,119,488 | 82.85% | 145.77 | 90,001 | 160,976 | True | False |
| cosmic-tree-62 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,299,968 | 51.35% | 107.19 | 86,762 | 160,976 | True | False |
| lucky-fox-63 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.61% | 146.23 | 90,797 | 160,976 | True | False |
| cosmic-fox-64 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,448,320 | 66.23% | 128.46 | 89,120 | 160,976 | True | False |
| lucky-badger-65 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,236,224 | 84.78% | 147.60 | 89,676 | 160,976 | True | False |
| brave-robin-66 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,652,544 | 82.37% | 137.60 | 91,952 | 160,976 | True | False |
| mystic-comet-67 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=64, residual_1=0, stride_1=4 | 7,933,952 | 43.41% | 88.18 | 89,975 | 160,352 | True | True |
| golden-falcon-68 | connect4-flex-v1 | channels_1=8, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 73.43% | 146.09 | 90,884 | 109,768 | True | False |
| merry-badger-69 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,530,240 | 81.75% | 128.66 | 89,618 | 160,976 | True | False |
| calm-fox-70 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,737,664 | 67.40% | 113.67 | 94,463 | 160,528 | True | False |
| quiet-maple-71 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.59% | 146.77 | 90,463 | 160,976 | True | False |
| golden-fox-72 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,650,496 | 80.69% | 131.54 | 96,172 | 160,528 | True | False |
| golden-river-73 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,237,376 | 72.07% | 124.29 | 90,413 | 160,976 | True | False |
| brave-otter-74 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.51% | 146.26 | 90,778 | 160,976 | True | False |
| bright-fox-75 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,317,824 | 65.13% | 115.67 | 89,201 | 160,976 | True | False |
| mystic-otter-76 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,412,928 | 66.46% | 139.42 | 89,033 | 160,976 | True | False |
| cosmic-bear-77 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,366,976 | 75.23% | 110.37 | 93,929 | 160,528 | True | False |
| brave-bear-78 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,148,736 | 77.10% | 133.71 | 90,859 | 160,736 | True | False |
| wild-cat-79 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,489,280 | 70.99% | 125.37 | 91,643 | 160,528 | True | False |
| silver-robin-80 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,126,784 | 84.94% | 122.90 | 90,535 | 160,736 | True | True |
| merry-fox-81 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 8,706,048 | 42.94% | 97.60 | 89,201 | 253,648 | True | False |
| bright-river-82 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.87% | 146.91 | 90,376 | 160,976 | True | False |
| clever-fox-83 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,531,712 | 84.22% | 138.16 | 90,704 | 160,736 | True | False |
| bright-owl-84 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.59% | 146.01 | 90,933 | 160,976 | True | False |
| brave-panda-85 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,800,576 | 64.78% | 132.46 | 89,088 | 160,736 | True | False |
| brave-bear-86 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 80.85% | 146.16 | 90,840 | 160,736 | True | False |
| cosmic-wolf-87 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 8,167,424 | 35.95% | 89.40 | 91,358 | 253,472 | True | False |
| brave-robin-88 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,375,168 | 61.66% | 117.89 | 88,007 | 160,976 | True | False |
| brave-tiger-89 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,827,776 | 73.14% | 113.88 | 95,081 | 160,528 | True | False |
| golden-robin-90 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,692,608 | 69.56% | 115.76 | 92,369 | 160,352 | True | False |
| calm-otter-91 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,730,368 | 87.32% | 136.99 | 92,929 | 160,528 | True | True |
| sunny-tiger-92 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.21% | 146.63 | 90,549 | 160,976 | True | False |
| clever-bear-93 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,980,800 | 73.43% | 134.13 | 89,322 | 160,976 | True | False |
| brave-otter-94 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.51% | 146.83 | 90,426 | 160,976 | True | False |
| brave-wolf-95 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=4, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,748,480 | 65.49% | 107.08 | 91,039 | 160,208 | True | False |
| gentle-otter-96 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,448,896 | 70.82% | 110.84 | 94,270 | 160,528 | True | False |
| lucky-maple-97 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 80.30% | 142.08 | 93,449 | 160,528 | True | False |
| golden-tiger-98 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,823,104 | 70.77% | 127.82 | 92,498 | 160,352 | True | False |
| golden-maple-99 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.68% | 146.77 | 90,463 | 160,976 | True | False |
| quiet-owl-100 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 8,654,848 | 54.24% | 99.27 | 87,185 | 262,912 | True | False |
| sunny-river-101 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,383,360 | 74.71% | 114.06 | 91,034 | 160,736 | True | False |
| gentle-bear-102 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,413,504 | 77.47% | 127.37 | 89,609 | 160,976 | True | False |
| merry-badger-103 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 11,134,976 | 79.79% | 117.06 | 95,122 | 253,472 | True | False |
| happy-wolf-104 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,481,088 | 76.12% | 120.03 | 95,652 | 160,528 | True | False |
| lucky-maple-105 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 10,960,896 | 70.00% | 114.59 | 95,653 | 160,528 | True | False |
| lucky-owl-106 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 8,896,512 | 58.01% | 98.61 | 90,219 | 253,648 | True | True |
| swift-fox-107 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.62% | 145.99 | 90,946 | 160,976 | True | False |
| sunny-owl-108 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,655,744 | 81.10% | 117.30 | 90,842 | 253,648 | True | False |
| silver-owl-109 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 12,013,568 | 86.69% | 126.98 | 94,610 | 253,472 | True | True |
| gentle-tree-110 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,854,400 | 83.59% | 115.45 | 94,018 | 253,472 | True | True |
| swift-fox-111 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,513,280 | 76.76% | 137.30 | 91,138 | 160,736 | True | False |
| golden-falcon-112 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 81.25% | 145.95 | 90,971 | 160,736 | True | False |
| bright-robin-113 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,939,840 | 78.84% | 127.48 | 93,660 | 160,528 | True | False |
| swift-robin-114 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=4, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,651,648 | 75.69% | 115.78 | 91,999 | 253,328 | True | False |
| swift-cedar-115 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 12,326,912 | 75.99% | 128.06 | 96,259 | 160,528 | True | False |
| cosmic-wolf-116 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=4, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,484,288 | 81.28% | 103.54 | 91,600 | 261,856 | True | True |
| lucky-cedar-117 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=64, residual_1=0, stride_1=4 | 13,277,184 | 88.59% | 146.63 | 90,549 | 160,976 | True | False |
| swift-otter-118 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,719,232 | 78.80% | 116.72 | 91,837 | 253,648 | True | False |
| cosmic-cedar-119 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,164,224 | 70.39% | 115.73 | 87,827 | 456,992 | True | False |
| mystic-panda-120 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=4, pool_1=0, projection=128, residual_1=0, stride_1=4 | 10,174,464 | 72.53% | 112.40 | 90,520 | 456,352 | True | False |
| cosmic-wolf-121 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 13,277,184 | 88.06% | 142.42 | 93,226 | 253,648 | True | False |
| mystic-fox-122 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=3, pool_1=0, projection=64, residual_1=0, stride_1=4 | 8,708,096 | 62.75% | 100.74 | 86,441 | 261,632 | True | True |
| wild-otter-123 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=8, pool_1=0, projection=32, residual_1=0, stride_1=4 | 13,277,184 | 44.38% | 146.49 | 90,635 | 106,160 | True | False |
| swift-tree-124 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=7, pool_1=0, projection=64, residual_1=0, stride_1=4 | 9,476,096 | 56.47% | 108.32 | 87,482 | 262,912 | True | False |
| cosmic-wolf-125 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=128, residual_1=0, stride_1=4 | 13,277,184 | 87.80% | 142.34 | 93,278 | 253,648 | True | False |
| wild-cat-126 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=1, kernel_1=5, pool_1=0, projection=128, residual_1=0, stride_1=4 | 12,568,576 | 0.00% | 139.07 | 90,376 | 52,768 | True | False |
| cosmic-cedar-127 | connect4-flex-v1 | channels_1=32, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 7,942,144 | 17.61% | 90.75 | 87,517 | 262,496 | True | False |
| lucky-maple-128 | connect4-flex-v1 | channels_1=16, depth=1, global_pool=0, kernel_1=6, pool_1=0, projection=64, residual_1=0, stride_1=4 | 11,988,992 | 77.98% | 125.30 | 95,682 | 160,528 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
