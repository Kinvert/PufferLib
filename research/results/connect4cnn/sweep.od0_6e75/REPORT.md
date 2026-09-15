# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 919.912 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | C / depth / stride / projection / blocks / GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| bright-badger-1 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 3,317,760 | 0.33% | 38.03 | 87,241 | 75,432 | False | True |
| wild-wolf-2 | connect4-compact-v1 | 8 / 1 / 2 / 64 / – / – | 972,800 | 0.03% | 13.42 | 72,489 | 205,000 | False | False |
| happy-bear-3 | connect4-compact-v1 | 16 / 2 / 4 / 64 / – / – | 1,042,432 | 0.03% | 12.70 | 82,081 | 92,400 | False | False |
| silver-owl-4 | connect4-compact-v1 | 16 / 2 / 2 / 32 / – / – | 1,036,288 | 0.06% | 16.41 | 63,150 | 112,848 | False | False |
| bright-robin-5 | connect4-compact-v1 | 16 / 2 / 2 / 32 / – / – | 1,038,336 | 0.05% | 16.49 | 62,968 | 112,848 | False | False |
| gentle-wolf-6 | connect4-compact-v1 | 16 / 2 / 4 / 64 / – / – | 872,448 | 0.03% | 10.66 | 81,843 | 92,400 | False | False |
| silver-otter-7 | connect4-compact-v1 | 8 / 3 / 2 / 64 / – / – | 935,936 | 0.04% | 14.26 | 65,634 | 88,040 | False | False |
| quiet-falcon-8 | connect4-compact-v1 | 32 / 1 / 4 / 128 / – / – | 880,640 | 0.19% | 10.61 | 83,001 | 380,064 | False | True |
| mystic-wolf-9 | connect4-compact-v1 | 32 / 3 / 2 / 32 / – / – | 1,050,624 | 0.00% | 21.19 | 49,581 | 175,424 | False | False |
| sunny-fox-10 | connect4-compact-v1 | 8 / 1 / 4 / 64 / – / – | 999,424 | 0.10% | 12.02 | 83,147 | 100,040 | False | False |
| swift-river-11 | connect4-compact-v1 | 16 / 2 / 2 / 64 / – / – | 868,352 | 0.00% | 13.87 | 62,606 | 166,128 | False | False |
| swift-falcon-12 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 2,697,216 | 0.14% | 30.55 | 88,289 | 75,432 | True | False |
| bright-maple-13 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 4,538,368 | 0.75% | 51.01 | 88,970 | 75,432 | True | True |
| wild-tiger-14 | connect4-compact-v1 | 8 / 1 / 4 / 64 / – / – | 4,730,880 | 0.68% | 54.03 | 87,560 | 100,040 | True | False |
| cosmic-maple-15 | connect4-compact-v1 | 32 / 1 / 4 / 128 / – / – | 1,232,896 | 0.00% | 14.96 | 82,413 | 380,064 | True | False |
| clever-cat-16 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 3,991,552 | 0.18% | 45.19 | 88,328 | 75,432 | True | False |
| lucky-river-17 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 4,298,752 | 0.31% | 48.36 | 88,891 | 75,432 | True | False |
| gentle-cedar-18 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,074,368 | 9.70% | 68.48 | 88,703 | 75,432 | True | True |
| bright-fox-19 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,639,616 | 22.54% | 73.25 | 90,643 | 75,432 | True | False |
| happy-robin-20 | connect4-compact-v1 | 16 / 1 / 4 / 32 / – / – | 4,859,904 | 1.84% | 55.74 | 87,189 | 96,432 | True | True |
| clever-robin-21 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,639,616 | 20.63% | 72.87 | 91,116 | 75,432 | True | False |
| golden-maple-22 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,639,616 | 22.54% | 72.67 | 91,367 | 75,432 | True | False |
| merry-maple-23 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,639,616 | 22.76% | 72.66 | 91,379 | 75,432 | True | True |
| gentle-cedar-24 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 6,639,616 | 22.08% | 72.69 | 91,342 | 75,432 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
