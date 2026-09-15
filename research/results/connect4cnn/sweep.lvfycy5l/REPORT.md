# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 8.033 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | C / depth / stride / projection / blocks / GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| happy-robin-1 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 32,768 | 0.06% | 0.52 | 63,015 | 75,432 | False | True |
| lucky-otter-2 | connect4-compact-v1 | 8 / 1 / 2 / 64 / – / – | 16,384 | 0.05% | 0.32 | 51,200 | 205,000 | False | False |
| mystic-badger-3 | connect4-compact-v1 | 16 / 2 / 4 / 64 / – / – | 18,432 | 0.00% | 0.32 | 57,600 | 92,400 | False | False |
| silver-cat-4 | connect4-compact-v1 | 16 / 2 / 2 / 32 / – / – | 18,432 | 0.04% | 0.38 | 48,505 | 112,848 | False | False |
| gentle-panda-5 | connect4-compact-v1 | 16 / 2 / 2 / 32 / – / – | 18,432 | 0.04% | 0.39 | 47,262 | 112,848 | False | False |
| silver-cedar-6 | connect4-compact-v1 | 16 / 2 / 4 / 64 / – / – | 16,384 | 0.00% | 0.29 | 56,497 | 92,400 | False | False |
| wild-falcon-7 | connect4-compact-v1 | 8 / 3 / 2 / 64 / – / – | 16,384 | 0.00% | 0.34 | 48,188 | 88,040 | False | False |
| cosmic-falcon-8 | connect4-compact-v1 | 32 / 1 / 4 / 128 / – / – | 16,384 | 0.05% | 0.29 | 56,497 | 380,064 | False | True |
| silver-panda-9 | connect4-compact-v1 | 32 / 3 / 2 / 32 / – / – | 18,432 | 0.00% | 0.46 | 40,070 | 175,424 | False | False |
| wild-tiger-10 | connect4-compact-v1 | 8 / 1 / 4 / 64 / – / – | 18,432 | 0.00% | 0.30 | 61,440 | 100,040 | False | False |
| sunny-otter-11 | connect4-compact-v1 | 16 / 2 / 2 / 64 / – / – | 16,384 | 0.00% | 0.35 | 46,811 | 166,128 | False | False |
| clever-bear-12 | connect4-compact-v1 | 8 / 1 / 4 / 32 / – / – | 32,768 | 0.06% | 0.52 | 63,015 | 75,432 | True | True |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
