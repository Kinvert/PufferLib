# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 191.242 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | C / depth / stride / projection / blocks / GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| bright-tree-1 | connect4-nature-v1 | – / – / – / – / – / – | 3,317,760 | 0.11% | 43.18 | 76,836 | 138,528 | False | False |
| wild-wolf-2 | connect4-nature-v1 | – / – / – / – / – / – | 972,800 | 0.04% | 12.95 | 75,120 | 138,528 | False | False |
| swift-river-3 | connect4-nature-v1 | – / – / – / – / – / – | 1,042,432 | 0.00% | 13.66 | 76,313 | 138,528 | False | False |
| clever-wolf-4 | connect4-nature-v1 | – / – / – / – / – / – | 1,036,288 | 0.04% | 13.52 | 76,649 | 138,528 | False | False |
| quiet-tree-5 | connect4-nature-v1 | – / – / – / – / – / – | 1,038,336 | 0.00% | 13.53 | 76,743 | 138,528 | False | False |
| lucky-tree-6 | connect4-nature-v1 | – / – / – / – / – / – | 872,448 | 0.12% | 11.40 | 76,531 | 138,528 | False | True |
| sunny-panda-7 | connect4-nature-v1 | – / – / – / – / – / – | 935,936 | 0.00% | 12.11 | 77,286 | 138,528 | False | False |
| golden-tree-8 | connect4-nature-v1 | – / – / – / – / – / – | 880,640 | 0.05% | 11.32 | 77,795 | 138,528 | False | False |
| brave-tree-9 | connect4-nature-v1 | – / – / – / – / – / – | 1,050,624 | 0.07% | 13.55 | 77,537 | 138,528 | False | False |
| lucky-fox-10 | connect4-nature-v1 | – / – / – / – / – / – | 999,424 | 0.03% | 12.92 | 77,355 | 138,528 | False | False |
| silver-tree-11 | connect4-nature-v1 | – / – / – / – / – / – | 868,352 | 0.05% | 11.25 | 77,187 | 138,528 | False | True |
| cosmic-cat-12 | connect4-nature-v1 | – / – / – / – / – / – | 1,368,064 | 0.01% | 17.98 | 76,088 | 138,528 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
