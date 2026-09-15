# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 7.932 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Run | Family | C / depth / stride / projection / blocks / GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| golden-maple-1 | connect4-nature-v1 | – / – / – / – / – / – | 32,768 | 0.00% | 0.57 | 57,488 | 138,528 | False | False |
| quiet-cat-2 | connect4-nature-v1 | – / – / – / – / – / – | 16,384 | 0.00% | 0.37 | 44,281 | 138,528 | False | False |
| wild-tiger-3 | connect4-nature-v1 | – / – / – / – / – / – | 18,432 | 0.00% | 0.36 | 51,200 | 138,528 | False | False |
| brave-cat-4 | connect4-nature-v1 | – / – / – / – / – / – | 18,432 | 0.00% | 0.33 | 55,855 | 138,528 | False | False |
| wild-fox-5 | connect4-nature-v1 | – / – / – / – / – / – | 18,432 | 0.00% | 0.33 | 55,855 | 138,528 | False | False |
| bright-bear-6 | connect4-nature-v1 | – / – / – / – / – / – | 16,384 | 0.00% | 0.31 | 52,852 | 138,528 | False | False |
| golden-river-7 | connect4-nature-v1 | – / – / – / – / – / – | 16,384 | 0.00% | 0.30 | 54,613 | 138,528 | False | True |
| bright-cedar-8 | connect4-nature-v1 | – / – / – / – / – / – | 16,384 | 0.00% | 0.31 | 52,852 | 138,528 | False | False |
| happy-panda-9 | connect4-nature-v1 | – / – / – / – / – / – | 18,432 | 0.00% | 0.33 | 55,855 | 138,528 | False | False |
| silver-maple-10 | connect4-nature-v1 | – / – / – / – / – / – | 18,432 | 0.00% | 0.36 | 51,200 | 138,528 | False | False |
| clever-comet-11 | connect4-nature-v1 | – / – / – / – / – / – | 16,384 | 0.00% | 0.31 | 52,852 | 138,528 | False | False |
| brave-robin-12 | connect4-nature-v1 | – / – / – / – / – / – | 22,528 | 0.00% | 0.43 | 52,391 | 138,528 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
