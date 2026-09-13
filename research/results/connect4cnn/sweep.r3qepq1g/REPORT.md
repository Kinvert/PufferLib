# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 19.409 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Canary budgets validate infrastructure, not learning quality. Pareto flags use rounded final observations.

| Trial | Channels | Blocks | GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 8 | 0 | 0 | 32,768 | 0.00% | 0.98 | 33,437 | 115,312 | False | False |
| 1 | 8 | 0 | 0 | 16,384 | 0.00% | 0.51 | 32,125 | 115,312 | False | True |
| 2 | 16 | 1 | 1 | 18,432 | 0.00% | 1.26 | 14,629 | 110,080 | False | False |
| 3 | 16 | 1 | 0 | 18,432 | 0.00% | 1.26 | 14,629 | 228,864 | False | False |
| 4 | 16 | 1 | 0 | 18,432 | 0.00% | 1.26 | 14,629 | 228,864 | False | False |
| 5 | 16 | 1 | 1 | 16,384 | 0.00% | 1.13 | 14,499 | 110,080 | False | False |
| 6 | 8 | 2 | 0 | 16,384 | 0.00% | 1.04 | 15,754 | 136,208 | False | False |
| 7 | 32 | 0 | 1 | 16,384 | 0.31% | 0.92 | 17,809 | 114,240 | False | True |
| 8 | 32 | 2 | 0 | 18,432 | 0.00% | 3.24 | 5,689 | 684,224 | False | False |
| 9 | 8 | 0 | 1 | 18,432 | 0.00% | 0.56 | 32,914 | 55,920 | False | False |
| 10 | 16 | 1 | 0 | 16,384 | 0.00% | 1.13 | 14,499 | 228,864 | False | False |
| 11 | 32 | 1 | 1 | 14,336 | 0.22% | 1.84 | 7,791 | 280,448 | True | False |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and post-run W&B synchronization.
