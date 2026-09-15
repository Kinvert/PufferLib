# Experimental CNN native PROTEIN sweep

Status: ok. Sweep process wall: 11611.189 seconds.
Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.
Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.
Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Pareto flags use rounded final observations.

| Trial | Channels | Blocks | GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 8 | 0 | 0 | 6,639,616 | 14.83% | 165.30 | 40,167 | 115,312 | False | True |
| 1 | 8 | 0 | 0 | 3,690,496 | 0.18% | 92.99 | 39,687 | 115,312 | False | True |
| 2 | 16 | 1 | 1 | 3,868,672 | 0.12% | 244.87 | 15,799 | 110,080 | False | False |
| 3 | 16 | 1 | 0 | 3,850,240 | 21.32% | 241.88 | 15,918 | 228,864 | False | True |
| 4 | 16 | 1 | 0 | 3,856,384 | 10.70% | 243.42 | 15,843 | 228,864 | False | False |
| 5 | 16 | 1 | 1 | 3,434,496 | 0.00% | 217.44 | 15,795 | 110,080 | False | False |
| 6 | 8 | 2 | 0 | 3,596,288 | 1.85% | 205.47 | 17,503 | 136,208 | False | False |
| 7 | 32 | 0 | 1 | 3,457,024 | 0.00% | 176.52 | 19,584 | 114,240 | False | False |
| 8 | 32 | 2 | 0 | 3,885,056 | 52.54% | 664.87 | 5,843 | 684,224 | False | False |
| 9 | 8 | 0 | 1 | 3,762,176 | 0.03% | 95.05 | 39,581 | 55,920 | False | False |
| 10 | 16 | 1 | 0 | 3,424,256 | 1.00% | 216.14 | 15,843 | 228,864 | False | False |
| 11 | 32 | 2 | 0 | 5,459,968 | 97.44% | 934.75 | 5,841 | 684,224 | True | False |
| 12 | 32 | 1 | 0 | 7,579,648 | 100.00% | 838.63 | 9,038 | 518,016 | True | False |
| 13 | 32 | 1 | 0 | 10,352,640 | 98.80% | 1142.55 | 9,061 | 518,016 | True | False |
| 14 | 32 | 1 | 0 | 6,039,552 | 98.12% | 668.48 | 9,035 | 518,016 | True | False |
| 15 | 16 | 1 | 0 | 5,588,992 | 62.00% | 349.62 | 15,986 | 228,864 | True | True |
| 16 | 16 | 1 | 0 | 6,356,992 | 69.00% | 397.20 | 16,005 | 228,864 | True | True |
| 17 | 32 | 1 | 0 | 6,508,544 | 100.00% | 722.24 | 9,012 | 518,016 | True | False |
| 18 | 32 | 1 | 0 | 6,215,680 | 94.27% | 686.42 | 9,055 | 518,016 | True | False |
| 19 | 32 | 1 | 0 | 5,867,520 | 97.82% | 649.50 | 9,034 | 518,016 | True | False |
| 20 | 32 | 1 | 0 | 5,785,600 | 100.00% | 641.40 | 9,020 | 518,016 | True | True |
| 21 | 32 | 1 | 0 | 6,633,472 | 100.00% | 735.59 | 9,018 | 518,016 | True | False |
| 22 | 32 | 1 | 0 | 5,904,384 | 90.59% | 651.63 | 9,061 | 518,016 | True | False |
| 23 | 32 | 1 | 0 | 5,595,136 | 96.59% | 619.05 | 9,038 | 518,016 | True | True |

Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.
Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.
Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.
