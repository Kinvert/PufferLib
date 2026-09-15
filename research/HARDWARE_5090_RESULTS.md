# RTX 5090 hardware comparison — received and audited

Source: the separate 5090 session's report and transferred evidence archive. **The G240 receipt audit passed:** all 42 captured source files match the recorded Git revision, all 52 CSV observations match raw evaluation logs, and all four models have matching non-encoder settings. The remote session additionally reports passing numerical/gradient, repeatability, sanitizer and checkpoint audits; those tests were not rerun here.

Reported revision: `9b829e071f6d3d56064465dc8284a4b0f03c82a4`. Hardware: RTX 5090, Ryzen 9 9950X3D, CUDA compiler **13.1.115**, driver **580.105.08**. Float32, train seed **173**, **13,312,000 decisions per model**. Runtime setup used a machine-local helper before the standard runner:

```bash
source build/connect4cnn/runtime-5090.sh
bash ocean/connect4cnn/hardware_compare.sh --full
```

The [received runtime helper](results/connect4cnn/compare.vtew3n12/runtime-5090.sh) captures machine-local settings outside that revision. It reads the existing F-Zero NCCL package and creates a linker alias inside the checkout. It was inspected, not executed on G240; no system CUDA changes were made.

## Final checkpoint results

| Model | Held-out wins | Train wall seconds | Process SPS | Native average SPS | Parameters |
|---|---:|---:|---:|---:|---:|
| Ours quality | 71.78% | 77.819 | 171,063 | 171,897 | 160,736 |
| Nature | 66.11% | 83.790 | 158,873 | 159,390 | 138,528 |
| IMPALA | 96.49% | 461.455 | 28,848 | 28,866 | 270,496 |
| Impoola | 82.77% | 476.149 | 27,958 | 27,974 | 151,712 |

Training-process wall sums to **1,099.214 seconds = 18 minutes 19.214 seconds**. This excludes compilation, validation/canaries and post-training evaluation. The reported agent work duration of 24m49s is a different timing boundary.

Ours used **7.13% less training wall time** than Nature, corresponding to **7.67% greater process throughput**, and scored **5.67 percentage points higher** in this seed. At the final checkpoints, ours dominates Nature in time/score and IMPALA dominates Impoola. This does not establish statistically reliable score superiority.

## Whole observed frontier

The [complete frontier tables](results/connect4cnn/compare.vtew3n12/analysis/REPORT.md) use all 52 checkpoints, preserving declines in [observations.csv](results/connect4cnn/compare.vtew3n12/analysis/observations.csv). Representative points from the time frontier:

| Model | Decisions | Checkpoint wall seconds | Held-out wins |
|---|---:|---:|---:|
| Ours quality | 9,216,000 | 53.960 | 72.14% |
| Impoola | 3,072,000 | 104.057 | 79.46% |
| Impoola | 4,096,000 | 138.215 | 83.76% |
| IMPALA | 6,144,000 | 218.682 | 92.61% |
| IMPALA | 10,240,000 | 357.207 | 96.88% |

Ours occupies the lowest-cost region, **Impoola contributes the middle region**, and IMPALA reaches the highest scores. Nature contributes no observed frontier points in this seed. The decision-count frontier contains IMPALA and Impoola only: ours' advantage here is in wall time. These are observed points, not confidence bounds or an interpolated guarantee between measurements. Earlier checkpoints share the full 13.312M learning-rate schedule; they are not independently trained short-budget runs. Do not confuse this single-seed frontier with the historical five-seed mean frontier.

## Historical 5060 comparison

Matching seed 173 / final 13.312M-step rows from [the archived 5060 confirmation](results/connect4cnn/confirm.ol9tcj5k/results.csv):

| Model | Historical 5060 wins | Historical 5060 train seconds | Reported 5090 train seconds | Approximate wall-time ratio (5060 / 5090) |
|---|---:|---:|---:|---:|
| Ours quality | 66.73% | 145.820 | 77.819 | 1.87× |
| Nature | 60.89% | 159.652 | 83.790 | 1.91× |
| IMPALA | 96.30% | 1,227.800 | 461.455 | 2.66× |
| Impoola | 78.50% | 1,228.714 | 476.149 | 2.58× |

These are **historical whole-machine ratios**, not isolated GPU speedups: the 5060 experiment used the older [captured source](BENCHMARK_STATE.md), a CUDA 12.8 compiler and a different host. A matching integer seed across GPU/toolchain changes does not imply identical trajectories or an additional independent training seed. Do not pool these runs as independent seed replication.

## Evidence and next verification

The received small evidence files are preserved under [compare.vtew3n12](results/connect4cnn/compare.vtew3n12/RECEIVED.md), including the original report, source snapshots, configs, raw logs and runtime helper. [Audit JSON](results/connect4cnn/compare.vtew3n12/analysis/audit.json) records checks and limitations. Reproduce the receipt audit from the original local tarball:

```bash
.venv/bin/python research/audit_hardware_archive.py build/hardware-artifacts/compare.vtew3n12.khf7LOJr.tar.gz --output research/results/connect4cnn/compare.vtew3n12/analysis
```

Compiled binaries and checkpoint arrays were excluded from the transfer. Their recorded hashes and remote audit survive, but local byte-hash/finiteness verification requires those actual files. A same-revision G240 run after Pong finishes remains future work for a controlled cross-host comparison. Fresh-seed score/frontier confirmation remains separate work under the [paper evidence standard](PAPER_PLAN.md#required-evidence-standard-defend-the-claim-like-a-thesis). No additional GPU run was launched for this audit.
