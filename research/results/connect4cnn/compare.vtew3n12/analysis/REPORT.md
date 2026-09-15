# RTX 5090: received evidence audit and complete observed frontier

Revision `9b829e071f6d3d56064465dc8284a4b0f03c82a4`; source hashes verified: 42; 52 CSV observations match their raw evaluation logs; four non-encoder configs match.
Captured files differing from the recorded Git revision: none.
One training seed (173). No seed confidence intervals or population-dominance claim. All 52 chronological observations remain in observations.csv, including declines.
Executable binaries and checkpoint arrays were omitted from the transfer. Their recorded hashes and the remote audit are retained, but this audit cannot independently repeat byte-hash/finiteness checks on absent files.

## Complete observed seconds frontier

| Model | Decisions | Checkpoint wall seconds | Held-out wins |
|---|---:|---:|---:|
| Ours quality | 1,024,000 | 6.631 | 0.08% |
| Ours quality | 2,048,000 | 12.929 | 0.37% |
| Ours quality | 3,072,000 | 19.177 | 12.89% |
| Ours quality | 4,096,000 | 25.292 | 29.70% |
| Ours quality | 5,120,000 | 31.312 | 47.85% |
| Ours quality | 6,144,000 | 37.161 | 64.15% |
| Ours quality | 7,168,000 | 42.924 | 68.36% |
| Ours quality | 8,192,000 | 48.657 | 69.89% |
| Ours quality | 9,216,000 | 53.960 | 72.14% |
| Impoola | 3,072,000 | 104.057 | 79.46% |
| Impoola | 4,096,000 | 138.215 | 83.76% |
| IMPALA | 6,144,000 | 218.682 | 92.61% |
| IMPALA | 7,168,000 | 253.288 | 94.13% |
| IMPALA | 8,192,000 | 287.935 | 95.15% |
| IMPALA | 9,216,000 | 322.583 | 95.52% |
| IMPALA | 10,240,000 | 357.207 | 96.88% |

## Complete observed steps frontier

| Model | Decisions | Checkpoint wall seconds | Held-out wins |
|---|---:|---:|---:|
| IMPALA | 1,024,000 | 38.300 | 0.36% |
| IMPALA | 2,048,000 | 76.648 | 45.84% |
| Impoola | 3,072,000 | 104.057 | 79.46% |
| Impoola | 4,096,000 | 138.215 | 83.76% |
| IMPALA | 6,144,000 | 218.682 | 92.61% |
| IMPALA | 7,168,000 | 253.288 | 94.13% |
| IMPALA | 8,192,000 | 287.935 | 95.15% |
| IMPALA | 9,216,000 | 322.583 | 95.52% |
| IMPALA | 10,240,000 | 357.207 | 96.88% |

Checkpoint timing includes process startup and checkpoint writes; it is distinct from native uptime. Early checkpoints share the full-run learning-rate schedule and are not independent short-budget training runs.
Archive provenance and hardware metadata remain in the received archive. Historical 5060 data used different source/compiler/host settings; matching-source cross-host confirmation remains pending.
