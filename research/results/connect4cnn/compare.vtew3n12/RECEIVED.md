# RTX 5090 evidence received on G240

Original archive: `build/hardware-artifacts/compare.vtew3n12.khf7LOJr.tar.gz`.
SHA256: `ffd4e83fda2cb57ff5cc846115345a02af03adc910adfd0df5cab10d62c1da87`.

Separately received `runtime-5090.sh` SHA256: `99ca283a2474be08d621292b79c7285593cd3a4f5059b5cd7b952c4dab2d44dc`.
It records existing 5090 toolchain/NCCL paths and a checkout-local linker alias; inspected without execution on G240.

All 168 archive members were validated as regular files with safe, unique relative paths before extraction. Original filenames and contents are retained; the scoped `.gitignore` preserves `.log` files. `analysis/`, this note, the ignore file and the separately received runtime helper are receipt-side additions. Original `REPORT.md` is unchanged.

[Local audit and full observed frontiers](analysis/REPORT.md) were generated with `research/audit_hardware_archive.py`. All 42 source files match recorded revision `9b829e071f6d3d56064465dc8284a4b0f03c82a4`; all 52 observations match evaluation logs; all four non-encoder configurations match. See [interpretation and limitations](../../../HARDWARE_5090_RESULTS.md).

The transfer omits executable binaries and checkpoint arrays. Their hashes and remote audit claims are retained, but receipt verification cannot independently validate absent bytes or rerun numerical/gradient/sanitizer tests. Large artifacts remain on the 5090 under its original `build/connect4cnn/compare.vtew3n12/` directory. The compressed transfer stays in ignored local `build/hardware-artifacts/`; small evidence here can be committed with the research.
