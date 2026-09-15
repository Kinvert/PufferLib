# RTX 5090 fixed comparison

Completed four jobs and 52 held-out evaluations with zero failures.

| Model | Decisions | Held-out wins | Train wall seconds | Process SPS | Native average SPS | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Ours quality | 13,312,000 | 71.78% | 77.819 | 171,063 | 171,897 | 160,736 |
| Nature | 13,312,000 | 66.11% | 83.790 | 158,873 | 159,390 | 138,528 |
| IMPALA | 13,312,000 | 96.49% | 461.455 | 28,848 | 28,866 | 270,496 |
| Impoola | 13,312,000 | 82.77% | 476.149 | 27,958 | 27,974 | 151,712 |

## Protocol and hardware

- Revision: `9b829e071f6d3d56064465dc8284a4b0f03c82a4`; measured source hashes verified against the captured files.
- RTX 5090, AMD Ryzen 9 9950X3D, CUDA compiler 13.1.115, NVIDIA driver 580.105.08.
- Native C/CUDA float32; seed 173; held-out seed 20173; common H128/L1 core and representation 0.
- Command: `source build/connect4cnn/runtime-5090.sh && bash ocean/connect4cnn/hardware_compare.sh --full`.
- Existing F-Zero NCCL reused read-only; linker alias lives in this checkout. Exact paths/hashes are in `validation-5090/setup.txt`.
- Source, recipe, binary, and all 52 checkpoint hashes verified. All checkpoint arrays have the expected parameter count and finite float32 values.

## Validation

- 17 configuration/runner tests passed.
- Ten representation pixel/reset/parity/repeatability checks and ASan/UBSan passed.
- Nature, Flex, IMPALA and Impoola numerical/gradient and eager/graph repeatability checks passed.
- Canary `compare.bfi2gjs_`: four successful jobs and 16 evaluations. Canary scores/timing are excluded from this table.

## Interpretation limits

The table uses each model's final 13.312M-decision checkpoint. Process SPS includes trainer startup and checkpoint writes; native average SPS uses the trainer's adjusted timer. Builds and post-training evaluations are excluded from training wall time. Parameter counts differ.

This is one training seed on one game. It does not establish score superiority or a general-purpose CNN ranking. No same-revision RTX 5060 comparison has been run here; historical 5060 results use different captured source. Desktop graphics processes were present, with no competing compute job at launch.

Full checkpoint curves and actual batched evaluation counts are in `REPORT.md` and `results.csv`. Raw evidence is retained in this run directory.
