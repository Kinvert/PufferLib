# Optional autoencoder: implementation and bounded verification

October 20, 2026, G240. This is code/build/host-contract evidence, not a trained
backbone or downstream RL result. See [5090 commands](../../../NEXT_5090_AUTOENCODER.md).

Implemented: native full-frame collection; one shared variable-shape CUDA encoder
and reconstruction decoder; fixed Adam/shuffle schedule; complete optimizer-state
checkpoints/resume; validation and encoder-only PufferLib initialization export.
The encoder has 36,776 parameters and a 128-value bottleneck; the discarded
decoder has 34,987 parameters. Full RGB Flappy 640x420 and Pong 640x620 preserve
their own shapes. The pixel-space labeled-object dataset remains a future design.

Two native CUDA binaries compile. Native host registration agrees with independent
counts. Twelve autoencoder artifact/data/resume/export/weight-inventory tests and seventeen
compression regression tests pass. These tests do not compute a CPU neural model.
Source/binary manifests and compact receipts are in `receipts/`; raw builds remain
under ignored `build/autoencoder-5060-20261020-v1`.

GPU qualification was **not executed**. Its bounded queue exhausted its 600-second
wait cap while another campaign owned the GPU; `gpu_launched=false`, supervisor
session 61521 exited 1. The queue is terminal, not a live reservation, and must
not automatically restart. No corpus or long pretraining has run. Native math,
repeat/resume/import and memory checks are
mandatory on the 5090 before optional pretraining. The earlier full-frame
compression packet's WDDM memory-instrumentation failure remains a failure.

Released TAESD and MobileNetV3-Small-050 weights were actually downloaded and
verified against pinned revisions/LFS SHA256 values. Compact acquisition receipts
are included; the 9.8 MB and 6.42 MB weights stay under ignored `build/backbones`.
They were inspected as safetensors data only. Neither architecture has been
ported/executed through native PufferLib yet. No Torch, CUDA, system or dependency
changes were made. Neither acquisition nor reconstruction implies a Pareto win.
