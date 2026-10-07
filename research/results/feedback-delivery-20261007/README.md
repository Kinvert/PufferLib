# Clean-checkout feedback handoff verification

October 7, 2026, G240. Tested implementation commit
`b61127cb979354c9db3e829933dda3c4777a1919` in a separate clean Git clone at
`/tmp/cnn-feedback-clean-P3OXmBv8/repo`, containing only committed source.
Historical untracked results and the main workspace's venv/build directory were
not copied. Local Git object storage was shared; the checkout and builds were
separate. Reused the existing system toolkit and existing NCCL by process-local
paths. Created a new `uv venv --python 3.12 .venv`, installing NumPy 2.5.3 from
the uv cache. No CUDA/Torch/system changes or GPU work.

Actual command:

```bash
CUDA_HOME=/usr/local/cuda \
NCCL_ROOT=/home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl \
bash research/run_feedback_5090.sh prepare build/cross-game-feedback/clean-delivery
```

Result: exit 0. All 44 host/scalar/mock tests and three native pre-CUDA validation
tests pass. Native feedback bridge, six normal float32 game targets and three
host metadata tools build. Frozen recipe/build/source/metadata preparation and
offline inspection pass; 512 immutable input files / 291 owned source files,
one width search dimension. Build registry declares `gpu_queried=false` and
`policy_executed=false`; no execution directory exists. The clean checkout
remained clean after preparation. It independently downloaded the ordinary
Raylib build dependency; rendering/GPU execution was not required.

Prepared plan SHA256:
`9b4cbdbba205db69d590e98f7b36ba65e38d7a25197d8a50c144fa8fc014e116`.

Small native/host/build logs are retained in `receipts/`. Full locally bound build
and input packets remain in the temporary clone's
`build/cross-game-feedback/clean-delivery/`; the initial workspace preparation
is preserved separately under `build/cross-game-feedback/delivery-20261007/`.
The clean clone rebuild includes later whitespace cleanup of two newly copied
game headers. Remote agents must regenerate packets from their own local build.
Do not launch either G240 packet or treat these paths as remote inputs.

The subsequent receipt/documentation commit does not change the tested executable
source or launcher. These checks qualify fresh-checkout build/preparation only.
**Actual 5090 GPU suggestions, replay, training, deterministic evaluations and
feedback are pending.** No learning, full-learner math, Pareto or SOTA result.
Vendor/system/link closure remains incomplete. Follow
[NEXT_5090_FEEDBACK.md](../../../NEXT_5090_FEEDBACK.md) for the remote canary.
