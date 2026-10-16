# Sprite Forage: roughly ten-minute custom CNN run, RTX 5060, October 15

The requested new allocation completes without failures: one fresh native
221,184-decision training and both assigned final evaluations. The process
survived the assistant server restart; it was checked and never relaunched.
No completed prior training was resumed, replaced or automatically extended.

| Measurement | Result |
|---|---:|
| Decisions | 221,184 |
| Training process seconds | 612.087 (10m12s) |
| Process SPS | 361.36 |
| Native adjusted-average SPS | 361.77 |
| Native final-bin SPS | 261.35 |
| Final training-bin perf | 0% |
| Parameters | 19,721,472 |
| Full train/evaluation campaign seconds | 622.562 |

Process time uses monotonic launch/end stamps and includes startup/checkpoint
writes. The native adjusted timer has a different scope and uses CLOCK_REALTIME;
its611.393s uptime falls inside the process duration. The low final-bin SPS is
retained, not replaced with the overall average. Its cause is not established
by this receipt check.

## Fixed setup

Our locked Flex quality graph: padded7x7/s4 C16 convolution, flatten/linear64,
linear128, H128/L1 standard policy; full640x480 RGB float32. No state input,
downsampling, pretrained backbone, pooling or residual block. Depth-one graph;
stored layer-2/3 INI coordinates remain inactive.

Fixed policy seed7, drawing0, world offset0,8 agents, one buffer, two threads,
horizon8, minibatch8, LR.001, original starting rewards and512-step episode
cap. Seeds are never swept. Binary/core/game/default INI match the preceding
custom pilot; the launcher checks env/vec/policy and all other train values.
Only total budget, checkpoint cadence and output paths change. The unchanged
annealing rule now spans the longer budget. This is a fresh initialization,
not a weights-only continuation or byte-identical prefix of the short run.

## Evaluation

Final scheduled checkpoint,16 sequential episodes per world offset, one slot,
fixed action/policy seed7, drawing0, same512-step cap. The legacy native serial
aggregate evaluator is used. These world offsets have existing development
use; they are not a new held-out confirmation set or per-ID action-trace
qualification. The original untrained control is inherited from the exact
same binary/seed/graph pilot, with raw-log hashes checked. It is not executed
again and is not counted as new evidence.

| World offset | Untrained control (reused) | Final221,184 decisions |
|---|---:|---:|
| 420 | 4.6875% | 6.25% |
| 1337 | 0% | 12.5% |
| Mean over32 episodes | 2.34375% | 9.375% |

All32 newly assigned final episodes finish. Perf is the collection-based
metric, not win rate. Both means rise relative to the untrained control, but
the absolute performance remains poor. Final training-bin perf is0, retained
alongside the evaluation scores. This is independent training n=1. It does
not establish reliable learning, convergence, generalization or Pareto
superiority.

Earlier custom32,768-decision pilot:89.842s,364.73 process SPS, final eval
4.6875%. Earlier Nature32,768-decision pilot:147.219s,222.58 SPS, final eval
9.375%. The longer custom run's score equals that small Nature pilot, but
the unequal budgets and single seeds do not support an architecture-quality
claim. No longer Nature run was authorized/launched here. Earlier state runs
also use a different learner/vector/batch and are not matched comparators.

All four scheduled checkpoints (55,296/110,592/165,888/221,184 decisions) have
the expected78,885,888-byte size, finite float32 values and distinct hashes.
Only the final scheduled checkpoint is evaluated; no best-checkpoint choice.
Intermediate checkpoints are retained without inferred evaluation scores.

## Evidence

Raw packet: `build/sprite-forage-flex-ten-minute-5060-20261015/`, including
owned source snapshots, exact reused binary, requested/resolved INIs, plan,
commands, GPU-idle queries, native logs/metrics, process times and checkpoint
hashes. Compact results, evaluations, plan and training receipt are copied
here with a raw SHA256 manifest. Native checkpoint post-write receipts remain
in the raw training log. Weights remain ignored under build.

The existing shared GPU lock covers native policy work. No training TUI was
streamed, no competing GPU job was launched, and no heavy CPU work overlapped
timed training. Python is process/artifact glue only; all neural work is
native CUDA. No system/CUDA/Torch/dependency change, CPU CNN, new dataset,
5090 job, production-core edit, commit or push. No automatic retry or further
training follows from this report; all prior allocations remain intact.
