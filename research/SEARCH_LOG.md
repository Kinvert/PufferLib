# Literature search scope and follow-up

Research date: September 11, 2026, America/Los_Angeles. This was a focused primary-source search plus local PufferLib 5.0 inspection, not a systematic review or a complete current leaderboard audit.

## Search passes

| Pass | Representative queries / source traversal | Result |
|---|---|---|
| Verify the supplied conversation | Nature DQN 2015; IMPALA 2018; BBF width scaling; Impoola 2503.05546; Hilton RL scaling | Established encoder references and corrected the BBF interpretation. |
| Encoder/core allocation | `reinforcement learning encoder recurrent compute allocation`, `encoder recurrent compute-optimal`, DreamerV3 size appendix, IL scaling | Found related scaling methodology, but no directly transferable optimum for this native PufferLib setting. |
| Encoder-specific improvements | Follow Impoola references; observational overfitting; global average pooling; SEER stored embeddings | Added spatial-readout, observation-generalization, and freezing/caching evidence. |
| Hardware efficiency | MobileNetV2, ShuffleNet V2, NVIDIA convolution/CUTLASS/cuBLAS/cuDNN documentation | Separated FLOPs, parameter count, actual runtime, and determinism conditions. |
| Current work | Pixel RL efficiency and policy compute queries with 2025/2026; primary arXiv pages | Added PlayTrain and On the Role of Computation in RL. Search results about using RL to optimize unrelated CNN applications were excluded. |
| Evaluation and memory | Procgen, Atari 100K versus broad ALE, rliable, POPGym Arcade, Memory Gym, original POPGym | Added protocols and paired observability tasks for the encoder/core study. |
| Existing efficient systems | Cleanba and EnvPool papers/author code | Established relevant compiled and distributed baseline families. |
| Local source audit | `src/algo.cu`, `src/pufferl.cu`, `src/ocean.cu`, `src/protein.cu`, NMMO3, Breakout, default configs/tests | Identified the actual core/interface, nonpixel Breakout observations, and native sweep semantics. |

Primary papers are linked in [READING_LIST.md](READING_LIST.md); source receipts and checksums are beside the local PDFs. Research agents used official author implementations and vendor documentation for implementation details. Blog/search snippets were discovery aids, not authority for final technical conclusions.

## Remaining work before claiming novelty or SOTA

- Citation-chain search around encoder architecture and recurrent compute allocation, including newer work citing Impoola and Hilton. Failure to find a universal rule in this pass is not evidence that all component allocation studies are absent.
- Verify current benchmark leaderboards and exact comparable implementations once the target protocol is fixed. Atari 100K, multi-game world models, Procgen generalization, and high-throughput online PPO do not share one leaderboard.
- Read relevant paper code/configurations at pinned commits before reproduction. The local library pins paper copies; it does not clone every author repository.
- Inspect available GPU hardware/software without modifying the system stack. This pass did not establish a target-GPU performance baseline; `nvidia-smi` was unavailable in the current shell view.
- Determine the lowest-effort faithful original pixel-environment integration with this native branch. Do not assume old Python PufferLib wrappers still exist in 5.0.
- Validate numerical precision, optimizer behavior, asynchronous scheduling, and complete deterministic training on actual hardware. Static inspection alone cannot establish those properties.

## Conversion quality notes

Twenty-five selected papers were acquired, covering 646 PDF pages. HTML conversion was preferred for structure and TeX math. Nature DQN, DreamerV3, and POPGym Arcade used PDF extraction; exact methods and counts are in each receipt. DreamerV3's HTML had substantially less text than its PDF, triggering the completeness fallback.

No conversion is an exact typeset substitute. PDF extraction can damage math/table alignment; HTML can omit content or preserve awkward figure/table structures. Retain source PDFs and use their page/figure/table identifiers when recording scientific evidence. The library verifier checks receipt hashes and retrieval provenance; it does not prove every formula was converted correctly.
