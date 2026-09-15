# Pong evaluation/backend audit received from RTX 5090

The [received evidence and local audit](results/pongcnn/eval-profile-audit.VjfOaDBg/RECEIVED.md) preserve the complete 12m51s remote diagnostic campaign. Package integrity passed for all 764 manifest entries. G240 independently checked 57 raw attempt receipts and 90 captured source hashes. Twelve successful old/new evaluation pairs match exactly; eight originally timed-out checkpoint pairs timed out in both diagnostic arms. Actual checkpoints, binaries and full Nsight databases remain on the 5090, so their deeper remote audits have not been repeated here.

## Evaluation finding

The eight timeout checkpoints advanced **14.64–18.22 million decisions** by their last progress sample around 25 seconds, but completed only **0–13 matches** against the 512-match target. Reported interval throughput was approximately **586k–733k decisions/s**. This establishes active simulation with sparse match completion, not a hard execution stall. Whether particular trajectories are indefinitely nonterminating remains unresolved; aggregate logs do not prove that.

The telemetry patch preserved all completed paired results, but successful-control old/new wall ratios had a median near **0.998**, with no consistent speedup. Keep the fix as reduced unnecessary telemetry and useful progress reporting, not as a solution to long Pong evaluation.

Original evaluation consumed 6,611.45s versus 4,923.91s training; eight original timeout attempts consumed 2,400.75s. Eight checkpoint evaluations per training run plus substantial timeout cost explain the aggregate duration. Diagnostic partial matches do not replace missing original outcomes.

## Baseline profiling finding

[Profiler table and limits](results/pongcnn/eval-profile-audit.VjfOaDBg/audit/PROFILE.md): the IMPALA/Impoola traces attribute approximately **61–64% of summed kernel time to shared GEMMs**, **19–22% to patch materialization**, **5–6% to bias gradients**, and about **2% to pooling**. These are traced GPU-duration fractions, not wall-time fractions. Shared matrix kernels serve several callers, so per-layer and forward/input-gradient/weight-gradient attribution is still incomplete. No occupancy/bandwidth hardware counters or isolated telemetry CPU cost were measured.

Short diagnostic runs used 131,072 decisions; recipe A executed 64 effective updates, B 192. The native replay setting truncates these recipes to one versus three updates per rollout. This accounts for extra learner work under B and must not be confused with a different architecture. The remote audit reports all eight instrumented/unprofiled checkpoint pairs bit-identical; bytes are not transferred here.

For ours/Nature, coarse copy time around 1.1–1.25ms per rollout is comparable to 1.3–1.5ms inference. Larger references have much greater model compute. These measurements prioritize caller attribution, patch/reduction paths and small-model transfer overhead; they do not prove that current reference speed is an intrinsic architectural limitation.

## Original quality conclusions stay bounded

Original totals remain **40 training runs, 312 successful evaluations, eight timeouts**. [Original reviewed report](results/pongcnn/eval-profile-audit.VjfOaDBg/original-context/REVIEWED_REPORT.md) and [curves](results/pongcnn/eval-profile-audit.VjfOaDBg/original-context/curves.html) are now available locally as transferred evidence. Recipe A has ours/Nature in the observed low-time mean frontier and Impoola at high performance; B favors Nature across nearly all observed mean-frontier points. Missing evaluation outcomes leave parts of the frontier unidentified. These are not superiority/SOTA results, and pointwise five-seed intervals do not establish dominance across a frontier.

## Recommended next work — not launched

1. Review and implement a separately versioned evaluator with fixed episode identities/allocation, explicit match-decision limits, recurrent-reset correctness, per-episode records and honest truncation bounds. [Remote proposal](results/pongcnn/eval-profile-audit.VjfOaDBg/audit/PONG_EVALUATION_POLICY_PROPOSAL.md) is a starting design; numeric caps and exact allocation/masking behavior remain to be frozen and tested. Never splice its scores into v1 results.
2. Attribute the hot matrix multiplications to layers/directions, then test same-math backend optimizations with numerical/gradient/repeatability checks and unprofiled measurements. Pooling alone is a small fraction of this measured cost.
3. Continue the portable native constructor and established-benchmark integration after the evaluation/measurement foundation is dependable. Do not launch another full quality sweep simply to fill missing values.

No archived scripts/patches were executed or applied, no new training was launched, and no commit/push was performed for this receipt audit.
