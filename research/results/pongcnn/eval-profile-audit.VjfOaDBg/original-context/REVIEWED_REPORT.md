# Pong RTX 5090: reviewed results

**40/40 training runs finished; 312/320 evaluations completed. Eight evaluations hit their fixed 300-second caps.** No retries or extra seeds were run. All 320 checkpoint arrays and hashes, all 312 successful evaluation logs, frozen configs/sources/binaries, final bootstrap statistics and observed mean frontiers passed independent review.

End-to-end launcher time: **3.217 hours (11581.80 s)**. Training sum: **4923.91 s**; evaluation sum: **6611.45 s**; build sum: **16.58 s**. The original campaign elapsed excludes final analysis/package; the launcher elapsed includes them. All jobs were serial.

Native PufferLib PongCNN, not Atari/ALE. Scores below are episode-averaged **fraction of points won**, not match win rate. Each model/recipe has five paired training seeds and 4,194,304 decisions per seed, with H128/L1 and float32.

## Final quality and target attainment

| Recipe | Model | Final seeds observed | Final mean point fraction [pointwise 95% CI] | Missing-result bounds | Mean point-score difference | Observed ≥95% seeds | Target unknown | Sustained after first crossing |
|---|---|---:|---|---|---:|---:|---:|---:|
| A | Ours quality | 4/5 | Unavailable | 68.42%–88.42% | Unavailable | 3/5 | 1 | 3/5 |
| A | Nature | 4/5 | Unavailable | 69.46%–89.46% | Unavailable | 3/5 | 1 | 3/5 |
| A | IMPALA | 5/5 | 79.64% [39.66%, 99.96%] | — | 12.52 | 4/5 | 0 | 3/5 |
| A | Impoola | 5/5 | 99.91% [99.85%, 99.95%] | — | 20.98 | 5/5 | 0 | 5/5 |
| B | Ours quality | 5/5 | 29.90% [0.36%, 69.55%] | — | -8.61 | 1/5 | 0 | 0/5 |
| B | Nature | 4/5 | Unavailable | 79.66%–99.66% | Unavailable | 4/5 | 1 | 3/5 |
| B | IMPALA | 5/5 | 59.37% [19.68%, 99.07%] | — | 4.06 | 3/5 | 0 | 3/5 |
| B | Impoola | 5/5 | 0.00% [0.00%, 0.00%] | — | -21.00 | 0/5 | 0 | 0/5 |

Missing-result bounds are the possible mean over the five observed training seeds if each missing point fraction lies anywhere in [0,1]. **They are not confidence intervals.** No mean over only the available four seeds substitutes for a five-seed result. Unknown target attainment is separate from a recorded failure to reach 95%.

Intervals use the frozen 10,000 paired whole-seed bootstrap draws. The observed-attainment intervals in the original generated report describe the recorded crossings; they do not resolve missing evaluations. Degenerate bootstrap intervals at 0/5 or 5/5 are not population guarantees. First crossings and subsequent declines are distinct; a later recovery does not erase a decline. The original per-seed crossing table and `target-attainment.csv` retain all 40 seeds, non-achievers and checkpoint-resolution brackets.

## Training efficiency (means across all five seeds)

| Recipe | Model | Parameters | Train wall s | Process SPS | Native average SPS | Final reported VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| A | Ours quality | 160,224 | 8.69 | 483,277 | 498,604 | 3.198 |
| A | Nature | 138,016 | 9.36 | 448,386 | 461,506 | 3.281 |
| A | IMPALA | 269,984 | 131.92 | 31,810 | 31,875 | 5.239 |
| A | Impoola | 151,200 | 132.61 | 31,651 | 31,716 | 5.215 |
| B | Ours quality | 160,224 | 13.52 | 310,340 | 316,616 | 3.137 |
| B | Nature | 138,016 | 14.79 | 284,628 | 290,016 | 3.304 |
| B | IMPALA | 269,984 | 338.47 | 12,411 | 12,421 | 5.257 |
| B | Impoola | 151,200 | 335.42 | 12,510 | 12,520 | 5.254 |

Process timing includes startup and checkpoint writes. Native SPS uses adjusted trainer time. These are means of per-seed SPS, and VRAM is the final reported sample, not peak memory. Native point-score differences and complete raw metric histories remain in the original summary and per-job metrics.

## Interpretation of the entire observed frontier

- **Recipe A:** ours and Nature occupy portions of the low-time mean frontier; Impoola reaches the high-score region most consistently. Impoola reaches 95% in all five seeds and finishes at 99.91% mean point fraction. Ours and Nature each have three observed target crossings and one unknown seed. Their full final means are unresolved by the timed-out evaluations.
- **Recipe B:** Nature owns nearly all of the observed mean time and decision frontiers; ours contributes the cheapest initial time point. Nature reaches the target in four seeds, with one unknown, while ours reaches it in one. Impoola fails to learn under this recipe despite its strong recipe-A result.
- The mean frontiers use only checkpoints with all five paired evaluations; later missing ours/Nature points can change the complete frontier. All 312 valid individual observations and their per-seed fronts remain present. Pointwise score/time intervals and bootstrap membership frequencies are descriptive uncertainty, not a frontier-wide dominance test.
- These results do not establish a quality advantage over Nature. Speed depends on these native backends and the locked learner recipes. The architectures were selected on Connect4, but Pong informed learner development. Five seeds and two data-informed shared recipes do not establish an unbiased best-tuned or universal/SOTA comparison.

## Preserved timeout failures

| Recipe | Model | Seed | Checkpoint decisions | Timeout seconds |
|---|---|---:|---:|---:|
| A | Ours quality | 31002 | 2,621,440 | 300.10 |
| A | Ours quality | 31002 | 3,145,728 | 300.09 |
| A | Ours quality | 31002 | 3,670,016 | 300.09 |
| A | Ours quality | 31002 | 4,194,304 | 300.09 |
| B | Nature | 31004 | 4,194,304 | 300.09 |
| A | Nature | 31005 | 3,145,728 | 300.10 |
| A | Nature | 31005 | 3,670,016 | 300.10 |
| A | Nature | 31005 | 4,194,304 | 300.09 |

## Reproduction and evidence

Native source commit: `1d9e9e0dca88706c7a8495644c8dc584a4446f11`. Frozen protocol: `7627852d943d2b2777db6fd6874cd6eaffaca64c8a6a95b6c6acd44771125571`. Review script SHA256: `bdac3eec307b27f408b17e1ebf1222069270e9a127e7270a7964465ecea209f6`.
Hardware: RTX 5090 / Ryzen 9 9950X3D, CUDA compiler 13.1.115, driver 580.105.08; exact receipts are in the parent directory.

The original REPORT.md, results.csv, protocol, source snapshots, weights, analysis arrays and first archive are unchanged. This review supplies continuous Markdown tables and explicit missing-target labels. Original score/failure data and the frozen analysis rules were not altered.

- [Original full report and per-seed crossing times](../REPORT.md)
- [All planned checkpoint observations and failures](../results.csv)
- [All raw and mean curves](../curves.html)
- [Mean points/frontiers and uncertainty](../mean-curves.csv)
- [Individual seed frontiers](../seed-frontiers.csv)
- [Target crossings, non-achievers and later declines](../target-attainment.csv)
- [Independent audit](audit.json)

```bash
source build/connect4cnn/runtime-5090.sh
bash ocean/pongcnn/confirm.sh --full
bash ocean/pongcnn/confirm.sh --resume /home/keith/Git/ml/cnn-5090/build/pongcnn/confirm-5090.full.JXYKB06r
.venv/bin/python research/review_pong_5090.py /home/keith/Git/ml/cnn-5090/build/pongcnn/confirm-5090.full.JXYKB06r
```

A finished campaign resumes as a no-op; a fresh --full would create a new experiment and is a reproduction command, not a request to rerun this one. No additional training was performed for this review.
