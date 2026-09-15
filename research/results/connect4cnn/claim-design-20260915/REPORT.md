# Connect4 claim design: development evidence only

**Superseded design:** these time slices and nine-contrast power calculations do not define or size the current full-frontier study. See research/CONNECT4_CLAIM_PROTOCOL.md. Historical numbers are retained, not new confirmation evidence.

Existing results now inform design; they are not new confirmation data. Hardware/source cohorts are kept separate.
Latest eligible checkpoint is selected without consulting its score. No interpolation, best-checkpoint selection, or zero imputation.

| Cohort | Seconds | Model | Available seeds | Mean wins |
|---|---:|---|---:|---:|
| 5060_historical | 60 | flex_quality | 5/5 | 52.83% |
| 5060_historical | 60 | nature_cnn | 5/5 | 24.66% |
| 5060_historical | 60 | impala_cnn | 0/5 | missing |
| 5060_historical | 60 | impoola_cnn | 0/5 | missing |
| 5060_historical | 80 | flex_quality | 5/5 | 69.91% |
| 5060_historical | 80 | nature_cnn | 5/5 | 56.32% |
| 5060_historical | 80 | impala_cnn | 0/5 | missing |
| 5060_historical | 80 | impoola_cnn | 0/5 | missing |
| 5060_historical | 100 | flex_quality | 5/5 | 77.36% |
| 5060_historical | 100 | nature_cnn | 5/5 | 68.04% |
| 5060_historical | 100 | impala_cnn | 5/5 | 0.18% |
| 5060_historical | 100 | impoola_cnn | 5/5 | 0.10% |
| 5060_historical | 120 | flex_quality | 5/5 | 78.83% |
| 5060_historical | 120 | nature_cnn | 5/5 | 71.20% |
| 5060_historical | 120 | impala_cnn | 5/5 | 0.18% |
| 5060_historical | 120 | impoola_cnn | 5/5 | 0.10% |
| 5060_historical | 140 | flex_quality | 5/5 | 79.64% |
| 5060_historical | 140 | nature_cnn | 5/5 | 73.60% |
| 5060_historical | 140 | impala_cnn | 5/5 | 0.18% |
| 5060_historical | 140 | impoola_cnn | 5/5 | 0.10% |
| 5090_single_seed | 30 | flex_quality | 1/1 | 29.70% |
| 5090_single_seed | 30 | nature_cnn | 1/1 | 20.59% |
| 5090_single_seed | 30 | impala_cnn | 0/1 | missing |
| 5090_single_seed | 30 | impoola_cnn | 0/1 | missing |
| 5090_single_seed | 45 | flex_quality | 1/1 | 68.36% |
| 5090_single_seed | 45 | nature_cnn | 1/1 | 35.49% |
| 5090_single_seed | 45 | impala_cnn | 1/1 | 0.36% |
| 5090_single_seed | 45 | impoola_cnn | 1/1 | 0.00% |
| 5090_single_seed | 60 | flex_quality | 1/1 | 71.58% |
| 5090_single_seed | 60 | nature_cnn | 1/1 | 59.78% |
| 5090_single_seed | 60 | impala_cnn | 1/1 | 0.36% |
| 5090_single_seed | 60 | impoola_cnn | 1/1 | 0.00% |
| 5090_single_seed | 75 | flex_quality | 1/1 | 71.84% |
| 5090_single_seed | 75 | nature_cnn | 1/1 | 65.17% |
| 5090_single_seed | 75 | impala_cnn | 1/1 | 0.36% |
| 5090_single_seed | 75 | impoola_cnn | 1/1 | 12.48% |

## Power sensitivity — assumed effects, not observed significance

One-sided alpha .05/9 for three references at three eligible budgets. Null margin is zero; this does not test that improvement exceeds five percentage points.
| Paired SD | Assumed advantage | Approximate seeds for 80% power | Approximate power at 40 |
|---:|---:|---:|---:|
| 5% | 5% | 12 | 100.0% |
| 5% | 10% | 3 | 100.0% |
| 5% | 15% | 2 | 100.0% |
| 10% | 5% | 46 | 73.3% |
| 10% | 10% | 12 | 100.0% |
| 10% | 15% | 6 | 100.0% |
| 15% | 5% | 103 | 33.3% |
| 15% | 10% | 26 | 95.3% |
| 15% | 15% | 12 | 100.0% |
| 20% | 5% | 183 | 16.9% |
| 20% | 10% | 46 | 73.3% |
| 20% | 15% | 21 | 98.6% |

The five-seed historical variance is unstable and not a 5090 variance estimate. The single 5090 seed cannot estimate training-seed variance.
Old checkpoints use a full 13.312M-decision annealing schedule and coarse checkpoint intervals. A new wall-budget protocol must validate timing/availability and cannot inherit these scores.
See research/CONNECT4_CLAIM_PROTOCOL.md for the replacement full-frontier/common-learner design and open validation gates.
