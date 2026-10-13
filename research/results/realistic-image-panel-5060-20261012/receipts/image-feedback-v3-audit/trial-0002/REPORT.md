# Native CNN multi-game development panel

Status: ok. 6/6 training jobs; 41/41 drawing evaluations; 12 exact-byte repeat/eager checks.

One CNN architecture per candidate across all six games; learners are fixed per game. Each checkpoint is evaluated on every fixed drawing with paired episode identities/RNG. Training is charged once per job. All scores, including zero scores and censoring, are retained. No cross-game raw-score average or best-game winner.

Short plumbing results are not learning evidence, independent numerical acceptance, calibrated budgets/caps or a publication frontier.

| Game | Candidate | Process seconds | Process SPS | Parameters |
|---|---|---:|---:|---:|
| connect4cnn | swift-fox-3 | 0.866 | 74 | 56,232 |
| pongcnn | swift-fox-3 | 0.716 | 89 | 55,720 |
| flappycnn | swift-fox-3 | 0.665 | 96 | 55,592 |
| breakoutcnn | swift-fox-3 | 0.565 | 113 | 55,720 |
| snakecnn | swift-fox-3 | 0.716 | 89 | 55,848 |
| mazecnn | swift-fox-3 | 0.866 | 74 | 55,976 |

Native metrics retain SPS/uptime/performance arrays in their original INIs. Process SPS above includes startup/checkpoints and is not native average SPS. See observations.csv for every drawing/score and analysis.json for complete counts and missing cells.
