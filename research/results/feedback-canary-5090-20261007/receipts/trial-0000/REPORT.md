# Native CNN multi-game development panel

Status: ok. 6/6 training jobs; 41/41 drawing evaluations; 12 exact-byte repeat/eager checks.

One CNN architecture per candidate across all six games; learners are fixed per game. Each checkpoint is evaluated on every fixed drawing with paired episode identities/RNG. Training is charged once per job. All scores, including zero scores and censoring, are retained. No cross-game raw-score average or best-game winner.

Short plumbing results are not learning evidence, independent numerical acceptance, calibrated budgets/caps or a publication frontier.

| Game | Candidate | Process seconds | Process SPS | Parameters |
|---|---|---:|---:|---:|
| connect4cnn | happy-cat-1 | 0.765 | 85,691 | 160,736 |
| pongcnn | happy-cat-1 | 0.464 | 141,122 | 160,224 |
| flappycnn | happy-cat-1 | 0.464 | 141,113 | 160,096 |
| breakoutcnn | happy-cat-1 | 0.464 | 141,200 | 160,224 |
| snakecnn | happy-cat-1 | 0.464 | 141,139 | 160,352 |
| mazecnn | happy-cat-1 | 0.514 | 127,380 | 160,480 |

Native metrics retain SPS/uptime/performance arrays in their original INIs. Process SPS above includes startup/checkpoints and is not native average SPS. See observations.csv for every drawing/score and analysis.json for complete counts and missing cells.
