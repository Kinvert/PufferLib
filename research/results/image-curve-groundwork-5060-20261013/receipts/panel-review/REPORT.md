# Native CNN multi-game development panel

Status: ok. 6/6 training jobs; 369/369 drawing evaluations; 12 exact-byte repeat/eager checks.

One CNN architecture per candidate across all six games; learners are fixed per game. Each checkpoint is evaluated on every fixed drawing with paired episode identities/RNG. Training is charged once per job. All scores, including zero scores and censoring, are retained. No cross-game raw-score average or best-game winner.

Short plumbing results are not learning evidence, independent numerical acceptance, calibrated budgets/caps or a publication frontier.

| Game | Candidate | Process seconds | Process SPS | Parameters |
|---|---|---:|---:|---:|
| connect4cnn | happy-cat-1 | 2.574 | 199 | 60,616 |
| pongcnn | happy-cat-1 | 2.523 | 203 | 60,104 |
| flappycnn | happy-cat-1 | 1.973 | 259 | 59,976 |
| breakoutcnn | happy-cat-1 | 1.468 | 349 | 60,104 |
| snakecnn | happy-cat-1 | 2.421 | 212 | 60,232 |
| mazecnn | happy-cat-1 | 2.571 | 199 | 60,360 |

Native metrics retain SPS/uptime/performance arrays in their original INIs. Process SPS above includes startup/checkpoints and is not native average SPS. See observations.csv for every drawing/score and analysis.json for complete counts and missing cells.
