# Native CNN multi-game development panel

Status: ok. 6/6 training jobs; 41/41 drawing evaluations; 12 exact-byte repeat/eager checks.

One CNN architecture per candidate across all six games; learners are fixed per game. Each checkpoint is evaluated on every fixed drawing with paired episode identities/RNG. Training is charged once per job. All scores, including zero scores and censoring, are retained. No cross-game raw-score average or best-game winner.

Short plumbing results are not learning evidence, independent numerical acceptance, calibrated budgets/caps or a publication frontier.

| Game | Candidate | Process seconds | Process SPS | Parameters |
|---|---|---:|---:|---:|
| connect4cnn | mystic-tree-2 | 0.715 | 91,705 | 160,736 |
| pongcnn | mystic-tree-2 | 0.464 | 141,138 | 160,224 |
| flappycnn | mystic-tree-2 | 0.414 | 158,225 | 160,096 |
| breakoutcnn | mystic-tree-2 | 0.464 | 141,114 | 160,224 |
| snakecnn | mystic-tree-2 | 0.414 | 158,201 | 160,352 |
| mazecnn | mystic-tree-2 | 0.514 | 127,434 | 160,480 |

Native metrics retain SPS/uptime/performance arrays in their original INIs. Process SPS above includes startup/checkpoints and is not native average SPS. See observations.csv for every drawing/score and analysis.json for complete counts and missing cells.
