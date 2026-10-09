# Independent analysis of committed 5090 exports

All seven export tables pass cross-file consistency checks. This is not an independent raw-packet audit.

Final fixed-anchor aggregate frontier (all 14 models retained in analysis.json):

| Model | Score | Total paired training seconds |
|---|---:|---:|
| swift-fox-3 | 0.102099 | 564.535 |
| happy-cat-1 | 0.311465 | 590.555 |
| quality-reference | 0.353913 | 631.885 |

Aggregate scores are supplied audited-export values. Clipping occurs before averaging;
they cannot generally be reconstructed from game means alone.

Quality versus Nature at final budgets:

| Game | Quality score/bounds | Nature score/bounds | Quality time reduction | Quality / Nature process SPS |
|---|---:|---:|---:|---:|
| connect4cnn | 0.4379 | 0.4258 | 3.14% | 177,006 / 171,427 |
| pongcnn | 0.4157–0.7269 | 0.1563–0.6306 | 4.48% | 372,839 / 356,136 |
| flappycnn | 24.3961 | 33.0411 | 2.23% | 474,686 / 464,323 |
| breakoutcnn | 9.4455 | 6.4606 | 6.46% | 369,120 / 345,539 |
| snakecnn | 6.1465 | 7.1894 | 5.15% | 442,844 / 419,036 |
| mazecnn | 0.2828 | 0.2298 | 6.61% | 297,495 / 277,987 |

Complete checkpoint frontiers follow: every model's four scheduled checkpoints participates,
including declines and duplicate trials. Drawings are averaged equally within each game,
then the two paired seeds are averaged. Time is mean once-per-job checkpoint cost.
No drawings or seeds are treated as independent training replicas.

## breakoutcnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| happy-cat-1 | 4,194,304 | 11.035 | 5.4273 |
| quiet-owl-11 | 4,194,304 | 11.044 | 5.7788 |
| nature-cnn | 4,194,304 | 12.304 | 5.9242 |
| quiet-owl-7 | 4,194,304 | 13.621 | 6.8061 |
| quiet-owl-8 | 4,194,304 | 14.258 | 6.8152 |
| quality-reference | 12,582,912 | 33.865 | 8.6909 |
| quality-reference | 16,777,216 | 45.297 | 9.4455 |

## connect4cnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| swift-fox-3 | 3,328,000 | 19.470 | 0.0000 |
| happy-cat-1 | 3,328,000 | 19.587 | 0.0030 |
| quality-reference | 3,328,000 | 19.925 | 0.0045 |
| quiet-owl-11 | 3,328,000 | 20.597 | 0.0136 |
| quiet-owl-4 | 3,328,000 | 31.591 | 0.0409 |
| happy-cat-1 | 6,656,000 | 37.930 | 0.2530 |
| quality-reference | 6,656,000 | 38.638 | 0.3182 |
| nature-cnn | 6,656,000 | 39.953 | 0.3667 |
| quality-reference | 9,984,000 | 56.733 | 0.4182 |
| nature-cnn | 9,984,000 | 58.632 | 0.4227 |
| quality-reference | 13,312,000 | 75.076 | 0.4379 |

## flappycnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| swift-fox-3 | 4,980,736 | 9.270 | 0.8355 |
| happy-cat-1 | 4,980,736 | 9.630 | 3.2338 |
| quiet-owl-12 | 4,980,736 | 9.970 | 4.3225 |
| quality-reference | 4,980,736 | 10.653 | 5.2056 |
| happy-cat-1 | 9,961,472 | 19.060 | 6.1667 |
| quiet-owl-11 | 9,961,472 | 19.113 | 6.8810 |
| quiet-owl-12 | 9,961,472 | 19.718 | 7.1212 |
| quality-reference | 9,961,472 | 21.090 | 13.2706 |
| nature-cnn | 9,961,472 | 21.679 | 18.4221 |
| quiet-owl-12 | 14,942,208 | 29.431 | 19.0108 |
| nature-cnn | 14,942,208 | 32.223 | 20.0714 |
| quiet-owl-7 | 14,942,208 | 38.294 | 21.6017 |
| quality-reference | 19,922,944 | 41.811 | 24.3961 |
| nature-cnn | 19,922,944 | 42.765 | 33.0411 |

## mazecnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| swift-fox-3 | 8,388,608 | 22.982 | 0.1364 |
| quiet-owl-11 | 8,388,608 | 25.550 | 0.1717 |
| quiet-owl-12 | 8,388,608 | 26.601 | 0.1742 |
| quiet-owl-11 | 16,777,216 | 50.841 | 0.2348 |
| quiet-owl-11 | 33,554,432 | 102.918 | 0.2702 |
| quality-reference | 33,554,432 | 112.596 | 0.2828 |

## pongcnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| swift-fox-3 | 2,097,152 | 4.972 | 0.0000 |
| happy-cat-1 | 2,097,152 | 5.124 | 0.5295–0.5357 |
| happy-cat-1 | 4,194,304 | 10.042 | 0.5635 |
| happy-cat-1 | 6,291,456 | 14.947 | 0.5980–0.6141 |

The lower-endpoint frontier uses conservative censoring scores. It does not certify
true win-rate dominance; overlapping lower/upper bounds remain unresolved. The CSV
also retains the upper-endpoint frontier and dominance using nonoverlapping bounds.

## snakecnn

Lower-endpoint descriptive frontier:

| Model | Decisions | Mean train seconds | Score/bounds |
|---|---:|---:|---:|
| swift-fox-3 | 2,097,152 | 4.375 | 2.2551 |
| happy-cat-1 | 2,097,152 | 4.464 | 3.2702 |
| quiet-owl-12 | 2,097,152 | 4.624 | 3.3460 |
| nature-cnn | 2,097,152 | 5.124 | 3.3636 |
| quiet-owl-10 | 2,097,152 | 7.703 | 3.5985 |
| happy-cat-1 | 4,194,304 | 8.689 | 4.2222 |
| quiet-owl-12 | 4,194,304 | 9.003 | 4.5227 |
| nature-cnn | 4,194,304 | 10.003 | 5.1515 |
| happy-cat-1 | 6,291,456 | 12.951 | 5.6212 |
| quiet-owl-12 | 6,291,456 | 13.433 | 5.6793 |
| quality-reference | 6,291,456 | 14.141 | 5.8056 |
| nature-cnn | 6,291,456 | 14.903 | 6.9848 |
| nature-cnn | 8,388,608 | 19.811 | 7.1894 |

Adaptive discovery seeds, two seeds per model, no independent episode/appearance
rows here and no full-trainer numerical qualification: no publication/SOTA claim.
