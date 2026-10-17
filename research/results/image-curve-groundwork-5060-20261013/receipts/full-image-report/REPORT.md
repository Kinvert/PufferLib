# Audited realistic-image results

Termination: ok. 6 training jobs; 54 checkpoints. All declared games/drawings remain.

[Process times/SPS](training.csv) · [Monotonic checkpoint costs](checkpoints.csv) · [Raw native SPS/uptime bins](native-bins.csv) · [Every anchor](anchors.csv) · [Selected affordable checkpoints](anchor-selections.csv)

- [panel: all games/drawings/checkpoints](panels/panel/curves.html)

Process SPS includes startup and checkpoint writes. Checkpoint intervals include writes. Native bins retain the trainer's own clock and are not averaged into process SPS. An uptime exceeding process duration is retained and flagged, not corrected.

The initial policy has a measured positive startup/checkpoint cost and no invented training SPS. Discovery curves are descriptive; fresh paired confirmation and statistical inference remain required. No superiority, generalization or SOTA claim follows from this report.
