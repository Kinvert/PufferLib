# Full-frame compression first-increment report

Status: **incomplete**. Encoder correctness and short deterministic integration only.
Source frames: Flappy640x420 RGB; Pong640x620 RGB. No sprite/ROI shortcuts.

ProcessSPS includes startup and checkpoint writes; these8192-decision jobs do not establish steady-state speed or learning.

|Game|Model|Repeat|Process seconds|Process SPS|Parameters|
|---|---:|---:|---:|---:|---:|

Model0: learned strided compression. Model1: area4 plus matched small CNN. Model2: area4 plus resized RGB Nature/H128.

Read REPORT.json and native receipts for exact coverage. Numerical checks cover the declared fixtures, not arbitrary architectures or independent whole-learner mathematics.
No pretrained learning, held-out generalization, Pareto or SOTA result is claimed.
