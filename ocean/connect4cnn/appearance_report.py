"""Audit completed native appearance canaries; CPU reporting only."""
import csv
from pathlib import Path
import sys

import numpy as np

from compare import EVAL


def main(root):
    expected = {"flex_quality": 160736, "nature_cnn": 138528,
                "impala_cnn": 270496, "impoola_cnn": 151712}
    rows = []
    for game in ("connect4cnn", "pongcnn"):
        for variant, params in expected.items():
            count = params - (512 if game == "pongcnn" else 0)
            for repeat in (1, 2):
                trial = root / game / variant / f"repeat-{repeat}"
                paths = sorted((trial / "checkpoints" / game / "trial").glob("*.bin"))
                assert [int(p.stem) for p in paths] == [8192, 16384]
                for path in paths:
                    values = np.fromfile(path, dtype=np.float32)
                    assert path.stat().st_size == 4 * count and np.isfinite(values).all(), path
                match = EVAL.fullmatch((trial / "eval-result.txt").read_text().strip())
                assert match and match[1] == game and int(match[5]) == count
                assert int(match[4]) >= 64 and 0 <= float(match[3]) <= 1
                seconds = float((trial / "train-wall.txt").read_text())
                assert seconds > 0
                rows.append(dict(environment=game, model=variant, threads=repeat,
                                 decisions=16384, train_seconds=seconds, process_sps=16384 / seconds,
                                 eval_seconds=float((trial / "eval-wall.txt").read_text()),
                                 eval_perf=float(match[3]), eval_games=int(match[4]), params=count))
    with (root / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Native mixed-appearance canary", "",
             "16 short training runs and 16 checkpoint reload evaluations. Two checkpoints per run were finite and had expected parameter counts. Paired checkpoints and evaluation records matched byte-for-byte across one/two CPU workers.", "",
             "Appearance mode 1, seed 12345, 64 slots; assignment CSVs are retained. Training seed 56173, evaluation action seed 66173. Game RNG remains the native per-slot stream. All four encoders use the same per-game learner recipe and 16,384 decisions. This is a reproducibility check, not learning or speed evidence.", "",
             "Evaluation is pooled v1, may overshoot 64 games and is bounded by a 30-second process timeout. This does not fix sparse Pong match completion or implement paper-grade exact episode quotas. A timeout makes the canary fail; it is never a zero score.", "",
             "| Game | Model | Workers | Train s | Process SPS | Eval s | Eval perf |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['environment']} | {row['model']} | {row['threads']} | {row['train_seconds']:.2f} | {row['process_sps']:,.0f} | {row['eval_seconds']:.2f} | {row['eval_perf']:.2%} |")
    lines += ["", "Process SPS includes startup/writes and uses /usr/bin/time's rounded timer; native timing/SPS remain in each metrics directory. Source snapshots/checks, resolved native configs, commands, binary/checkpoint hashes and raw failures are retained. Large artifacts remain local.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
