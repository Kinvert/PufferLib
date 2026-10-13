"""Synthetic file/report checks; no game or neural model executes."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
import report_hardware_run as hardware


class HardwareReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "synthetic"
        self.run.mkdir()
        selected = json.loads((ROOT / "ocean/connect4cnn/confirmation.json").read_text())
        models = ["flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn"]
        steps = list(range(1024000, 13312001, 1024000))
        source = self.run / "source/ocean/connect4cnn/confirmation.json"
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps(selected))
        protocol = dict(seeds=[173], eval_seed=20173, steps=13312000, completed_steps=13312000,
                        checkpoint_steps=steps, variants={m: "connect4cnn" for m in models},
                        precision="float32", evaluation_protocol="pooled-v1", revision="synthetic-not-a-run",
                        environment_rules="connect4-full-board-draw-v2",
                        source_sha256={"ocean/connect4cnn/confirmation.json": hardware.sha256(source)}, binary_sha256={})
        rows, jobs = [], []
        for model in models:
            binary = self.run / model
            binary.write_bytes(b"synthetic, not executable")
            protocol["binary_sha256"][model] = hardware.sha256(binary)
            directory = self.run / f"{model}-s173"
            checkpoints = directory / "checkpoints/connect4cnn/trial"
            checkpoints.mkdir(parents=True)
            job = dict(variant=model, seed=173, status="ok", checkpoint_sha256={})
            count = selected["parameters"][model]
            for index, step in enumerate(steps, 1):
                checkpoint = checkpoints / f"{step:016d}.bin"
                np.zeros(count, dtype=np.float32).tofile(checkpoint)
                job["checkpoint_sha256"][checkpoint.name] = hardware.sha256(checkpoint)
                perf = index / 20
                (directory / f"eval-{step:016d}.log").write_text(
                    f"CUDA_EVAL env=connect4cnn score=0 perf={perf} games=1024 params={count}\n")
                rows.append(dict(variant=model, seed=173, eval_seed=20173, steps=step, score=0,
                                 win_rate=perf, games=1024, params=count, checkpoint=str(checkpoint.relative_to(self.run)),
                                 checkpoint_wall_s=index, train_process_wall_s=14, process_sps=13312000/14,
                                 native_avg_sps=13312000/13))
            jobs.append(job)
        for name, value in (("protocol.json", protocol), ("jobs.json", jobs),
                            ("finished.json", dict(status="ok", jobs=4, evaluations=52, logging_failures=0))):
            (self.run / name).write_text(json.dumps(value))
        (self.run / "gpu.txt").write_text("NVIDIA GeForce RTX 5060\n")
        with (self.run / "results.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_archive_has_whole_curves_and_excludes_weights(self):
        out = self.root / "report"
        hardware.archive(self.run, out, "5060")
        result = json.loads((out / "analysis/analysis.json").read_text())
        self.assertEqual(len(result["points"]), 52)
        self.assertEqual(result["environment_rules"], "connect4-full-board-draw-v2")
        self.assertEqual(result["claim_status"], "exploratory_only_inference_and_runtime_gates_pending")
        self.assertTrue((out / "analysis/curves.html").is_file())
        self.assertFalse(list(out.rglob("*.bin")))
        with self.assertRaises(ValueError):
            hardware.archive(self.run, out, "5060")

    def test_rejects_wrong_host_and_changed_checkpoint(self):
        out = self.root / "report"
        with self.assertRaisesRegex(ValueError, "Wrong hardware"):
            hardware.archive(self.run, out, "5090")
        checkpoint = next(self.run.rglob("*.bin"))
        checkpoint.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
            hardware.archive(self.run, out, "5060")
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
