"""CPU tests for locked recipes, missing-data semantics and frozen receipts."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ocean/pongcnn"))
import confirm_report as report


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = subprocess.check_output(["bash", "ocean/pongcnn/confirm.sh", "--canary", "--prepare-only"], cwd=ROOT, text=True)
        cls.root = Path(output.split("Pong confirmation: ", 1)[1].strip())

    def test_exact_matrix_and_recipes(self):
        protocol = json.loads((self.root / "protocol.prepared.json").read_text())
        self.assertEqual(len(protocol["order"]), 8)
        self.assertEqual(len(list(self.root.glob("*/evaluations/*/config/pongcnn.ini"))), 32)
        self.assertEqual(protocol["native_update_semantics"]["updates_per_rollout"], {"a": 1, "b": 3})
        for job in report.jobs(self.root):
            ini = report.validate_config(self.root, job)
            self.assertEqual(ini.get("train", "replay_ratio"), report.RECIPES[job["recipe"]]["replay_ratio"])
            self.assertFalse(any(s.startswith("sweep.") for s in ini))
            self.assertIn("bot_eval", ini)
        self.assertIn("fraction of points", protocol["metric"])
        self.assertEqual(protocol["analysis"]["target"], .95)

    def test_model_recipe_environment_and_eval_seed_drift_rejected(self):
        job = report.jobs(self.root)[0]
        for old, new in (("cnn_kernel_1 = 7", "cnn_kernel_1 = 3"),
                         ("frameskip = 8", "frameskip = 4"),
                         ("replay_ratio = 1.91617525", "replay_ratio = 2")):
            path = self.root / job["id"] / "config/default.ini"
            original = path.read_text()
            self.assertIn(old, original)
            try:
                path.write_text(original.replace(old, new))
                with self.assertRaises(AssertionError): report.validate_config(self.root, job)
            finally:
                path.write_text(original)

    def test_failed_and_interrupted_attempts_are_not_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            self.assertEqual(report.attempt_status(p), "pending")
            (p / "started.txt").write_text("1\n")
            self.assertEqual(report.attempt_status(p), "interrupted")
            for code, expected in ((124, "timeout"), (137, "timeout"), (1, "failed"), (0, "ok")):
                (p / "exit-code.txt").write_text(str(code))
                self.assertEqual(report.attempt_status(p), expected)

    def test_collect_preserves_complete_grid_and_missing_scores(self):
        panel, rows = report.collect(self.root)
        self.assertEqual(len(panel), 8)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(r["status"] == "pending" and r["point_fraction"] is None for r in rows))

    def test_evaluation_timeout_retained_without_zero_score(self):
        job = report.jobs(self.root)[0]
        directory = self.root / job["id"] / "evaluations/0000000000016384"
        try:
            (directory / "started.txt").write_text("1\n")
            (directory / "exit-code.txt").write_text("124\n")
            (directory / "wall.txt").write_text("Command exited with non-zero status 124\n120.00\n")
            result = report.audit_eval(self.root, job, "0000000000016384")
            self.assertEqual(result, {"status": "timeout", "eval_wall_s": 120.0})
            self.assertNotIn("point_fraction", result)
        finally:
            for name in ("started.txt", "exit-code.txt", "wall.txt", "audit.json"):
                (directory / name).unlink(missing_ok=True)

    def test_captured_config_tampering_detected(self):
        protocol = json.loads((self.root / "protocol.prepared.json").read_text())
        report.verify(self.root, protocol)
        path = self.root / "budget.tsv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(AssertionError): report.verify(self.root, protocol)
        finally:
            path.write_bytes(original)

    def test_complete_observed_pareto_sets(self):
        points = [dict(time=1, score=.5), dict(time=2, score=.4), dict(time=3, score=.9), dict(time=4, score=.9)]
        self.assertEqual(report.pareto(points), [True, False, True, False])
        self.assertEqual(report.pareto([]), [])

    def test_bootstrap_retains_paired_seed_dependence(self):
        indices = np.random.default_rng(1).integers(0, 5, (10000, 5))
        nature = np.array([.1, .3, .6, .8, .5])
        ours = nature + .1
        mean, low, high = report.interval(ours-nature, indices)
        np.testing.assert_allclose([mean, low, high], [.1, .1, .1])

    def test_gpu_lookup_prefers_path_and_busy_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "nvidia-smi"
            fake.write_text("#!/bin/sh\necho 999\n")
            fake.chmod(0o755)
            command = f'source ocean/pongcnn/gpu_env.sh; PATH={directory}; PONG_SMI=$(pong_find_smi); echo "$PONG_SMI"; pong_gpu_idle'
            result = subprocess.run(["bash", "-c", command], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), str(fake))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GPU busy", result.stderr)


if __name__ == "__main__":
    unittest.main()
