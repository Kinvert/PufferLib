"""Native descriptor/input rejection only; no CUDA query or model computation.

Set PROTEIN_FEEDBACK_HOST_BINARY to a fresh research build to enable these.
All propose cases fail in validation before cudaGetDeviceCount; valid ledgers
are deliberately excluded. CUDA_VISIBLE_DEVICES is additionally empty.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = os.environ.get("PROTEIN_FEEDBACK_HOST_BINARY")


@unittest.skipUnless(BINARY, "Explicit compiled research binary required")
class NativeHostChecks(unittest.TestCase):
    def test_descriptor_returns_before_gpu_branch(self):
        result = subprocess.run([BINARY, "research_protein_describe",
            str(ROOT / "research/recipes/cross_game_feedback_smoke.ini")],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["coordinates"], ["cnn_channels_1"])
        self.assertEqual(value["dimensions"], 1)
        self.assertFalse(value["gpu_queried"])
        self.assertFalse(value["policy_executed"])

    def test_malformed_ledger_is_rejected_before_gpu_query(self):
        cases = [("PUFFER_CROSS_GAME_V1 2\n", "header"),
            ("PUFFER_CROSS_GAME_V1 1 extra\n", "header"),
            ("PUFFER_CROSS_GAME_V1 1", "header"),
            ("PUFFER_CROSS_GAME_V1 1\n0 nan 1 1\n", "score"),
            ("PUFFER_CROSS_GAME_V1 1\n1 inf 1 1\n", "score"),
            ("PUFFER_CROSS_GAME_V1 1\n0 0.5 0 1\n", "cost"),
            ("PUFFER_CROSS_GAME_V1 1\n0 0.5 1 2\n", "coordinate"),
            ("PUFFER_CROSS_GAME_V1 1\n0 0.5 1 1 extra\n", "extra history"),
            ("PUFFER_CROSS_GAME_V1 1\n0 0.5 1 1", "truncated")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (text, error) in enumerate(cases):
                ledger = root / f"history-{index}.tsv"; ledger.write_text(text)
                output = root / f"proposal-{index}.json"
                result = subprocess.run([BINARY, "research_protein_propose",
                    str(ROOT / "research/recipes/cross_game_feedback_smoke.ini"), str(ledger), str(output)],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, timeout=10)
                with self.subTest(index=index):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn("scheduled GPU", result.stderr)
                    self.assertFalse(output.exists())

    def test_full_search_descriptor_returns_before_gpu_branch(self):
        result = subprocess.run([BINARY, "research_protein_describe",
            str(ROOT / "research/recipes/cross_game_feedback_prepare.ini")],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["dimensions"], 18)
        self.assertEqual(len(set(value["coordinates"])), 18)
        self.assertIn("cnn_depth", value["coordinates"])
        self.assertIn("cnn_residual_3", value["coordinates"])
        self.assertFalse(value["gpu_queried"])
        self.assertFalse(value["policy_executed"])

    def test_fractional_optimizer_control_is_rejected_before_gpu_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); recipe = root / "recipe.ini"
            recipe.write_text((ROOT / "research/recipes/cross_game_feedback_smoke.ini").read_text().replace(
                "gp_training_iter = 50", "gp_training_iter = 1.5"))
            history = root / "history.tsv"; history.write_text("PUFFER_CROSS_GAME_V1 1\n")
            result = subprocess.run([BINARY, "research_protein_propose", str(recipe), str(history), str(root / "output.json")],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bad optimizer controls", result.stderr)


if __name__ == "__main__": unittest.main()
