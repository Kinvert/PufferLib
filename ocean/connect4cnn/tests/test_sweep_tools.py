"""CPU checks for config isolation and accurate, repeatable sidecar ingestion."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep
import wandb_sidecar as sidecar


class ToolsTest(unittest.TestCase):
    def test_isolated_dimensions(self):
        original = (sweep.ROOT / "config/default.ini").read_bytes()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"NVCC_PREPEND_FLAGS": ""}):
            root = Path(tmp)
            config = sweep.prepare(root, sweep.HERE / "sweep.ini", 12)
            self.assertEqual({s[6:] for s in config if s.startswith("sweep.")}, sweep.DIMENSIONS)
            self.assertEqual(config.getint("policy", "hidden_size"), 128)
            self.assertEqual(config.getfloat("train", "learning_rate"), .001)
            self.assertEqual(config.getint("policy", "encoder"), 1)
            self.assertEqual(original, (sweep.ROOT / "config/default.ini").read_bytes())

    def fixture(self, root):
        run_id = "sweep_1000_0000"
        logdir = root / "metrics/connect4cnn"
        logdir.mkdir(parents=True)
        (logdir / (run_id + ".ini")).write_text("""[base]
env_name = connect4cnn
[policy]
encoder = 1
cnn_channels = 8
cnn_blocks = 0
cnn_global_pool = 1
hidden_size = 128
num_layers = 1
[metrics]
agent_steps = 2048,4096
uptime = 1,2
SPS = 2000,2100
env/perf = .1,.2
""")
        checkpoints = root / "checkpoints/connect4cnn" / run_id
        checkpoints.mkdir(parents=True)
        (checkpoints / "0000000000004096.bin").write_bytes(b"\0" * 16)
        return run_id

    def test_final_observation_not_binned_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = self.fixture(root)
            (root / "sweep.log").write_text("sweep run=0 score=0.8000 cost=3.00 steps=4096 random=1 gp_obs=0 pareto=0\n")
            trial, = sidecar.trials(root)
            self.assertEqual(trial["score"], .8)
            self.assertEqual(trial["cost"], 3)
            self.assertEqual(trial["history"][-1]["binned/env/perf"], .2)
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 1)
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 0)
            payload = json.loads((root / "sidecar" / (run_id + ".json")).read_text())
            self.assertEqual(payload["checkpoint_sha256"], trial["checkpoint_sha256"])

    def test_partial_trial_is_not_uploaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "sweep.log").write_text("sweep run=0 score=0.8")
            self.assertEqual(sidecar.trials(root), [])
            (root / "sweep.log").write_text("sweep run=0 score=nan cost=3.00 steps=4096 random=1 gp_obs=0 pareto=0\n")
            self.assertEqual(sidecar.trials(root), [])

    def test_missing_checkpoint_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = self.fixture(root)
            (root / "sweep.log").write_text("sweep run=0 score=0.8000 cost=3.00 steps=8192 random=1 gp_obs=0 pareto=0\n")
            with self.assertRaises(ValueError):
                sidecar.trials(root)


if __name__ == "__main__":
    unittest.main()
