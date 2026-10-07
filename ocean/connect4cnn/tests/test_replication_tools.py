"""Frozen design/artifact tests, without CPU inference or GPU execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replication as rp
from claim import common_settings, ini


class ReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_full_design_and_corruption(self):
        binaries = self.root/"binaries"
        binaries.mkdir()
        for name in ("timing-default", "timing-nature"):
            (binaries/name).write_bytes(b"Artifact fixture, never executed")
        out = self.root/"campaign"
        p = rp.prepare(out, binaries, False)
        self.assertEqual(p["training_seeds"], [53101, 53102, 53103, 53104, 53105])
        self.assertEqual(len(p["jobs"]), 10)
        self.assertEqual(len(p["checkpoint_steps"]), 51)
        self.assertEqual(p["checkpoint_steps"][-1], 13312000)
        self.assertFalse(p["candidate_bands"])
        suite = rp.ev.load_suite(out/"suite/suite.json")
        self.assertEqual(suite["episodes"], 1000)
        for index, job in enumerate(p["jobs"]):
            config = ini(out/job["id"]/"config/default.ini")
            self.assertEqual(common_settings(config), p["common_settings"])
            self.assertEqual(config.getint("train", "total_timesteps"), 13312000)
            self.assertEqual(config.getint("base", "checkpoint_interval"), 128)
            self.assertEqual(config.getint("policy", "hidden_size"), 128)
            self.assertEqual(config.getint("policy", "num_layers"), 1)
            self.assertEqual(config.getint("policy", "encoder"), 4 if job["model"] == "flex_quality" else 0)
        self.assertEqual(rp.load(out)["jobs"], p["jobs"])
        config_path = out/p["jobs"][0]["id"]/"config/default.ini"
        config_path.write_text(config_path.read_text()+"\n# mutation\n")
        with self.assertRaises(ValueError):
            rp.load(out)

    def test_partial_receipt_requires_config_and_valid_weights(self):
        directory = self.root/"job"
        config = directory/"config"
        checkpoints = directory/"checkpoints/connect4cnn/trial"
        config.mkdir(parents=True)
        checkpoints.mkdir(parents=True)
        (config/"default.ini").write_text("[base]\nseed = 3\n[policy]\nhidden_size = 128\nnum_layers = 1\n")
        (checkpoints/"resolved.ini").write_bytes((config/"default.ini").read_bytes())
        np.zeros(10, dtype=np.float32).tofile(checkpoints/"0000000000032768.bin")
        (directory/"train.log").write_text("PUFFER_CHECKPOINT steps=32768 bytes=40 monotonic_ns=2000\n")
        timed = dict(status="failed", launch_monotonic_ns=1000, end_monotonic_ns=3000)
        recovered = rp.verify_checkpoints(directory, timed, [32768, 65536], 10)
        self.assertEqual(list(recovered), ["32768"])
        self.assertEqual(recovered["32768"]["seconds"], .000001)
        bad = np.zeros(10, dtype=np.float32); bad[0] = np.nan
        bad.tofile(checkpoints/"0000000000032768.bin")
        with self.assertRaises(ValueError):
            rp.verify_checkpoints(directory, timed, [32768, 65536], 10)


if __name__ == "__main__":
    unittest.main()
