"""Configuration/provenance gates only. No neural inference or GPU access."""
import configparser
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_pixel_robustness as panel


class PanelTest(unittest.TestCase):
    def test_full_appearance_panel_is_paired_and_preserves_one_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "panel"
            protocol = panel.prepare(root, list(panel.ENVIRONMENTS), "all", [53111, 53112], 65536, 32768)
            self.assertEqual(len(protocol["jobs"]), 8*sum(panel.ENVIRONMENTS.values()))
            self.assertEqual(len(protocol["builds"]), 18)
            self.assertFalse(protocol["publication_confirmation_launchable"])
            grouped = {}
            for job in protocol["jobs"]:
                path = root / job["config"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), job["config_sha256"])
                config = configparser.ConfigParser(interpolation=None); config.read(path)
                self.assertEqual(config.get("base", "load_model_path"), "None")
                self.assertEqual(config.getint("base", "seed"), job["seed"])
                self.assertEqual(config.getint("env", "representation"), job["representation"])
                self.assertEqual(config.getint("env", "representation_mode"), 0)
                self.assertEqual(config.getint("policy", "hidden_size"), 128)
                self.assertEqual(config.getint("policy", "num_layers"), 1)
                self.assertEqual(config.getint("train", "total_timesteps"), 65536)
                self.assertEqual(job["checkpoint_steps"], [32768, 65536])
                self.assertFalse(any(s.startswith("sweep.") for s in config))
                if job["model"] == "flex_quality":
                    self.assertEqual({k: config.getint("policy", k) for k in protocol["architecture"]}, protocol["architecture"])
                # Architecture-independent settings match within every task/render/seed.
                common = {s: dict(config[s]) for s in ("env", "vec", "train", "selfplay")}
                common["base"] = {k: v for k, v in config["base"].items() if k not in ("checkpoint_dir", "log_dir")}
                key = job["environment"], job["representation"], job["seed"]
                grouped.setdefault(key, []).append(common)
                expected = {"connect4cnn": "connect4-exact-available", "flappycnn": "flappy-exact-compiled-gpu-validation-pending",
                            "pongcnn": "pong-exact-compiled-gpu-validation-pending", "breakoutcnn": "breakout-exact-compiled-gpu-validation-pending",
                            "snakecnn": "snake-exact-compiled-gpu-validation-pending",
                            "mazecnn": "maze-exact-compiled-gpu-validation-pending"}[job["environment"]]
                self.assertEqual(job["evaluation_gate"], expected)
                self.assertEqual(job["rules"], panel.ENVIRONMENT_RULES[job["environment"]])
            self.assertEqual(len(grouped), 2*sum(panel.ENVIRONMENTS.values()))
            for configs in grouped.values():
                self.assertEqual(len(configs), 4)
                self.assertTrue(all(config == configs[0] for config in configs))
            for name, digest in protocol["source_sha256"].items():
                self.assertEqual(hashlib.sha256((root / "source" / name).read_bytes()).hexdigest(), digest)
            self.assertIn("ocean/snakebench/snakebench.h", protocol["source_sha256"])
            self.assertIn("ocean/snakecnn/game.h", protocol["source_sha256"])
            self.assertIn("ocean/maze/maze.h", protocol["source_sha256"])
            self.assertIn("ocean/mazecnn/mazecnn.h", protocol["source_sha256"])
            snake = next(job for job in protocol["jobs"] if job["environment"] == "snakecnn")
            config.read(root / snake["config"])
            self.assertEqual(config.getint("env", "rule_version"), 1)
            self.assertEqual(config.getint("env", "num_agents"), 1)
            self.assertEqual(config.getint("env", "vision"), 5)
            self.assertEqual(config.getint("env", "max_steps"), 2048)
            builds = (root / "build-only.sh").read_text()
            self.assertNotIn(" train ", builds)
            self.assertNotIn(" eval ", builds)
            self.assertIn("sha256sum --check", builds)
            with self.assertRaises(FileExistsError):
                panel.prepare(root, ["connect4cnn"], "default", [53111], 65536, 32768)

    def test_bad_budgets_seeds_and_environment_lists_fail_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [([], [1], 65536, 32768), (["unknown"], [1], 65536, 32768),
                     (["connect4cnn", "connect4cnn"], [1], 65536, 32768),
                     (["connect4cnn"], [1, 1], 65536, 32768),
                     (["connect4cnn"], [-1], 65536, 32768),
                     (["connect4cnn"], [2**32], 65536, 32768),
                     (["connect4cnn"], [1], 123, 32768),
                     (["connect4cnn"], [1], 65536, 0)]
            for index, (envs, seeds, steps, cadence) in enumerate(cases):
                out = Path(tmp) / str(index)
                with self.assertRaises(ValueError):
                    panel.prepare(out, envs, "default", seeds, steps, cadence)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
