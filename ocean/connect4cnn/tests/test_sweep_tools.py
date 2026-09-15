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
    def test_representation_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = sweep.prepare(Path(tmp), sweep.HERE / "sweep_representation.ini", 2)
            self.assertEqual({s for s in config if s.startswith("sweep.")}, {"sweep.env.representation"})
            self.assertEqual(config.getint("env", "representation"), 0)
            self.assertEqual(config.getint("sweep.env.representation", "max"), 9)
            self.assertEqual(config.getint("train", "total_timesteps"), 13312000)
        for value in ("-1", "10", "1.5", "nan"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                recipe = sidecar.read_ini(sweep.HERE / "sweep_representation.ini")
                recipe.set("env", "representation", value)
                path = root / "bad.ini"
                with path.open("w") as f:
                    recipe.write(f)
                with self.assertRaises(ValueError):
                    sweep.prepare(root, path, 2)

    def test_representation_recorded_separately_from_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = self.fixture(root)
            path = root / "metrics/connect4cnn" / (run_id + ".ini")
            with path.open("a") as f:
                f.write("\n[env]\nrepresentation = 5\n")
            (root / "sweep.log").write_text("sweep run=0 score=0.8000 cost=3.00 steps=4096 random=1 gp_obs=0 pareto=0\n")
            trial, = sidecar.trials(root)
            self.assertEqual(trial["representation"], 5)
            spec = sidecar.architecture(trial["config"])
            trial["config"]["env.representation"] = 7
            self.assertEqual(sidecar.architecture(trial["config"]), spec)
            second = {**trial, "index": 1, "representation": 7, "score": .9, "cost": 2}
            sweep.write_report(root, [trial, second], 5, "ok")
            import csv
            with (root / "results.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["representation"] for r in rows], ["5", "7"])
            self.assertEqual([r["pareto"] for r in rows], ["True", "True"])

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

    def test_discovery_recipe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"NVCC_PREPEND_FLAGS": ""}):
            config = sweep.prepare(Path(tmp), sweep.HERE / "sweep_discovery.ini", None)
            self.assertEqual(config.getint("sweep", "max_runs"), 24)
            self.assertEqual(config.getint("base", "checkpoint_interval"), 1000)
            self.assertEqual(config.getint("train", "total_timesteps"), 6639616)
            self.assertEqual(config.getint("sweep.train.total_timesteps", "min"), 3319808)
            self.assertEqual(config.getint("sweep.train.total_timesteps", "max"), 13279232)
            self.assertEqual(config.getint("policy", "hidden_size"), 128)

    def test_nature_and_compact_recipes(self):
        configs = []
        for name, encoder, count in (("nature", 2, 1), ("compact", 3, 5)):
            with tempfile.TemporaryDirectory() as tmp:
                config = sweep.prepare(Path(tmp), sweep.HERE / f"sweep_{name}.ini", None)
                self.assertEqual(config.getint("policy", "encoder"), encoder)
                self.assertEqual(sum(s.startswith("sweep.") for s in config), count)
                self.assertNotIn("cnn_blocks", config["policy"])
                if encoder == 2:
                    self.assertNotIn("cnn_channels", config["policy"])
                configs.append(config)
        for section in ("train", "vec", "env", "selfplay"):
            self.assertEqual(dict(configs[0][section]), dict(configs[1][section]))
        self.assertEqual(dict(configs[0]["sweep.train.total_timesteps"]), dict(configs[1]["sweep.train.total_timesteps"]))

    def test_architecture_identity_ignores_inactive_fields(self):
        config = {"policy.encoder": 2, "policy.hidden_size": 128, "policy.num_layers": 1}
        first = sidecar.architecture(config)
        config["policy.cnn_blocks"] = 2
        self.assertEqual(sidecar.architecture(config), first)
        config.update({"policy.encoder": 3, "policy.cnn_channels": 8, "policy.cnn_depth": 1,
                       "policy.cnn_stride": 4, "policy.cnn_projection": 32})
        self.assertNotEqual(sidecar.architecture(config), first)

    def test_flexible_subset_and_fixed_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = sidecar.read_ini(sweep.HERE / "sweep_flex.ini")
            for section in list(recipe.sections()):
                if section.startswith("sweep.") and section != "sweep.policy.cnn_kernel_1":
                    recipe.remove_section(section)
            path = root / "subset.ini"
            with path.open("w") as f:
                recipe.write(f)
            out = root / "out"; out.mkdir()
            config = sweep.prepare(out, path, 2)
            self.assertEqual({s for s in config if s.startswith("sweep.")}, {"sweep.policy.cnn_kernel_1"})
            self.assertEqual(config.getint("policy", "cnn_channels_2"), 16)
            self.assertEqual(config.getint("train", "total_timesteps"), recipe.getint("train", "total_timesteps"))
            values = {f"policy.{k}": sidecar.scalar(v) for k, v in config["policy"].items()}
            spec = sidecar.architecture(values)
            self.assertNotIn("policy.cnn_channels_2", spec)
            values["policy.cnn_channels_2"] = 32
            self.assertEqual(sidecar.architecture(values), spec)
            values["policy.cnn_kernel_1"] = 4
            self.assertNotEqual(sidecar.architecture(values), spec)

    def test_flexible_rejects_invalid_fixed_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = sidecar.read_ini(sweep.HERE / "sweep_flex.ini")
            recipe.remove_section("sweep.policy.cnn_stride_1")
            recipe.set("policy", "cnn_stride_1", "3")
            path = root / "bad.ini"
            with path.open("w") as f:
                recipe.write(f)
            with self.assertRaises(ValueError):
                sweep.prepare(root, path, 2)

    def test_final_observation_not_binned_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = self.fixture(root)
            (root / "sweep.log").write_text("sweep run=0 score=0.8000 cost=3.00 steps=4096 random=1 gp_obs=0 pareto=0\n")
            trial, = sidecar.trials(root)
            self.assertEqual(trial["score"], .8)
            self.assertEqual(trial["cost"], 3)
            self.assertEqual(trial["history"][-1]["env/perf"], .2)
            self.assertEqual(trial["history"][-1]["SPS"], 2100)
            self.assertEqual(trial["history"][-1]["uptime"], 2)
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 1)
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 0)
            payload = json.loads((root / "sidecar" / (run_id + ".json")).read_text())
            self.assertEqual(payload["checkpoint_sha256"], trial["checkpoint_sha256"])

    def test_names_and_legacy_state_migration(self):
        self.assertRegex(sidecar.display_name("sweep.example", 0), r"^[a-z]+-[a-z]+-1$")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = self.fixture(root)
            (root / "sweep.log").write_text("sweep run=0 score=0.8000 cost=3.00 steps=4096 random=1 gp_obs=0 pareto=0\n")
            sidecar.sync(root, "disabled", "test")
            state_path, = (root / "sidecar").glob("state-*.json")
            state_path.write_text(json.dumps({run_id: "legacy-id"}))
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 1)
            self.assertEqual(sidecar.sync(root, "disabled", "test"), 0)

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
