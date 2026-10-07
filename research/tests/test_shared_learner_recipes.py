"""Shared numeric learner overlays; config/native scalar checks, no policies."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_pixel_panel as audit
import prepare_pixel_robustness as panel


class SharedLearnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stock = panel.ROOT / "ocean/flappycnn/learner_stock.ini"

    def recipe(self, text):
        directory = Path(tempfile.mkdtemp(dir=self.root)); path = directory / "recipe.ini"
        path.write_text(text); return path

    def prepare(self, name="panel", **kwargs):
        values = dict(out=self.root / name, environments=["flappycnn"], appearances="default",
            seeds=[53121], steps=19922944, checkpoint_steps=1048576, learner_recipes={"flappycnn": self.stock})
        values.update(kwargs); return panel.prepare(**values)

    def test_stock_candidate_all_drawings_two_seeds_is_frozen_and_paired(self):
        value = self.prepare(appearances="all", seeds=[53121, 53122])
        self.assertEqual(value["version"], "pixel-robustness-preparation-v2")
        self.assertEqual(len(value["jobs"]), 8*panel.ENVIRONMENTS["flappycnn"]); self.assertFalse(value["learning_calibrated"])
        self.assertFalse(value["gpu_execution_authorized"])
        root = self.root / "panel"; self.assertEqual(audit.load(root / "protocol.json"), value)
        geometry = json.loads((root / value["rollout_geometry"]["flappycnn"]["receipt"]).read_text())
        self.assertEqual(geometry["epochs"], 152); self.assertEqual(geometry["optimizer_updates_per_rank"], 1216)
        self.assertEqual(geometry["checkpoint_count"], 19); self.assertFalse(geometry["policy_executed"])
        groups = {}
        for job in value["jobs"]:
            c = audit.config(root / job["config"])
            self.assertEqual(c.getint("vec", "total_agents"), 2048)
            self.assertEqual(c.getint("vec", "num_buffers"), 4); self.assertEqual(c.getint("vec", "num_threads"), 8)
            self.assertEqual(c.getint("base", "async"), 1); self.assertEqual(c.getint("base", "checkpoint_interval"), 8)
            self.assertEqual(c.getint("train", "horizon"), 64); self.assertEqual(c.getint("train", "minibatch_size"), 16384)
            self.assertEqual(c.getfloat("train", "learning_rate"), .01); self.assertEqual(c.getfloat("train", "max_grad_norm"), 1.5)
            self.assertEqual(c.getint("policy", "hidden_size"), 128); self.assertEqual(c.getint("policy", "num_layers"), 1)
            self.assertEqual(c.getint("env", "representation"), job["representation"])
            self.assertEqual(c.getfloat("env", "gravity"), .45)
            self.assertEqual(job["checkpoint_steps"], list(range(1048576, 19922944+1, 1048576)))
            groups.setdefault((job["representation"], job["seed"]), []).append(audit.common(c))
        self.assertTrue(all(len(g) == 4 and all(c == g[0] for c in g) for g in groups.values()))
        relocated = self.root / "relocated"; shutil.copytree(root, relocated)
        (relocated / "validation/learner_geometry").unlink()
        self.assertEqual(audit.load(relocated / "protocol.json"), value)
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertFalse((root / "failure.json").exists())  # Refused overwrite never changes an existing packet.

    def test_mixed_games_have_per_game_geometry_and_same_actual_budget(self):
        value = self.prepare(environments=["connect4cnn", "flappycnn"], steps=1048576, checkpoint_steps=131072)
        root = self.root / "panel"; audit.load(root / "protocol.json")
        self.assertEqual(len(value["jobs"]), 8)
        self.assertEqual(value["rollout_geometry"]["connect4cnn"]["batch_steps"], 2048)
        self.assertEqual(value["rollout_geometry"]["flappycnn"]["batch_steps"], 131072)
        for job in value["jobs"]:
            c = audit.config(root / job["config"])
            self.assertEqual(c.getint("train", "total_timesteps"), 1048576)
            self.assertEqual(c.getint("base", "checkpoint_interval"), 64 if job["environment"] == "connect4cnn" else 1)

    def test_stock_small_batch_changes_only_shared_minibatch_and_update_geometry(self):
        path = panel.ROOT / "ocean/flappycnn/learner_stock_small_batch.ini"
        stock = panel.read(self.stock); small = panel.read(path)
        differences = {(s, k) for s in stock.sections() for k in stock[s] if stock[s][k] != small[s][k]}
        self.assertEqual(set(stock.sections()), set(small.sections()))
        self.assertTrue(all(set(stock[s]) == set(small[s]) for s in stock.sections()))
        self.assertEqual(differences, {("train", "minibatch_size")})
        value = self.prepare(seeds=[53121, 53122, 53123], learner_recipes={"flappycnn": path})
        root = self.root / "panel"; audit.load(root / "protocol.json")
        self.assertEqual(len(value["jobs"]), 12)
        result = json.loads((root / value["rollout_geometry"]["flappycnn"]["receipt"]).read_text())
        self.assertEqual(result["optimizer_minibatches_per_epoch"], 64)
        self.assertEqual(result["optimizer_updates_per_rank"], 9728)
        self.assertEqual(result["effective_replay_ratio"], 1)
        self.assertEqual(result["actual_global_decisions"], 19922944)
        groups = {}
        for job in value["jobs"]:
            c = audit.config(root / job["config"])
            self.assertEqual(c.getint("vec", "total_agents"), 2048)
            self.assertEqual(c.getint("train", "minibatch_size"), 2048)
            self.assertEqual(c.getfloat("train", "learning_rate"), .01)
            self.assertEqual(c.getint("train", "horizon"), 64)
            groups.setdefault(job["seed"], []).append(audit.common(c))
        self.assertTrue(all(len(g) == 4 and all(c == g[0] for c in g) for g in groups.values()))

    def test_world_policy_seed_gpu_budget_and_inherited_controls_are_forbidden(self):
        cases = ("[env]\ngravity=.5\n", "[policy]\nencoder=2\n", "[policy]\nhidden_size=64\n",
            "[base]\nseed=73\n", "[base]\ncheckpoint_interval=8\n", "[base]\nload_model_path=old.bin\n",
            "[train]\ngpus=2\n", "[train]\ntotal_timesteps=65536\n", "[vec]\nnum_policies=2\n",
            "[selfplay]\nenabled=1\n", "[sweep.train.learning_rate]\nmin=.001\n",
            "[DEFAULT]\nlearning_rate=.01\n[train]\ngamma=.995\n",
            "[train]\nent_coef=nan\n", "[train]\nlearning_rate=1,2\n", "[train]\n", "")
        for i, text in enumerate(cases):
            with self.subTest(text=text):
                out = self.root / f"invalid-{i}"
                with self.assertRaises(ValueError): self.prepare(name=out.name, learner_recipes={"flappycnn": self.recipe(text)})
                self.assertFalse(out.exists())

    def test_budgets_cadence_and_recipe_task_allocation_fail_before_creation(self):
        cases = (dict(steps=20000000), dict(checkpoint_steps=32768), dict(steps=65536),
            dict(steps=19922944.0), dict(learner_recipes={"pongcnn": self.stock}), dict(learner_recipes=[]))
        for i, kwargs in enumerate(cases):
            out = self.root / f"invalid-{i}"
            with self.assertRaises(ValueError): self.prepare(name=out.name, **kwargs)
            self.assertFalse(out.exists())

    def test_zero_updates_and_bad_minibatch_retain_failed_native_preflight(self):
        for i, text in enumerate(("[train]\nreplay_ratio=.25\n", "[train]\nminibatch_size=0\n")):
            out = self.root / f"failed-{i}"
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                self.prepare(name=out.name, steps=65536, checkpoint_steps=32768, learner_recipes={"flappycnn": self.recipe(text)})
            failure = json.loads((out / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed-learner-preflight"); self.assertFalse(failure["policy_executed"])
            self.assertFalse((out / "protocol.json").exists())
            self.assertTrue((out / "validation/build.txt").exists())
            if i == 0:
                result = json.loads((out / "validation/flappycnn.json").read_text())
                self.assertFalse(result["nonzero_training_updates"])
                self.assertEqual(result["optimizer_minibatches_per_epoch"], 0)

    def test_rehashed_all_model_drift_and_native_preflight_changes_are_rejected(self):
        value = self.prepare(); root = self.root / "panel"; path = root / "protocol.json"
        original = path.read_text()
        for job in value["jobs"]:
            config = root / job["config"]; c = audit.config(config); c["train"]["learning_rate"] = ".02"
            with config.open("w") as stream: c.write(stream)
            job["config_sha256"] = audit.sha(config)
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "captured recipe"): audit.load(path)
        # The untouched copied scalar input also remains bound to its own SHA.
        path.write_text(original); record = value["rollout_geometry"]["flappycnn"]
        config = root / record["config"]; config.write_text(config.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "geometry input/result"): audit.load(path)

    def test_changed_recipe_legacy_upgrade_and_claim_flags_are_rejected(self):
        value = self.prepare(); root = self.root / "panel"; path = root / "protocol.json"
        original = path.read_text()
        for key in ("learning_calibrated", "gpu_execution_authorized"):
            changed = json.loads(original); changed[key] = True; path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "upgraded"): audit.load(path)
        path.write_text(original)
        snapshot = root / "source" / value["learner_recipes"]["flappycnn"]["snapshot"]
        snapshot.write_text(snapshot.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "source snapshot"): audit.load(path)
        legacy = self.root / "legacy"
        value = panel.prepare(legacy, ["flappycnn"], "default", [53121], 65536, 32768)
        value["learner_recipes"] = {}; (legacy / "protocol.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Legacy panel upgraded"): audit.load(legacy / "protocol.json")

    def test_external_overlay_snapshot_and_repository_build_receipts_are_separate(self):
        external = self.recipe("[train]\nlearning_rate=.01\n")
        self.prepare(learner_recipes={"flappycnn": external})
        root = self.root / "panel"; external.unlink()
        # Snapshot audit/build checks must not need the original external INI.
        audit.load(root / "protocol.json")
        subprocess.run(["sha256sum", "--check", "--quiet", str(root / "source.sha256")],
                       cwd=root / "source", check=True, capture_output=True, text=True, timeout=20)
        subprocess.run(["sha256sum", "--check", "--quiet", str(root / "source.repo.sha256")],
                       cwd=panel.ROOT, check=True, capture_output=True, text=True, timeout=20)
        subprocess.run(["bash", "-n", str(root / "build-only.sh")], check=True, capture_output=True, timeout=20)
        self.assertNotIn("learner-recipes/", (root / "source.repo.sha256").read_text())
        with (root / "source.repo.sha256").open("a") as stream:
            stream.write("0"*64+"  learner-recipes/flappycnn.ini\n")
        with self.assertRaisesRegex(ValueError, "Repository build manifest"): audit.load(root / "protocol.json")

    def test_changed_build_script_and_failed_packet_are_not_accepted(self):
        self.prepare(); root = self.root / "panel"
        script = root / "build-only.sh"; original = script.read_text()
        script.write_text(original+"\necho altered\n")
        with self.assertRaisesRegex(ValueError, "build-only script"): audit.load(root / "protocol.json")
        script.write_text(original); (root / "failure.json").write_text('{"status":"failed"}\n')
        with self.assertRaisesRegex(ValueError, "Failed panel preparation"): audit.load(root / "protocol.json")


if __name__ == "__main__": unittest.main()
