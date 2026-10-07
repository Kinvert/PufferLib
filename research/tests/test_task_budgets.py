"""Per-game fixed-model budget/configuration checks; no GPU or model."""
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


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, name="panel", **kwargs):
        values = dict(out=self.root / name, environments=["connect4cnn", "flappycnn"],
            appearances="default", seeds=[53131], steps=65536, checkpoint_steps=32768,
            learner_recipes={"flappycnn": panel.ROOT / "ocean/flappycnn/learner_stock_small_batch.ini"},
            task_budgets={"flappycnn": dict(steps=262144, checkpoint_steps=131072)})
        values.update(kwargs); return panel.prepare(**values)

    def test_full_inventory_per_task_budgets_and_native_geometry_are_paired(self):
        overrides = dict(flappycnn=dict(steps=262144, checkpoint_steps=131072),
                         mazecnn=dict(steps=196608, checkpoint_steps=65536))
        value = self.prepare(environments=list(panel.ENVIRONMENTS), appearances="all", seeds=[53131, 53132], task_budgets=overrides)
        root = self.root / "panel"; self.assertEqual(audit.load(root / "protocol.json"), value)
        self.assertEqual(value["version"], "pixel-robustness-preparation-v3")
        self.assertEqual(value["learner_contract"], "shared-native-geometry-v2")
        self.assertEqual(len(value["jobs"]), 8*sum(panel.ENVIRONMENTS.values()))
        self.assertEqual(value["purpose"], "development-design-not-calibrated")
        self.assertEqual(set(value["task_budgets"]), set(panel.ENVIRONMENTS))
        self.assertFalse(value["gpu_execution_authorized"]); self.assertFalse(value["learning_calibrated"])
        groups = {}
        for job in value["jobs"]:
            task = job["environment"]; steps, cadence = audit.task_budget(value, task)
            c = audit.config(root / job["config"])
            self.assertEqual(c.getint("train", "total_timesteps"), steps)
            self.assertEqual(c.getint("base", "checkpoint_interval"), cadence // audit.batch_steps(value, task))
            self.assertEqual(job["checkpoint_steps"], list(range(cadence, steps, cadence)) + [steps])
            groups.setdefault((task, job["representation"], job["seed"]), []).append(audit.common(c))
        self.assertTrue(all(len(g) == 4 and all(c == g[0] for c in g) for g in groups.values()))
        for task, record in value["rollout_geometry"].items():
            geometry = json.loads((root / record["receipt"]).read_text())
            self.assertEqual(geometry["actual_global_decisions"], value["task_budgets"][task]["steps"])
            self.assertEqual(geometry["dropped_requested_decisions"], 0)
            self.assertTrue(geometry["nonzero_training_updates"])
        self.assertEqual(value["rollout_geometry"]["flappycnn"]["batch_steps"], 131072)
        relocated = self.root / "relocated"; shutil.copytree(root, relocated)
        (relocated / "validation/learner_geometry").unlink()
        self.assertEqual(audit.load(relocated / "protocol.json"), value)

    def test_budget_only_v3_and_nondivisible_final_checkpoint(self):
        value = self.prepare(environments=["connect4cnn", "snakecnn"], learner_recipes={},
            task_budgets={"connect4cnn": dict(steps=98304, checkpoint_steps=65536)})
        audit.load(self.root / "panel/protocol.json")
        self.assertEqual(value["learner_recipes"], {})
        for job in value["jobs"]:
            self.assertEqual(job["checkpoint_steps"], [65536, 98304] if job["environment"] == "connect4cnn" else [32768, 65536])
        final = self.prepare(name="final-only", environments=["connect4cnn"], learner_recipes={},
            task_budgets={"connect4cnn": dict(steps=2048, checkpoint_steps=4096)})
        audit.load(self.root / "final-only/protocol.json")
        self.assertTrue(all(j["checkpoint_steps"] == [2048] for j in final["jobs"]))

    def test_invalid_declarations_and_alignment_fail_before_creation(self):
        cases = ([], {"other": dict(steps=65536, checkpoint_steps=32768)},
            {"flappycnn": dict(steps=True, checkpoint_steps=131072)},
            {"flappycnn": dict(steps=262144.0, checkpoint_steps=131072)},
            {"flappycnn": dict(steps=0, checkpoint_steps=131072)},
            {"flappycnn": dict(steps=262144)}, {"flappycnn": dict(steps=262144, checkpoint_steps=131072, seed=3)},
            {"flappycnn": dict(steps=262145, checkpoint_steps=131072)},
            {"flappycnn": dict(steps=262144, checkpoint_steps=32768)})
        for i, declaration in enumerate(cases):
            with self.subTest(declaration=declaration):
                with self.assertRaises(ValueError): self.prepare(name=f"bad-{i}", task_budgets=declaration)
                self.assertFalse((self.root / f"bad-{i}").exists())

    def test_budget_declaration_drift_and_rehashed_job_change_reject(self):
        value = self.prepare(); root = self.root / "panel"; path = root / "protocol.json"; original = path.read_text()
        value["task_budgets"]["flappycnn"]["steps"] = 393216
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Resolved task budgets"): audit.load(path)
        path.write_text(original); value = json.loads(original)
        job = next(j for j in value["jobs"] if j["environment"] == "flappycnn")
        config = root / job["config"]; c = audit.config(config); c["train"]["total_timesteps"] = "393216"
        with config.open("w") as stream: c.write(stream)
        job["config_sha256"] = audit.sha(config); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "captured recipe"): audit.load(path)

    def test_resolved_budget_integer_types_cannot_pass_by_numeric_equality(self):
        self.prepare(); path = self.root / "panel/protocol.json"; original = path.read_text()
        for field in ("steps", "checkpoint_steps"):
            value = json.loads(original)
            value["task_budgets"]["flappycnn"][field] = float(value["task_budgets"]["flappycnn"][field])
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "positive integer"): audit.load(path)

    def test_legacy_formats_stay_legacy_and_reject_added_budget_fields(self):
        for version, recipes in (("v1", {}), ("v2", {"flappycnn": panel.ROOT / "ocean/flappycnn/learner_stock_small_batch.ini"})):
            value = self.prepare(name=version, steps=262144, checkpoint_steps=131072,
                learner_recipes=recipes, task_budgets={})
            path = self.root / version / "protocol.json"; audit.load(path)
            self.assertEqual(value["version"], "pixel-robustness-preparation-"+version)
            value["task_budgets"] = {}; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Legacy panel upgraded"): audit.load(path)

    def test_cli_syntax_duplicates_and_unsigned_task_allocation_reject(self):
        cases = (["flappycnn=262144:131072", "flappycnn=262144:131072"],
            ["flappycnn=262144"], ["flappycnn=262144:131072:1"], ["flappycnn=3.5:2"],
            ["unknown=65536:32768"], ["connect4cnn=0:32768"])
        for i, budgets in enumerate(cases):
            out = self.root / f"cli-{i}"
            command = [sys.executable, str(panel.ROOT / "research/prepare_pixel_robustness.py"), "--out", str(out)]
            for item in budgets: command.extend(["--task-budget", item])
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__": unittest.main()
