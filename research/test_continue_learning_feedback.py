"""Host-only continuation tests: fake native/process adapters, no GPU queries."""
import configparser
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("continuation", Path(__file__).with_name("continue_learning_feedback.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ContinuationTests(unittest.TestCase):
    def test_budget_bounds(self):
        c.check_budget(36, 12, 12)
        for args in ((0, 12, 12), (65, 12, 12), (36, 0, 12), (36, float("nan"), 12), (36, 12, 500)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                c.check_budget(*args)

    def test_history_preserves_cost_and_coordinate_roundtrip(self):
        entry = dict(feedback=dict(score=0.35391309916541164, cost=631.884930176),
                     proposal=dict(normalized=[0.123456789, 1.0]))
        fields = c.history_row(entry).split()
        self.assertEqual(fields[0], "0")
        self.assertEqual(float(fields[1]), entry["feedback"]["score"])
        self.assertEqual(float(fields[2]), entry["feedback"]["cost"])
        self.assertEqual([float(v) for v in fields[3:]], entry["proposal"]["normalized"])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "search").mkdir()
        (self.root / "build/connect4cnn").mkdir(parents=True)
        c.save(self.root / "continuation.json", {})
        c.save(self.root / "search/registry-location.json", dict(original="/registry.json"))
        self.entries = [(self.root / "parent", dict(index=i, architecture={"width": 8})) for i in range(12)]
        self.history = "PUFFER_CROSS_GAME_V1 1\n" + "0 0.25 40 0.5\n" * 12
        self.next = dict(normalized=[0.5], policy={"cnn_c1": 8})
        self.config = configparser.ConfigParser(interpolation=None)
        self.config["policy"] = dict(encoder="4")
        self.config["search"] = {k: str(v) for k, v in dict(proposal_timeout=120, steps=100,
            checkpoint_steps=25, train_timeout=900, eval_timeout=120, appearance_seed=3,
            eval_seed=4, episodes=33, slots=16, pong_max_decisions=8192, breakout_max_frames=16384).items()}
        self.hardware = {"gpu": "5090"}
        self.value = dict(optimizer="/worker", dimensions=1, description={"coordinates": ["cnn_c1"]},
            encoder_validation="/validation", policy_metadata="/metadata", learner_recipes=[])
        self.plan = dict(parent=str(self.root / "parent"), inherited_observations=12,
            inherited_history_sha256=c.hashlib.sha256(self.history.encode()).hexdigest(), additional_trials=1, hours=12)
        self.ledgers = []
        self.changed_next = False
        panels = SimpleNamespace(hardware=Mock(return_value=self.hardware), process=Mock(side_effect=self.process),
            panel=SimpleNamespace(read=self.config_read), architecture=Mock(return_value={"width": 8}),
            prepare=Mock(return_value={}), run=Mock(), audit=Mock(return_value={}))
        self.fb = SimpleNamespace(ROOT=self.root, panels=panels, inspect=Mock(return_value=self.value),
            recipe=Mock(return_value=(self.config, [1, 2])), task_budgets=Mock(return_value={"game": {"steps": 100, "checkpoint_steps": 25}}),
            per_game_learners=Mock(return_value=True), aggregate=Mock(return_value=dict(score=0.3, cost=41)))
        self.checks = SimpleNamespace(run=Mock(side_effect=self.math_check))

    def config_read(self, *args):
        cfg = configparser.ConfigParser(interpolation=None)
        cfg["policy"] = dict(encoder="4")
        return cfg

    def process(self, command, cwd, log, timeout):
        ledger = Path(command[3]).read_text()
        self.ledgers.append(ledger)
        count = len(ledger.splitlines()) - 1
        proposal = dict(self.next, observations_replayed=count)
        if self.changed_next:
            proposal["normalized"] = [0.75]
        c.save(command[4], proposal)
        return {"status": "ok"}

    def math_check(self, validation, candidate, output, expected, timeout):
        output.mkdir()
        c.save(output / "REPORT.json", dict(status="ok"))
        return dict(hardware=self.hardware)

    def audit_panel(self, plan, out):
        out.mkdir(parents=True)
        c.save(out / "analysis.json", {})
        return {}

    def execute(self, allow=True):
        self.fb.panels.audit.side_effect = self.audit_panel
        with patch.object(c, "inspect", return_value=self.plan), patch.object(c, "parent_records", return_value=(
                self.entries, self.history, self.next, self.hardware, self.root / "parent")), \
                patch.object(c.importlib, "import_module", return_value=self.checks), patch.object(c, "audit", return_value={}) as review:
            result = c.run(SimpleNamespace(out=self.root, allow_gpu=allow), self.fb)
            return result, review

    def test_complete_panel_appends_to_all_twelve_and_retains_duplicate(self):
        result, review = self.execute()
        self.assertEqual(self.ledgers[0], self.history)
        self.assertEqual(self.ledgers[1], self.history + "0 0.29999999999999999 41 0.5\n")
        self.assertEqual(result["final_observations_replayed"], 13)
        self.assertEqual(result["trials"][0]["index"], 12)
        self.assertEqual(result["trials"][0]["duplicate_of"], 0)
        self.fb.panels.run.assert_called_once()
        review.assert_called_once()

    def test_first_proposal_mismatch_stops_before_model_checks_or_training(self):
        self.changed_next = True
        with self.assertRaisesRegex(ValueError, "parent's final unused"):
            self.execute()
        self.checks.run.assert_not_called()
        self.fb.panels.run.assert_not_called()
        result = c.read(self.root / "execution/result.json")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["history"], self.history)
        self.assertEqual(result["trials"], [])

    def test_failed_math_gate_preserves_history_and_does_not_train(self):
        self.checks.run.side_effect = ValueError("numerical gate failed")
        with self.assertRaisesRegex(ValueError, "numerical gate failed"):
            self.execute()
        self.fb.panels.run.assert_not_called()
        self.assertEqual(c.read(self.root / "execution/result.json")["history"], self.history)

    def test_failed_training_does_not_submit_partial_feedback(self):
        self.fb.panels.run.side_effect = RuntimeError("native timeout")
        with self.assertRaisesRegex(RuntimeError, "native timeout"):
            self.execute()
        self.fb.aggregate.assert_not_called()
        result = c.read(self.root / "execution/result.json")
        self.assertEqual(result["history"], self.history)
        self.assertEqual(result["trials"], [])

    def test_no_gpu_authorization(self):
        with self.assertRaisesRegex(ValueError, "allow-gpu"):
            self.execute(allow=False)
        self.fb.panels.hardware.assert_not_called()
        self.fb.panels.process.assert_not_called()

    def test_existing_execution_is_never_overwritten(self):
        (self.root / "search/execution").mkdir()
        sentinel = self.root / "search/execution/original.txt"
        sentinel.write_text("retained")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(sentinel.read_text(), "retained")
        self.fb.panels.process.assert_not_called()

    def test_time_cap_between_panels_observes_prefix_without_new_training(self):
        self.plan["hours"] = 0.001
        result, _ = self.execute()
        self.assertEqual(result["stop_reason"], "time-cap-between-trials")
        self.assertEqual(result["final_observations_replayed"], 12)
        self.assertEqual(result["trials"], [])
        self.fb.panels.run.assert_not_called()

    def test_changed_imported_history_rejected_before_execution(self):
        self.plan["inherited_history_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Changed imported history"):
            self.execute()
        self.assertFalse((self.root / "search/execution").exists())
        self.fb.panels.process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
