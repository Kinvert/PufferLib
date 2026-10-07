"""Synthetic artifact/control-flow tests; no policies or native processes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import candidate_acceptance_cases as tool


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.plan = self.root / "plan.json"
        self.jobs = []
        self.hashes = {}
        self.builds = {}
        drawings = dict(connect4cnn=10, pongcnn=7, flappycnn=7, breakoutcnn=5, snakecnn=6, mazecnn=6)
        for task, count in drawings.items():
            for name, family in (("happy-cat-1", "flex"), ("nature-cnn", "nature")):
                directory = f"jobs/{task}/{name}"
                config = directory + "/config/default.ini"
                binary = self.root / (task + "-binary")
                binary.write_bytes(b"synthetic-not-executable")
                self.builds[task] = dict(path=str(binary), sha256=tool.sha(binary))
                files = [self.root / config, binary]
                for step in (32, 64): files.append(self.root / directory / "checkpoints" / task / name / f"{step:016d}.bin")
                targets = [dict(representation=r, suite=f"suites/{task}/r{r}/suite.json") for r in range(count)]
                files += [self.root / t["suite"] for t in targets]
                for source in files:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if not source.exists(): source.write_bytes(b"synthetic-artifact")
                    self.hashes[str(source)] = tool.sha(source)
                self.jobs.append(dict(id=directory, environment=task, candidate=name, seed=173,
                    policy_family=family, config=config, command=[str(binary), "train"],
                    checkpoint_steps=[32, 64], targets=targets))
        self.write_plan()
        self.analysis = dict(status="ok", audited_jobs=12, planned_jobs=12,
            audited_evaluations=82, planned_evaluations=82, missing_evaluations=[],
            inputs_sha256=self.hashes, plan_sha256=tool.sha(self.plan))

    def write_plan(self):
        self.plan.write_text(json.dumps(dict(jobs=self.jobs,
            native_registry=dict(protocol="native-cnn-candidate-build-v1", binaries=self.builds))))

    def audit(self, path, out):
        out.mkdir(); tool.save(out / "analysis.json", self.analysis); return self.analysis

    def export(self, all_drawings=False):
        with patch.object(tool.panel, "audit", side_effect=self.audit), \
             patch("subprocess.Popen", side_effect=AssertionError("Process prohibited")), \
             patch("subprocess.run", side_effect=AssertionError("Process prohibited")), \
             patch("subprocess.check_output", side_effect=AssertionError("Process prohibited")):
            return tool.export(self.plan, self.root / "export", all_drawings)

    def test_drawing_zero_all_models_final_scheduled_checkpoint(self):
        record = self.export(); cases = json.loads((self.root / "export/cases.json").read_text())
        self.assertEqual(record["cases"], 10); self.assertEqual(len(record["excluded"]), 2)
        self.assertEqual({c["task"] for c in cases}, set(tool.gate.TASKS))
        self.assertEqual({c["family"] for c in cases}, {"flex", "nature"})
        self.assertTrue(all(c["checkpoint"].endswith("0000000000000064.bin") for c in cases))
        self.assertTrue(all(b["checkpoint_step"] == 64 and b["evaluation_representation"] == 0 for b in record["bindings"]))
        self.assertFalse(record["gpu_queried"]); self.assertFalse(record["quality_selection_allowed"])

    def test_all_drawing_export_keeps_every_cell(self):
        record = self.export(True)
        self.assertEqual(record["cases"], 62)
        self.assertEqual(len({b["name"] for b in record["bindings"]}), 62)
        self.assertEqual({b["evaluation_representation"] for b in record["bindings"]}, set(range(7)))

    def test_incomplete_panel_rejected_before_case_output(self):
        self.analysis["audited_evaluations"] -= 1
        with self.assertRaisesRegex(ValueError, "complete successful"): self.export()
        self.assertFalse((self.root / "export/cases.json").exists())
        self.assertTrue((self.root / "export/failure.json").exists())

    def test_changed_input_rejected_even_with_successful_audit_status(self):
        (self.root / self.jobs[2]["config"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing/changed"): self.export()

    def test_native_binary_hash_binding_rejected(self):
        Path(self.jobs[2]["command"][0]).write_bytes(b"changed-binary")
        with self.assertRaisesRegex(ValueError, "native binary differs"): self.export()

    def test_unrecognized_family_not_silently_dropped(self):
        self.jobs[2]["policy_family"] = "flex2"; self.write_plan()
        self.analysis["plan_sha256"] = tool.sha(self.plan)
        with self.assertRaisesRegex(ValueError, "Unsupported"): self.export()

    def test_wrong_checkpoint_order_rejected(self):
        self.jobs[2]["checkpoint_steps"] = [64, 32]; self.write_plan()
        self.analysis["plan_sha256"] = tool.sha(self.plan)
        with self.assertRaisesRegex(ValueError, "ordered scheduled"): self.export()

    def test_plan_drift_during_audit_rejected(self):
        self.analysis["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Plan changed"): self.export()

    def test_oversized_allocation_is_not_truncated(self):
        for _ in range(6): self.jobs.append(copy.deepcopy(self.jobs[2]))
        self.write_plan(); self.analysis["plan_sha256"] = tool.sha(self.plan)
        self.analysis["planned_jobs"] = self.analysis["audited_jobs"] = len(self.jobs)
        with self.assertRaisesRegex(ValueError, "1–64"): self.export(True)
        self.assertFalse((self.root / "export/cases.json").exists())


if __name__ == "__main__": unittest.main()
