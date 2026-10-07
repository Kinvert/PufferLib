"""Allocation/scalar/display fixtures; no model, GPU or training."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_viewer as viewer
from research.tests import test_candidate_frontiers as fixtures


def fixture():
    plan, rows = fixtures.fixture()
    plan["planned_evaluations"] = len(rows)
    plan["task_budgets"] = {task: dict(steps=196608) for task in plan["assignments"]}
    training = []
    for index, job in enumerate(plan["jobs"]):
        job["id"] = f"fixture-job-{index}"
        ci = next(i for i, c in enumerate(plan["candidates"]) if c["name"] == job["candidate"])
        seconds = 3*(ci+1)
        training.append(dict(environment=job["environment"], candidate=job["candidate"], seed=job["seed"],
            parameters=4, train_seconds=seconds, process_sps=196608/seconds))
    for row in rows: row["counts"] = {"episodes": 17}
    analysis = dict(status="ok", mode="development", training=training, observations=rows,
                    hardware={"gpu_name": "SYNTHETIC; no GPU query"}, result_sha256="synthetic", inputs_sha256={})
    return plan, analysis


class CandidateViewerTests(unittest.TestCase):
    def test_allocation_keeps_arbitrary_models_all_drawings_and_every_checkpoint(self):
        plan, _ = fixture(); result = viewer.assemble(plan)
        self.assertEqual(result["model_catalog"], [c["name"] for c in plan["candidates"]])
        self.assertEqual(len(result["planned_cells"]), 738)
        self.assertEqual(len(result["missing"]), 738)
        self.assertEqual(len(result["conditions"]), 41)
        self.assertEqual(result["points"], [])
        self.assertEqual(result["execution_status"], "prepared-not-executed")
        self.assertEqual(result["accounted_training_process_seconds"], 0)
        self.assertFalse(result["training_cost_accounting_complete"])

    def test_full_paired_curves_keep_declines_bounds_and_charge_training_once(self):
        plan, analysis = fixture(); result = viewer.assemble(plan, analysis)
        self.assertEqual(len(result["points"]), 738); self.assertEqual(len(result["means"]), 369)
        self.assertEqual(len(result["training_job_costs"]), 36)
        self.assertEqual(result["accounted_training_process_seconds"], sum(row["train_seconds"] for row in analysis["training"]))
        self.assertTrue(result["training_cost_accounting_complete"])
        self.assertTrue(all(row["native_sps"] is None for row in result["training_job_costs"]))
        self.assertTrue(all(row["episodes"] == 17 and row["parameters"] == 4 for row in result["points"]))
        self.assertFalse(result["dominance_certified"]); self.assertFalse(result["quality_selection_allowed"])
        pong = [row for row in result["points"] if row["environment"] == "pongcnn"]
        self.assertTrue(all(abs(row["score_upper"]-row["score_lower"]-.1) < 1e-12 for row in pong))

    def test_smoke_and_failed_campaigns_have_no_frontier_marks(self):
        plan, analysis = fixture()
        for status, mode in (("ok", "smoke"), ("failed", "development")):
            with self.subTest(status=status, mode=mode):
                analysis.update(status=status, mode=mode)
                result = viewer.assemble(plan, analysis)
                self.assertTrue(all(row["frontier_seconds_lower"] is None and row["frontier_seconds_upper"] is None
                                    for row in result["means"]))

    def test_missing_seed_does_not_hide_candidate_or_invent_a_paired_mean(self):
        plan, analysis = fixture(); removed = analysis["observations"].pop()
        result = viewer.assemble(plan, analysis)
        self.assertEqual(len(result["missing"]), 1)
        self.assertIn(removed["candidate"], result["model_catalog"])
        condition = [row for row in result["means"] if row["environment"] == removed["environment"]
                     and row["evaluation_representation"] == removed["representation"]]
        self.assertTrue(all(row["frontier_seconds_lower"] is None for row in condition))

    def test_duplicate_training_and_inconsistent_sps_are_rejected(self):
        plan, analysis = fixture(); duplicate = copy.deepcopy(analysis)
        duplicate["training"].append(duplicate["training"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"): viewer.assemble(plan, duplicate)
        analysis["training"][0]["process_sps"] += 1
        with self.assertRaisesRegex(ValueError, "SPS"): viewer.assemble(plan, analysis)

    def test_prepare_renders_empty_html_without_audit_or_native_execution(self):
        plan, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "plan.json"; path.write_text("{}")
            with patch.object(viewer.candidates, "inspect", return_value=plan), \
                 patch.object(viewer.candidates, "audit", side_effect=AssertionError("Preparation must not audit execution")):
                result = viewer.render(path, root / "output")
                with self.assertRaisesRegex(ValueError, "fresh"): viewer.render(path, root / "output")
            html = (root / "output/curves.html").read_text()
            payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)[1])
            self.assertEqual(payload, result); self.assertFalse(payload["gpu_queried"])
            self.assertEqual(len(payload["planned_cells"]), 738)

    def test_report_requires_audit_and_keeps_its_checkpoint_identities(self):
        plan, analysis = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "plan.json"; path.write_text("{}")
            for job in plan["jobs"]:
                for step in job["checkpoint_steps"]:
                    checkpoint = root / job["id"] / "checkpoints" / job["environment"] / job["candidate"] / f"{step:016d}.bin"
                    analysis["inputs_sha256"][str(checkpoint)] = "synthetic-checkpoint-identity"
            with patch.object(viewer.candidates, "inspect", return_value=plan), \
                 patch.object(viewer.candidates, "audit", return_value=analysis) as audited:
                result = viewer.render(path, root / "output", executed=True)
            audited.assert_called_once_with(path, root / "output/audit")
            self.assertTrue(all(row["checkpoint_sha256"] == "synthetic-checkpoint-identity" for row in result["points"]))
            self.assertEqual(len(result["means"]), 369)

    def test_failed_audit_preserves_failure_without_rendering_scores(self):
        plan, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "plan.json"; path.write_text("{}")
            with patch.object(viewer.candidates, "inspect", return_value=plan), \
                 patch.object(viewer.candidates, "audit", side_effect=ValueError("bad raw episode receipts")):
                with self.assertRaisesRegex(ValueError, "raw episode"): viewer.render(path, root / "output", executed=True)
            self.assertTrue((root / "output/failure.json").is_file())
            self.assertFalse((root / "output/curves.html").exists())


if __name__ == "__main__": unittest.main()
