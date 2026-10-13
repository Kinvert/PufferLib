"""Fixed calibration assignments and negative gating; no GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_image_calibration as tool
import research.tests.test_image_learning_gate as learning_fixtures


class CalibrationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.recipe=tool.panels.read_ini(tool.panels.ROOT/"research/recipes/image_feedback_smoke.ini")
        self.recipe["search"].update(curve_objective="balanced-eight-nature-time-anchors-v1",
            training_seeds="7,42,69",evaluation_seeds="7007,7008,7009",seed_role="development",
            steps="512",checkpoint_steps="64")
        self.roles=copy.deepcopy(tool.seed_roles.DEFAULT)
        self.args=SimpleNamespace(registry=self.root/"registry.json",policy_metadata=self.root/"metadata",
            quality_config=tool.panels.ROOT/"research/recipes/panel_smoke_quality.ini",
            seed_roles=self.root/"roles.json",
            learner_recipe=[task+"="+str(self.root/(task+".ini")) for task in tool.panels.TASKS])

    def arguments(self):
        return tool.panel_arguments(self.args,self.root,self.recipe,self.roles,self.root/"math.json")

    def test_all_six_matched_controls_receive_exact_development_assignments_and_initial_stage(self):
        value=self.arguments()
        self.assertEqual(value.seeds,[7,42,69]); self.assertEqual(value.evaluation_seeds,[7007,7008,7009])
        self.assertEqual(value.baselines,["nature_cnn"]); self.assertTrue(value.include_initial)
        self.assertEqual(len(value.learner_recipe),6); self.assertEqual(len(value.task_budget),6)
        self.assertTrue(all(entry.endswith("=512:64") for entry in value.task_budget))
        self.assertEqual(value.appearance_seed,17); self.assertEqual(value.seed_role,"development")

    def test_seed_selection_missing_learners_duplicate_learners_and_seven_checkpoints_fail(self):
        for key,value in (("training_seeds","7,42"),("seed_role","confirmation"),
                          ("checkpoint_steps","128")):
            original=self.recipe["search"][key]; self.recipe["search"][key]=value
            with self.assertRaises(ValueError): self.arguments()
            self.recipe["search"][key]=original
        original=list(self.args.learner_recipe)
        for changed in (original[:-1],original+[original[0]]):
            self.args.learner_recipe=changed
            with self.assertRaises(ValueError): self.arguments()

    def test_control_math_manifest_preserves_original_raw_allocations_without_renaming_controls(self):
        proofs=[]
        for name in ("quality-reference","nature-cnn"):
            path=self.root/(name+".json"); path.write_text(json.dumps(dict(schema=tool.checks.SCHEMA,
                candidates={name:{task:"raw/"+task for task in tool.panels.TASKS}})))
            proofs.append(path)
        output=self.root/"merged.json"; tool.math_manifest(*proofs,output)
        result=json.loads(output.read_text())
        self.assertEqual(set(result["candidates"]),{"quality-reference","nature-cnn"})
        self.assertEqual(result["candidates"]["nature-cnn"]["flappycnn"],"raw/flappycnn")
        value=json.loads(proofs[1].read_text());value["candidates"]={"fake-name":value["candidates"]["nature-cnn"]}
        proofs[1].write_text(json.dumps(value))
        with self.assertRaises(ValueError): tool.math_manifest(*proofs,self.root/"invalid.json")

    def test_failed_learning_keeps_review_and_never_creates_reference_anchors(self):
        fixture=learning_fixtures.LearningGateTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        allocation=self.root/"controls"; (allocation/"panel").mkdir(parents=True)
        role_path=allocation/"panel/seed-roles.json"; role_path.write_text(json.dumps(self.roles))
        fixture.plan["seed_roles"]["file"]=role_path.name
        fixture.plan["task_budgets"]={t:dict(steps=512,checkpoint_steps=64) for t in tool.panels.TASKS}
        for job in fixture.plan["jobs"]: job["checkpoint_steps"]=[step*64 for step in job["checkpoint_steps"]]
        for row in fixture.analysis["observations"]: row["decisions"]*=64
        fixture.plan.update(episodes=self.recipe.getint("search","episodes"),slots=self.recipe.getint("search","slots"),
            caps={key:self.recipe.getint("search",key) for key in ("pong_max_decisions","breakout_max_frames")})
        plan_path=allocation/"panel/plan.json";plan_path.write_text(json.dumps(fixture.plan))
        recipe_path=allocation/"recipe.ini"
        with recipe_path.open("x") as stream: self.recipe.write(stream)
        (allocation/"ALLOCATION.json").write_text(json.dumps(dict(
            schema="realistic-image-calibration-allocation-v1",panel_plan_sha256=tool.sha(plan_path),
            recipe_sha256=tool.sha(recipe_path),normalization=fixture.normalization,seed_roles_sha256=tool.sha(role_path))))
        def write_review(plan,out,panel_only):
            (out/"review").mkdir(parents=True)
            (out/"review/analysis.json").write_text(json.dumps(fixture.analysis))
        import report_image_feedback as reporting
        import image_curve_feedback as curves
        for eligible in (False,True):
            for row in fixture.analysis["observations"]:
                row["score_lower"]=row["score_upper"]=(.1 if row["decisions"]==0 else .3) if eligible else .8
            destination=self.root/("eligible" if eligible else "unlearned")
            with patch.object(reporting,"report",side_effect=write_review), \
                 patch.object(tool.panels,"inspect",return_value=fixture.plan), \
                 patch.object(curves,"freeze") as freeze:
                gate=tool.qualify(allocation,destination)
                self.assertEqual(freeze.call_count,1 if eligible else 0)
                self.assertEqual(gate["status"],"eligible-development-search" if eligible else "blocked-learning-calibration")
                self.assertTrue((destination/"learning-eligibility.json").is_file())
                self.assertTrue((destination/"report/review/analysis.json").is_file())


if __name__ == "__main__": unittest.main()
