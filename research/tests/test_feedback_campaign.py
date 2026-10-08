"""Scalar selection, recipes, mocked stage/failure checks; no GPU/CPU CNN."""
import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import feedback_campaign as tool


class CampaignTests(unittest.TestCase):
    def fixture(self):
        recipe=tool.panels.panel.read(tool.ROOT/"research/recipes/cross_game_feedback_prepare.ini")
        plan=dict(candidates=[dict(name=n) for n in ("quality-reference","nature-cnn")],seeds=[1,2],
            jobs=[dict(candidate=n,environment=t,seed=s) for n in ("quality-reference","nature-cnn") for t in tool.panels.TASKS for s in (1,2)],
            task_budgets={t:dict(steps=100,checkpoint_steps=25) for t in tool.panels.TASKS},learner_recipes={})
        rows=[]
        for n in ("quality-reference","nature-cnn"):
            for t,count in tool.panels.panel.ENVIRONMENTS.items():
                offset,target=recipe.getfloat("objective."+t,"offset"),recipe.getfloat("objective."+t,"target")
                for seed in (1,2):
                    for drawing in range(count):
                        for step in (25,50,75,100):
                            score=offset+(target-offset)*(.2 if step==100 else .9)
                            rows.append(dict(candidate=n,environment=t,seed=seed,representation=drawing,decisions=step,
                                             score_lower=score,score_upper=score,train_seconds=step/10))
        plan["planned_evaluations"]=len(rows)
        analysis=dict(status="ok",missing_evaluations=[],audited_jobs=len(plan["jobs"]),observations=rows)
        return plan,analysis,recipe

    def test_real_recipe_has_disjoint_seeds_four_stages_and_positive_limits(self):
        config,seeds,budgets=tool.study(tool.ROOT/"research/recipes/feedback_learning.ini")
        self.assertEqual(config.getint("campaign","max_runs"),12)
        self.assertEqual(len(budgets),6)
        self.assertTrue(all(b["steps"]==4*b["checkpoint_steps"] for b in budgets.values()))
        self.assertFalse(set(seeds["calibration_seeds"])&set(seeds["search_seeds"]))

    def test_complete_controls_can_start_development_and_keep_full_budgets(self):
        plan,analysis,recipe=self.fixture(); result=tool.eligibility(plan,analysis,recipe,.1,.025)
        self.assertEqual(result["status"],"eligible-development-search")
        self.assertEqual(result["task_budgets"],plan["task_budgets"])
        self.assertTrue(all(abs(c["mean"]-.2)<1e-12 for c in result["controls"]))

    def test_one_unlearned_game_blocks_without_best_checkpoint_rescue(self):
        plan,analysis,recipe=self.fixture()
        for row in analysis["observations"]:
            if row["environment"]=="mazecnn" and row["decisions"]==100: row["score_lower"]=row["score_upper"]=0
        result=tool.eligibility(plan,analysis,recipe,.1,.025)
        self.assertEqual(result["unresolved_games"],["mazecnn"])
        self.assertEqual(result["status"],"blocked-learning-calibration")

    def test_lucky_seed_and_missing_cell_cannot_qualify(self):
        plan,analysis,recipe=self.fixture()
        for row in analysis["observations"]:
            if row["environment"]=="mazecnn" and row["decisions"]==100 and row["seed"]==2: row["score_lower"]=row["score_upper"]=0
        self.assertIn("mazecnn",tool.eligibility(plan,analysis,recipe,.1,.025)["unresolved_games"])
        analysis["observations"].pop()
        with self.assertRaises(ValueError): tool.eligibility(plan,analysis,recipe,.1,.025)

    def test_failures_never_turn_into_zero_scores(self):
        plan,analysis,recipe=self.fixture()
        for change in (dict(status="failed"),dict(missing_evaluations=[1]),dict(audited_jobs=23)):
            with self.subTest(change=change),self.assertRaises(ValueError): tool.eligibility(plan,{**analysis,**change},recipe,.1,.025)

    def test_stage_failure_is_retained_and_search_not_called(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copyfile(tool.ROOT/"research/recipes/feedback_learning.ini",root/"study.ini")
            (root/"campaign.json").write_text("{}")
            with patch.object(tool,"inspect",return_value=dict(validation=str(root))), \
                 patch.object(tool.graph_checks,"run",side_effect=ValueError("math failed")), \
                 patch.object(tool.feedback,"run") as search:
                with self.assertRaisesRegex(ValueError,"math failed"): tool.run(SimpleNamespace(out=root,allow_gpu=True))
            search.assert_not_called()
            result=json.loads((root/"execution/result.json").read_text())
            self.assertEqual(result["status"],"failed"); self.assertIn("math failed",result["error"])
            with self.assertRaises(FileExistsError):
                with patch.object(tool,"inspect",return_value=dict(validation=str(root))): tool.run(SimpleNamespace(out=root,allow_gpu=True))

    def test_unlearned_control_stops_before_native_search_and_reference_training(self):
        plan,analysis,_=self.fixture(); analysis["result_sha256"]="synthetic"
        for row in analysis["observations"]:
            if row["environment"]=="mazecnn": row["score_lower"]=row["score_upper"]=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"calibration").mkdir()
            for src,name in (("feedback_learning.ini","study.ini"),("cross_game_feedback_prepare.ini","search-space.ini")):
                shutil.copyfile(tool.ROOT/"research/recipes"/src,root/name)
            (root/"campaign.json").write_text("{}"); (root/"calibration/plan.json").write_text("{}")
            with patch.object(tool,"inspect",return_value=dict(validation=str(root))), \
                 patch.object(tool.graph_checks,"run"),patch.object(tool.panels,"run") as training, \
                 patch.object(tool.panels,"audit",return_value=analysis),patch.object(tool.panels,"inspect",return_value=plan), \
                 patch.object(tool.feedback,"prepare") as search:
                with self.assertRaisesRegex(ValueError,"calibration failed.*mazecnn"):
                    tool.run(SimpleNamespace(out=root,allow_gpu=True))
            self.assertEqual(training.call_count,1); search.assert_not_called()
            gate=json.loads((root/"execution/calibration-gate.json").read_text())
            self.assertEqual(gate["unresolved_games"],["mazecnn"])

    def test_combined_view_preserves_all_candidates_points_and_costs(self):
        from research.tests.test_candidate_viewer import fixture
        datasets=[]
        for prefix in ("search-","reference-"):
            plan,analysis=fixture()
            for record in plan["candidates"]: record["name"]=prefix+record["name"]
            for record in plan["jobs"]: record["candidate"]=prefix+record["candidate"]
            for record in analysis["observations"]+analysis["training"]: record["candidate"]=prefix+record["candidate"]
            datasets.append((plan,analysis))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"execution/search/execution").mkdir(parents=True); (root/"review").mkdir()
            (root/"execution/search/execution/result.json").write_text(json.dumps(dict(trials=[dict(index=0)])))
            def selected(path): return datasets[1 if "references" in str(path) else 0]
            with patch.object(tool.panels,"inspect",side_effect=lambda path:selected(path)[0]), \
                 patch.object(tool.panels,"audit",side_effect=lambda path,out:selected(path)[1]):
                result=tool.combined_view(root,{})
            self.assertEqual(result["observations"],1476); self.assertEqual(result["training_jobs"],72)
            self.assertTrue((root/"review/combined/curves.html").is_file())
            curves=json.loads((root/"review/combined/curves.json").read_text())
            self.assertEqual(len(curves["model_catalog"]),6)
            self.assertFalse(curves["publication_claim_qualified"])

    def test_successful_controller_freezes_full_budgets_and_runs_paired_references(self):
        plan,analysis,_=self.fixture(); analysis["result_sha256"]="synthetic"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"calibration").mkdir()
            for src,name in (("feedback_learning.ini","study.ini"),("cross_game_feedback_prepare.ini","search-space.ini")):
                shutil.copyfile(tool.ROOT/"research/recipes"/src,root/name)
            (root/"campaign.json").write_text("{}"); (root/"calibration/plan.json").write_text("{}")
            captured=dict(validation=str(root),registry=str(root),metadata=str(root),optimizer=str(root),
                          learners={"mazecnn":"learners/maze.ini"})
            with patch.object(tool,"inspect",return_value=captured),patch.object(tool.graph_checks,"run") as math_check, \
                 patch.object(tool.panels,"run") as training,patch.object(tool.panels,"audit",return_value=analysis), \
                 patch.object(tool.panels,"inspect",return_value=plan),patch.object(tool.feedback,"prepare") as prepare, \
                 patch.object(tool.feedback,"run") as search,patch.object(tool.feedback,"audit") as audit, \
                 patch.object(tool,"combined_view",return_value=dict(curves="synthetic")):
                result=tool.run(SimpleNamespace(out=root,allow_gpu=True))
            self.assertEqual(result["status"],"ok"); self.assertEqual(math_check.call_count,2)
            self.assertEqual([c.args[0].plan for c in training.call_args_list],
                             [root/"calibration/plan.json",root/"references/plan.json"])
            cfg,seeds=tool.feedback.recipe(root/"execution/search.ini")
            self.assertEqual(seeds,[64173,64174]); self.assertEqual(cfg.getint("search","max_runs"),12)
            self.assertEqual(tool.feedback.task_budgets(cfg),plan["task_budgets"])
            self.assertTrue(tool.feedback.per_game_learners(cfg))
            self.assertEqual(prepare.call_args.args[0].encoder_validation,root)
            self.assertEqual(prepare.call_args.args[0].learner_recipe,["mazecnn="+str(root/"learners/maze.ini")])
            search.assert_called_once(); audit.assert_called_once()
            self.assertEqual(result["stages"],["calibration-audited","native-search-audited","paired-references-audited"])


if __name__=="__main__": unittest.main()
