"""Scalar objective/config checks only: no GPU, NN, optimizer or dataset."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
import time
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cross_game_feedback as tool


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.path = tool.ROOT / "research/recipes/cross_game_feedback_smoke.ini"
        self.config, seeds = tool.recipe(self.path)
        self.plan = dict(candidates=[dict(name="happy-cat-1")], seeds=[1,2],
            task_budgets={t:dict(steps=100) for t in tool.panels.TASKS}, jobs=list(range(12)))
        self.rows = []
        for task,count in tool.panels.panel.ENVIRONMENTS.items():
            lo = self.config.getfloat("objective."+task,"offset")
            hi = self.config.getfloat("objective."+task,"target")
            for seed in (1,2):
                for drawing in range(count):
                    self.rows.append(dict(environment=task,seed=seed,representation=drawing,decisions=100,
                        score_lower=lo+(hi-lo)*0.5,score_upper=lo+(hi-lo)*0.5,train_seconds=float(seed)))
        self.plan["planned_evaluations"] = len(self.rows)
        self.analysis = dict(status="ok",missing_evaluations=[],audited_jobs=12,observations=self.rows)

    def test_equal_games_not_equal_drawing_count_and_training_charged_once(self):
        for row in self.rows:
            if row["environment"] == "connect4cnn": row["score_lower"]=row["score_upper"]=1
        result=tool.aggregate(self.plan,self.analysis,self.config)
        self.assertAlmostEqual(result["score"],(1+5*0.5)/6)
        self.assertEqual(result["cost"],18)
        self.assertEqual(result["training_jobs_charged"],12)

    def test_pong_uses_lower_bound_and_keeps_upper(self):
        for row in self.rows:
            if row["environment"] == "pongcnn": row["score_lower"]=0;row["score_upper"]=1
        result=tool.aggregate(self.plan,self.analysis,self.config)
        self.assertAlmostEqual(result["score"],2.5/6)
        self.assertAlmostEqual(result["score_upper"],3.5/6)

    def test_failure_missing_duplicate_nonfinite_and_changed_cost_refuse_feedback(self):
        mutations=[lambda a:a.update(status="failed"),lambda a:a.update(missing_evaluations=[1]),
            lambda a:a["observations"].pop(),
            lambda a:a["observations"].__setitem__(0,copy.deepcopy(a["observations"][1])),
            lambda a:a["observations"][0].update(score_lower=float("nan")),
            lambda a:a["observations"][0].update(train_seconds=100)]
        for mutation in mutations:
            analysis=copy.deepcopy(self.analysis);mutation(analysis)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                tool.aggregate(self.plan,analysis,self.config)

    def test_clipping_and_no_best_checkpoint_selection(self):
        for row in self.rows:row["score_lower"]=row["score_upper"]=-100
        result=tool.aggregate(self.plan,self.analysis,self.config)
        self.assertEqual(result["score"],0)
        older=[{**row,"decisions":50,"score_lower":1e9,"score_upper":1e9} for row in self.rows]
        plan={**self.plan,"planned_evaluations":2*len(self.rows)}
        self.assertEqual(tool.aggregate(plan,{**self.analysis,"observations":self.rows+older},self.config)["score"],0)

    def test_learner_core_inactive_unknown_ranges_cannot_be_search_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/"recipe.ini"
            text=self.path.read_text()
            for original,replacement in (("sweep.policy.cnn_channels_1","sweep.train.learning_rate"),
                ("max = 16","max = 15"),("hidden_size = 128","hidden_size = 256"),
                ("sweep.policy.cnn_channels_1","sweep.policy.cnn_channels_2")):
                target.write_text(text.replace(original,replacement))
                with self.subTest(replacement=replacement),self.assertRaises((ValueError,KeyError)):
                    tool.recipe(target)

    def test_objective_anchors_require_finite_positive_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/"recipe.ini"
            for replacement in ("target = nan","target = 0"):
                target.write_text(self.path.read_text().replace("target = 100",replacement))
                with self.assertRaises(ValueError):tool.recipe(target)

    def test_invalid_controls_and_seeds_rejected_before_native_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/"recipe.ini"
            for before, after in (("steps = 65536", "steps = 0"), ("gp_training_iter = 50", "gp_training_iter = 1.5"),
                    ("eval_seed = 70173", "eval_seed = 60173"), ("slots = 16", "slots = -1")):
                target.write_text(self.path.read_text().replace(before, after))
                with self.subTest(after=after), self.assertRaises(ValueError):tool.recipe(target)

    def test_reversed_bounds_cannot_supply_successful_observation(self):
        self.rows[0].update(score_lower=0.9, score_upper=0.1)
        with self.assertRaisesRegex(ValueError,"Reversed"):tool.aggregate(self.plan,self.analysis,self.config)

    def test_three_mock_panels_feed_native_ledger_and_keep_duplicate(self):
        # Process/search/NN calls are mocked. This tests only orchestration.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"recipe.ini").write_text(self.path.read_text())
            (root/"plan.json").write_text("{}")
            (root/"registry-location.json").write_text(json.dumps(dict(original=str(root/"registry.json"))))
            value=dict(dimensions=1,optimizer="fake-native",policy_metadata=str(root),learner_recipes=[],
                description=dict(coordinates=["cnn_channels_1"]))
            plan=copy.deepcopy(self.plan);plan["seeds"]=[60173]
            analysis=copy.deepcopy(self.analysis)
            analysis["observations"]=[{**r,"seed":60173} for r in self.rows if r["seed"]==1]
            plan["planned_evaluations"]=len(analysis["observations"])
            plan["task_budgets"]={t:dict(steps=100) for t in tool.panels.TASKS}
            plan["jobs"]=[dict(environment=t,seed=60173,config=t+".ini") for t in tool.panels.TASKS]
            analysis["audited_jobs"]=6
            histories=[]; panel_args=[]
            def process(command,cwd,log,timeout):
                history=Path(command[3]).read_text();histories.append(history)
                index=len(history.splitlines())-1
                output=dict(protocol="native-cross-game-protein-v1",trial=index,observations_replayed=index,
                    gp_observations=index,success_observations=index,failure_observations=0,
                    normalized=[1.0 if index==0 else -1.0],policy=dict(cnn_channels_1=16 if index==0 else 8))
                Path(command[4]).write_text(json.dumps(output))
                timing=dict(status="ok",returncode=0,command=command,cwd=str(cwd),timeout=timeout,
                    launch_monotonic_ns=time.monotonic_ns())
                timing["end_monotonic_ns"]=time.monotonic_ns()
                log.with_suffix(log.suffix+".json").write_text(json.dumps(timing))
                return timing
            def prepare(args):
                panel_args.append(args);args.out.mkdir()
                trial_plan=copy.deepcopy(plan)
                trial_plan["candidates"]=[dict(architecture=tool.panels.architecture(Path(args.candidate[0].split("=",1)[1])))]
                (args.out/"plan.json").write_text(json.dumps(trial_plan))
                for job in trial_plan["jobs"]:
                    source=tool.panels.panel.read(Path(args.candidate[0].split("=",1)[1]))
                    source["base"]={"seed":"60173","run_id":args.candidate[0].split("=",1)[0]}
                    source["train"]={"learning_rate":"0.001"}
                    with (args.out/job["config"]).open("w") as stream:source.write(stream)
                return trial_plan
            def audit(path,out):
                out.mkdir();(out/"analysis.json").write_text("{}")
                return analysis
            def panel_run(args):
                destination=args.plan.parent/"execution";destination.mkdir()
                start=time.monotonic_ns();end=time.monotonic_ns()
                (destination/"result.json").write_text(json.dumps(dict(started_monotonic_ns=start,ended_monotonic_ns=end)))
            with patch.object(tool,"ROOT",root),patch.object(tool,"inspect",return_value=value),\
                 patch.object(tool.panels,"hardware",return_value="fake-5090") as hardware,\
                 patch.object(tool.panels,"process",side_effect=process),\
                 patch.object(tool.panels,"prepare",side_effect=prepare),\
                 patch.object(tool.panels,"run",side_effect=panel_run),\
                 patch.object(tool.panels,"audit",side_effect=audit):
                result=tool.run(SimpleNamespace(plan=root/"plan.json",allow_gpu=True,mode="canary5090"))
            self.assertTrue((root/"build/connect4cnn/hardware-benchmark.lock").is_file())
            self.assertTrue(all(call.args == ("5090",) for call in hardware.call_args_list))
            self.assertEqual(result["status"],"ok")
            self.assertEqual([len(h.splitlines())-1 for h in histories],[0,1,2,3])
            self.assertEqual(result["final_observations_replayed"],3)
            self.assertEqual(result["trials"][2]["duplicate_of"],1)
            self.assertEqual(len(panel_args),3)
            self.assertTrue(all(args.steps==65536 and args.seeds==[60173] for args in panel_args))
            self.assertIn("0 0.5 6 1",histories[1])
            with patch.object(tool,"inspect",return_value=value),\
                 patch.object(tool.panels,"inspect",side_effect=lambda p:json.loads(p.read_text())),\
                 patch.object(tool.panels,"audit",side_effect=audit):
                reviewed=tool.audit(SimpleNamespace(plan=root/"plan.json",out=root/"review"))
                self.assertEqual(reviewed["status"],"ok")
                # Changing the history cannot pass a fresh offline audit.
                (root/"execution/final-feedback/history.tsv").write_text("changed")
                with self.assertRaisesRegex(ValueError,"history"):
                    tool.audit(SimpleNamespace(plan=root/"plan.json",out=root/"review-changed"))

    def test_gpu_permission_is_checked_before_inspection(self):
        with patch.object(tool,"inspect") as inspection,self.assertRaisesRegex(ValueError,"allow-gpu"):
            tool.run(SimpleNamespace(allow_gpu=False))
        inspection.assert_not_called()


if __name__ == "__main__":unittest.main()
