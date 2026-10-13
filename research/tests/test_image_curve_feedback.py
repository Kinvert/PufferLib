"""Synthetic scalar curves only. No native process, GPU, CPU CNN or dataset."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_curve_feedback as tool


class CurveFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.plan=dict(protocol=tool.panels.IMAGE_PROTOCOL,per_game_learners=True,seeds=[7,42],
            image_contracts={t:dict(version="human-geometry-v1",shape_hwc=[640,480,3],layout="NHWC",dtype="float32",preprocessing="none")
                             for t in tool.panels.TASKS},
            task_budgets={t:dict(steps=8,checkpoint_steps=1) for t in tool.panels.TASKS},
            learner_recipes={t:dict(sha256="ab"*32) for t in tool.panels.TASKS},
            appearance_seed=17,eval_seed=7007,episodes=33,slots=4,caps=dict(pong_max_decisions=512,breakout_max_frames=2048),
            source_sha256={"src/ocean.cu":"cd"*32},jobs=[],
            candidates=[dict(name="nature-cnn",kind="baseline",baseline="nature_cnn",architecture=dict(encoder=2))])
        for task in tool.panels.TASKS:
            config=self.root/(task+".ini"); cfg=tool.panels.panel.read(tool.panels.ROOT/"config/default.ini")
            cfg["base"].update(env_name=task,seed="7",run_id="nature-cnn",checkpoint_dir="checkpoints",log_dir="logs")
            cfg["policy"].update(encoder="2",hidden_size="128",num_layers="1")
            with config.open("x") as f: cfg.write(f)
            for seed in (7,42):
                self.plan["jobs"].append(dict(environment=task,seed=seed,candidate="nature-cnn",
                                              config=config.name,checkpoint_steps=list(range(1,9))))
        rows=[dict(candidate="nature-cnn",environment=t,seed=s,representation=d,decisions=step,
                   train_seconds=2.*step,score_lower=.5,score_upper=.5)
              for t in tool.panels.TASKS for s in (7,42) for step in range(1,9) for d in range(tool.panels.panel.ENVIRONMENTS[t])]
        self.plan["planned_evaluations"]=len(rows)
        self.analysis=dict(status="ok",missing_evaluations=[],audited_jobs=12,audited_evaluations=len(rows),
                           observations=rows,result_sha256="de"*32,hardware=dict(gpu_name="synthetic-fixture"))
        anchors={t:dict(offset=0.,target=1.) for t in tool.panels.TASKS}
        self.reference=tool.freeze(self.plan,self.analysis,self.root,anchors,self.root/"reference.json")
        self.plan["candidates"]=[dict(name="happy-cat-1",architecture=dict(encoder=4))]
        for job in self.plan["jobs"]: job["candidate"]="happy-cat-1"
        for row in self.analysis["observations"]: row["candidate"]="happy-cat-1"

    def result(self,analysis=None,reference=None,initial=None):
        return tool.aggregate(self.plan,analysis or self.analysis,self.root,reference or self.reference,initial)

    def test_eight_points_equal_game_seed_weights_and_once_per_job_cost(self):
        result=self.result()
        self.assertAlmostEqual(result["score"],.5)
        self.assertEqual(result["training_jobs_charged"],12); self.assertEqual(result["cost"],192.)
        self.assertEqual(len(result["points"]),8)
        self.assertEqual(result["points"][0]["coverage"],dict(wins=0,ties=6,losses=0))

    def test_declines_and_last_affordable_not_best_so_far(self):
        for row in self.analysis["observations"]:
            if row["decisions"]==7: row["score_lower"]=row["score_upper"]=1.
            if row["decisions"]==8: row["score_lower"]=row["score_upper"]=0.
        result=self.result()
        self.assertAlmostEqual(result["points"][-2]["balanced_lower"],1.)
        self.assertAlmostEqual(result["points"][-1]["balanced_lower"],0.)
        self.assertTrue(all(s["selected_decisions"]==8 for g in result["points"][-1]["games"] for s in g["selections"]))

    def test_no_interpolation_or_extrapolated_training(self):
        for row in self.analysis["observations"]: row["train_seconds"]*=.5
        result=self.result()
        self.assertEqual(result["points"][0]["games"][0]["selections"][0]["selected_decisions"],2)
        self.assertEqual(result["points"][-1]["games"][0]["selections"][0]["selected_decisions"],8)
        self.assertEqual(result["cost"],96.)

    def test_pong_censoring_lower_and_upper_remain_separate(self):
        for row in self.analysis["observations"]:
            if row["environment"]=="pongcnn": row["score_lower"]=0.;row["score_upper"]=1.
        result=self.result(); self.assertLess(result["score"],.5); self.assertGreater(result["score_upper"],.5)
        self.assertTrue(result["conservative_pong"])

    def test_single_game_gain_gets_less_reward_than_balanced_gain(self):
        one=tool.balanced([1.,.2,.2,.2,.2,.2]); broad=tool.balanced([1/3]*6)
        self.assertAlmostEqual(sum([1.,.2,.2,.2,.2,.2])/6,1/3)
        self.assertLess(one,broad)

    def test_missing_duplicate_nan_changed_cost_and_failure_never_submit(self):
        def duplicate(rows): rows[-1]=copy.deepcopy(rows[0])
        modifications=[lambda a:a.update(status="failed"),lambda a:a.update(missing_evaluations=[1]),
            lambda a:a["observations"].pop(),lambda a:duplicate(a["observations"]),
            lambda a:a["observations"][0].update(score_lower=float("nan")),
            lambda a:a["observations"][0].update(train_seconds=2.1)]
        for modification in modifications:
            bad=copy.deepcopy(self.analysis); modification(bad)
            with self.subTest(modification=modification),self.assertRaises(ValueError): self.result(analysis=bad)

    def test_no_affordable_checkpoint_requires_measured_own_initial_policy(self):
        for row in self.analysis["observations"]: row["train_seconds"]+=1.
        with self.assertRaisesRegex(ValueError,"measured untrained"): self.result()
        initial={(t,s):[dict(candidate="happy-cat-1",environment=t,seed=s,representation=d,decisions=0,
                            train_seconds=1.,score_lower=.1,score_upper=.1)
                       for d in range(tool.panels.panel.ENVIRONMENTS[t])]
                 for t in tool.panels.TASKS for s in (7,42)}
        result=self.result(initial=initial); self.assertAlmostEqual(result["points"][0]["balanced_lower"],.1)
        initial[("pongcnn",7)][0]["candidate"]="nature-cnn"
        with self.assertRaisesRegex(ValueError,"measured untrained"): self.result(initial=initial)

    def test_v5_uses_its_audited_initial_rows_and_refuses_external_replacements(self):
        self.plan["protocol"] = tool.panels.IMAGE_CURVE_PROTOCOL
        self.plan["initial_checkpoint"] = "same-policy-before-first-update-v1"
        for job in self.plan["jobs"]: job["checkpoint_steps"].insert(0,0)
        initial = [dict(candidate="happy-cat-1",environment=t,seed=s,representation=d,decisions=0,
                        train_seconds=1.,score_lower=.1,score_upper=.1)
                   for t in tool.panels.TASKS for s in (7,42) for d in range(tool.panels.panel.ENVIRONMENTS[t])]
        for row in self.analysis["observations"]: row["train_seconds"]+=1.
        self.analysis["observations"]+=initial
        self.plan["planned_evaluations"]=self.analysis["audited_evaluations"]=len(self.analysis["observations"])
        reference=copy.deepcopy(self.reference)
        reference["conditions"]=tool.conditions(self.plan,self.root)
        result=self.result(reference=reference)
        self.assertAlmostEqual(result["points"][0]["balanced_lower"],.1)
        self.assertEqual(result["cost"],204.)
        with self.assertRaisesRegex(ValueError,"complete audited panel"):
            self.result(reference=reference,initial={})
        self.analysis["observations"].pop()
        with self.assertRaises(ValueError): self.result(reference=reference)

    def test_pixel_seed_learner_cap_native_source_and_hardware_mismatches_refuse(self):
        for field,value in (("appearance_seed",19),("eval_seed",1337),("seeds",[42,7]),("slots",16),
                            ("source_sha256",{"src/ocean.cu":"ab"*32})):
            original=self.plan[field]; self.plan[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.result()
            self.plan[field]=original
        bad=copy.deepcopy(self.analysis); bad["hardware"]["gpu_name"]="different-fixture"
        with self.assertRaises(ValueError): self.result(analysis=bad)
        cfg=tool.panels.panel.read(self.root/"flappycnn.ini");cfg["train"]["learning_rate"]="0.9"
        with (self.root/"flappycnn.ini").open("w") as f: cfg.write(f)
        with self.assertRaises(ValueError): self.result()

    def test_reference_cannot_change_objective_resolution_or_drop_drawing(self):
        for field,value in (("epsilon",.1),("checkpoints",4),("objective","best-seed")):
            bad=copy.deepcopy(self.reference); bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.result(reference=bad)
        bad=copy.deepcopy(self.reference);bad["nature"]["pongcnn"]["7"][0]["drawings"].pop()
        with self.assertRaises(ValueError): self.result(reference=bad)


if __name__=="__main__": unittest.main()
