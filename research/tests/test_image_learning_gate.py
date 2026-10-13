"""Synthetic scalar learning receipts; no GPU, policy or CPU neural work."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research.tests.test_image_curve_feedback as curve_fixtures
import image_learning_gate as tool


class LearningGateTests(unittest.TestCase):
    def setUp(self):
        fixture=curve_fixtures.CurveFeedbackTests(); fixture.setUp(); self.addCleanup(fixture.temp.cleanup)
        self.root=fixture.root
        self.plan=copy.deepcopy(fixture.plan); self.analysis=copy.deepcopy(fixture.analysis)
        self.normalization=fixture.reference["normalization"]
        self.plan["protocol"]=tool.curves.panels.IMAGE_CURVE_PROTOCOL
        self.plan["initial_checkpoint"]="same-policy-before-first-update-v1"
        self.plan["seed_roles"]={"role":"development"}
        self.plan["seeds"].append(69)
        jobs=self.plan["jobs"]
        jobs.extend({**j,"seed":69} for j in list(jobs) if j["seed"]==7)
        for job in jobs: job.update(candidate="quality-reference",checkpoint_steps=[0]+list(range(1,9)))
        jobs.extend({**j,"candidate":"nature-cnn"} for j in list(jobs))
        self.plan["candidates"]=[dict(name="quality-reference",architecture={"encoder":4}),
            dict(name="nature-cnn",kind="baseline",baseline="nature_cnn",architecture={"encoder":2})]
        rows=[dict(candidate=j["candidate"],environment=j["environment"],seed=j["seed"],representation=d,
                   decisions=step,train_seconds=step+1.,score_lower=.1 if step==0 else .3,score_upper=.1 if step==0 else .3)
              for j in jobs for step in j["checkpoint_steps"] for d in range(tool.curves.panels.panel.ENVIRONMENTS[j["environment"]])]
        self.plan["planned_evaluations"]=len(rows)
        self.analysis.update(observations=rows,audited_jobs=len(jobs),audited_evaluations=len(rows))

    def qualify(self): return tool.qualify(self.plan,self.analysis,self.normalization)

    def test_all_games_need_a_fixed_control_with_paired_learning(self):
        result=self.qualify()
        self.assertEqual(result["status"],"eligible-development-search")
        self.assertEqual(len(result["controls"]),12)
        self.assertFalse(result["publication_claim_qualified"])

    def test_high_untrained_scores_cannot_qualify_without_learning(self):
        for row in self.analysis["observations"]: row["score_lower"]=row["score_upper"]=.8
        result=self.qualify()
        self.assertEqual(set(result["unresolved_games"]),set(tool.curves.panels.TASKS))

    def test_best_checkpoint_and_two_good_replicas_cannot_rescue_final_failure(self):
        for row in self.analysis["observations"]:
            if row["environment"]=="pongcnn" and row["seed"]==69 and row["decisions"]==8:
                row["score_lower"]=row["score_upper"]=0.
        result=self.qualify()
        self.assertEqual(result["unresolved_games"],["pongcnn"])
        self.assertEqual(result["status"],"blocked-learning-calibration")

    def test_another_predeclared_control_can_demonstrate_game_learning(self):
        for row in self.analysis["observations"]:
            if row["candidate"]=="quality-reference" and row["environment"]=="pongcnn":
                row["score_lower"]=row["score_upper"]=0.
        result=self.qualify()
        self.assertEqual(result["status"],"eligible-development-search")
        self.assertEqual(sum(c["eligible"] for c in result["controls"]),11)

    def test_partial_or_confirmation_receipts_cannot_start_search(self):
        self.analysis["observations"].pop()
        with self.assertRaises(ValueError): self.qualify()
        self.plan["seed_roles"]["role"]="confirmation"
        with self.assertRaises(ValueError): self.qualify()


if __name__=="__main__": unittest.main()
