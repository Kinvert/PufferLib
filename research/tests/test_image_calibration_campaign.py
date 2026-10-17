"""Calibration admission/process/packaging fixtures; no GPU or neural math."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_calibration_campaign as tool


def probe():
    plan=dict(protocol=tool.panels.IMAGE_PROTOCOL,candidates=[dict(name=n) for n in tool.MODELS],
        seeds=[7],evaluation_seeds=[7007],episodes=tool.EPISODES,slots=tool.SLOTS,
        task_budgets={t:dict(steps=tool.PROBE_STEPS,checkpoint_steps=tool.PROBE_STEPS) for t in tool.panels.TASKS})
    training=[dict(environment=t,candidate=n,seed=7,train_seconds=1.+i,
        process_sps=tool.PROBE_STEPS/(1.+i)) for t in tool.panels.TASKS for i,n in enumerate(tool.MODELS)]
    analysis=dict(status="ok",audited_jobs=18,audited_evaluations=123,missing_evaluations=[],training=training)
    result=dict(status="ok",started_monotonic_ns=10**9,ended_monotonic_ns=100*10**9)
    return plan,analysis,result


class AdmissionTests(unittest.TestCase):
    def test_measured_slowest_model_sets_one_budget_per_game_for_all_models_and_seeds(self):
        plan,analysis,result=probe()
        value=tool.choose_budgets(plan,analysis,result,5.,40000.)
        self.assertEqual(value["status"],"admit-fixed-learning-panel")
        self.assertEqual(set(value["budgets"]),set(tool.panels.TASKS))
        self.assertLessEqual(value["estimated_total_seconds"],40000)
        for task,budget in value["budgets"].items():
            self.assertEqual(value["slowest_seconds_per_decision"][task],3./tool.PROBE_STEPS)
            self.assertEqual(budget["steps"]%tool.QUANTUM,0)
            self.assertEqual(budget["checkpoint_steps"]*8,budget["steps"])
            self.assertGreaterEqual(budget["steps"],tool.MIN_STEPS)
            self.assertLessEqual(budget["steps"],tool.MAX_STEPS)

    def test_insufficient_time_keeps_full_seed_drawing_and_minimum_training_requirements(self):
        value=tool.choose_budgets(*probe(),5.,100.)
        self.assertEqual(value["status"],"insufficient-time-for-minimum-learning-panel")
        self.assertIsNone(value["budgets"])
        self.assertGreater(value["required_seconds"],100.)

    def test_partial_foreign_or_tampered_probe_cannot_admit_main_learning(self):
        plan,analysis,result=probe()
        for key,value in (("status","failed"),("audited_evaluations",122),("missing_evaluations",[1])):
            bad=copy.deepcopy(analysis);bad[key]=value
            with self.assertRaises(ValueError):tool.choose_budgets(plan,bad,result,5.,40000.)
        bad=copy.deepcopy(analysis);bad["training"][0]["process_sps"]=123.
        with self.assertRaises(ValueError):tool.choose_budgets(plan,bad,result,5.,40000.)
        bad=copy.deepcopy(analysis);bad["training"][-1]=copy.deepcopy(bad["training"][0])
        with self.assertRaises(ValueError):tool.choose_budgets(plan,bad,result,5.,40000.)
        for values in ((float("nan"),40000.),(5.,float("nan"))):
            with self.assertRaises(ValueError):tool.choose_budgets(plan,analysis,result,*values)

    def test_probe_and_calibration_keep_same_learners_models_and_pixel_caps(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"seed-roles.json").write_text(json.dumps(tool.roles.DEFAULT))
            value=dict(models=[dict(name=n,config=n+".ini") for n in tool.MODELS],
                learners={t:dict(file=t+".ini") for t in tool.panels.TASKS})
            budgets={t:dict(steps=tool.MIN_STEPS,checkpoint_steps=tool.MIN_STEPS//8) for t in tool.panels.TASKS}
            short=tool.arguments(root,value,root/"probe",root/"math.json",budgets,False)
            main=tool.arguments(root,value,root/"main",root/"math.json",budgets,True)
            self.assertEqual(short.candidate,main.candidate);self.assertEqual(short.learner_recipe,main.learner_recipe)
            self.assertEqual(main.seeds,[7,42,69]);self.assertEqual(main.evaluation_seeds,[7007,7008,7009])
            self.assertEqual(short.seeds,[7]);self.assertEqual(short.evaluation_seeds,[7007])
            self.assertTrue(main.include_initial);self.assertFalse(short.include_initial)
            self.assertEqual(main.pong_max_decisions,short.pong_max_decisions)


class SupervisorTests(unittest.TestCase):
    def test_completed_probe_stops_before_main_if_minimum_budget_does_not_fit_and_never_restarts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"CAMPAIGN.json").write_text("{}\n")
            (root/"seed-roles.json").write_text(json.dumps(tool.roles.DEFAULT))
            value=dict(models=[dict(name=n,config=n+".ini") for n in tool.MODELS],
                learners={t:dict(file=t+".ini") for t in tool.panels.TASKS})
            device=dict(gpu_name="RTX5090",gpu_uuid="fixture")
            def mathematics(args):
                args.out.mkdir(parents=True)
                (args.out/"evidence.json").write_text(json.dumps(dict(candidates={args.name:{t:t for t in tool.panels.TASKS}})))
                (args.out/"REPORT.json").write_text(json.dumps(dict(status="passed",hardware=device)))
                return dict(hardware=device)
            def prepare(args):
                args.out.mkdir();(args.out/"plan.json").write_text("{}\n");return probe()[0]
            def execute(args):
                folder=args.plan.parent/"execution";folder.mkdir()
                (folder/"result.json").write_text(json.dumps(probe()[2]))
            def report(root,plan,out,deadline):
                (out/"review").mkdir(parents=True)
                analysis=probe()[1];analysis["hardware"]=device
                analysis["plan_sha256"]=tool.sha(plan)
                analysis["result_sha256"]=tool.sha(plan.parent/"execution/result.json")
                (out/"review/analysis.json").write_text(json.dumps(analysis))
                (out/"REPORT.json").write_text(json.dumps(dict(status="ok",hardware=device,
                    plan_sha256=analysis["plan_sha256"],result_sha256=analysis["result_sha256"],
                    files_sha256={"review/analysis.json":tool.sha(out/"review/analysis.json")})))
                return dict(launch_monotonic_ns=10**9,end_monotonic_ns=2*10**9)
            with patch.object(tool,"inspect",return_value=value),patch.object(tool.catalog,"run",side_effect=mathematics), \
                 patch.object(tool.panels,"prepare",side_effect=prepare) as preparations, \
                 patch.object(tool.panels,"run",side_effect=execute) as executions, \
                 patch.object(tool,"report_panel",side_effect=report), \
                 patch.object(tool,"choose_budgets",return_value=dict(budgets=None,status="insufficient-time")):
                result=tool.run(SimpleNamespace(prepared=root,allow_gpu=True))
                self.assertEqual(result["status"],"stopped-insufficient-time")
                self.assertFalse(result["protein_started"])
                self.assertEqual(preparations.call_count,1);self.assertEqual(executions.call_count,1)
                self.assertTrue((root/"execution/RESULT.json").is_file())
                with self.assertRaises(FileExistsError):tool.run(SimpleNamespace(prepared=root,allow_gpu=True))

    def test_report_refuses_changed_result_and_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/"panel/plan.json";plan.parent.mkdir()
            plan.write_text("{}\n");(plan.parent/"execution").mkdir()
            result=plan.parent/"execution/result.json";result.write_text('{"status":"ok"}\n')
            out=root/"report";(out/"review").mkdir(parents=True)
            device=dict(gpu_uuid="fixture")
            analysis=dict(hardware=device,plan_sha256=tool.sha(plan),result_sha256=tool.sha(result))
            (out/"review/analysis.json").write_text(json.dumps(analysis))
            summary=dict(status="ok",**analysis,files_sha256={"review/analysis.json":tool.sha(out/"review/analysis.json")})
            (out/"REPORT.json").write_text(json.dumps(summary))
            self.assertEqual(tool.read_report(plan,out,device)[1]["status"],"ok")
            with self.assertRaises(ValueError):tool.read_report(plan,out,dict(gpu_uuid="other"))
            result.write_text('{"status":"failed"}\n')
            with self.assertRaises(ValueError):tool.read_report(plan,out,device)

    def test_numerical_failure_is_preserved_and_no_training_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"CAMPAIGN.json").write_text("{}\n")
            value=dict(models=[dict(name=n,config=n+".ini") for n in tool.MODELS])
            with patch.object(tool,"inspect",return_value=value), \
                 patch.object(tool.catalog,"run",side_effect=ValueError("numerical fixture failure")), \
                 patch.object(tool.panels,"prepare") as preparation:
                with self.assertRaisesRegex(ValueError,"numerical fixture failure"):
                    tool.run(SimpleNamespace(prepared=root,allow_gpu=True))
                preparation.assert_not_called()
                result=json.loads((root/"execution/RESULT.json").read_text())
                self.assertEqual(result["status"],"failed")
                self.assertFalse(result["protein_started"])
                self.assertIn("numerical fixture failure",result["error"])
                self.assertGreaterEqual(result["overall_seconds"],0)

    def test_complete_panel_freezes_anchors_only_after_learning_eligibility(self):
        for eligible in (False,True):
            with self.subTest(eligible=eligible),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/"CAMPAIGN.json").write_text("{}\n")
                (root/"seed-roles.json").write_text(json.dumps(tool.roles.DEFAULT))
                value=dict(models=[dict(name=n,config=n+".ini") for n in tool.MODELS],
                    learners={t:dict(file=t+".ini") for t in tool.panels.TASKS})
                device=dict(gpu_uuid="fixture")
                def mathematics(args):
                    args.out.mkdir(parents=True)
                    (args.out/"evidence.json").write_text(json.dumps(dict(candidates={args.name:{t:t for t in tool.panels.TASKS}})))
                    (args.out/"REPORT.json").write_text("{}\n")
                    return dict(hardware=device)
                def prepare(args):
                    args.out.mkdir();(args.out/"plan.json").write_text("{}\n");return probe()[0]
                def report(root,plan,out,deadline):
                    (out/"review").mkdir(parents=True)
                    (out/"review/analysis.json").write_text("{}\n")
                    (out/"REPORT.json").write_text("{}\n")
                    return dict(launch_monotonic_ns=10**9,end_monotonic_ns=2*10**9)
                budgets={t:dict(steps=tool.MIN_STEPS,checkpoint_steps=tool.MIN_STEPS//8) for t in tool.panels.TASKS}
                gate=dict(status="eligible-development-search" if eligible else "blocked-learning-calibration")
                with patch.object(tool,"inspect",return_value=value), \
                     patch.object(tool.catalog,"run",side_effect=mathematics), \
                     patch.object(tool.panels,"prepare",side_effect=prepare) as preparation, \
                     patch.object(tool.panels,"run") as execution, \
                     patch.object(tool,"report_panel",side_effect=report), \
                     patch.object(tool,"read_report",return_value=(probe()[1],probe()[2])), \
                     patch.object(tool,"choose_budgets",return_value=dict(budgets=budgets)), \
                     patch.object(__import__("image_learning_gate"),"qualify",return_value=gate), \
                     patch.object(__import__("image_curve_feedback"),"freeze") as freeze:
                    result=tool.run(SimpleNamespace(prepared=root,allow_gpu=True))
                    self.assertEqual(preparation.call_count,2);self.assertEqual(execution.call_count,2)
                    self.assertEqual(freeze.call_count,int(eligible))
                    self.assertEqual(result["status"],"calibration-complete" if eligible else "stopped-unlearned-games")
                    self.assertFalse(result["protein_started"])
                    self.assertEqual(json.loads((root/"execution/learning-eligibility.json").read_text()),gate)

    def test_handoff_omits_large_arrays_weights_binaries_and_checksums_every_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"campaign";root.mkdir();(root/"CAMPAIGN.json").write_text("{}\n")
            (root/"weights.bin").write_bytes(b"fake");(root/"grad.npy").write_bytes(b"fake")
            (root/"native.so").write_bytes(b"fake");(root/"REPORT.md").write_text("fixture report\n")
            result=tool.handoff(root,Path(temp)/"archives")
            self.assertEqual(tool.sha(result["archive"]),result["sha256"])
            with tarfile.open(result["archive"]) as archive:
                names=archive.getnames()
                self.assertNotIn("campaign/weights.bin",names);self.assertNotIn("campaign/grad.npy",names)
                self.assertNotIn("campaign/native.so",names)
                declaration=json.load(archive.extractfile("campaign/ARCHIVE.json"))
                self.assertFalse(declaration["self_contained_raw_audit"])
                self.assertEqual(declaration["files_sha256"]["REPORT.md"],tool.sha(root/"REPORT.md"))


if __name__=="__main__":unittest.main()
