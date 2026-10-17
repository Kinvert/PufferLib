"""Scalar clock/control-flow fixtures; no neural computation or GPU use."""
import copy
from pathlib import Path
import sys
import json
import tempfile
import configparser
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_campaign_budget as tool


def process(start, end):
    return dict(status="ok", returncode=0, launch_monotonic_ns=start * 10**9,
                end_monotonic_ns=end * 10**9)


class CampaignBudgetTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(status="ok", hardware={"gpu_name": "RTX 5090", "gpu_uuid": "paired"},
            trials=[dict(index=i, optimizer_process=process(t, t+1),
                         feedback=dict(score=0.2, cost=20)) for i, t in enumerate((10, 110))],
            final_observations_replayed=2, final_optimizer_process=process(230, 232))
        self.audit = dict(status="ok", trials=[{}, {}], result_sha256="a"*64)
        self.value = tool.calibration(self.result, self.audit,
            "a"*64, "b"*64, "c"*64, [process(240, 250)])

    def admit(self, **kwargs):
        defaults = dict(hardware=self.result["hardware"], conditions_sha256="b"*64,
            envelope_sha256="c"*64, started_ns=10**9, now_ns=10**9)
        defaults.update(kwargs)
        return tool.admit(self.value, **defaults)

    def test_whole_trial_and_post_replay_reporting_are_charged(self):
        self.assertEqual(self.value["measured_panel_seconds"], [100, 120])
        self.assertEqual(self.value["next_panel_seconds"], 240)
        self.assertEqual(self.value["post_panel_seconds"], 24)
        self.assertEqual(self.admit(budget_seconds=264)["status"], "admit-complete-panel")
        receipt = self.admit(budget_seconds=263)
        self.assertEqual(receipt["status"], "budget_stop")
        self.assertFalse(receipt["partial_feedback_allowed"])

    def test_elapsed_time_and_slower_observations_stop_before_proposal(self):
        self.assertEqual(self.admit(budget_seconds=264, now_ns=2*10**9)["status"], "budget_stop")
        self.assertEqual(self.admit(budget_seconds=264, observed_panel_seconds=[121])["status"], "budget_stop")
        self.assertEqual(self.admit(budget_seconds=264, observed_panel_seconds=[1])["next_panel_seconds"], 240)

    def test_hardware_input_envelope_and_tampered_reserves_are_refused(self):
        for kwargs in (dict(hardware={"gpu_name": "RTX 5060"}), dict(conditions_sha256="d"*64),
                       dict(envelope_sha256="d"*64), dict(now_ns=0), dict(budget_seconds=43201),
                       dict(observed_panel_seconds=[float("nan")])):
            with self.assertRaises(ValueError): self.admit(**kwargs)
        self.value["next_panel_seconds"] = 1
        with self.assertRaises(ValueError): self.admit()

    def test_partial_failed_or_unmeasured_pilots_never_calibrate(self):
        for field, value in (("status", "failed"), ("final_observations_replayed", 1)):
            changed = copy.deepcopy(self.result); changed[field] = value
            with self.assertRaises(ValueError):
                tool.calibration(changed, self.audit, "a"*64, "b"*64, "c"*64, [process(1, 2)])
        with self.assertRaises(ValueError):
            tool.calibration(self.result, self.audit, "a"*64, "b"*64, "c"*64, [])
        failed = process(1, 2); failed["returncode"] = -9
        with self.assertRaises(ValueError): tool.seconds(failed)

    def test_reporting_growth_and_slower_optimizer_increase_the_admission_reserve(self):
        receipt=self.admit(budget_seconds=264,observed_panel_seconds=[100,100,100])
        self.assertEqual(receipt["status"],"budget_stop")
        self.assertEqual(receipt["reporting_observations"],4)
        self.assertEqual(receipt["post_panel_seconds"],44)
        slower=self.admit(budget_seconds=264,observed_optimizer_seconds=[5])
        self.assertEqual(slower["post_panel_seconds"],30)
        self.assertEqual(slower["status"],"budget_stop")

    def test_pilot_proof_binds_full_report_and_refuses_audit_only_measurement(self):
        import cross_game_feedback as feedback
        from claim import sha
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"execution").mkdir(); report=root/"report"; (report/"review").mkdir(parents=True)
            plan=root/"plan.json";plan.write_text("{}\n")
            recipe=configparser.ConfigParser();recipe["policy"]={"encoder":"4"}
            (root/"nature-reference.json").write_text(json.dumps(dict(conditions={"fixed":"conditions"})))
            fixed,envelope=tool.identities(recipe,dict(conditions={"fixed":"conditions"}))
            result=copy.deepcopy(self.result)
            result.update(mode="curve-pilot5090",plan_sha256=sha(plan))
            result_path=root/"execution/result.json";result_path.write_text(json.dumps(result))
            audit=copy.deepcopy(self.audit);audit["result_sha256"]=sha(result_path)
            audit_path=report/"review/analysis.json";audit_path.write_text(json.dumps(audit))
            timing=process(240,250)
            timing.update(command=[sys.executable,str(feedback.ROOT/"research/report_image_feedback.py"),
                "--plan",str(plan),"--out",str(report)],cwd=str(feedback.ROOT))
            timing_path=root/"report.txt.json";timing_path.write_text(json.dumps(timing))
            (report/"training.csv").write_text("observed,SPS\n1,123\n")
            summary=dict(schema="audited-image-feedback-report-v1",status="ok",plan_sha256=sha(plan),
                result_sha256=sha(result_path),policy_executed=False,gpu_queried=False,
                files_sha256={"training.csv":sha(report/"training.csv")})
            (report/"REPORT.json").write_text(json.dumps(summary))
            frozen=tool.calibration(result,audit,sha(result_path),fixed,envelope,[timing])
            frozen["pilot_proof"]={key:str(path) for key,path in
                (("plan",plan),("result",result_path),("audit",audit_path),("report_process",timing_path))}
            frozen["pilot_proof"].update({key+"_sha256":sha(path) for key,path in
                (("plan",plan),("result",result_path),("audit",audit_path),("report_process",timing_path))})
            with patch.object(feedback,"inspect",return_value=dict(protocol=feedback.CURVE_PROTOCOL)), \
                 patch.object(feedback,"recipe",return_value=(recipe,[7,42,69])):
                tool.verify_pilot(frozen)
                (report/"training.csv").write_text("observed,SPS\n1,99999\n")
                with self.assertRaisesRegex(ValueError,"report exports"): tool.verify_pilot(frozen)
                timing["command"]=[sys.executable,str(feedback.ROOT/"research/cross_game_feedback.py"),
                    "audit","--plan",str(plan),"--out",str(report)]
                timing_path.write_text(json.dumps(timing))
                frozen["pilot_proof"]["report_process_sha256"]=sha(timing_path)
                with self.assertRaisesRegex(ValueError,"reporting command"): tool.verify_pilot(frozen)


if __name__ == "__main__": unittest.main()
