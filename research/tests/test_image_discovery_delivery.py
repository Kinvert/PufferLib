"""Discovery recipe/history/launch boundaries; no GPU or neural computation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_image_discovery as delivery
import image_initial_history as inherited
from claim import sha


class DiscoveryDeliveryTests(unittest.TestCase):
    def plan(self):
        return dict(seeds=[7,42,69],evaluation_seeds=[7007,7008,7009],appearance_seed=17,
            eval_seed=7007,episodes=5,slots=16,caps=dict(pong_max_decisions=8192,breakout_max_frames=16384),
            task_budgets={task:dict(steps=2097152,checkpoint_steps=262144) for task in delivery.panels.TASKS})

    def recipe(self,phase):
        return delivery.fixed_recipe(delivery.TEMPLATE,dict(normalization=delivery.calibration.NORMALIZATION),self.plan(),phase)

    def test_actual_calibration_conditions_replace_template_examples_and_only_count_deadline_change(self):
        pilot=self.recipe("pilot");sweep=self.recipe("sweep")
        inherited.same_recipe(pilot,sweep)
        self.assertEqual(pilot.getint("search","max_runs"),3)
        self.assertEqual(sweep.getint("search","max_runs"),128)
        self.assertEqual(sweep.getint("search","campaign_timeout"),43200)
        self.assertEqual(delivery.feedback.task_budgets(sweep),self.plan()["task_budgets"])
        for task in delivery.panels.TASKS:
            self.assertEqual(sweep.getint("task."+task,"checkpoint_steps")*8,sweep.getint("task."+task,"steps"))
        sweep["policy"]["cnn_stride_1"]="8"
        with self.assertRaisesRegex(ValueError,"envelope"):inherited.same_recipe(pilot,sweep)

    def test_tiny_budgets_and_changed_training_eval_or_optimizer_seeds_are_refused(self):
        for change in (dict(seeds=[7,42,70]),dict(evaluation_seeds=[7007,7008,7010]),dict(appearance_seed=18)):
            plan=self.plan();plan.update(change)
            with self.assertRaises(ValueError):delivery.fixed_recipe(delivery.TEMPLATE,
                dict(normalization=delivery.calibration.NORMALIZATION),plan,"pilot")
        plan=self.plan();plan["task_budgets"]["flappycnn"]=dict(steps=512,checkpoint_steps=64)
        with self.assertRaisesRegex(ValueError,"tiny smoke"):delivery.fixed_recipe(delivery.TEMPLATE,
            dict(normalization=delivery.calibration.NORMALIZATION),plan,"pilot")
        plan=self.plan();plan["task_budgets"].pop("mazecnn")
        with self.assertRaisesRegex(ValueError,"every calibrated game"):delivery.fixed_recipe(delivery.TEMPLATE,
            dict(normalization=delivery.calibration.NORMALIZATION),plan,"pilot")

    def test_history_import_requires_identical_actual_nature_reference_not_just_same_ini(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"new";parent=Path(temp)/"pilot";root.mkdir();parent.mkdir()
            data=dict(conditions=dict(pixels="RGB",learner_sha256="same"),nature=dict(measured_seconds=10))
            for path in (root,parent):(path/"nature-reference.json").write_text(json.dumps(data))
            inherited.same_reference(root,parent)
            data["nature"]["measured_seconds"]=11
            (root/"nature-reference.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"Nature anchors"):inherited.same_reference(root,parent)

    def test_gpu_permission_and_completed_allocation_guards_precede_native_execution(self):
        with patch.object(delivery,"inspect") as check:
            with self.assertRaisesRegex(ValueError,"allow-gpu"):delivery.run(SimpleNamespace(allow_gpu=False))
            check.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"RUN.json").write_text('{"retained":true}\n')
            with patch.object(delivery,"inspect",return_value=dict(phase="pilot")),patch.object(delivery.feedback,"run") as run:
                with self.assertRaisesRegex(ValueError,"restart"):delivery.run(SimpleNamespace(allow_gpu=True,delivery=root))
                run.assert_not_called()
                self.assertEqual(json.loads((root/"RUN.json").read_text()),dict(retained=True))

    def test_missing_or_failed_learning_calibration_cannot_prepare_a_native_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.object(delivery.calibration,"inspect",return_value={}),patch.object(delivery,"process") as native:
                with self.assertRaisesRegex(ValueError,"calibration first"):delivery.calibrated_inputs(root)
                native.assert_not_called()

    def test_pilot_ledger_preserves_all_observations_and_refuses_failures_or_reordering(self):
        result=dict(status="ok",mode="curve-pilot5090",final_observations_replayed=3,
            trials=[dict(index=i,proposal=dict(observations_replayed=i,normalized=[.25,.75]),
                feedback=dict(score=.1*i,cost=12.)) for i in range(3)])
        history="PUFFER_CROSS_GAME_V1 2\n"
        for entry in result["trials"]:
            history+=" ".join(["0",format(entry["feedback"]["score"],".17g"),"12","0.25","0.75"])+"\n"
        result["history"]=history
        self.assertEqual(inherited.ledger(result,2),history)
        for field,number in (("status","failed"),("final_observations_replayed",2),("history","changed")):
            changed=copy.deepcopy(result);changed[field]=number
            with self.assertRaises(ValueError):inherited.ledger(changed,2)
        result["trials"][1]["index"]=0
        with self.assertRaises(ValueError):inherited.ledger(result,2)

    def test_pilot_launch_must_finish_successfully_within_its_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"feedback/execution").mkdir(parents=True)
            result=root/"feedback/execution/result.json";result.write_text('{"status":"ok"}\n')
            with self.assertRaisesRegex(ValueError,"launcher must finish"):delivery.completed_pilot(root)
            receipt=dict(phase="pilot",status="ok",native_status="ok",observations=3,
                final_observations_replayed=3,inherited_observations=0,native_result_sha256=sha(result),
                started_monotonic_ns=1000000000,ended_monotonic_ns=3000000000,
                overall_seconds=2.,declared_window_exceeded=False)
            def write(value):(root/"RUN.json").write_text(json.dumps(value))
            write(receipt)
            self.assertEqual(delivery.completed_pilot(root),dict(delivery=str(root),run_sha256=sha(root/"RUN.json")))
            for changes in (dict(status="running"),dict(status="failed"),dict(observations=2),
                    dict(native_result_sha256="altered"),dict(overall_seconds=3.),
                    dict(ended_monotonic_ns=14402000000000,overall_seconds=14401.,declared_window_exceeded=True)):
                value=dict(receipt);value.update(changes);write(value)
                with self.assertRaises(ValueError):delivery.completed_pilot(root)

    def test_report_failure_retains_native_success_but_fails_delivery_and_refuses_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            result=dict(status="ok",trials=[{}]*3,final_observations_replayed=3)
            def native_run(args):
                (root/"feedback/execution").mkdir(parents=True)
                (root/"feedback/execution/result.json").write_text(json.dumps(result))
                return result
            with patch.object(delivery,"inspect",return_value=dict(phase="pilot")),\
                    patch.object(delivery.feedback,"run",side_effect=native_run) as native,\
                    patch.object(delivery,"process",side_effect=RuntimeError("report failure")):
                args=SimpleNamespace(allow_gpu=True,delivery=root)
                with self.assertRaisesRegex(RuntimeError,"report failure"):delivery.run(args)
                receipt=json.loads((root/"RUN.json").read_text())
                self.assertEqual(receipt["status"],"failed")
                self.assertEqual(receipt["native_status"],"ok")
                self.assertEqual(receipt["observations"],3)
                with self.assertRaisesRegex(ValueError,"successful complete"):delivery.completed_pilot(root)
                with self.assertRaisesRegex(ValueError,"restart"):delivery.run(args)
                self.assertEqual(native.call_count,1)


if __name__=="__main__":unittest.main()
