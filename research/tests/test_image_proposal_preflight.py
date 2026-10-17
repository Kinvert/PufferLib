"""Admission order/failure tests; scalar fixtures only, no GPU/model/reference."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_proposal_preflight as gate
import cross_game_feedback as feedback


class ImageProposalTests(unittest.TestCase):
    def recipe(self):
        config,seeds=feedback.recipe(feedback.ROOT/"research/recipes/image_feedback_smoke.ini")
        self.assertEqual(seeds,[7])
        return config

    def test_all_games_registered_before_pass_no_cuda_query(self):
        config=self.recipe(); contracts={t:{"shape_hwc":[616,616,3]} for t in gate.panels.TASKS}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); candidate=root/"candidate.ini"; candidate.write_text("[policy]\nencoder=4\n")
            registry=dict(schema=gate.panels.policy_layout.IMAGE_SCHEMA,image_contracts=contracts)
            calls=[]
            def describe(metadata,path,task,family,out,timeout):
                calls.append((task,"layout")); return dict(parameters=123)
            def storage(metadata,path,task,family,out,timeout,parameters):
                self.assertEqual(parameters,123); calls.append((task,"storage"))
                return dict(storage_floor_bytes=2**30)
            with patch.object(gate.panels.policy_layout,"inspect_registry",return_value=registry), \
                 patch.object(gate,"configuration",return_value=config), \
                 patch.object(gate.panels.policy_layout,"describe",side_effect=describe), \
                 patch.object(gate.panels.policy_layout,"storage",side_effect=storage), \
                 patch.object(gate.panels,"hardware",side_effect=AssertionError("No GPU query")), \
                 patch.object(gate.time,"monotonic",return_value=100):
                record=gate.run(root,dict(image_contracts=contracts),config,candidate,root/"preflight",200)
            self.assertEqual(record["status"],"passed")
            self.assertEqual(calls,[(t,k) for t in gate.panels.TASKS for k in ("layout","storage")])
            self.assertFalse(record["peak_memory_certified"])

    def test_oversized_graph_preserves_failed_admission_and_stops(self):
        config=self.recipe(); contracts={t:{"shape_hwc":[616,616,3]} for t in gate.panels.TASKS}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); candidate=root/"candidate.ini"; candidate.write_text("[policy]\nencoder=4\n")
            registry=dict(schema=gate.panels.policy_layout.IMAGE_SCHEMA,image_contracts=contracts)
            with patch.object(gate.panels.policy_layout,"inspect_registry",return_value=registry), \
                 patch.object(gate,"configuration",return_value=config), \
                 patch.object(gate.panels.policy_layout,"describe",return_value=dict(parameters=123)), \
                 patch.object(gate.panels.policy_layout,"storage",return_value=dict(storage_floor_bytes=4*2**30)), \
                 patch.object(gate.time,"monotonic",return_value=100):
                with self.assertRaisesRegex(ValueError,"storage floor"):
                    gate.run(root,dict(image_contracts=contracts),config,candidate,root/"preflight",200)
            import json
            record=json.loads((root/"preflight/REPORT.json").read_text())
            self.assertEqual(record["status"],"failed"); self.assertEqual(len(record["cases"]),1)

    def test_expired_campaign_does_not_register_or_allocate(self):
        config=self.recipe(); contracts={t:{} for t in gate.panels.TASKS}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); candidate=root/"candidate.ini"; candidate.write_text("[policy]\nencoder=4\n")
            registry=dict(schema=gate.panels.policy_layout.IMAGE_SCHEMA,image_contracts=contracts)
            with patch.object(gate.panels.policy_layout,"inspect_registry",return_value=registry), \
                 patch.object(gate,"configuration",return_value=config), \
                 patch.object(gate.panels.policy_layout,"describe",side_effect=AssertionError("Do not register")), \
                 patch.object(gate.time,"monotonic",return_value=200):
                with self.assertRaisesRegex(ValueError,"Campaign deadline"):
                    gate.run(root,dict(image_contracts=contracts),config,candidate,root/"preflight",200)

    def test_missing_native_scale_fails_before_optimizer_or_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"recipe.ini"
            path.write_text((feedback.ROOT/"research/recipes/image_feedback_smoke.ini").read_text().replace("scale = auto\n",""))
            with self.assertRaisesRegex(ValueError,"requires a positive scale"):
                feedback.recipe(path)


if __name__=="__main__": unittest.main()
