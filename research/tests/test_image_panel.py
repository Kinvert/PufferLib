"""Image panel contract/geometry checks; no model execution."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_panel as panels
import prepare_image_inputs as images


def registry():
    contracts={task:dict(version="human-geometry-v1",shape_hwc=[*images.SHAPES[task],3],
                        layout="NHWC",dtype="float32",preprocessing="none") for task in panels.TASKS}
    return dict(protocol=panels.IMAGE_BUILD,image_contracts=contracts,
                binaries={task:{"default":dict(info=dict(image_contract=contract))} for task,contract in contracts.items()})


class ImagePanelTests(unittest.TestCase):
    def test_image_builds_need_every_game_and_match_inputs_across_families(self):
        value=registry(); self.assertEqual(set(panels.registry_builds(value)),set(panels.TASKS))
        changed=copy.deepcopy(value); changed["binaries"].pop("pongcnn")
        with self.assertRaises(ValueError): panels.registry_builds(changed)
        changed=copy.deepcopy(value); changed["binaries"]["flappycnn"]["impala"]=copy.deepcopy(changed["binaries"]["flappycnn"]["default"])
        changed["binaries"]["flappycnn"]["impala"]["info"]["image_contract"]["shape_hwc"][2]=1
        with self.assertRaises(ValueError): panels.registry_builds(changed)

    def test_nature_reuses_default_but_residual_baselines_need_their_builds(self):
        value=registry()
        self.assertTrue(panels.matching_registry(value,["nature_cnn"],True))
        self.assertFalse(panels.matching_registry(value,["impala_cnn"],True))
        self.assertFalse(panels.matching_registry(value,[],False))

    def test_legacy_build_matching_remains_separate(self):
        value=dict(protocol="native-cnn-candidate-build-v1")
        self.assertTrue(panels.matching_registry(value,[],False))
        self.assertTrue(panels.matching_registry(value,["nature_cnn"],True))
        self.assertFalse(panels.matching_registry(value,["impoola_cnn"],True))

    def test_fixture_learner_overlays_cannot_change_architecture_world_or_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"inputs"; value=images.prepare(root)
            self.assertEqual(value["schema"],"realistic-image-validation-inputs-v2")
            self.assertEqual(set(value["learner_recipes"]),set(panels.TASKS))
            for task,record in value["learner_recipes"].items():
                overlay=panels.panel.learner_recipe(root/record["path"],panels.panel.read(panels.ROOT/"config/default.ini"))
                self.assertEqual(set(overlay.sections()),{"base","train","vec"})
                self.assertEqual(overlay.getint("train","minibatch_size"),16)
                self.assertEqual(overlay.getint("vec","total_agents"),4)
                self.assertNotIn("seed",overlay["base"])

    def test_local_image_mode_allows_eight_checkpoints_but_not_long_training(self):
        value=dict(protocol=panels.IMAGE_PROTOCOL,candidates=[dict(architecture=dict(encoder=4))],
                   seeds=[7],task_budgets={t:dict(steps=512) for t in panels.TASKS},episodes=5,slots=4,
                   planned_evaluations=328,jobs=[dict(checkpoint_steps=list(range(64,513,64)),policy_layout=dict(bytes=239904),
                                                    image_storage=dict(storage_floor_bytes=2**30))],image_encoder_checks={})
        self.assertEqual(panels.run_hardware(value,"image5060"),"5060")
        for field,data in (("slots",16),("protocol",panels.LEARNER_PROTOCOL),("planned_evaluations",657)):
            with self.assertRaises(ValueError): panels.run_hardware({**value,field:data},"image5060")
        value["task_budgets"]["pongcnn"]["steps"]=2048
        with self.assertRaises(ValueError): panels.run_hardware(value,"image5060")


if __name__=="__main__": unittest.main()
