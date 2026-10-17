"""Scalar role/condition rejection fixtures; no policy or GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_curve_bindings as tool
import image_seed_roles as roles
import research.tests.test_image_learning_gate as learning_fixtures
import image_learning_gate as learning


class BindingTests(unittest.TestCase):
    def recipe(self):
        config=tool.panels.panel.read()
        config["search"]=dict(training_seeds="7,42,69",evaluation_seeds="7007,7008,7009",eval_seed="7007",
            seed_role="development",appearance_seed="17",episodes="33",slots="4",pong_max_decisions="512",breakout_max_frames="2048")
        config["protein"]=dict(seed="83")
        return config

    def test_confirmation_reordering_subsets_or_optimizer_seed_changes_are_refused(self):
        recipe=self.recipe(); tool.check_roles(recipe,roles.DEFAULT)
        for key,value in (("training_seeds","69,42,7"),("evaluation_seeds","7007,7009,7008"),
                          ("seed_role","confirmation")):
            bad=copy.deepcopy(recipe);bad["search"][key]=value
            with self.assertRaises(ValueError): tool.check_roles(bad,roles.DEFAULT)
        bad=copy.deepcopy(recipe);bad["protein"]["seed"]="84"
        with self.assertRaises(ValueError): tool.check_roles(bad,roles.DEFAULT)

    def test_fixed_conditions_are_checked_before_any_gpu_admission(self):
        recipe=self.recipe()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); learners=[]; bases={}; hashes={}
            for task in tool.panels.TASKS:
                path=root/(task+".ini");path.write_text("[train]\nlearning_rate=.001\n")
                learners.append(dict(environment=task,file=path.name));hashes[task]=tool.sha(path)
                config=tool.panels.panel.read(tool.panels.ROOT/"config/default.ini")
                config["base"]["env_name"]=task;config["policy"].update(hidden_size="128",num_layers="1")
                bases[task]=config
            value=dict(image_contracts={t:dict(shape_hwc=[640,480,3]) for t in tool.panels.TASKS},
                task_budgets={t:dict(steps=512,checkpoint_steps=64) for t in tool.panels.TASKS},
                learner_recipes=learners,source_sha256={"src/ocean.cu":"a"*64})
            fixed=dict(image_contracts=value["image_contracts"],task_budgets=value["task_budgets"],episodes=33,slots=4,
                caps=dict(pong_max_decisions=512,breakout_max_frames=2048),game_settings={t:tool.common_settings(c) for t,c in bases.items()},
                learner_sha256=hashes,native_sources=value["source_sha256"])
            with patch.object(tool,"inspect",return_value=dict(conditions=fixed)):
                tool.feedback_conditions(root,value,recipe,bases)
                for field,changed in (("image_contracts",{}),("task_budgets",{}),("source_sha256",{"src/ocean.cu":"b"*64})):
                    bad=copy.deepcopy(value);bad[field]=changed
                    with self.assertRaises(ValueError): tool.feedback_conditions(root,bad,recipe,bases)
                bases["pongcnn"]["train"]["learning_rate"]="0.9"
                with self.assertRaisesRegex(ValueError,"learners/world/core"):
                    tool.feedback_conditions(root,value,recipe,bases)

    def test_learning_gate_is_reconstructed_even_if_its_modified_receipt_is_rehashed(self):
        fixture=learning_fixtures.LearningGateTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root;plan=fixture.plan;analysis=fixture.analysis
        plan["evaluation_seeds"]=[7007,7008,7009]
        recipe=self.recipe()
        for task in tool.panels.TASKS: recipe["objective."+task]={"offset":"0","target":"1"}
        def write(path,value): path.write_text(json.dumps(value,sort_keys=True)+"\n")
        write(root/"nature-calibration-plan.json",plan)
        analysis["plan_sha256"]=tool.sha(root/"nature-calibration-plan.json")
        (root/"nature-replay").mkdir();write(root/"nature-replay/analysis.json",analysis)
        write(root/"seed-roles.json",roles.DEFAULT)
        gate=learning.qualify(plan,analysis,fixture.normalization);write(root/"learning-eligibility.json",gate)
        tool.curves.freeze(plan,analysis,root,fixture.normalization,root/"nature-reference.json")
        binding=dict(nature_plan_sha256=analysis["plan_sha256"],
            calibration_plan_sha256=tool.sha(root/"nature-calibration-plan.json"),
            replay_sha256=tool.sha(root/"nature-replay/analysis.json"),
            learning_eligibility_sha256=tool.sha(root/"learning-eligibility.json"),
            nature_reference_sha256=tool.sha(root/"nature-reference.json"),seed_roles_sha256=tool.sha(root/"seed-roles.json"))
        write(root/"curve-bindings.json",binding)
        tool.inspect(root,recipe)
        gate["controls"][0]["mean"]=.99
        write(root/"learning-eligibility.json",gate)
        binding["learning_eligibility_sha256"]=tool.sha(root/"learning-eligibility.json")
        write(root/"curve-bindings.json",binding)
        with self.assertRaisesRegex(ValueError,"captured native calibration"): tool.inspect(root,recipe)


if __name__=="__main__":unittest.main()
