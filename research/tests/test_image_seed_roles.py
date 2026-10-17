"""Scalar seed-assignment checks only, no environment or neural execution."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_seed_roles as tool


class SeedRoleTests(unittest.TestCase):
    def test_fixed_lists_and_both_roles_are_valid(self):
        value=copy.deepcopy(tool.DEFAULT); tool.validate(value)
        for role in ("development","confirmation"):
            tool.check_panel(value,role,value[role+"_training"],value[role+"_evaluation"],17)

    def test_confirmation_can_be_evaluated_but_never_observed_by_search(self):
        value=copy.deepcopy(tool.DEFAULT)
        tool.check_search(value,"development",[7,42,69],[7007,7008,7009],17,83)
        with self.assertRaisesRegex(ValueError,"never feed"):
            tool.check_search(value,"confirmation",value["confirmation_training"],value["confirmation_evaluation"],17,83)

    def test_overlap_lucky_subsets_reordering_and_seed_sweep_are_refused(self):
        value=copy.deepcopy(tool.DEFAULT)
        for before,after in (([7,42,69],[7,42]),([7,42,69],[69,42,7])):
            with self.assertRaises(ValueError): tool.check_panel(value,"development",after,[7007,7008,7009],17)
        changed=copy.deepcopy(value);changed["confirmation_evaluation"][0]=7
        with self.assertRaises(ValueError): tool.validate(changed)
        changed=copy.deepcopy(value);changed["seed_sweep_allowed"]=True
        with self.assertRaises(ValueError): tool.validate(changed)
        changed=copy.deepcopy(value);changed["development_training"][0]=True
        with self.assertRaises(ValueError): tool.validate(changed)


if __name__=="__main__": unittest.main()
