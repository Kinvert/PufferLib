"""VRAM/recipe/receipt scalar fixtures, with no actual GPU queries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cross_game_feedback as feedback
import image_memory_budget as memory
from claim import sha


class MemoryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config,_=feedback.recipe(feedback.ROOT/"research/recipes/image_cnn_discovery_5090.ini")
        self.hardware=dict(gpu_name="NVIDIA GeForce RTX 5090",gpu_uuid="GPU-fixture")
        self.text="NVIDIA GeForce RTX 5090, GPU-fixture, 32607, 32000\n"

    def test_legacy_limits_remain_and_only_explicit_policy_allows_24gib(self):
        config,_=feedback.recipe(feedback.ROOT/"research/recipes/image_feedback_smoke.ini")
        self.assertEqual(memory.budget(config),(memory.LEGACY,4096))
        config["search"]["image_storage_limit_mib"]="24576"
        with self.assertRaises(ValueError): memory.budget(config)
        self.assertEqual(memory.budget(self.config),(memory.POLICY,24576))
        self.config.remove_option("search","image_storage_limit_mib")
        with self.assertRaisesRegex(ValueError,"explicit"): memory.budget(self.config)

    def test_receipt_requires_correct_single_gpu_and_free_headroom(self):
        result=memory.measurement(self.text,self.hardware,self.config)
        self.assertEqual(result["required_headroom_mib"],4096)
        self.assertFalse(result["peak_memory_certified"])
        for text in (self.text.replace("GPU-fixture","GPU-wrong"),self.text*2,
                     self.text.replace("32000","28000"),self.text.replace("32000","40000"),
                     self.text.replace("32000","N/A")):
            with self.assertRaises(ValueError): memory.measurement(text,self.hardware,self.config)
        bad=copy.deepcopy(self.hardware);bad["gpu_name"]="RTX 5060"
        with self.assertRaises(ValueError): memory.measurement(self.text,bad,self.config)

    def test_all_18_coordinates_are_cnn_only_and_first_stride_includes_one_two_four_eight(self):
        sections=[s for s in self.config.sections() if s.startswith("sweep.")]
        self.assertEqual(len(sections),18)
        self.assertTrue(all(s.startswith("sweep.policy.cnn_") for s in sections))
        self.assertEqual(self.config["sweep.policy.cnn_stride_1"]["distribution"],"uniform_pow2")
        self.assertEqual([self.config.getint("sweep.policy.cnn_stride_1",k) for k in ("min","max")],[1,8])
        self.assertEqual(self.config.get("search","training_seeds"),"7,42,69")
        self.assertEqual(self.config.get("search","evaluation_seeds"),"7007,7008,7009")

    def test_query_artifact_inspection_refuses_tampering_and_wrong_chronology(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/"query.txt").write_text(self.text)
            command=["/usr/bin/nvidia-smi","--query-gpu=name,uuid,memory.total,memory.free","--format=csv,noheader,nounits"]
            timing=dict(status="ok",returncode=0,command=command,cwd="/fixture",
                launch_monotonic_ns=20,end_monotonic_ns=30)
            (out/"query.txt.json").write_text(json.dumps(timing))
            receipt=memory.measurement(self.text,self.hardware,self.config)
            receipt.update(command=command,repository="/fixture",timing=timing,
                query_sha256=sha(out/"query.txt"),process_sha256=sha(out/"query.txt.json"))
            (out/"receipt.json").write_text(json.dumps(receipt))
            memory.inspect(self.config,self.hardware,out,10,40)
            with self.assertRaises(ValueError):memory.inspect(self.config,self.hardware,out,25,40)
            receipt["free_mib"]=33000
            (out/"receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"admission values"):
                memory.inspect(self.config,self.hardware,out,10,40)


if __name__=="__main__":unittest.main()
