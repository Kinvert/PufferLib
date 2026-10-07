"""Host file/configuration/supervision checks; no neural library or GPU load."""
import copy
import ctypes
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_encoder_smoke as tool


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(out=self.root / "packet",
            native=tool.ROOT / "build/connect4cnn/gpu-oracle-native-20261006.so",
            reference=tool.ROOT / "build/connect4cnn/gpu-oracle-reference-20261006.so")

    def prepare(self):
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")), \
             patch.object(tool.fixtures, "idle_gpu", side_effect=AssertionError("No GPU query")), \
             patch("subprocess.Popen", side_effect=AssertionError("No native process")):
            return tool.prepare(self.args)

    def test_real_compiled_libraries_prepare_inspect_without_load_or_gpu(self):
        self.prepare(); value = tool.load(self.args.out, current=True)
        self.assertEqual(len(value["cases"]), 18)
        self.assertEqual({c["batch"] for c in value["cases"]}, {1, 3, 64})
        self.assertEqual({c["hidden"] for c in value["cases"]}, {128})
        self.assertEqual(value["oracle_device"], "cuda"); self.assertFalse(value["cpu_neural_reference"])
        self.assertFalse((self.args.out / "execution").exists())

    def test_mutated_library_or_source_copy_fails(self):
        self.prepare()
        path = self.args.out / "libraries/reference.so"; path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Immutable"): tool.load(self.args.out)

    def test_case_tolerance_and_worker_allocations_cannot_be_relaxed(self):
        self.prepare(); path = self.args.out / "packet.json"; original = json.loads(path.read_text())
        for key, value in (("cases", original["cases"][:-1]), ("workers", ["repeat-0"]),
                           ("tolerances", {"flex_quality": [1, 1], "nature_cnn": [1, 1]}),
                           ("oracle_device", "cpu")):
            changed = copy.deepcopy(original); changed[key] = value; path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "protocol/allocation"): tool.load(self.args.out)

    def test_worker_fails_before_library_load_without_scheduled_parent(self):
        self.prepare(); execution = self.args.out / "execution"; execution.mkdir()
        tool.save(execution / "request.json", dict(supervisor_pid=-1, nonce="wrong", packet_sha256="wrong"))
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")):
            with self.assertRaisesRegex(ValueError, "not a child"): tool.worker(self.args.out, "repeat-0")

    def test_query_failure_retained_without_worker_or_library_load(self):
        self.prepare(); self.args.timeout = 30
        with patch.object(tool.fixtures, "idle_gpu", side_effect=ValueError("query rejected")), \
             patch.object(tool.fixtures.claim, "process", side_effect=AssertionError("No worker")), \
             patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")):
            with self.assertRaisesRegex(ValueError, "query rejected"): tool.run(self.args)
        self.assertTrue((self.args.out / "execution/failure.json").is_file())
        self.assertFalse((self.args.out / "execution/REPORT.json").exists())

    def test_failed_or_incomplete_execution_cannot_audit_as_success(self):
        self.prepare(); (self.args.out / "execution").mkdir()
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")):
            with self.assertRaises(FileNotFoundError): tool.audit(self.args.out)

    def test_overwrite_rejected(self):
        self.prepare()
        with self.assertRaises(FileExistsError): self.prepare()

    def test_learner_panel_prepares_all_72_fixed_cases_without_neural_load(self):
        self.args.learner_panel = True
        value = self.prepare(); tool.load(self.args.out, current=True)
        self.assertEqual(value["schema"], tool.LEARNER_SCHEMA)
        self.assertEqual(len(value["cases"]), 72)
        self.assertEqual({c["batch"] for c in value["cases"]}, {1, 3, 64, 2048})
        self.assertEqual({c["hidden"] for c in value["cases"]}, {64, 128, 256})
        self.assertEqual(len([c for c in value["cases"] if c["batch"] == 2048]), 18)
        self.assertEqual(len([c for c in value["cases"] if c["model"] == "flex_quality" and c["hidden"] == 64]), 12)
        self.assertFalse(value["learner_batch_2048_qualified"])
        self.assertFalse((self.args.out / "execution").exists())

    def test_v1_schema_does_not_gain_v2_cases_or_qualifications(self):
        value = self.prepare(); path = self.args.out / "packet.json"
        value["cases"] = tool.cases(True); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "protocol/allocation"): tool.load(self.args.out)
        value["cases"] = tool.cases(); value["learner_batch_2048_qualified"] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Preparation cannot certify"): tool.load(self.args.out)


if __name__ == "__main__": unittest.main()
