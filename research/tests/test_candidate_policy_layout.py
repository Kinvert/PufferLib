"""Native registration/config and synthetic-artifact guards; no neural tests."""
import copy
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.tests import test_candidate_panel as fixtures
import candidate_panel as tool

METADATA = Path(os.environ.get("POLICY_METADATA_DIR", tool.ROOT / "build/policy-metadata/candidate-layout-20261006"))


class CandidateLayoutTests(unittest.TestCase):
    setUp = fixtures.CandidatePanelTests.setUp
    with_baselines = fixtures.CandidatePanelTests.with_baselines

    def prepare(self, baselines=False):
        if baselines: self.with_baselines()
        self.args.policy_metadata = METADATA
        registry = json.loads(self.registry.read_text())
        registry["source_sha256"] = tool.capture(self.registry.parent)
        tool.save(self.registry, registry)
        with patch.object(tool.bridge, "adapter", side_effect=fixtures.SuiteFixture), \
             patch.object(tool, "check_starts", side_effect=fixtures.fixture_start_check), \
             patch.object(tool, "hardware", side_effect=AssertionError("GPU query prohibited")):
            return tool.prepare(self.args)

    def inspect(self, root=None, current=False):
        with patch.object(tool.bridge, "adapter", side_effect=fixtures.SuiteFixture):
            return tool.inspect((root or self.args.out) / "plan.json", current)

    def rehash(self, value):
        for name in value["files_sha256"]:
            value["files_sha256"][name] = tool.sha(self.args.out / name)
        tool.save(self.args.out / "plan.json", value)

    def test_all_six_game_candidate_reference_layouts_match_independent_counts(self):
        value = self.prepare(baselines=True)
        self.assertEqual(len(value["jobs"]), 30)
        actions = {"connect4cnn": 7, "pongcnn": 3, "flappycnn": 2, "breakoutcnn": 3, "snakecnn": 4, "mazecnn": 5}
        for index, job in enumerate(value["jobs"]):
            family = job["policy_family"]; config = tool.panel.read(self.args.out / job["config"])
            hidden = 128
            if family == "flex":
                channels = config.getint("policy", "cnn_channels_1")
                encoder = channels * (49 + 1) + 64 * (9*11*channels+1) + hidden*(64+1)
            elif family == "nature":
                encoder = 32*(64+1) + 64*(32*16+1) + 64*(64*9+1) + hidden*(128+1)
            else:
                encoder = 16*(9+1) + 4*16*(16*9+1) + 32*(16*9+1) + 4*32*(32*9+1)
                encoder += 32*(32*9+1) + 4*32*(32*9+1)
                encoder += hidden*((960 if family == "impala" else 32)+1)
            expected = encoder + (actions[job["environment"]]+1)*hidden + 3*hidden*hidden
            self.assertEqual(job["policy_layout"]["parameters"], expected)
            self.assertEqual(job["policy_layout"]["bytes"], expected*4)
            descriptor = json.loads((self.args.out / f"policy-layouts/job-{index:04d}/descriptor.json").read_text())
            self.assertIsNone(descriptor["checkpoint_bytes"])
            for flag in tool.policy_layout.FALSE_FLAGS: self.assertIs(descriptor[flag], False)
        self.assertFalse(value["policy_layout"]["checkpoint_bytes_checked_at_preparation"])
        self.assertFalse(value["policy_layout"]["full_learner_memory_accepted"])
        self.inspect()

    def test_relocation_is_offline_only(self):
        self.prepare(); relocated = self.root / "moved"
        shutil.copytree(self.args.out, relocated)
        with patch.object(tool.policy_layout, "process", side_effect=AssertionError("Native execution prohibited")):
            self.inspect(relocated)
            with self.assertRaisesRegex(ValueError, "relocated layout"):
                self.inspect(relocated, current=True)

    def test_rehashed_tensor_and_component_proof_is_rejected(self):
        value = self.prepare(); job = value["jobs"][0]
        directory = self.args.out / "policy-layouts/job-0000"
        descriptor = json.loads((directory / "descriptor.json").read_text())
        descriptor["parameter_tensors"][0]["shape"][0] += 1
        tool.save(directory / "descriptor.json", descriptor); tool.save(directory / "native.txt", descriptor)
        job["policy_layout"]["descriptor_sha256"] = tool.sha(directory / "descriptor.json")
        tool.save(directory / "result.json", job["policy_layout"]); self.rehash(value)
        with self.assertRaisesRegex(ValueError, "shape/offset"):
            self.inspect()

    def test_rehashed_host_command_or_clock_is_rejected(self):
        value = self.prepare(); directory = self.args.out / "policy-layouts/job-0000"
        record = value["jobs"][0]["policy_layout"]
        record["timing"]["end_monotonic_ns"] = record["timing"]["launch_monotonic_ns"]
        tool.save(directory / "native.txt.json", record["timing"])
        tool.save(directory / "result.json", record); self.rehash(value)
        with self.assertRaisesRegex(ValueError, "process receipt"):
            self.inspect()

    def test_native_build_source_or_action_mismatch_is_rejected(self):
        registry = tool.policy_layout.freeze(METADATA, self.root / "metadata")
        native = dict(source_sha256=tool.capture(self.registry.parent))
        sources = dict(native["source_sha256"])
        tool.metadata_source_binding(registry, native, sources)
        for name in ("src/algo.cu", "ocean/flappycnn/flappycnn.h"):
            altered = copy.deepcopy(native); altered["source_sha256"][name] = "0"*64
            with self.assertRaisesRegex(ValueError, "source differ"):
                tool.metadata_source_binding(registry, altered, sources)

    def test_wrong_size_checkpoint_rejected_without_reading_weights(self):
        job = self.prepare()["jobs"][0]; checkpoint = self.root / "synthetic.bin"
        for byte_count, valid in ((job["policy_layout"]["bytes"], True),
                (job["policy_layout"]["bytes"]+4, False), (0, False), (3, False)):
            with checkpoint.open("wb") as stream: stream.truncate(byte_count)
            with patch.object(tool.np, "fromfile", side_effect=AssertionError("No weight read")):
                if valid: self.assertEqual(tool.checkpoint_size(job, checkpoint), byte_count//4)
                else:
                    with self.assertRaises(ValueError): tool.checkpoint_size(job, checkpoint)

    def test_repeated_host_descriptor_and_clock_are_audited(self):
        value = self.prepare(); job = value["jobs"][0]
        destination = self.args.out / "execution/policy-layouts/job-0000"
        tool.policy_layout.describe(self.args.out / "policy-metadata", self.args.out / job["config"],
            job["environment"], "flex", destination, 60)
        tool.inspect_job_layout(value, self.args.out, 0, destination, execution=True)
        timing = json.loads((destination / "native.txt.json").read_text())
        timing["command"][1] = "train"
        record = json.loads((destination / "result.json").read_text())
        record["command"] = timing["command"]; record["timing"] = timing
        tool.save(destination / "result.json", record); tool.save(destination / "native.txt.json", timing)
        with self.assertRaisesRegex(ValueError, "process receipt"):
            tool.inspect_job_layout(value, self.args.out, 0, destination, execution=True)

    def test_failed_repeated_layout_precedes_gpu_query_and_retains_failure(self):
        value = self.prepare()
        args = SimpleNamespace(plan=self.args.out / "plan.json", mode="development", allow_gpu=True,
                               timeout=900, train_timeout=60, eval_timeout=30)
        with patch.object(tool, "inspect", return_value=value), \
             patch.object(tool.policy_layout, "describe", side_effect=ValueError("host shape mismatch")), \
             patch.object(tool, "hardware", side_effect=AssertionError("No GPU query")), \
             patch.object(tool, "process", side_effect=AssertionError("No trainer")):
            with self.assertRaisesRegex(ValueError, "host shape mismatch"): tool.run(args)
        result = json.loads((self.args.out / "execution/result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["jobs"], [])
        self.assertEqual(result["layout_checks"], [])

    def test_offline_audit_refuses_training_outside_registered_prefix(self):
        self.prepare(); directory = self.args.out / "execution"; directory.mkdir()
        tool.save(directory / "result.json", dict(status="failed", mode="development",
            plan_sha256=tool.sha(self.args.out / "plan.json"), jobs=[dict(job_index=1)],
            layout_checks=[dict(job_index=0)], evaluations=[], repeat_checks=[]))
        with patch.object(tool.bridge, "adapter", side_effect=fixtures.SuiteFixture), \
             self.assertRaisesRegex(ValueError, "declared prefix"):
            tool.audit(self.args.out / "plan.json", self.root / "audit")


if __name__ == "__main__": unittest.main()
