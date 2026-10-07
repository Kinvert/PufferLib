"""Native host descriptor and synthetic artifact guards, no neural execution."""
import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_acceptance as gate
import checkpoint_preflight as preflight
from research.tests import test_eval_acceptance as fixtures

METADATA = Path(os.environ.get("POLICY_METADATA_DIR", gate.ROOT / "build/policy-metadata/native-acceptance-20261006"))


class CheckpointPreflightTests(unittest.TestCase):
    case = fixtures.AcceptanceTest.case

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def size_artifact(self, case):
        config = gate.adapter(case["task"]).read(Path(case["config"]))
        policy = config["policy"]; h = int(policy["hidden_size"]); layers = int(policy["num_layers"])
        # Independent scalar parameter arithmetic only: legacy SAME conv,
        # optional residual, parameterless pooling, projections, head and core.
        height, width, channels, count = 36, 44, 1, 0
        for stage in range(1, int(policy["cnn_depth"])+1):
            out = int(policy[f"cnn_channels_{stage}"]); k = int(policy[f"cnn_kernel_{stage}"])
            stride = int(policy[f"cnn_stride_{stage}"])
            count += out * (channels*k*k+1)
            height = (height+stride-1)//stride; width = (width+stride-1)//stride; channels = out
            if int(policy[f"cnn_residual_{stage}"]): count += out*(out*9+1)
            if int(policy[f"cnn_pool_{stage}"]): height = (height+1)//2; width = (width+1)//2
        if int(policy["cnn_global_pool"]): height = width = 1
        projection = int(policy["cnn_projection"]); count += projection*(height*width*channels+1)
        if projection != h: count += h*(projection+1)
        actions = int(re.search(r"^#define ACT_SIZES \{([0-9]+)\}$",
            (gate.ROOT / preflight.HEADERS[case["task"]]).read_text(), re.M)[1])
        count += (actions+1)*h + layers*3*h*h
        # Sparse zeros are artifact fixtures, not initialized/executed policies.
        with Path(case["checkpoint"]).open("wb") as stream: stream.truncate(count*4)
        return count

    def prepare(self, cases, name="packet"):
        source = self.root / f"{name}-cases.json"; source.write_text(json.dumps(cases))
        gate.prepare(argparse.Namespace(cases=source, out=self.root / name, timeout=30, policy_metadata=METADATA))
        return self.root / name / "packet.json"

    def test_all_five_game_heads_match_independent_counts_before_gpu(self):
        cases = [self.case(task, task) for task in gate.TASKS]
        counts = [self.size_artifact(case) for case in cases]
        # Only host info/manifests/registration are permitted.
        with patch.object(gate.adapter("flappycnn").shutil, "which", side_effect=AssertionError("GPU query")):
            packet = self.prepare(cases)
            value = gate.load_packet(packet, current_inputs=True)
        self.assertEqual(value["protocol"], gate.LAYOUT_PROTOCOL)
        self.assertEqual([case["checkpoint_layout"]["parameters"] for case in value["cases"]], counts)
        self.assertTrue(all(not case["checkpoint_layout"]["numerical_acceptance"] for case in value["cases"]))
        moved = self.root / "moved"; shutil.copytree(packet.parent, moved)
        self.assertEqual(gate.load_packet(moved / "packet.json"), value)

    def test_configurable_depth_skips_pooling_projection_and_gap_counts(self):
        case = self.case(); config = gate.adapter("flappycnn").read(Path(case["config"]))
        config["policy"].update(cnn_depth="3", cnn_projection="16", cnn_global_pool="1", cnn_channels_1="8",
            cnn_kernel_1="1", cnn_stride_1="8", cnn_residual_1="1", cnn_pool_1="2",
            cnn_channels_2="32", cnn_kernel_2="5", cnn_stride_2="2", cnn_residual_2="1", cnn_pool_2="1",
            cnn_channels_3="16", cnn_kernel_3="3", cnn_stride_3="1", cnn_residual_3="0", cnn_pool_3="0")
        with Path(case["config"]).open("w") as stream: config.write(stream)
        expected = self.size_artifact(case); packet = self.prepare([case])
        self.assertEqual(gate.load_packet(packet)["cases"][0]["checkpoint_layout"]["parameters"], expected)

    def test_nature_impala_and_impoola_head_counts_match_independent_c(self):
        scalar = subprocess.check_output([str(gate.ROOT / "build/policy-metadata/costs-20261006"), "128", "1", "2"], text=True)
        counts = {row["model"]: int(row["total_parameters"]) for row in csv.DictReader(io.StringIO(scalar))}
        cases = []
        for family, model in (("nature", "nature_cnn"), ("impala", "impala_cnn"), ("impoola", "impoola_cnn")):
            case = self.case(name=family); config = gate.adapter("flappycnn").read(Path(case["config"]))
            for key in list(config["policy"]):
                if key.startswith("cnn_"): config.remove_option("policy", key)
            config["policy"]["encoder"] = "2" if family == "nature" else "0"
            with Path(case["config"]).open("w") as stream: config.write(stream)
            case["family"] = family
            if family != "nature": case["binary"] = str(Path(case["binary"]).parent / family)
            with Path(case["checkpoint"]).open("wb") as stream: stream.truncate(counts[model]*4)
            cases.append(case)
        value = gate.load_packet(self.prepare(cases))
        self.assertEqual([c["checkpoint_layout"]["parameters"] for c in value["cases"]],
            [counts[preflight.FAMILIES[c["family"]][1]] for c in cases])

    def test_truncated_extra_and_wrong_core_bytes_fail_with_retained_host_receipts(self):
        for index, change in enumerate((-4, 4, "core")):
            case = self.case(name=f"bad-{index}"); count = self.size_artifact(case)
            if change == "core":
                config = gate.adapter("flappycnn").read(Path(case["config"]))
                config["policy"]["hidden_size"] = "256"
                with Path(case["config"]).open("w") as stream: config.write(stream)
            else:
                with Path(case["checkpoint"]).open("wb") as stream: stream.truncate(count*4+change)
            with self.assertRaisesRegex(ValueError, "Process failed"):
                self.prepare([case], f"failed-{index}")
            out = self.root / f"failed-{index}"
            self.assertTrue((out / "preparation_failure.json").is_file())
            self.assertFalse((out / "packet.json").exists())
            native = json.loads((out / "inputs" / case["name"] / "layout/native.txt.json").read_text())
            self.assertNotEqual(native["returncode"], 0)

    def test_matching_size_nan_still_fails_existing_finite_guard(self):
        case = self.case(); self.size_artifact(case)
        with Path(case["checkpoint"]).open("r+b") as stream: stream.write(struct.pack("<f", float("nan")))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.prepare([case])
        out = self.root / "packet"
        self.assertTrue((out / "inputs/quality/layout/descriptor.json").exists())
        self.assertTrue((out / "preparation_failure.json").exists())
        self.assertFalse((out / "packet.json").exists())

    def test_rehashed_descriptor_action_and_clock_edits_reject(self):
        case = self.case(); self.size_artifact(case); packet = self.prepare([case])
        original = json.loads(packet.read_text())
        for label in ("descriptor", "action", "clock"):
            output = self.root / label; shutil.copytree(packet.parent, output)
            value = copy.deepcopy(original)
            if label == "descriptor":
                path = output / "inputs/quality/layout/descriptor.json"; data = json.loads(path.read_text())
                data["discrete_actions"] = 3; path.write_text(json.dumps(data))
            elif label == "action": value["cases"][0]["checkpoint_layout"]["actions"] = 3
            else: value["cases"][0]["checkpoint_layout"]["timing"]["end_monotonic_ns"] = 0
            value["files_sha256"] = gate.hashes(output)
            value["files_sha256"].pop("packet.json", None)
            (output / "packet.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError): gate.load_packet(output / "packet.json")

    def test_unsupported_state_and_encoder5_do_not_bypass_new_gate(self):
        for label in ("state", "encoder5"):
            case = self.case(name=label); self.size_artifact(case)
            if label == "state": case["family"] = "state"
            else:
                config = gate.adapter("flappycnn").read(Path(case["config"]))
                config["policy"]["encoder"] = "5"
                with Path(case["config"]).open("w") as stream: config.write(stream)
            with self.assertRaises(ValueError): self.prepare([case], label+"-packet")
            self.assertTrue((self.root / (label+"-packet") / "preparation_failure.json").exists())

    def test_failed_repeated_host_check_precedes_any_gpu_adapter(self):
        case = self.case(); self.size_artifact(case); packet = self.prepare([case])
        ev = gate.adapter("flappycnn")
        with patch.object(preflight, "check", side_effect=ValueError("retained host mismatch")), \
             patch.object(ev, "run", side_effect=AssertionError("No GPU adapter")):
            with self.assertRaisesRegex(ValueError, "host mismatch"):
                gate.run(argparse.Namespace(packet=packet, out=self.root / "execution"))
        self.assertTrue((self.root / "execution/failure.json").exists())
        self.assertFalse((self.root / "execution/acceptance.json").exists())

    def test_execution_layout_audit_reparses_raw_descriptor_and_clocks(self):
        case = self.case(); self.size_artifact(case); packet = self.prepare([case])
        value = gate.load_packet(packet); entry = value["cases"][0]
        layout = packet.parent / "inputs/quality/layout"
        gate.inspect_execution_layout(entry, layout)
        data = json.loads((layout / "native.txt.json").read_text()); data["end_monotonic_ns"] = 0
        (layout / "native.txt.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError): gate.inspect_execution_layout(entry, layout)


if __name__ == "__main__": unittest.main()
