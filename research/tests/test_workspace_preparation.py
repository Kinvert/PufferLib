"""Native host metadata and synthetic exports only; never CUDA/GEMM/model runs."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_workspace_acceptance as tool

BINARY = tool.ROOT / os.environ.get("WORKSPACE_TEST_BINARY", "build/workspace-acceptance/native-20261006-final/workspace_acceptance")


class WorkspacePreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare(self, name="packet"):
        args = argparse.Namespace(binary=BINARY, out=self.root / name)
        tool.prepare(args); return args.out / "packet.json"

    def test_real_host_description_geometry_and_dyadic_bound(self):
        value = tool.metadata(BINARY)
        self.assertEqual(len(value["fixtures"]), 6)
        for fixture in value["fixtures"]:
            self.assertEqual(fixture["input_bytes"], fixture["output_bytes"])
            self.assertEqual(fixture["input_bytes"], 8*(fixture["m"]*fixture["k"]+
                fixture["n"]*fixture["k"]+fixture["m"]*fixture["n"]))
            self.assertLess(16*max(fixture["m"], fixture["n"], fixture["k"]), 2**24)

    def test_preparation_only_calls_native_describe(self):
        original = subprocess.check_output
        def host_only(command, **kwargs):
            self.assertEqual(command, [str(BINARY.resolve()), "--describe"])
            return original(command, **kwargs)
        with patch.object(tool.subprocess, "check_output", side_effect=host_only):
            packet = self.prepare(); value = tool.load_packet(packet)
        self.assertEqual(len(value["cases"]), 96)
        self.assertFalse(value["runtime_supervisor_implemented"])
        self.assertFalse(value["gpu_execution_authorized"])
        self.assertFalse(any(packet.parent.rglob("actual.f32")))
        self.assertEqual(len(set(c["name"] for c in value["cases"])), 96)

    def test_overwrite_and_frozen_binary_or_source_changes_rejected(self):
        packet = self.prepare()
        with self.assertRaises(FileExistsError): self.prepare()
        native = packet.parent / "native"; size = native.stat().st_size
        with native.open("ab") as stream: stream.write(b"x")
        with self.assertRaises(ValueError): tool.load_packet(packet)
        with native.open("r+b") as stream: stream.truncate(size)
        source = packet.parent / "source/src/algo.cu"
        with source.open("a") as stream: stream.write("\n")
        with self.assertRaises(ValueError): tool.load_packet(packet)

    def test_panel_subset_protocol_and_unsupported_gate_rejected(self):
        packet = self.prepare(); original = json.loads(packet.read_text())
        for key, value in (("cases", original["cases"][:-1]), ("protocol", "wrong"),
                           ("gpu_execution_authorized", True), ("backend_math_certified", True),
                           ("runtime_supervisor_implemented", True)):
            changed = copy.deepcopy(original); changed[key] = value; packet.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): tool.load_packet(packet)

    def test_stale_build_dependency_receipt_rejects_without_modifying_libraries(self):
        build = self.root / "copied-build"; shutil.copytree(BINARY.parent, build)
        receipt = build / "dependencies.sha256"; lines = receipt.read_text().splitlines()
        _, name = lines[0].split(None, 1); lines[0] = "0"*64 + "  " + name
        receipt.write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError, "Build dependency changed"):
            tool.prepare(argparse.Namespace(binary=build / BINARY.name, out=self.root / "bad-packet"))
        self.assertFalse((self.root / "bad-packet").exists())

    def synthetic(self, case):
        directory = self.root / case["name"]; directory.mkdir()
        fixture = tool.fixtures()[case["fixture"]]
        # Deliberately synthetic zeros, not a native/GPU oracle or policy.
        for filename, key in (("input.f32", "input_bytes"), ("actual.f32", "output_bytes"), ("reference.f32", "output_bytes")):
            np.zeros(fixture[key]//4, dtype=np.float32).tofile(directory / filename)
        submissions = 2 if case["mode"] == "graph" else 4
        receipt = dict(protocol=tool.PROTOCOL, status="scoped-gpu-checks-passed", fixture=case["fixture"],
            workspace=case["workspace"], schedule=case["schedule"], mode=case["mode"], lanes=2, iterations=3,
            main_api_calls=4*submissions, dw_api_calls=2*submissions, gemm_api_calls=6*submissions,
            workspace_api_calls=6*submissions if case["workspace"] == "reassign" else 0,
            input_bytes=fixture["input_bytes"], output_bytes=fixture["output_bytes"], runtime=12080,
            driver=13000, cublas=120800, measured_overlap=False, encoder_math_certified=False,
            frontier_superiority_certified=False)
        (directory / "native.json").write_text(json.dumps(receipt))
        return directory, receipt

    def test_synthetic_all_schedule_mode_workspace_audits_without_certification(self):
        value = tool.load_packet(self.prepare())
        for case in [c for c in value["cases"] if c["fixture"] == 0]:
            directory, _ = self.synthetic(case)
            result = tool.audit_case(value, case["name"], directory)
            self.assertEqual(result["status"], "export-consistency-passed")
            self.assertFalse(result["backend_math_certified"])
            self.assertFalse(result["encoder_math_certified"])

    def test_mismatched_nonfinite_nondyadic_and_truncated_exports_rejected(self):
        value = tool.load_packet(self.prepare()); case = value["cases"][0]
        directory, _ = self.synthetic(case)
        mutations = (("actual.f32", np.float32(.125)), ("reference.f32", np.float32(np.nan)),
                     ("input.f32", np.float32(.2)), ("input.f32", np.float32(1)))
        for filename, number in mutations:
            path = directory / filename; data = np.fromfile(path, dtype=np.float32)
            data[0] = number; data.tofile(path)
            with self.assertRaises(ValueError, msg=filename): tool.audit_case(value, case["name"], directory)
            data[0] = 0; data.tofile(path)
        with (directory / "actual.f32").open("r+b") as stream: stream.truncate(4)
        with self.assertRaises(ValueError): tool.audit_case(value, case["name"], directory)

    def test_missing_handle_wrong_iteration_library_case_and_claim_rejected(self):
        value = tool.load_packet(self.prepare()); case = value["cases"][0]
        directory, receipt = self.synthetic(case); path = directory / "native.json"
        for key, changed in (("dw_api_calls", 0), ("main_api_calls", 0), ("workspace_api_calls", 1),
                             ("iterations", 2), ("fixture", 1), ("cublas", 0), ("measured_overlap", True)):
            replacement = dict(receipt); replacement[key] = changed; path.write_text(json.dumps(replacement))
            with self.assertRaises(ValueError, msg=key): tool.audit_case(value, case["name"], directory)

    def test_invalid_native_arguments_reject_before_cuda_or_output_creation(self):
        cases = (("6", "default", "serial", "eager"), ("-1", "default", "serial", "eager"),
                 ("0", "bad", "serial", "eager"), ("0", "default", "bad", "eager"),
                 ("0", "default", "serial", "bad"))
        for fixture, workspace, schedule, mode in cases:
            out = self.root / "unused"
            result = subprocess.run([str(BINARY), "--run", fixture, workspace, schedule, mode, str(out)],
                                    text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0); self.assertFalse(out.exists())
        result = subprocess.run([str(BINARY), "--run", "0", "default", "serial", "eager", str(self.root)],
                                text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Need fresh receipt directory", result.stderr)

    def test_relocated_frozen_preparation_still_inspects(self):
        packet = self.prepare(); moved = self.root / "archive"
        shutil.copytree(packet.parent, moved)
        value = tool.load_packet(moved / "packet.json")
        self.assertEqual(value["binary_sha256"], tool.sha(BINARY))


if __name__ == "__main__": unittest.main()
