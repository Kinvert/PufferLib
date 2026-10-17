"""Host/synthetic receipt tests only: no GPU query, CUDA or CPU policy execution."""
import argparse
import configparser
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("encoder_profile_tools", HERE / "encoder_profile.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


class ProfileTests(unittest.TestCase):
    def test_runner_does_not_shadow_standard_python_profiler(self):
        result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,sys.argv[1]); import cProfile; assert callable(cProfile.runctx)", str(HERE)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build"
        self.build.mkdir()
        for family in set(p.MODELS.values()):
            (self.build / family).write_bytes(b"SYNTHETIC NONEXECUTABLE " + family.encode())
        (self.build / "binaries.sha256").write_text("".join(
            f"{p.sha(self.build / family)}  {self.build / family}\n" for family in set(p.MODELS.values())))
        source = p.ROOT / "ocean/connect4cnn/encoder_profile.cu"
        (self.build / "source.sha256").write_text(f"{p.sha(source)}  {source.relative_to(p.ROOT)}\n")
        for name in ("compiler.txt", "revision.txt", "worktree.txt"):
            (self.build / name).write_text("synthetic unit-test fixture; not a compilation\n")
        self.configs = self.root / "configs"
        for model in p.MODELS:
            directory = self.configs / f"{model}-s1/config"
            directory.mkdir(parents=True)
            config = configparser.ConfigParser(interpolation=None)
            config["policy"] = dict(hidden_size="128", encoder="4" if model == "flex_quality" else "2" if model == "nature_cnn" else "0")
            config["vec"] = dict(total_agents="64", num_buffers="1")
            config["train"] = dict(minibatch_size="2048")
            with (directory / "default.ini").open("w") as stream:
                config.write(stream)
        self.args = argparse.Namespace(binaries=self.build, configs=self.configs, out=self.root / "prepared",
                                       warmup=5, samples=5, repetitions=2)

    def describe(self, binary, config, batch):
        model = config.stem
        return dict(model=model, batch=batch, hidden=128, encoder_parameters=p.PARAMS[model],
                    parameter_payload_bytes=4 * p.PARAMS[model], receipt_exports=True, host_hash=True,
                    cuda_executed=False, policy_executed=False)

    def prepared(self):
        with patch.object(p, "describe", side_effect=self.describe) as desc, patch.object(p, "idle_gpu", side_effect=AssertionError("No GPU query")):
            protocol = p.prepare(self.args)
        self.assertEqual(desc.call_count, 8)
        return protocol

    def execution_args(self):
        return argparse.Namespace(out=self.args.out, timeout=1)

    def stub_audit(self, result, case, protocol, native, binary):
        # Small retained byte fixtures bind the full supervision path, not neural results.
        values = {"input.f32": case["phase"], "upstream.f32": case["phase"],
                  "parameters.f32": case["model"], "output-grad.f32": case["model"] + case["phase"]}
        files = {}
        for name, text in values.items():
            path = native / name
            path.write_bytes(text.encode())
            files[name] = dict(bytes=path.stat().st_size, sha256=p.sha(path))
        return dict(native=result, files=files)

    def stub_process(self, command, directory, log, timeout):
        result = dict(status="ok", launch_monotonic_ns=10, end_monotonic_ns=1000000010)
        log.write_text("{}\n")
        log.with_suffix(".log.json").write_text(json.dumps(result))
        return result

    def test_prepare_full_panel_copies_inputs_and_never_queries_gpu(self):
        protocol = self.prepared()
        self.assertEqual(len(protocol["cases"]), 64)
        self.assertEqual(protocol["cases"], p.case_panel(2))
        self.assertFalse(protocol["runtime_qualified"])
        (self.build / "default").write_bytes(b"Changed original; copies stay frozen")
        self.assertEqual(p.load(self.args.out), protocol)
        with self.assertRaises(FileExistsError):
            p.prepare(self.args)

    def test_reject_stale_build_and_old_receipt_interface_without_gpu(self):
        (self.build / "default").write_bytes(b"stale")
        with patch.object(p, "describe", side_effect=self.describe), self.assertRaisesRegex(ValueError, "Binary changed"):
            p.prepare(self.args)
        self.args.out = self.root / "old-interface"
        (self.build / "default").write_bytes(b"SYNTHETIC NONEXECUTABLE default")
        def old(binary, config, batch):
            info = self.describe(binary, config, batch)
            info.pop("receipt_exports")
            return info
        with patch.object(p, "describe", side_effect=old), self.assertRaisesRegex(ValueError, "metadata"):
            p.prepare(self.args)

    def test_reject_changed_compiled_source(self):
        text = (self.build / "source.sha256").read_text()
        (self.build / "source.sha256").write_text("0" * 64 + text[64:])
        with self.assertRaisesRegex(ValueError, "Compiled source"):
            p.prepare(self.args)

    def test_invalid_panel_controls_fail_before_preparation(self):
        for key, value in (("warmup", 4), ("samples", 0), ("repetitions", 1)):
            args = copy.copy(self.args)
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                p.prepare(args)
        self.assertFalse(self.args.out.exists())

    def test_tampered_frozen_input_prevents_queries_and_execution(self):
        self.prepared()
        (self.args.out / "configs/flex_quality.ini").write_text("changed")
        with patch.object(p, "idle_gpu", side_effect=AssertionError("No query")), self.assertRaisesRegex(ValueError, "Immutable"):
            p.run(self.execution_args())
        self.assertFalse((self.args.out / "execution").exists())

    def test_busy_and_query_errors_preserve_failure_without_child(self):
        for i, error in enumerate((ValueError("GPU busy"), subprocess.CalledProcessError(1, ["synthetic-smi"]))):
            self.args.out = self.root / f"prepared-{i}"
            self.prepared()
            with patch.object(p, "LOCK_PATH", self.root / "lock"), patch.object(p, "idle_gpu", side_effect=error), patch.object(p.claim, "process") as child:
                with self.assertRaises(type(error)):
                    p.run(self.execution_args())
                child.assert_not_called()
            self.assertEqual(json.loads((self.args.out / "execution/FAILURE.json").read_text())["status"], "failed")
            self.assertFalse((self.args.out / "execution/REPORT.json").exists())

    def test_reservation_prevents_queries_and_children(self):
        self.prepared()
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(p, "LOCK_PATH", self.root / "lock"), patch.object(p, "idle_gpu") as idle, patch.object(p.claim, "process") as child:
                with self.assertRaises(BlockingIOError):
                    p.run(self.execution_args())
                idle.assert_not_called()
                child.assert_not_called()
        self.assertTrue((self.args.out / "execution/FAILURE.json").exists())

    def test_full_synthetic_panel_and_immutable_receipts(self):
        self.prepared()
        with patch.object(p, "LOCK_PATH", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC RTX 5090, UUID, driver"), patch.object(p.claim, "process", side_effect=self.stub_process) as process, patch.object(p, "audit_result", side_effect=self.stub_audit):
            report = p.run(self.execution_args())
        self.assertEqual(process.call_count, 64)
        self.assertEqual(report["paired_groups"], 8)
        self.assertFalse(report["runtime_qualified"])
        for call in process.call_args_list:
            self.assertEqual(call.args[0][1], "--run")
            self.assertEqual(len(call.args[0]), 10)
        with self.assertRaises(FileExistsError):
            p.run(self.execution_args())

    def test_mid_panel_failure_keeps_first_case_and_never_aggregates(self):
        protocol = self.prepared()
        calls = []
        def failure(*args):
            calls.append(args)
            if len(calls) == 2:
                args[2].write_text("retained partial native log\n")
                raise subprocess.TimeoutExpired(args[0], 1)
            return self.stub_process(*args)
        with patch.object(p, "LOCK_PATH", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC RTX 5090, UUID, driver"), patch.object(p.claim, "process", side_effect=failure), patch.object(p, "audit_result", side_effect=self.stub_audit):
            with self.assertRaises(subprocess.TimeoutExpired):
                p.run(self.execution_args())
        failure = json.loads((self.args.out / "execution/FAILURE.json").read_text())
        self.assertEqual(failure["completed_cases"], [protocol["cases"][0]["id"]])
        self.assertFalse((self.args.out / "execution/REPORT.json").exists())
        self.assertTrue(calls[1][2].exists())

    def test_mutation_during_child_is_rejected(self):
        self.prepared()
        def mutate(*args):
            result = self.stub_process(*args)
            (self.args.out / "configs/flex_quality.ini").write_text("mutation")
            return result
        with patch.object(p, "LOCK_PATH", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC RTX 5090, UUID, driver"), patch.object(p.claim, "process", side_effect=mutate), patch.object(p, "audit_result") as audit:
            with self.assertRaisesRegex(ValueError, "Immutable"):
                p.run(self.execution_args())
            audit.assert_not_called()
        self.assertTrue((self.args.out / "execution/FAILURE.json").exists())

    def result_fixture(self, mode="eager", workspace="default", phase="rollout"):
        # Small file-shaped scalar fixtures; no model or policy forward occurs.
        case = dict(id="fixture", model="nature_cnn", batch=1, phase=phase, mode=mode, workspace=workspace)
        protocol = dict(samples=5, warmup=5,
                        metadata={f"nature_cnn/{phase}": self.describe(None, Path("nature_cnn.ini"), 1)})
        count = p.PARAMS[case["model"]]
        native = self.root / f"native-{mode}-{workspace}-{phase}"
        native.mkdir()
        sizes = {"input.f32": 1584 * 4, "upstream.f32": 128 * 4, "parameters.f32": count * 4,
                 "output-grad.f32": (128 + (count if phase == "learner" else 0)) * 4}
        hashes = {}
        for name, size in sizes.items():
            path = native / name
            path.write_bytes(b"\0" * size)
            hashes[name] = fnv(path.read_bytes())
        calls = 4 if phase == "rollout" else 11
        total = calls * 11 if mode == "eager" else 2 * calls
        result = dict(schema=1, model="nature_cnn", batch=1, hidden=128, phase=phase, mode=mode,
                      workspace=workspace, seed=173, warmup=5, samples=5, encoder_parameters=count,
                      parameters_unchanged=True, eager_mode_repeat_bytes_equal=True, receipts_written=True,
                      runtime_qualified=False, samples_ms=[1.0] * 5, mean_ms=1.0,
                      input_fnv64=hashes["input.f32"], upstream_fnv64=hashes["upstream.f32"],
                      parameter_fnv64=hashes["parameters.f32"], output_grad_fnv64=hashes["output-grad.f32"],
                      timed_host_gemm_calls=calls * 5 if mode == "eager" else 0, stream_api_calls=total,
                      dw_stream_api_calls=0, workspace_api_calls=total if workspace == "reassign" else 0)
        return result, case, protocol, native

    def test_native_receipt_shapes_durations_counters_and_all_modes(self):
        for mode in ("eager", "graph"):
            for workspace in ("default", "reassign"):
                for phase in ("rollout", "learner"):
                    fixture = self.result_fixture(mode, workspace, phase)
                    with patch.object(p, "file_fnv", side_effect=lambda binary, path: fnv(path.read_bytes())):
                        result = p.audit_result(*fixture, self.build / "default")
                    self.assertEqual(set(result["files"]), set(p.FILES))

    def test_bad_numerical_timing_and_identity_receipts_rejected(self):
        fixture = self.result_fixture()
        for key, value in (("runtime_qualified", True), ("samples_ms", [math.nan] * 5),
                           ("samples_ms", [0.0] * 5), ("mean_ms", 2.0), ("seed", 174),
                           ("parameters_unchanged", False), ("stream_api_calls", 0),
                           ("workspace_api_calls", 1), ("dw_stream_api_calls", 1),
                           ("encoder_parameters", True), ("input_fnv64", "bad")):
            result = copy.deepcopy(fixture[0])
            result[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.audit_result(result, *fixture[1:], self.build / "default")

    def test_candidate_identity_must_match_frozen_host_descriptor(self):
        result, case, protocol, native = self.result_fixture()
        markers = dict(dense_patch_alias_candidate=True, dense_patch_alias_active=True,
                       dense_patch_alias_layers=1, skipped_forward_patch_payload_bytes=512)
        protocol["metadata"]["nature_cnn/rollout"].update(markers)
        with self.assertRaisesRegex(ValueError, "presence"):
            p.audit_result(result, case, protocol, native, self.build / "default")
        result.update(markers)
        with patch.object(p, "file_fnv", side_effect=lambda binary, path: fnv(path.read_bytes())):
            p.audit_result(result, case, protocol, native, self.build / "default")
        for key, value in markers.items():
            changed = dict(result)
            changed[key] = not value if type(value) is bool else value + 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                p.audit_result(changed, case, protocol, native, self.build / "default")
        protocol["metadata"]["nature_cnn/rollout"] = self.describe(None, Path("nature_cnn.ini"), 1)
        with self.assertRaisesRegex(ValueError, "presence"):
            p.audit_result(result, case, protocol, native, self.build / "default")

    def test_bad_byte_size_nonfinite_and_fingerprint_rejected(self):
        fixture = self.result_fixture()
        native = fixture[3]
        path = native / "input.f32"
        original = path.read_bytes()
        path.write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "byte count"):
            p.audit_result(*fixture, self.build / "default")
        path.write_bytes(b"\0\0\xc0\x7f" + original[4:])  # Quiet NaN float32.
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            p.audit_result(*fixture, self.build / "default")
        path.write_bytes(original)
        with patch.object(p, "file_fnv", return_value="0" * 16), self.assertRaisesRegex(ValueError, "fingerprint/file"):
            p.audit_result(*fixture, self.build / "default")

    def test_panel_never_drops_cases_or_accepts_pair_differences(self):
        protocol = dict(cases=p.case_panel(2))
        records = {}
        for case in protocol["cases"]:
            values = {"input.f32": case["phase"], "upstream.f32": case["phase"],
                      "parameters.f32": case["model"], "output-grad.f32": case["model"] + case["phase"]}
            records[case["id"]] = dict(files={name: dict(sha256=hashlib.sha256(value.encode()).hexdigest()) for name, value in values.items()})
        self.assertEqual(p.audit_panel(protocol, records)["cases"], 64)
        missing = copy.deepcopy(records)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            p.audit_panel(protocol, missing)
        changed = copy.deepcopy(records)
        changed[protocol["cases"][-1]["id"]]["files"]["output-grad.f32"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "byte mismatch"):
            p.audit_panel(protocol, changed)

    def test_real_host_process_timeout_kills_child_and_retains_receipt(self):
        # Exercise shared process-group cleanup with a plain host sleep, not a policy.
        log = self.root / "timeout.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            p.claim.process([sys.executable, "-c", "import time; time.sleep(5)"], self.root, log, 0.05)
        record = json.loads(log.with_suffix(".log.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertLess(record["returncode"], 0)
        self.assertGreater(record["end_monotonic_ns"], record["launch_monotonic_ns"])
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)


if __name__ == "__main__":
    unittest.main()
