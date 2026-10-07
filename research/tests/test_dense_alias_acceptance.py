"""Host/synthetic supervision tests; never load a model or evaluate a reference."""
import argparse
import copy
import ctypes
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("alias_acceptance", Path(__file__).parents[1] / "dense_alias_acceptance.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(out=self.root / "prepared", baseline=self.root / "baseline.so", candidate=self.root / "candidate.so")
        self.source = p.ROOT / "ocean/connect4cnn/tests/test_nature.cu"
        for marker, variant in enumerate(p.VARIANTS):
            library = getattr(self.args, variant)
            library.write_bytes(b"SYNTHETIC NONEXECUTABLE LIBRARY " + variant.encode())
            receipts = Path(str(library) + ".build")
            receipts.mkdir()
            (receipts / "source.sha256").write_text(f"{p.sha(self.source)}  {self.source.relative_to(p.ROOT)}\n")
            (receipts / "binary.sha256").write_text(f"{p.sha(library)}  {library}\n")
            (receipts / "environment.txt").write_text(f"C4_DENSE_PATCH_ALIAS={marker}\nNVCC_ARCH=sm_120\nNVCC_PREPEND_FLAGS=--threads 1\nNVCC_APPEND_FLAGS=\n")
            (receipts / "command.txt").write_text("synthetic nvcc -shared ocean/connect4cnn/tests/test_nature.cu " + ("-DC4_DENSE_PATCH_ALIAS" if marker else "") + "\n")
            (receipts / "compiler.txt").write_text("synthetic compiler receipt; not compilation\n")

    def prepare(self):
        with patch.object(p, "idle_gpu", side_effect=AssertionError("No GPU query")), patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")):
            return p.prepare(self.args)

    def run_args(self):
        return argparse.Namespace(out=self.args.out, timeout=1)

    def stub_process(self, command, cwd, log, timeout, environment):
        protocol = p.load(self.args.out)
        job = next(job for job in protocol["jobs"] if job["id"] == cwd.name)
        ordinal = protocol["jobs"].index(job)
        result = dict(status="ok", returncode=0, command=command, cwd=str(cwd),
                      launch_monotonic_ns=100 + ordinal * 20, end_monotonic_ns=110 + ordinal * 20)
        log.write_text("".join("PASS " + case["id"] + "\n" for case in protocol["cases"]) + "COMPLETE " + job["id"] + "\n")
        p.save(log.with_suffix(".log.json"), result)
        return result

    def stub_audit(self, protocol, job, directory):
        # This tests process/paired closure only; these aren't GPU arrays/results.
        return {case["id"]: {"input.f32": "synthetic-input", "eager-0-output.f32": "synthetic-output",
                              "result.json": job["variant"]} for case in protocol["cases"]}

    def test_fixed_panel_covers_quality_equal_projection_and_all_declared_dimensions(self):
        panel = p.cases()
        self.assertEqual(len(panel), 72)
        self.assertEqual(len({case["id"] for case in panel}), 72)
        self.assertEqual({case["batch"] for case in panel}, {1, 3, 64, 2048})
        expected = {"flex_quality": {64: 102240, 128: 110560, 256: 118880},
                    "nature_cnn": {64: 80096, 128: 88352, 256: 104864}}
        for case in panel:
            self.assertEqual(p.count(case), expected[case["model"]][case["hidden"]])
        self.assertEqual([job["variant"] for job in p.jobs()], ["baseline", "candidate", "candidate", "baseline"])

    def test_prepare_and_inspect_do_not_load_library_query_gpu_or_evaluate_oracle(self):
        protocol = self.prepare()
        with patch.object(p, "worker", side_effect=AssertionError("No model")), patch.object(p, "idle_gpu", side_effect=AssertionError("No query")):
            self.assertEqual(p.load(self.args.out), protocol)
        self.assertFalse((self.args.out / "execution").exists())
        self.assertFalse((self.args.out / "arrays").exists())
        self.assertFalse(protocol["cuda_executed"])
        self.assertFalse(protocol["performance_qualified"])

    def test_build_marker_override_and_stale_library_rejected(self):
        receipt = Path(str(self.args.candidate) + ".build")
        for name, text in (("environment.txt", "C4_DENSE_PATCH_ALIAS=0\nNVCC_ARCH=sm_120\nNVCC_APPEND_FLAGS=\nNVCC_PREPEND_FLAGS=\n"),
                           ("environment.txt", "C4_DENSE_PATCH_ALIAS=1\nNVCC_ARCH=native\nNVCC_APPEND_FLAGS=\nNVCC_PREPEND_FLAGS=\n"),
                           ("environment.txt", "C4_DENSE_PATCH_ALIAS=1\nNVCC_ARCH=sm_120\nNVCC_APPEND_FLAGS=-DC4_BOGUS\nNVCC_PREPEND_FLAGS=\n"),
                           ("binary.sha256", "0" * 64 + "  candidate.so\n")):
            original = (receipt / name).read_text()
            self.args.out = self.root / ("prepared-" + str(len(list(self.root.glob("prepared-*")))))
            (receipt / name).write_text(text)
            with self.subTest(name=name, text=text), self.assertRaises(ValueError): self.prepare()
            (receipt / name).write_text(original)

    def test_changed_copied_input_and_protocol_are_rejected(self):
        self.prepare()
        path = self.args.out / "libraries/candidate.so"
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Immutable"): p.load(self.args.out)
        path.write_bytes(original)
        protocol = self.args.out / "protocol.json"
        protocol.write_text(protocol.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "Protocol"): p.load(self.args.out)

    def test_boolean_cannot_replace_integer_case_field(self):
        self.prepare()
        path = self.args.out / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["cases"][0]["batch"] = True
        path.write_text(json.dumps(protocol))
        (self.args.out / "protocol.sha256").write_text(p.sha(path) + "\n")
        with self.assertRaisesRegex(ValueError, "type"): p.load(self.args.out)

    def test_worker_cannot_run_without_supervised_execution_receipts(self):
        self.prepare()
        with patch.object(ctypes, "CDLL", side_effect=AssertionError("No library load")), self.assertRaisesRegex(ValueError, "supervisor"):
            p.worker(self.args.out, p.jobs()[0]["id"])

    def test_shared_reservation_blocks_gpu_queries_and_children(self):
        self.prepare()
        lock_path = self.root / "lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(p, "LOCK", lock_path), patch.object(p, "idle_gpu") as idle, patch.object(p.claim, "process") as child:
                with self.assertRaises(BlockingIOError): p.run(self.run_args())
                idle.assert_not_called(); child.assert_not_called()
        self.assertTrue((self.args.out / "execution/FAILURE.json").exists())

    def test_busy_and_query_failure_preserve_failures_without_children(self):
        for index, error in enumerate((ValueError("GPU busy"), subprocess.CalledProcessError(1, ["synthetic-query"]))):
            self.args.out = self.root / f"prepared-{index}"
            self.prepare()
            with patch.object(p, "LOCK", self.root / "lock"), patch.object(p, "idle_gpu", side_effect=error), patch.object(p.claim, "process") as child:
                with self.assertRaises(type(error)): p.run(self.run_args())
                child.assert_not_called()
            self.assertFalse((self.args.out / "execution/REPORT.json").exists())

    def test_complete_synthetic_supervision_offline_audit_and_no_restart(self):
        self.prepare()
        with patch.object(p, "LOCK", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC GPU"), patch.object(p.claim, "process", side_effect=self.stub_process) as child, patch.object(p, "audit_worker", side_effect=self.stub_audit):
            summary = p.run(self.run_args())
            self.assertEqual(child.call_count, 4)
            self.assertEqual(summary["native_calls"], 1152)
            with patch.object(p, "idle_gpu", side_effect=AssertionError("Offline audit must not query")):
                self.assertEqual(p.audit(self.args.out), summary)
            with self.assertRaises(FileExistsError): p.run(self.run_args())
        log = self.args.out / "execution/r0-baseline/worker.log"
        log.write_text(log.read_text() + "changed\n")
        with self.assertRaisesRegex(ValueError, "Immutable"): p.audit(self.args.out)

    def test_worker_timeout_preserves_terminal_failure_and_never_qualifies(self):
        self.prepare()
        error = subprocess.TimeoutExpired(["synthetic-worker"], 1)
        with patch.object(p, "LOCK", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC GPU"), patch.object(p.claim, "process", side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired): p.run(self.run_args())
        self.assertFalse((self.args.out / "execution/REPORT.json").exists())
        with self.assertRaisesRegex(ValueError, "Failed"): p.audit(self.args.out)

    def test_changed_library_during_child_preserves_failure_despite_success_log(self):
        self.prepare()
        def child(*args):
            result = self.stub_process(*args)
            path = self.args.out / "libraries/candidate.so"
            path.write_bytes(path.read_bytes() + b"changed during synthetic process")
            return result
        with patch.object(p, "LOCK", self.root / "lock"), patch.object(p, "idle_gpu", return_value="SYNTHETIC GPU"), patch.object(p.claim, "process", side_effect=child) as launched, patch.object(p, "audit_worker") as audit:
            with self.assertRaisesRegex(ValueError, "Immutable"): p.run(self.run_args())
            self.assertEqual(launched.call_count, 1)
            audit.assert_not_called()
        self.assertTrue((self.args.out / "execution/FAILURE.json").exists())
        self.assertFalse((self.args.out / "execution/REPORT.json").exists())

    def test_process_command_clock_and_completion_guards(self):
        protocol = dict(cases=p.cases(), jobs=p.jobs())
        execution = self.root / "execution"
        execution.mkdir()
        bindings = {}
        for ordinal, job in enumerate(protocol["jobs"]):
            directory = execution / job["id"]
            directory.mkdir()
            command = ["synthetic", job["id"]]
            bindings[job["id"]] = dict(command=command, cwd=str(directory))
            p.save(directory / "worker.log.json", dict(status="ok", returncode=0, command=command, cwd=str(directory),
                launch_monotonic_ns=100 + ordinal * 20, end_monotonic_ns=110 + ordinal * 20))
            (directory / "worker.log").write_text("".join("PASS " + case["id"] + "\n" for case in protocol["cases"]) + "COMPLETE " + job["id"] + "\n")
        p.save(execution / "process-bindings.json", bindings)
        p.audit_processes(protocol, execution)
        path = execution / protocol["jobs"][1]["id"] / "worker.log.json"
        original = json.loads(path.read_text())
        for key, value in (("command", ["different"]), ("cwd", "/wrong"), ("returncode", True),
                           ("launch_monotonic_ns", 101), ("end_monotonic_ns", 1)):
            changed = dict(original); changed[key] = value
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(ValueError): p.audit_processes(protocol, execution)
        path.write_text(json.dumps(original))
        (execution / protocol["jobs"][-1]["id"] / "worker.log").write_text("COMPLETE only\n")
        with self.assertRaisesRegex(ValueError, "completion"): p.audit_processes(protocol, execution)

    def array_fixture(self):
        import numpy as np
        case = next(case for case in p.cases() if case["state"] == "zero-weights-and-observation")
        protocol = dict(cases=[case], tolerances={"flex_quality": [8e-4, 6e-5], "nature_cnn": [3e-4, 3e-5]})
        job = p.jobs()[0]
        directory = self.root / "array-worker"
        folder = directory / "arrays" / case["id"]
        folder.mkdir(parents=True)
        data = p.fixture(case)
        for name, values in zip(("input.f32", "parameters.f32", "upstream.f32"), data): values.tofile(folder / name)
        # Synthetic retained-array fixtures; no native or reference executes.
        expected = [np.zeros_like(data[2], dtype="<f8"), np.zeros_like(data[1], dtype="<f8")]
        for name, values in zip(("expected-output.f64", "expected-gradient.f64"), expected): values.tofile(folder / name)
        for mode in p.MODES:
            for field, values in zip(("output", "gradient", "parameters"), [expected[0], expected[1], data[1]]):
                values.astype("<f4").tofile(folder / f"{mode}-{field}.f32")
        p.save(folder / "result.json", dict(case=case, status="passed", candidate=False))
        return protocol, job, directory, folder

    def test_retained_array_oracle_seed_parameter_and_bitwise_guards(self):
        import numpy as np
        protocol, job, directory, folder = self.array_fixture()
        p.audit_worker(protocol, job, directory)
        for filename, mode in (("input.f32", "seed"), ("graph-0-output.f32", "signedzero"),
                               ("expected-gradient.f64", "nonfinite"), ("eager-0-gradient.f32", "numeric")):
            path = folder / filename
            original = path.read_bytes()
            dtype = "<f8" if filename.endswith("f64") else "<f4"
            values = np.fromfile(path, dtype=dtype)
            values[0] = -0.0 if mode == "signedzero" else np.nan if mode == "nonfinite" else 1.0
            values.tofile(path)
            with self.subTest(filename=filename), self.assertRaises((ValueError, AssertionError)):
                p.audit_worker(protocol, job, directory)
            path.write_bytes(original)
        path = folder / "eager-0-output.f32"
        path.write_bytes(b"wrong length")
        with self.assertRaisesRegex(ValueError, "size"): p.audit_worker(protocol, job, directory)

    def test_missing_case_wrong_variant_and_extra_array_rejected(self):
        protocol, job, directory, folder = self.array_fixture()
        path = folder / "result.json"
        original = path.read_text()
        wrong = json.loads(original); wrong["candidate"] = True
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "variant"): p.audit_worker(protocol, job, directory)
        path.write_text(original)
        (folder / "extra.f32").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "extra"): p.audit_worker(protocol, job, directory)

    def test_every_worker_pair_and_negative_difference_is_required(self):
        protocol = dict(cases=p.cases(), jobs=p.jobs())
        records = {job["id"]: self.stub_audit(protocol, job, None) for job in protocol["jobs"]}
        self.assertEqual(p.compare_workers(protocol, records)["cases"], 72)
        missing = dict(records); missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "Missing"): p.compare_workers(protocol, missing)
        changed = copy.deepcopy(records)
        changed[p.jobs()[-1]["id"]][p.cases()[-1]["id"]]["eager-0-output.f32"] = "changed"
        with self.assertRaisesRegex(ValueError, "byte mismatch"): p.compare_workers(protocol, changed)


if __name__ == "__main__": unittest.main()
