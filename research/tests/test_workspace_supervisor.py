"""Host/control-flow doubles only. No GPU query, native GEMM or CPU model."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workspace_supervisor as tool

PACKET = tool.ROOT / os.environ.get("WORKSPACE_TEST_PACKET", "build/workspace-acceptance/panel-20261006-final/packet.json")
HARDWARE = "Synthetic RTX 5090, GPU-unit-fixture, synthetic-driver\n"


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        context = patch.object(tool, "LOCK", self.root / "benchmark.lock")
        context.start(); self.addCleanup(context.stop)

    def prepare(self):
        out = self.root / "plan"
        tool.prepare(argparse.Namespace(packet=PACKET, out=out))
        return out / "plan.json"

    def synthetic_process(self, command, cwd, log, timeout, mutation=None):
        # Minimal fake exports test supervision only. Fixture sizes/array math
        # are deliberately mocked; the independent preparation tests cover them.
        directory = Path(command[-1]); directory.mkdir()
        data = struct.pack("<f", int(command[2]) / 8)
        for name in ("input.f32", "reference.f32", "actual.f32"): (directory / name).write_bytes(data)
        receipt = dict(runtime=12080, driver=13000, cublas=120800,
                       fixture=int(command[2]), workspace=command[3], schedule=command[4], mode=command[5])
        (directory / "native.json").write_text(json.dumps(receipt))
        log.write_text("Explicit synthetic fixture: no GPU, no native GEMM or policy\n")
        result = dict(command=command, cwd=str(cwd), status="ok", returncode=0, pid=123456,
                      launch_monotonic_ns=1, end_monotonic_ns=1000000001, timeout=timeout)
        log.with_suffix(log.suffix+".json").write_text(json.dumps(result))
        if mutation: mutation(command, cwd, log)
        return result

    def synthetic_audit(self, packet, name, directory):
        return dict(case=name, status="synthetic-export-fixture-only", files_sha256=tool.native.files(directory),
                    backend_math_certified=False, encoder_math_certified=False, frontier_superiority_certified=False,
                    note="Explicit synthetic control-flow fixture, not a GPU oracle result")

    def execute(self, plan, out, mutation=None, gpu=None):
        with patch.object(tool, "gpu", side_effect=gpu or (lambda smi: HARDWARE)), \
             patch.object(tool.claim, "process", side_effect=lambda command, cwd, log, timeout:
                          self.synthetic_process(command, cwd, log, timeout, mutation)), \
             patch.object(tool.native, "audit_case", side_effect=self.synthetic_audit):
            tool.run(argparse.Namespace(plan=plan, out=out, timeout=2))

    def test_actual_preparation_has_no_query_or_launch(self):
        with patch.object(tool.subprocess, "check_output", side_effect=AssertionError("No queries")), \
             patch.object(tool.claim, "process", side_effect=AssertionError("No launch")):
            plan = self.prepare(); value, _ = tool.load(plan, current_tools=True)
        self.assertEqual(len(value["cases"]), 96)
        self.assertTrue(value["runtime_supervisor_implemented"])
        self.assertFalse(value["gpu_execution_authorized"])
        self.assertFalse(any(plan.parent.rglob("actual.f32")))

    def test_full_synthetic_panel_and_relocated_offline_audit(self):
        plan = self.prepare(); out = self.root / "synthetic-execution"
        self.execute(plan, out)
        report = json.loads((out / "REPORT.json").read_text())
        self.assertEqual(report["cases"], 96); self.assertEqual(len(report["records"]), 96)
        self.assertFalse(report["backend_math_certified"]); self.assertFalse(report["encoder_math_certified"])
        with patch.object(tool.native, "audit_case", side_effect=self.synthetic_audit):
            tool.audit(argparse.Namespace(plan=plan, execution=out))
        moved_plan, moved_out = self.root / "archive-plan", self.root / "archive-execution"
        shutil.copytree(plan.parent, moved_plan); shutil.copytree(out, moved_out)
        with patch.object(tool.native, "audit_case", side_effect=self.synthetic_audit):
            tool.audit(argparse.Namespace(plan=moved_plan / "plan.json", execution=moved_out))
        with self.assertRaises(FileExistsError): self.execute(plan, out)

    def test_busy_query_failure_and_hardware_changes_retain_failure(self):
        plan = self.prepare()
        variants = (ValueError("busy synthetic GPU"), subprocess.CalledProcessError(1, ["synthetic-smi"]), None)
        for index, error in enumerate(variants):
            calls = []
            def gpu(smi):
                calls.append(smi)
                if error: raise error
                return HARDWARE if len(calls) == 1 else HARDWARE+"changed"
            out = self.root / f"query-failure-{index}"
            with self.assertRaises((ValueError, subprocess.CalledProcessError)): self.execute(plan, out, gpu=gpu)
            self.assertTrue((out / "FAILURE.json").exists()); self.assertFalse((out / "REPORT.json").exists())
            self.assertFalse(any(out.rglob("actual.f32")))

    def test_reservation_collision_never_queries_or_launches(self):
        plan = self.prepare(); out = self.root / "reserved"
        with tool.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(tool, "gpu", side_effect=AssertionError("No query while reserved")), \
                 patch.object(tool.claim, "process", side_effect=AssertionError("No launch while reserved")):
                with self.assertRaises(BlockingIOError): tool.run(argparse.Namespace(plan=plan, out=out, timeout=2))
        self.assertTrue((out / "FAILURE.json").exists()); self.assertFalse((out / "REPORT.json").exists())

    def test_partial_timeout_keeps_files_and_no_success(self):
        plan = self.prepare(); out = self.root / "timeout"
        def mutation(command, cwd, log): raise TimeoutError("Synthetic native deadline")
        with self.assertRaises(TimeoutError): self.execute(plan, out, mutation)
        self.assertTrue(next(out.rglob("native.log")).exists())
        self.assertTrue(next(out.rglob("actual.f32")).exists())
        self.assertEqual(json.loads((out / "FAILURE.json").read_text())["completed_cases"], [])
        self.assertFalse((out / "REPORT.json").exists())

    def test_plan_or_copied_binary_changes_during_case_fail(self):
        for key in ("plan", "binary"):
            # Fresh test workspace per mode; original native files remain untouched.
            with self.subTest(key=key):
                if (self.root / "plan").exists(): shutil.rmtree(self.root / "plan")
                plan = self.prepare(); out = self.root / ("changed-"+key)
                def mutation(command, cwd, log):
                    if key == "plan":
                        value = json.loads(plan.read_text()); value["extra"] = "modified"; plan.write_text(json.dumps(value))
                    else:
                        with (plan.parent / "inputs/native").open("ab") as stream: stream.write(b"x")
                with self.assertRaises(ValueError): self.execute(plan, out, mutation)
                self.assertFalse((out / "REPORT.json").exists()); self.assertTrue((out / "FAILURE.json").exists())

    def test_nonzero_bad_clock_wrong_command_and_deadline_fail(self):
        plan = self.prepare()
        for key, value in (("returncode", 1), ("end_monotonic_ns", 0), ("pid", 0), ("timeout", 601),
                           ("command", ["wrong"])):
            out = self.root / ("bad-process-"+key)
            def mutation(command, cwd, log):
                path = log.with_suffix(log.suffix+".json"); receipt = json.loads(path.read_text())
                receipt[key] = value; path.write_text(json.dumps(receipt))
            with self.assertRaises((ValueError, IndexError)): self.execute(plan, out, mutation)
            self.assertFalse((out / "REPORT.json").exists())
        with self.assertRaises(ValueError): tool.run(argparse.Namespace(plan=plan, out=self.root / "invalid-timeout", timeout=0))
        self.assertFalse((self.root / "invalid-timeout").exists())

    def test_live_execution_cannot_claim_archive_relocation(self):
        plan = self.prepare(); out = self.root / "wrong-cwd"
        def mutation(command, cwd, log):
            path = log.with_suffix(log.suffix+".json"); value = json.loads(path.read_text())
            value["cwd"] = str(self.root / "different" / cwd.name)
            value["command"][-1] = str(Path(value["cwd"]) / "native")
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "cwd changed during execution"):
            self.execute(plan, out, mutation)
        self.assertTrue((out / "FAILURE.json").exists()); self.assertFalse((out / "REPORT.json").exists())

    def test_paired_output_or_runtime_identity_difference_fails(self):
        plan = self.prepare()
        for name in ("bytes", "version"):
            out = self.root / ("paired-drift-"+name)
            def mutation(command, cwd, log):
                if cwd.name == "f0-default-serial-eager-r1":
                    if name == "bytes":
                        for file in ("reference.f32", "actual.f32"): (cwd / "native" / file).write_bytes(struct.pack("<f", .5))
                    else:
                        path = cwd / "native/native.json"; value = json.loads(path.read_text())
                        value["cublas"] += 1; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "bytes differ|identity changed"): self.execute(plan, out, mutation)
            self.assertFalse((out / "REPORT.json").exists())
            self.assertEqual(len(json.loads((out / "FAILURE.json").read_text())["completed_cases"]), 96)

    def test_completed_earlier_files_cannot_be_silently_rewritten(self):
        plan = self.prepare(); out = self.root / "past-mutation"
        def mutation(command, cwd, log):
            if cwd.name == "f0-default-serial-graph-r0":
                path = out / "f0-default-serial-eager-r0/native/actual.f32"
                path.write_bytes(struct.pack("<f", .5))
        with self.assertRaisesRegex(ValueError, "Changed/unsafe retained file"): self.execute(plan, out, mutation)
        self.assertFalse((out / "REPORT.json").exists())

    def test_query_helper_rejects_empty_multiple_devices_busy_and_errors(self):
        variants = (("", ""), (HARDWARE+HARDWARE, ""), (HARDWARE, "12345\n"))
        for hardware, busy in variants:
            with patch.object(tool.subprocess, "check_output", side_effect=[hardware, busy]):
                with self.assertRaises(ValueError): tool.gpu("synthetic-smi")
        with patch.object(tool.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "fixture")):
            with self.assertRaises(subprocess.CalledProcessError): tool.gpu("synthetic-smi")

    def test_offline_audit_rejects_changed_report_or_raw_receipts(self):
        plan = self.prepare(); out = self.root / "completed"; self.execute(plan, out)
        path = out / "REPORT.json"; original = path.read_text(); value = json.loads(original)
        value["cases"] = 95; path.write_text(json.dumps(value))
        with patch.object(tool.native, "audit_case", side_effect=self.synthetic_audit):
            with self.assertRaises(ValueError): tool.audit(argparse.Namespace(plan=plan, execution=out))
        path.write_text(original)
        with (out / "f0-default-serial-eager-r0/native.log").open("a") as stream: stream.write("tampered\n")
        with self.assertRaises(ValueError): tool.audit(argparse.Namespace(plan=plan, execution=out))

    def test_real_host_child_deadline_kills_process_group_and_retains_clock(self):
        script = self.root / "host-child.sh"; pidfile = self.root / "child.pid"
        script.write_text('#!/usr/bin/env bash\nsleep 60 &\nprintf "%s\\n" "$!" > "$1"\nwait\n')
        log = self.root / "host-deadline.txt"
        with self.assertRaises(subprocess.TimeoutExpired):
            tool.claim.process(["bash", str(script), str(pidfile)], self.root, log, 1)
        result = json.loads(log.with_suffix(log.suffix+".json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["end_monotonic_ns"], result["launch_monotonic_ns"])
        state = Path("/proc") / pidfile.read_text().strip() / "stat"
        if state.exists(): self.assertEqual(state.read_text().split()[2], "Z")


if __name__ == "__main__": unittest.main()
