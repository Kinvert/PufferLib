#!/usr/bin/env python3
"""Native host manifests and synthetic match-receipt audits; no policies run."""
import argparse
import configparser
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pong_exact", ROOT / "ocean/pongcnn/exact_eval.py")
exact = importlib.util.module_from_spec(spec); spec.loader.exec_module(exact)
BINARIES = Path(os.environ.get("PONG_EXACT_TEST_BINARIES", ROOT / "build/pongcnn/exact-preparation-20261005-v2"))


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = configparser.ConfigParser(interpolation=None)
        config.read([ROOT / "config/default.ini", ROOT / "config/pongcnn.ini"])
        config.set("base", "env_name", "pongcnn")
        self.config = self.root / "source.ini"
        with self.config.open("x") as stream: config.write(stream)
        self.args = SimpleNamespace(binary=BINARIES / "default", config=self.config, out=self.root / "suite",
            seed=2**32-1, episodes=65, offset=31, slots=64, representation=3, max_decisions=64, purpose="development")
        exact.create_suite(self.args)
        self.suite, self.manifest = exact.load_suite(self.args.out / "suite.json")

    def results(self):
        rows = []
        for i, start in enumerate(self.manifest):
            completed = int(i % 3 == 2)
            r = 21 if completed else (i % 10 if i % 3 == 1 else 0)
            l = i % 20 if completed else (i % 7 if i % 3 == 1 else 0)
            rally = 3 if completed else (0 if r+l else 64)
            lower, upper = exact.fraction_bounds(r, l, 21, completed)
            row = dict(start)
            row.update(version="1", block_seed=str(self.suite["seed"]), slot=str(i % 64), decisions="64",
                last_rally_decisions=str(rally), right_points=str(r), left_points=str(l), episode_return=float(r-l).hex(),
                completed=str(completed), capped=str(1-completed), point_fraction_lower=lower.hex(),
                point_fraction_upper=upper.hex(), native_perf=exact.f32(r/(r+l)).hex() if completed else "",
                action_hash="0123456789abcdef")
            rows.append(row)
        return rows

    def audit(self, rows):
        path = self.root / "results.csv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=exact.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return exact.audit_csv(path, self.suite, self.manifest)

    def run_args(self, name="run", prepare_only=False):
        checkpoint = self.root / "weights.bin"
        if not checkpoint.exists(): checkpoint.write_bytes(b"\0" * 16)
        return SimpleNamespace(binary=BINARIES / "default", config=self.config, checkpoint=checkpoint,
            suite=self.args.out / "suite.json", out=self.root / name, family="flex", timeout=10,
            eager=False, prepare_only=prepare_only)

    def test_prepare_only_preserves_all_family_core_settings_without_gpu_queries(self):
        source = exact.read(self.config)
        cases = [(BINARIES / "default", encoder, family) for encoder, family in exact.FAMILIES.items()]
        cases += [(BINARIES / "default", 0, "tiny"), (BINARIES / "impala", 0, "impala"),
                  (BINARIES / "impoola", 0, "impoola")]
        for case, (binary, encoder, family) in enumerate(cases):
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config = copy.deepcopy(source)
                config["policy"].update(encoder=str(encoder), hidden_size=str(hidden), num_layers=str(layers))
                config["env"].update(representation_mode="1", representation_seed="12345")
                path = self.root / f"config-{case}-{hidden}.ini"
                with path.open("x") as stream: config.write(stream)
                args = self.run_args(f"prepared-{case}-{hidden}", prepare_only=True)
                args.binary, args.config, args.family = binary, path, family
                args.eager = hidden == 32
                identity = exact.info(binary)
                with patch.object(exact, "info", return_value=identity), \
                     patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No GPU queries")), \
                     patch.object(exact, "process", side_effect=AssertionError("No policy execution")):
                    exact.run(args)
                effective = exact.read(args.out / "config/default.ini")
                self.assertEqual(dict(effective["policy"]), dict(config["policy"]))
                self.assertEqual(effective.getint("env", "representation_mode"), 0)
                self.assertEqual(effective.getint("env", "representation"), 3)
                self.assertEqual(effective.getint("eval_exact", "max_decisions"), 64)
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if args.eager else 1)
                record = json.loads((args.out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertFalse(record["gpu_runtime_qualified"])
                self.assertEqual(record["estimand"], exact.ESTIMAND)
                self.assertFalse((args.out / "episodes.csv").exists())
                self.assertFalse((args.out / "REPORT.md").exists())
                for name, digest in record["tooling_sha256"].items():
                    self.assertEqual(exact.sha(args.out / "tooling" / name), digest)
        with self.assertRaises(FileExistsError): exact.run(args)

    def test_state_checkpoint_preparation_preserves_core(self):
        config = exact.read(self.config)
        config["base"]["env_name"] = "pong"
        config["policy"].update(encoder="5", hidden_size="256", num_layers="2")
        path = self.root / "state.ini"
        with path.open("x") as stream: config.write(stream)
        suite_args = copy.copy(self.args)
        suite_args.binary, suite_args.config, suite_args.representation = BINARIES / "state", path, 0
        suite_args.out = self.root / "state-suite"
        exact.create_suite(suite_args)
        args = self.run_args("prepared-state", prepare_only=True)
        args.binary, args.config, args.suite, args.family = suite_args.binary, path, suite_args.out / "suite.json", "state"
        identity = exact.info(args.binary)
        with patch.object(exact, "info", return_value=identity), \
             patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No GPU queries")):
            exact.run(args)
        self.assertEqual(dict(exact.read(args.out / "config/default.ini")["policy"]), dict(config["policy"]))
        self.assertTrue((args.out / "config/pong.ini").exists())

    def test_bad_weights_busy_gpu_and_failed_queries_never_launch_policy(self):
        identity = exact.info(BINARIES / "default")
        for name, data in (("empty", b""), ("malformed", b"\0"*3), ("nonfinite", bytes.fromhex("0000c07f"))):
            args = self.run_args(name); args.checkpoint.write_bytes(data)
            with patch.object(exact, "info", return_value=identity), \
                 patch.object(exact.subprocess, "check_output", side_effect=AssertionError("Weights before GPU queries")), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(Exception): exact.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args.checkpoint.write_bytes(b"\0"*16)
        for name, answers in (("busy", ["12345\n"]), ("query-failed", [subprocess.CalledProcessError(1, ["smi"])]),
                              ("no-hardware", ["", ""])):
            args = self.run_args(name)
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=answers), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(Exception): exact.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def test_reservation_conflict_stops_before_gpu_queries(self):
        args = self.run_args("locked"); identity = exact.info(args.binary)
        directory = self.root / "build/connect4cnn"; directory.mkdir(parents=True)
        with (directory / "hardware-benchmark.lock").open("a") as lock:
            exact.fcntl.flock(lock, exact.fcntl.LOCK_EX | exact.fcntl.LOCK_NB)
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No query while reserved")), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy execution")):
                with self.assertRaises(BlockingIOError): exact.run(args)
        self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def fake_native(self, command, cwd, log, timeout):
        rows = self.results(); counts = self.audit(rows)
        log.write_text("synthetic fixture, no policy execution\n"
            f"PONG_EXACT_EVAL version=1 requested=65 evaluated=65 completed={counts['completed']} "
            f"capped={counts['capped']} wins={counts['wins']} right={counts['right_points']} "
            f"left={counts['left_points']} params=4\n"
            "PONG_EXACT_BOUNDS mean_lower=999 mean_upper=-999 kind=censoring-not-confidence\n")
        with (cwd / "episodes.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=exact.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        timing = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
        log.with_suffix(".log.json").write_text(json.dumps(timing))
        return timing

    def test_reporting_uses_independent_bounds_and_requires_summary_quota_input_closure(self):
        args = self.run_args("completed"); identity = exact.info(args.binary)
        with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
             patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(exact, "process", side_effect=self.fake_native):
            exact.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "ok"); self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertEqual(record["counts"], self.audit(self.results()))
        self.assertFalse(record["gpu_runtime_qualified"])
        self.assertIn("not confidence intervals", (args.out / "REPORT.md").read_text())
        # Auxiliary native floating-point bounds never override audited integer-derived per-match bounds.
        self.assertNotIn("999", (args.out / "REPORT.md").read_text())
        cases = ("missing-row", "wrong-params", "wrong-wins", "wrong-points", "duplicate-summary",
                 "changed-config", "changed-copied-suite", "changed-spawn", "missing-clock", "backwards-clock")
        for case in cases:
            args = self.run_args(case)
            def corrupt(command, cwd, log, timeout):
                timing = self.fake_native(command, cwd, log, timeout)
                if case == "missing-row":
                    rows = (cwd / "episodes.csv").read_text().splitlines()
                    (cwd / "episodes.csv").write_text("\n".join(rows[:-1])+"\n")
                elif case == "wrong-params": log.write_text(log.read_text().replace("params=4", "params=5"))
                elif case == "wrong-wins": log.write_text(log.read_text().replace("wins=21", "wins=22"))
                elif case == "wrong-points": log.write_text(log.read_text().replace("right=", "right=9"))
                elif case == "duplicate-summary": log.write_text(log.read_text()*2)
                elif case == "changed-config": (cwd / "config/pongcnn.ini").write_text("# tampered\n")
                elif case == "changed-copied-suite":
                    path = cwd / "suite/suite.json"; data = json.loads(path.read_text()); data["purpose"] = "heldout"
                    path.write_text(json.dumps(data))
                elif case == "changed-spawn":
                    path = cwd / "suite/episodes.csv"; path.write_text(path.read_text()+"\n")
                elif case == "missing-clock": log.with_suffix(".log.json").unlink()
                else:
                    timing["end_monotonic_ns"] = 0; log.with_suffix(".log.json").write_text(json.dumps(timing))
                return timing
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
                 patch.object(exact, "process", side_effect=corrupt):
                with self.assertRaises(Exception): exact.run(args)
            record = json.loads((args.out / "result.json").read_text())
            self.assertEqual(record["status"], "failed"); self.assertNotIn("counts", record)
            self.assertFalse((args.out / "REPORT.md").exists())
            self.assertTrue((args.out / "episodes.csv").exists())

    def test_timeout_retains_partial_rows_and_process_clock_without_success(self):
        args = self.run_args("timeout"); identity = exact.info(args.binary)
        def failure(command, cwd, log, timeout):
            self.fake_native(command, cwd, log, timeout)
            rows = (cwd / "episodes.csv").read_text().splitlines()
            (cwd / "episodes.csv").write_text("\n".join(rows[:2])+"\n")
            raise subprocess.TimeoutExpired(command, timeout)
        with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
             patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(exact, "process", side_effect=failure):
            with self.assertRaises(subprocess.TimeoutExpired): exact.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed"); self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertNotIn("counts", record); self.assertFalse((args.out / "REPORT.md").exists())
        self.assertEqual(len((args.out / "episodes.csv").read_text().splitlines()), 2)

    def test_all_assigned_zero_point_caps_and_final_fraction_bounds(self):
        rows = self.results(); counts = self.audit(rows)
        self.assertEqual(counts["episodes"], 65)
        self.assertEqual(counts["completed"], 21); self.assertEqual(counts["capped"], 44)
        self.assertEqual(counts["wins"], 21)
        self.assertEqual(counts, self.audit(list(reversed(rows))))
        self.assertFalse(counts["policy_runtime_certified"])
        self.assertEqual(counts["bounds_kind"], "deterministic-censoring-not-confidence")
        # Every capped zero-point match contributes [0,1], never a dropped row.
        for row in rows:
            row.update(right_points="0", left_points="0", episode_return="0x0.0p+0", completed="0", capped="1",
                       last_rally_decisions="64", point_fraction_lower="0x0.0p+0", point_fraction_upper="0x1.0p+0", native_perf="")
        counts = self.audit(rows)
        self.assertEqual(counts["mean_final_point_fraction_lower"], 0)
        self.assertEqual(counts["mean_final_point_fraction_upper"], 1)
        self.assertEqual(counts["capped"], 65)
        self.suite["episodes"] = 32; self.manifest = self.manifest[:32]
        self.assertEqual(self.audit(rows[:32])["episodes"], 32)

    def test_missing_duplicate_extra_and_corrupt_receipts(self):
        rows = self.results()
        for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(Exception): self.audit(bad)
        for key, value in (("episode_id", "999"), ("start_rng", "-1"), ("policy_seed", "0"),
                ("representation", "0"), ("slot", "1"), ("start_world_hash", "0"*16), ("decisions", "65"),
                ("last_rally_decisions", "0"), ("right_points", "22"), ("left_points", "-1"),
                ("completed", "2"), ("capped", "0"), ("point_fraction_lower", "nan"),
                ("point_fraction_upper", "0x0.0p+0"), ("native_perf", "0x0.0p+0"),
                ("episode_return", "nan"), ("action_hash", "bad")):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(Exception): self.audit(bad)
        for key, value in (("last_rally_decisions", "0"), ("native_perf", "nan"), ("left_points", "21")):
            bad = copy.deepcopy(rows); bad[2][key] = value
            with self.subTest(terminal=key), self.assertRaises(Exception): self.audit(bad)

    def test_bound_formula_against_all_possible_terminal_scores(self):
        for target in range(1, 22):
            for right in range(target):
                for left in range(target):
                    possible = [target/(target+l) for l in range(left, target)]
                    possible += [r/(target+r) for r in range(right, target)]
                    self.assertEqual(exact.fraction_bounds(right, left, target, 0), (min(possible), max(possible)))
                    with self.assertRaises(Exception): exact.fraction_bounds(right, left, target, 1)
            for loser in range(target):
                for r, l in ((target, loser), (loser, target)):
                    self.assertEqual(exact.fraction_bounds(r, l, target, 1), (r/(r+l), r/(r+l)))
            with self.assertRaises(Exception): exact.fraction_bounds(target, target, target, 1)

    def test_frozen_inputs_no_overwrite_and_changed_suite_controls(self):
        with self.assertRaises(FileExistsError): exact.create_suite(self.args)
        for name in ("source.ini", "manifest.ini", "episodes.csv"):
            path = self.args.out / name; saved = path.read_bytes(); path.write_bytes(saved + b"\n")
            with self.subTest(file=name), self.assertRaises(Exception): exact.load_suite(self.args.out / "suite.json")
            path.write_bytes(saved)
        path = self.args.out / "suite.json"; saved = path.read_bytes()
        path.write_text(json.dumps(dict(self.suite, max_decisions=65)))
        with self.assertRaises(Exception): exact.load_suite(path)
        path.write_bytes(saved)
        for key, value in (("seed", True), ("max_decisions", 0), ("offset", 2**32-32),
                           ("rules", "other"), ("estimand", "partial-fraction"), ("representation", 7)):
            bad = dict(self.suite); bad[key] = value
            with self.subTest(key=key), self.assertRaises(Exception): exact.validate_suite(bad)

    def native_rows(self, binary, config, name):
        path = self.root / (name + ".ini")
        with path.open("x") as stream: config.write(stream)
        result = subprocess.run([str(binary), "eval_exact_manifest", str(path)], capture_output=True,
                                text=True, timeout=120, check=True)
        return list(csv.DictReader(result.stdout.splitlines()))

    def test_native_all_appearances_families_core_and_state_worlds(self):
        source = exact.read(self.config)
        keys = ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash")
        base = None
        for rep in range(exact.info(BINARIES / "default")["representation_count"]):
            config = exact.suite_config(source, dict(self.suite, representation=rep))
            config.set("policy", "hidden_size", "512"); config.set("policy", "num_layers", "4")
            rows = self.native_rows(BINARIES / "default", config, f"r{rep}")
            worlds = [{k: r[k] for k in keys} for r in rows]
            if base is None: base = worlds
            self.assertEqual(worlds, base)
            for family in ("impala", "impoola"):
                self.assertEqual(rows, self.native_rows(BINARIES / family, config, f"r{rep}-{family}"))
        config = exact.suite_config(source, dict(self.suite, representation=0))
        config.set("base", "env_name", "pong")
        state = self.native_rows(BINARIES / "state", config, "state")
        self.assertEqual([{k: r[k] for k in keys} for r in state], base)
        self.assertNotEqual(state[0]["start_observation_hash"], rows[0]["start_observation_hash"])

    def test_old_binary_catalog_rejects_new_ids_before_preparation(self):
        identity = exact.info(BINARIES / "default")
        identity["representation_count"] = 5
        for representation in (5, 6):
            args = copy.copy(self.args)
            args.representation = representation
            args.out = self.root / f"old-catalog-{representation}"
            with patch.object(exact, "info", return_value=identity), self.assertRaisesRegex(ValueError, "does not support"):
                exact.create_suite(args)
            self.assertFalse(args.out.exists())

    def test_native_independent_tail_identity(self):
        source = exact.read(self.config)
        tail = dict(self.suite, episodes=1, offset=95, slots=1)
        self.assertEqual(self.native_rows(BINARIES / "default", exact.suite_config(source, tail), "tail"), [self.manifest[-1]])

    def test_native_invalid_world_cap_and_assignment_rejected_before_gpu(self):
        good = exact.suite_config(exact.read(self.config), self.suite)
        for section, key, value in (("env", "height", "nan"), ("env", "max_score", "1.5"),
                ("env", "paddle_speed", "1e-100"), ("env", "ball_initial_speed_y", "-1"),
                ("env", "ball_width", "999999"), ("env", "continuous", "1"),
                ("env", "representation_mode", "1"), ("eval_exact", "max_decisions", "16777217"),
                ("eval_exact", "episode_offset", "4294967295"), ("eval_exact", "slots", "0")):
            config = copy.deepcopy(good); config.set(section, key, value)
            path = self.root / (key + ".invalid.ini")
            with path.open("w") as stream: config.write(stream)
            result = subprocess.run([str(BINARIES / "default"), "eval_exact_manifest", str(path)],
                                    capture_output=True, text=True, timeout=30)
            with self.subTest(key=key):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Pong exact evaluation:", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binaries", type=Path, default=BINARIES)
    args, rest = parser.parse_known_args(); BINARIES = args.binaries.resolve()
    unittest.main(argv=[__file__] + rest)
