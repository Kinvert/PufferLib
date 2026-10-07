#!/usr/bin/env python3
"""Native host manifests and synthetic receipt audits; no policy executes."""
import argparse
import configparser
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("breakout_exact", ROOT / "ocean/breakoutcnn/exact_eval.py")
exact = importlib.util.module_from_spec(spec); spec.loader.exec_module(exact)
BINARIES = ROOT / "build/breakoutcnn/exact-preparation-20261005"


class ExactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = configparser.ConfigParser(interpolation=None)
        config.read([ROOT / "config/default.ini", ROOT / "config/breakoutcnn.ini"])
        config.set("base", "env_name", "breakoutcnn")
        self.config = self.root / "source.ini"
        with self.config.open("x") as stream: config.write(stream)
        self.args = SimpleNamespace(binary=BINARIES / "default", config=self.config,
            out=self.root / "suite", seed=2**32-1, episodes=65, offset=31, slots=64,
            representation=3, max_frames=5, purpose="development")
        exact.create_suite(self.args)
        self.suite, self.manifest = exact.load_suite(self.args.out / "suite.json")

    def results(self):
        rows = []
        for i, start in enumerate(self.manifest):
            frames = 1 if i % 3 == 0 else 5
            completed = int(i % 3 != 1)
            points = i % 10
            row = dict(start)
            row.update(version="1", block_seed=str(self.suite["seed"]), slot=str(i % 64),
                decisions=str(1+(frames-1)//3), frames=str(frames), score=str(points),
                perf=exact.f32(points/864).hex(), episode_return=float(points).hex(),
                completed=str(completed), capped=str(1-completed), action_hash="0123456789abcdef")
            rows.append(row)
        return rows

    def audit(self, rows):
        path = self.root / "results.csv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=exact.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return exact.audit_csv(path, self.suite, self.manifest)

    def test_exact_quota_and_all_caps(self):
        rows = self.results()
        summary = self.audit(rows)
        self.assertEqual(summary["episodes"], 65)
        self.assertEqual(summary["completed"] + summary["capped"], 65)
        self.assertEqual(summary["score"], sum(i % 10 for i in range(65)))
        self.assertFalse(summary["policy_runtime_certified"])
        # A terminal on the cap frame is completed, not a censored episode.
        for row in rows:
            row.update(frames="5", decisions="2", completed="0", capped="1")
        self.assertEqual(self.audit(rows)["capped"], 65)
        # A smaller-than-batch assigned quota still needs exactly that many rows.
        self.suite["episodes"] = 32; self.manifest = self.manifest[:32]
        self.assertEqual(self.audit(rows[:32])["episodes"], 32)

    def test_missing_extra_duplicate_and_accounting(self):
        rows = self.results()
        for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(Exception): self.audit(bad)
        for key, value in (("episode_id", "999"), ("env_seed", "0"), ("start_rng", "0"),
                ("representation", "0"), ("slot", "1"), ("start_world_hash", "0"*16),
                ("decisions", "0"), ("frames", "6"), ("score", "865"), ("perf", "nan"),
                ("episode_return", "nan"), ("completed", "2"), ("capped", "1"), ("action_hash", "bad")):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(Exception): self.audit(bad)

    def test_manifest_inputs_frozen_no_overwrite(self):
        with self.assertRaises(FileExistsError): exact.create_suite(self.args)
        for name in ("source.ini", "manifest.ini", "episodes.csv"):
            path = self.args.out / name; saved = path.read_bytes()
            path.write_bytes(saved + b"\n")
            with self.subTest(file=name), self.assertRaises(Exception): exact.load_suite(self.args.out / "suite.json")
            path.write_bytes(saved)
        for key, value in (("seed", True), ("offset", 2**32-32), ("max_frames", 0),
                           ("rules", "other"), ("evaluation", "pooled"), ("representation", 5)):
            bad = copy.deepcopy(self.suite); bad[key] = value
            with self.subTest(key=key), self.assertRaises(Exception): exact.validate_suite(bad)
        path = self.args.out / "suite.json"; saved = path.read_bytes()
        changed = dict(self.suite, max_frames=6); path.write_text(json.dumps(changed))
        with self.assertRaises(Exception): exact.load_suite(path)
        path.write_bytes(saved)

    def native_rows(self, binary, config, name):
        path = self.root / (name + ".ini")
        with path.open("x") as stream: config.write(stream)
        result = subprocess.run([str(binary), "eval_exact_manifest", str(path)], text=True,
                                capture_output=True, timeout=120, check=True)
        return list(csv.DictReader(result.stdout.splitlines()))

    def test_native_all_drawings_and_compiled_families(self):
        source = exact.read(self.config)
        world_keys = ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash")
        base = None
        for representation in range(5):
            suite = dict(self.suite, representation=representation)
            config = exact.suite_config(source, suite)
            config.set("policy", "hidden_size", "512"); config.set("policy", "num_layers", "4")
            observed = self.native_rows(BINARIES / "default", config, f"r{representation}")
            worlds = [{k: row[k] for k in world_keys} for row in observed]
            if base is None: base = worlds
            self.assertEqual(worlds, base)
            for family in ("impala", "impoola"):
                self.assertEqual(observed, self.native_rows(BINARIES / family, config, f"r{representation}-{family}"))
        config = exact.suite_config(source, dict(self.suite, representation=0))
        config.set("base", "env_name", "breakout")
        state = self.native_rows(BINARIES / "state", config, "state")
        self.assertEqual([{k: row[k] for k in world_keys} for row in state], base)
        self.assertNotEqual(state[0]["start_observation_hash"], observed[0]["start_observation_hash"])

    def test_native_independent_final_episode(self):
        source = exact.read(self.config)
        tail = dict(self.suite, episodes=1, offset=95, slots=1)
        observed = self.native_rows(BINARIES / "default", exact.suite_config(source, tail), "tail")
        self.assertEqual(observed, [self.manifest[-1]])

    def test_native_invalid_inputs_rejected_before_gpu(self):
        good = exact.suite_config(exact.read(self.config), self.suite)
        for section, key, value in (("env", "height", "1.5"), ("env", "brick_rows", "5"),
                ("env", "continuous", "1"), ("env", "initial_ball_speed", "nan"),
                ("env", "paddle_speed", "1e-100"), ("env", "ball_width", "999999"),
                ("env", "representation_mode", "1"), ("eval_exact", "max_frames", "16777217"),
                ("eval_exact", "episode_offset", "4294967295"), ("eval_exact", "slots", "0")):
            config = copy.deepcopy(good); config.set(section, key, value)
            path = self.root / (key + ".invalid.ini")
            with path.open("w") as stream: config.write(stream)
            proc = subprocess.run([str(BINARIES / "default"), "eval_exact_manifest", str(path)],
                                  text=True, capture_output=True, timeout=30)
            with self.subTest(key=key):
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("Breakout exact evaluation:", proc.stderr)

    def run_args(self, name="run", prepare_only=False):
        checkpoint = self.root / "weights.bin"
        if not checkpoint.exists(): checkpoint.write_bytes(b"\0" * 16)
        return SimpleNamespace(binary=BINARIES / "default", config=self.config, checkpoint=checkpoint,
            suite=self.args.out / "suite.json", out=self.root / name, family="flex", timeout=10,
            eager=False, prepare_only=prepare_only)

    def test_prepare_only_preserves_family_core_and_never_queries_gpu(self):
        source = exact.read(self.config)
        cases = [(BINARIES / "default", number, family) for number, family in exact.FAMILIES.items()]
        cases += [(BINARIES / "default", 0, "tiny"), (BINARIES / "impala", 0, "impala"),
                  (BINARIES / "impoola", 0, "impoola")]
        for case, (binary, encoder, family) in enumerate(cases):
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config = copy.deepcopy(source)
                config.set("policy", "encoder", str(encoder)); config.set("policy", "hidden_size", str(hidden))
                config.set("policy", "num_layers", str(layers))
                config.set("env", "representation_mode", "1"); config.set("env", "representation_seed", "12345")
                path = self.root / f"config-{case}-{hidden}.ini"
                with path.open("x") as stream: config.write(stream)
                args = self.run_args(f"prepared-{case}-{hidden}", prepare_only=True)
                args.binary, args.config, args.family = binary, path, family
                args.eager = hidden == 32
                identity = exact.info(binary)
                with patch.object(exact, "info", return_value=identity), \
                     patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No GPU queries in preparation")), \
                     patch.object(exact, "process", side_effect=AssertionError("No policy process in preparation")):
                    exact.run(args)
                effective = exact.read(args.out / "config/default.ini")
                self.assertEqual(dict(effective["policy"]), dict(config["policy"]))
                self.assertEqual(effective.getint("env", "representation_mode"), 0)
                self.assertEqual(effective.getint("env", "representation"), 3)
                self.assertEqual(effective.getint("eval_exact", "max_frames"), 5)
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if args.eager else 1)
                record = json.loads((args.out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertFalse(record["gpu_runtime_qualified"])
                self.assertFalse((args.out / "episodes.csv").exists())
                self.assertFalse((args.out / "REPORT.md").exists())
                for name, digest in record["tooling_sha256"].items():
                    self.assertEqual(exact.sha(args.out / "tooling" / name), digest)
        with self.assertRaises(FileExistsError): exact.run(args)

    def test_state_config_and_preparation(self):
        config = exact.read(self.config)
        config.set("base", "env_name", "breakout")
        # A state build ignores custom CNN selectors, matching native behavior.
        config.set("policy", "encoder", "5")
        config.set("policy", "hidden_size", "256"); config.set("policy", "num_layers", "2")
        state_config = self.root / "state.ini"
        with state_config.open("x") as stream: config.write(stream)
        suite_args = copy.copy(self.args)
        suite_args.binary, suite_args.config = BINARIES / "state", state_config
        suite_args.representation, suite_args.out = 0, self.root / "state-suite"
        exact.create_suite(suite_args)
        args = self.run_args("prepared-state", prepare_only=True)
        args.binary, args.config, args.suite, args.family = suite_args.binary, state_config, suite_args.out / "suite.json", "state"
        identity = exact.info(args.binary)
        with patch.object(exact, "info", return_value=identity), \
             patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No GPU query")):
            exact.run(args)
        effective = exact.read(args.out / "config/default.ini")
        self.assertEqual(dict(effective["policy"]), dict(config["policy"]))
        self.assertTrue((args.out / "config/breakout.ini").exists())

    def test_bad_weights_busy_gpu_and_query_failures_never_launch_policy(self):
        identity = exact.info(BINARIES / "default")
        for name, contents in (("empty", b""), ("malformed", b"\0" * 3), ("nonfinite", bytes.fromhex("0000c07f"))):
            args = self.run_args(name); args.checkpoint.write_bytes(contents)
            with patch.object(exact, "info", return_value=identity), \
                 patch.object(exact.subprocess, "check_output", side_effect=AssertionError("Weights before GPU")), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(Exception): exact.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args.checkpoint.write_bytes(b"\0" * 16)
        for name, answers in (("busy", ["12345\n"]), ("query-failed", [subprocess.CalledProcessError(1, ["smi"])]),
                              ("no-hardware", ["", ""])):
            args = self.run_args(name)
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=answers), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(Exception): exact.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def test_lock_conflict_blocks_before_gpu_query(self):
        args = self.run_args("locked"); identity = exact.info(args.binary)
        lock_dir = self.root / "build/connect4cnn"; lock_dir.mkdir(parents=True)
        with (lock_dir / "hardware-benchmark.lock").open("a") as lock:
            exact.fcntl.flock(lock, exact.fcntl.LOCK_EX | exact.fcntl.LOCK_NB)
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=AssertionError("No query while reserved")), \
                 patch.object(exact, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(BlockingIOError): exact.run(args)
        self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def fake_native(self, command, cwd, log, timeout):
        rows = self.results()
        counts = self.audit(rows)
        log.write_text("synthetic audit fixture, not GPU execution\n"
            f"BREAKOUT_EXACT_EVAL version=1 requested=65 evaluated=65 completed={counts['completed']} "
            f"capped={counts['capped']} score={counts['score']} params=4\n")
        with (cwd / "episodes.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=exact.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        result = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
        log.with_suffix(".log.json").write_text(json.dumps(result))
        return result

    def test_supervised_reporting_requires_matching_native_summary_and_quota(self):
        args = self.run_args("completed"); identity = exact.info(args.binary)
        with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
             patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(exact, "process", side_effect=self.fake_native):
            exact.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertEqual(record["counts"]["episodes"], 65)
        self.assertFalse(record["gpu_runtime_qualified"])
        for case in ("missing-row", "wrong-params", "wrong-summary", "changed-config", "changed-copied-suite",
                     "missing-clock", "backwards-clock"):
            args = self.run_args(case)
            def corrupt(command, cwd, log, timeout):
                result = self.fake_native(command, cwd, log, timeout)
                if case == "missing-row":
                    rows = (cwd / "episodes.csv").read_text().splitlines()
                    (cwd / "episodes.csv").write_text("\n".join(rows[:-1]) + "\n")
                elif case == "wrong-params": log.write_text(log.read_text().replace("params=4", "params=5"))
                elif case == "wrong-summary": log.write_text(log.read_text().replace("requested=65", "requested=64"))
                elif case == "changed-config": (cwd / "config/breakoutcnn.ini").write_text("# tampered\n")
                elif case == "changed-copied-suite":
                    path = cwd / "suite/suite.json"
                    data = json.loads(path.read_text()); data["purpose"] = "heldout"
                    path.write_text(json.dumps(data))
                elif case == "missing-clock": log.with_suffix(".log.json").unlink()
                else:
                    result["end_monotonic_ns"] = 0
                    log.with_suffix(".log.json").write_text(json.dumps(result))
                return result
            with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
                 patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
                 patch.object(exact, "process", side_effect=corrupt):
                with self.assertRaises(Exception): exact.run(args)
            record = json.loads((args.out / "result.json").read_text())
            self.assertEqual(record["status"], "failed"); self.assertNotIn("counts", record)
            self.assertFalse((args.out / "REPORT.md").exists())
            self.assertTrue((args.out / "episodes.csv").exists())

    def test_timeout_retains_partial_rows_and_process_clock(self):
        args = self.run_args("timeout"); identity = exact.info(args.binary)
        def failure(command, cwd, log, timeout):
            self.fake_native(command, cwd, log, timeout)
            rows = (cwd / "episodes.csv").read_text().splitlines()
            (cwd / "episodes.csv").write_text("\n".join(rows[:2]) + "\n")
            raise subprocess.TimeoutExpired(command, timeout)
        with patch.object(exact, "info", return_value=identity), patch.object(exact, "ROOT", self.root), \
             patch.object(exact.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(exact, "process", side_effect=failure):
            with self.assertRaises(subprocess.TimeoutExpired): exact.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertNotIn("counts", record)
        self.assertFalse((args.out / "REPORT.md").exists())
        self.assertEqual(len((args.out / "episodes.csv").read_text().splitlines()), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binaries", type=Path, default=BINARIES)
    args, rest = parser.parse_known_args(); BINARIES = args.binaries.resolve()
    unittest.main(argv=[__file__] + rest)
