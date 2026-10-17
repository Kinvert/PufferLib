"""Native host manifests and synthetic receipt audits only; no neural policy."""
import argparse
import configparser
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exact_eval as ev

BINARIES = ev.ROOT / "build/mazecnn/exact-native-20261006"


class ExactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = configparser.ConfigParser(interpolation=None)
        config.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/mazecnn.ini", ev.ROOT / "ocean/mazecnn/compare.ini"])
        config["env"].update(num_maps="32", map_size="-1")
        self.config = self.root / "source.ini"
        with self.config.open("w") as stream: config.write(stream)
        self.args = argparse.Namespace(binary=BINARIES / "default", config=self.config, out=self.root / "suite",
            seed=56111, offset=31, episodes=65, slots=64, representation=0, purpose="development",
            level_offset=3, level_count=29, training_level_count=0)
        ev.create_suite(self.args); self.path = self.args.out / "suite.json"
        self.suite, self.manifest = ev.load_suite(self.path)

    def native_manifest(self, binary, config, output):
        ini = output.with_suffix(".ini")
        with ini.open("w") as stream: config.write(stream)
        with output.open("w") as stream:
            ev.subprocess.run([str(binary), "eval_exact_manifest", str(ini)], stdout=stream,
                              stderr=ev.subprocess.PIPE, text=True, check=True, timeout=30)
        with output.open() as stream: return list(csv.DictReader(stream))

    def test_start_manifest_is_policy_core_appearance_and_slot_independent(self):
        config = ev.read(self.path.parent / "manifest.ini")
        reference = self.manifest
        for family in ("default", "impala", "impoola"):
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config["policy"].update(hidden_size=str(hidden), num_layers=str(layers), encoder="4")
                rows = self.native_manifest(BINARIES / family, config, self.root / f"{family}-{hidden}.csv")
                self.assertEqual(rows, reference)
        common = [key for key in ev.MANIFEST_FIELDS if key not in ("representation", "start_observation_hash")]
        for representation in range(6):
            config["env"]["representation"] = str(representation)
            rows = self.native_manifest(BINARIES / "default", config, self.root / f"r{representation}.csv")
            for a, b in zip(rows, reference, strict=True): self.assertEqual([a[k] for k in common], [b[k] for k in common])
        config["base"]["env_name"] = "maze"; config["env"]["representation"] = "0"
        rows = self.native_manifest(BINARIES / "state", config, self.root / "state.csv")
        for a, b in zip(rows, reference, strict=True): self.assertEqual([a[k] for k in common], [b[k] for k in common])
        config = ev.read(self.path.parent / "manifest.ini")
        config["eval_exact"].update(episode_offset="95", episodes="1", slots="1")
        self.assertEqual(self.native_manifest(BINARIES / "default", config, self.root / "tail.csv"), reference[-1:])

    def test_cyclic_coverage_and_declared_heldout_prefix(self):
        levels = [int(row["level_id"]) for row in self.manifest]
        self.assertEqual(set(levels[:29]), set(range(3, 32)))
        suite = dict(self.suite, purpose="heldout", training_level_count=4)
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        suite["training_level_count"] = 3; ev.validate_suite(suite)
        suite["training_level_count"] = 0
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        self.assertEqual(self.suite["level_exclusion_status"], "declared-prefix-not-verified-against-training-artifacts")

    def test_bad_native_world_panel_and_mixed_inputs_are_rejected(self):
        original = ev.read(self.path.parent / "manifest.ini")
        for i, (section, key, value) in enumerate((("env", "num_maps", "0"), ("env", "map_size", "nan"),
                ("env", "map_size", "1e100"), ("env", "map_size", "11.5"),
                ("env", "representation_mode", "1"), ("eval_exact", "level_offset", "31"),
                ("eval_exact", "level_count", "0"), ("eval_exact", "episodes", "0"),
                ("eval_exact", "episode_offset", "4294967295"))):
            config = configparser.ConfigParser(interpolation=None)
            config.read_dict({s: dict(original[s]) for s in original.sections()}); config[section][key] = value
            path = self.root / f"bad-{i}.ini"
            with path.open("w") as stream: config.write(stream)
            result = ev.subprocess.run([str(BINARIES / "default"), "eval_exact_manifest", str(path)],
                                      capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0); self.assertIn("Maze exact evaluation:", result.stderr)

    def test_source_config_hashes_and_protocol_closure(self):
        suite = dict(self.suite, rules="snake-local-episodic-v1")
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        suite = dict(self.suite, environment="maze", representation=1)
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        with self.assertRaises(FileExistsError): ev.create_suite(self.args)
        manifest = self.path.parent / "episodes.csv"
        rows = [dict(row) for row in self.manifest]; rows[0]["level_id"] = "2"
        self.write(manifest, rows, ev.MANIFEST_FIELDS)
        with self.assertRaises(ValueError): ev.load_suite(self.path)
        self.suite["episodes_sha256"] = ev.sha(manifest)
        self.path.write_text(json.dumps(self.suite))
        with self.assertRaises(ValueError): ev.load_suite(self.path)

    def rows(self):
        rows = []
        for i, start in enumerate(self.manifest):
            row = {k: v for k, v in start.items() if k != "terminal_rng"}
            success = i % 2; horizon = int(start["horizon"])
            decisions = horizon if i % 3 == 0 or not success else 2 * (int(start["width"]) - 3)
            row.update(version="1", block_seed=str(self.suite["seed"]), slot=str(i % 64),
                decisions=str(decisions), success=str(success), episode_return=float(success).hex(),
                end_reason=str(1 if success else 2), native_log_entries=str(1 + (success and decisions == horizon)),
                end_rng=start["terminal_rng"], action_hash="0123456789abcdef")
            rows.append(row)
        return rows

    def write(self, path, rows, fields=ev.RESULT_FIELDS):
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def test_complete_all_assigned_audit_handles_duplicates_without_double_counting(self):
        rows = self.rows(); path = self.root / "audit.csv"; self.write(path, rows)
        result = ev.audit_csv(path, self.suite, self.manifest)
        self.assertEqual(result["episodes"], 65); self.assertEqual(result["successes"], 32)
        self.assertEqual(result["timeouts"], 33); self.assertEqual(result["success_fraction"], 32/65)
        self.assertGreater(result["duplicate_native_log_entries"], 0)
        self.assertFalse(result["policy_runtime_certified"]); self.assertFalse(result["heldout_exclusion_certified"])

    def test_incomplete_extra_wrong_and_malformed_receipts_fail(self):
        original = self.rows(); path = self.root / "bad.csv"
        mutations = [("slot", "65"), ("end_rng", "0"), ("success", "2"), ("native_log_entries", "2"),
                     ("episode_return", "nan"), ("decisions", "1"), ("end_reason", "1"),
                     ("horizon", "1"), ("action_hash", "zzz"), ("representation", "5")]
        for key, value in mutations:
            rows = [dict(row) for row in original]; rows[0][key] = value; self.write(path, rows)
            with self.assertRaises(ValueError, msg=key): ev.audit_csv(path, self.suite, self.manifest)
        for rows in (original[:-1], original + original[:1]):
            self.write(path, rows)
            with self.assertRaises(ValueError): ev.audit_csv(path, self.suite, self.manifest)

    def run_args(self, name="run", prepare_only=False):
        checkpoint = self.root / "weights.bin"
        if not checkpoint.exists(): checkpoint.write_bytes(b"\0" * 16)
        return argparse.Namespace(binary=BINARIES / "default", config=self.config, checkpoint=checkpoint,
            suite=self.path, out=self.root / name, family="flex", timeout=10, eager=False, prepare_only=prepare_only)

    def test_prepare_only_preserves_families_core_and_fixed_level_panel_without_gpu(self):
        source = ev.read(self.config)
        cases = [(BINARIES / "default", encoder, family) for encoder, family in ev.FAMILIES.items()]
        cases += [(BINARIES / "default", 0, "tiny"), (BINARIES / "impala", 0, "impala"),
                  (BINARIES / "impoola", 0, "impoola")]
        for case, (binary, encoder, family) in enumerate(cases):
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config = copy.deepcopy(source)
                config["policy"].update(encoder=str(encoder), hidden_size=str(hidden), num_layers=str(layers))
                config["env"].update(representation_mode="1", representation_seed="12345")
                path = self.root / f"config-{case}-{hidden}.ini"
                with path.open("w") as stream: config.write(stream)
                args = self.run_args(f"prepared-{case}-{hidden}", prepare_only=True)
                args.binary, args.config, args.family = binary, path, family; args.eager = hidden == 32
                identity = ev.info(binary)
                with patch.object(ev, "info", return_value=identity), \
                     patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No GPU queries")), \
                     patch.object(ev, "process", side_effect=AssertionError("No policy execution")):
                    ev.run(args)
                effective = ev.read(args.out / "config/default.ini")
                self.assertEqual(dict(effective["policy"]), dict(config["policy"]))
                self.assertEqual(ev.world(effective), self.suite["world"])
                self.assertEqual(effective.getint("env", "representation_mode"), 0)
                self.assertEqual(effective.getint("eval_exact", "level_offset"), 3)
                self.assertEqual(effective.getint("eval_exact", "level_count"), 29)
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if args.eager else 1)
                self.assertNotIn("max_decisions", effective["eval_exact"])
                record = json.loads((args.out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertEqual(record["training_appearance"]["representation_mode"], "1")
                self.assertFalse(record["gpu_runtime_qualified"])
                self.assertFalse(record["heldout_exclusion_certified"])
                self.assertFalse((args.out / "episodes.csv").exists())
                self.assertFalse((args.out / "REPORT.md").exists())
                for name, digest in record["tooling_sha256"].items():
                    self.assertEqual(ev.sha(args.out / "tooling" / name), digest)
        with self.assertRaises(FileExistsError): ev.run(args)

    def test_state_preparation_and_wrong_family_world_rejected(self):
        config = ev.read(self.config); config["base"]["env_name"] = "maze"
        config["policy"].update(encoder="5", hidden_size="256", num_layers="2")
        for key in ("representation", "representation_mode", "representation_seed"):
            del config["env"][key]
        path = self.root / "state.ini"
        with path.open("w") as stream: config.write(stream)
        suite_args = copy.copy(self.args)
        suite_args.binary, suite_args.config, suite_args.out = BINARIES / "state", path, self.root / "state-suite"
        ev.create_suite(suite_args)
        args = self.run_args("prepared-state", prepare_only=True)
        args.binary, args.config, args.suite, args.family = suite_args.binary, path, suite_args.out / "suite.json", "state"
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No queries")), \
             patch.object(ev, "process", side_effect=AssertionError("No policy")):
            ev.run(args)
        self.assertEqual(dict(ev.read(args.out / "config/default.ini")["policy"]), dict(config["policy"]))
        self.assertTrue((args.out / "config/maze.ini").exists())
        with self.assertRaises(ValueError): ev.checkpoint_config(path, identity, "flex")
        with self.assertRaises(ValueError): ev.checkpoint_config(self.config, ev.info(BINARIES / "default"), "nature")
        args = self.run_args("changed-table", prepare_only=True)
        config = ev.read(self.config); config["env"]["num_maps"] = "31"
        with self.config.open("w") as stream: config.write(stream)
        with self.assertRaisesRegex(ValueError, "game settings differ"): ev.run(args)
        self.assertFalse(args.out.exists())

    def test_heldout_prefix_override_is_exact_and_remains_uncertified(self):
        suite_args = copy.copy(self.args)
        suite_args.out = self.root / "heldout-suite"; suite_args.purpose = "heldout"
        suite_args.training_level_count = 3; ev.create_suite(suite_args)
        config = ev.read(self.config); config["env"]["num_maps"] = "3"
        path = self.root / "prefix.ini"
        with path.open("w") as stream: config.write(stream)
        args = self.run_args("prefix-prepared", prepare_only=True)
        args.config, args.suite = path, suite_args.out / "suite.json"
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No queries")), \
             patch.object(ev, "process", side_effect=AssertionError("No policy")):
            ev.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["training_world"]["num_maps"], 3)
        self.assertEqual(record["evaluation_world"]["num_maps"], 32)
        self.assertEqual(ev.read(args.out / "training.ini").getint("env", "num_maps"), 3)
        self.assertEqual(ev.read(args.out / "config/default.ini").getint("env", "num_maps"), 32)
        self.assertFalse(record["heldout_exclusion_certified"])
        heldout, _ = ev.load_suite(args.suite)
        for maps, size in ((4, -1), (32, -1), (3, 11)):
            changed = copy.deepcopy(config); changed["env"].update(num_maps=str(maps), map_size=str(size))
            with self.assertRaises(ValueError): ev.evaluation_world(changed, heldout)

    def test_bad_weights_busy_gpu_query_failure_and_reservation_never_launch(self):
        identity = ev.info(BINARIES / "default")
        for name, data in (("empty", b""), ("malformed", b"\0"*3), ("nonfinite", bytes.fromhex("0000c07f"))):
            args = self.run_args(name); args.checkpoint.write_bytes(data)
            with patch.object(ev, "info", return_value=identity), \
                 patch.object(ev.subprocess, "check_output", side_effect=AssertionError("Weights before queries")), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy")):
                with self.assertRaises(ValueError): ev.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args.checkpoint.write_bytes(b"\0"*16)
        for name, answers in (("busy", ["12345\n"]), ("query-failed", [ev.subprocess.CalledProcessError(1, ["smi"])]),
                              ("no-hardware", ["", ""])):
            args = self.run_args(name)
            with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
                 patch.object(ev.subprocess, "check_output", side_effect=answers), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy")):
                with self.assertRaises(Exception): ev.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args = self.run_args("reserved")
        with (self.root / "build/connect4cnn/hardware-benchmark.lock").open("a") as lock:
            ev.fcntl.flock(lock, ev.fcntl.LOCK_EX | ev.fcntl.LOCK_NB)
            with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
                 patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No queries while reserved")), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy")):
                with self.assertRaises(BlockingIOError): ev.run(args)
        self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def fake_native(self, command, cwd, log, timeout):
        rows = self.rows(); counts = ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        log.write_text("synthetic fixture, no policy execution\n"
            f"MAZE_EXACT_EVAL version=1 requested=65 completed=65 successes={counts['successes']} "
            f"timeouts={counts['timeouts']} params=4\n")
        self.write(cwd / "episodes.csv", rows)
        timing = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
        log.with_suffix(".log.json").write_text(json.dumps(timing))
        return timing

    def write_rows(self, rows):
        path = self.root / "fixture.csv"; self.write(path, rows); return path

    def synthetic_run(self, args, fake=None):
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "synthetic-device,uuid,driver\n"]), \
             patch.object(ev, "process", side_effect=fake or self.fake_native):
            ev.run(args)

    def test_synthetic_supervised_receipts_are_not_policy_certification(self):
        args = self.run_args("synthetic")
        self.synthetic_run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertEqual(record["counts"]["successes"], 32)
        self.assertEqual(record["counts"]["timeouts"], 33)
        self.assertGreater(record["counts"]["duplicate_native_log_entries"], 0)
        self.assertFalse(record["gpu_runtime_qualified"])
        self.assertFalse(record["heldout_exclusion_certified"])

    def test_bad_summary_quota_parameters_and_clock_rejected(self):
        for label in ("summary", "duplicate", "parameters", "quota", "clock", "no-clock"):
            args = self.run_args("bad-" + label)
            def fake(command, cwd, log, timeout):
                timed = self.fake_native(command, cwd, log, timeout)
                if label == "summary": log.write_text(log.read_text().replace("successes=32", "successes=31"))
                elif label == "duplicate": log.write_text(log.read_text()*2)
                elif label == "parameters": log.write_text(log.read_text().replace("params=4", "params=5"))
                elif label == "quota": self.write(cwd / "episodes.csv", self.rows()[:-1])
                elif label == "clock":
                    timed["end_monotonic_ns"] = 1; log.with_suffix(".log.json").write_text(json.dumps(timed))
                elif label == "no-clock": log.with_suffix(".log.json").unlink()
                return timed
            with self.assertRaises(ValueError): self.synthetic_run(args, fake)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
            self.assertFalse((args.out / "REPORT.md").exists())

    def test_immutable_inputs_and_tooling_changes_rejected(self):
        for name in ("checkpoint", "effective", "training-copy", "suite-copy", "manifest-copy", "tooling", "spawn"):
            args = self.run_args("changed-" + name)
            def fake(command, cwd, log, timeout):
                result = self.fake_native(command, cwd, log, timeout)
                target = {"checkpoint": args.checkpoint, "effective": cwd / "config/default.ini",
                          "training-copy": cwd / "training.ini", "suite-copy": cwd / "suite/suite.json",
                          "manifest-copy": cwd / "suite/episodes.csv", "tooling": cwd / "tooling/ocean/mazecnn/exact_eval.py",
                          "spawn": cwd / "spawn-check.csv"}[name]
                with target.open("ab") as stream: stream.write(b"\0" if name == "checkpoint" else b"\n")
                return result
            with self.assertRaises(ValueError): self.synthetic_run(args, fake)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
            self.assertFalse((args.out / "REPORT.md").exists())
            args.checkpoint.write_bytes(b"\0"*16)

    def test_timeout_keeps_partial_evidence_without_success_report(self):
        args = self.run_args("timeout")
        def fake(command, cwd, log, timeout):
            log.write_text("synthetic timeout, no policy execution\n")
            self.write(cwd / "episodes.csv", self.rows()[:1])
            log.with_suffix(".log.json").write_text(json.dumps(dict(launch_monotonic_ns=10, end_monotonic_ns=3000000010)))
            raise TimeoutError("synthetic process-group timeout")
        with self.assertRaises(TimeoutError): self.synthetic_run(args, fake)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed"); self.assertEqual(record["evaluation_process_seconds"], 3)
        self.assertNotIn("counts", record); self.assertTrue((args.out / "episodes.csv").exists())
        self.assertFalse((args.out / "REPORT.md").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binaries", type=Path)
    args, rest = parser.parse_known_args()
    if args.binaries: BINARIES = args.binaries.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
