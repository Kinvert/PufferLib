"""Suite/audit and native host-spawn tests. Never executes a policy on CPU/GPU."""
import argparse
import configparser
import copy
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exact_eval as ev

BINARY_DIR = ev.ROOT / "build/flappycnn/exact-preparation-20261005-v2"


class ExactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = configparser.ConfigParser(interpolation=None)
        config.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/flappycnn.ini",
                     ev.ROOT / "ocean/flappycnn/compare.ini"])
        self.config = self.root / "source.ini"
        with self.config.open("w") as stream: config.write(stream)
        self.args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, out=self.root / "suite",
                                       seed=2**32-1, offset=31, episodes=65, slots=64, representation=3,
                                       purpose="development")
        ev.create_suite(self.args)
        self.suite_path = self.args.out / "suite.json"
        self.suite, self.manifest = ev.load_suite(self.suite_path)

    def write_rows(self, rows):
        path = self.root / "results.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return path

    def receipts(self):
        return [dict(**row, version=1, block_seed=self.suite["seed"],
                     slot=(int(row["episode_id"])-self.suite["offset"]) % self.suite["slots"],
                     decisions=4096 if i % 2 else 8, pipes=1, perf=float(ev.f32(.05)).hex(),
                     episode_return=float(ev.f32(-.9)).hex(), cap_reached=i % 2,
                     action_hash="1234567890abcdef") for i, row in enumerate(self.manifest)]

    def test_exact_quotas_caps_and_corruptions(self):
        rows = self.receipts()
        counts = ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        self.assertEqual(counts["episodes"], 65)
        self.assertEqual(counts["pipes"], 65)
        self.assertEqual(counts["capped"], 32)
        self.assertEqual(counts["mean_pipes"], 1)
        for corrupted in (rows[:-1], rows+[rows[0]], [rows[0]]+rows[:-1]):
            with self.assertRaises(ValueError): ev.audit_csv(self.write_rows(corrupted), self.suite, self.manifest)
        for key, value in (("start_rng", "0"), ("start_world_hash", "0"*16),
                           ("start_observation_hash", "0"*16), ("policy_seed", "0"),
                           ("representation", "0"), ("slot", 99), ("cap_reached", 1),
                           ("pipes", -1), ("decisions", 4097), ("perf", "nan"),
                           ("episode_return", "inf"), ("action_hash", "bad")):
            corrupted = [dict(row) for row in rows]; corrupted[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ev.audit_csv(self.write_rows(corrupted), self.suite, self.manifest)

    def test_suite_tamper_and_wrong_rules(self):
        manifest = self.suite_path.parent / "episodes.csv"
        changed = manifest.read_text().replace("31,", "32,", 1)
        manifest.write_text(changed)
        with self.assertRaises(ValueError): ev.load_suite(self.suite_path)
        self.suite["episodes_sha256"] = ev.sha(manifest)
        self.suite_path.write_text(json.dumps(self.suite))
        with self.assertRaises(ValueError): ev.load_suite(self.suite_path)
        suite = dict(self.suite, rules="connect4-full-board-draw-v2")
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        suite = dict(self.suite, environment="flappy", representation=3)
        with self.assertRaises(ValueError): ev.validate_suite(suite)
        with self.assertRaises(FileExistsError): ev.create_suite(self.args)

    def test_source_configuration_bytes_are_checked(self):
        path = self.suite_path.parent / "source.ini"
        with path.open("a") as stream: stream.write("\n# Changed after preparation\n")
        with self.assertRaisesRegex(ValueError, "source.ini"): ev.load_suite(self.suite_path)

    def test_manifest_configuration_bytes_are_checked(self):
        path = self.suite_path.parent / "manifest.ini"
        with path.open("a") as stream: stream.write("\n# Changed after preparation\n")
        with self.assertRaisesRegex(ValueError, "manifest.ini"): ev.load_suite(self.suite_path)

    def test_rehashed_source_world_drift_is_rejected(self):
        path = self.suite_path.parent / "source.ini"
        config = ev.read(path); config["env"]["gravity"] = "2"
        with path.open("w") as stream: config.write(stream)
        suite = dict(self.suite, source_ini_sha256=ev.sha(path))
        self.suite_path.write_text(json.dumps(suite))
        with self.assertRaisesRegex(ValueError, "world"): ev.load_suite(self.suite_path)

    def test_all_native_builds_and_policy_sizes_have_identical_spawns(self):
        ini = ev.read(self.suite_path.parent / "manifest.ini")
        for family in ("default", "impala", "impoola"):
            directory = self.root / family; directory.mkdir()
            ini["policy"].update(hidden_size="512", num_layers="4", encoder="4")
            ev.native_manifest(BINARY_DIR / family, ini, directory / "episodes.csv", directory)
            self.assertEqual((directory / "episodes.csv").read_bytes(),
                             (self.suite_path.parent / "episodes.csv").read_bytes())
        directory = self.root / "state"; directory.mkdir()
        ev.native_manifest(BINARY_DIR / "state", ini, directory / "episodes.csv", directory)
        with (directory / "episodes.csv").open() as stream: state = list(csv.DictReader(stream))
        for pixel, original in zip(self.manifest, state, strict=True):
            for key in ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash"):
                self.assertEqual(pixel[key], original[key])
            self.assertNotEqual(pixel["start_observation_hash"], original["start_observation_hash"])

    def test_last_wave_and_slot_count_do_not_change_episode_spawns(self):
        ini = ev.read(self.suite_path.parent / "manifest.ini")
        ini["eval_exact"].update(episodes="1", episode_offset="95", slots="1")
        directory = self.root / "tail"; directory.mkdir()
        ev.native_manifest(BINARY_DIR / "default", ini, directory / "episodes.csv", directory)
        with (directory / "episodes.csv").open() as stream: rows = list(csv.DictReader(stream))
        self.assertEqual(rows, self.manifest[-1:])

    def test_world_and_core_sizes_are_preserved(self):
        config = ev.read(self.config)
        for encoder, family in {0: "tiny", **ev.FAMILIES}.items():
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config["policy"].update(encoder=str(encoder), hidden_size=str(hidden), num_layers=str(layers))
                with self.config.open("w") as stream: config.write(stream)
                loaded = ev.load_config(self.config, ev.info(BINARY_DIR / "default"), family)
                self.assertEqual(loaded.getint("policy", "hidden_size"), hidden)
                self.assertEqual(loaded.getint("policy", "num_layers"), layers)
                self.assertEqual(ev.world(loaded), self.suite["world"])
        config["env"]["gravity"] = "nan"
        with self.assertRaises(ValueError): ev.world(config)

    def test_native_host_manifest_rejects_invalid_physics_before_reset(self):
        original = ev.read(self.suite_path.parent / "manifest.ini")
        for index, (key, value) in enumerate((("height", "1.5"), ("max_steps", "0"),
                                            ("gravity", "nan"), ("pipe_gap", "1000"),
                                            ("bird_x", "1e300"))):
            config = configparser.ConfigParser(interpolation=None)
            config.read_dict({s: dict(original[s]) for s in original.sections()})
            config["env"][key] = value
            path = self.root / f"invalid-{index}.ini"
            with path.open("w") as stream: config.write(stream)
            result = ev.subprocess.run([str(BINARY_DIR / "default"), "eval_exact_manifest", str(path)],
                                       capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Flappy exact evaluation:", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_busy_gpu_preserves_failure_without_starting_policy(self):
        checkpoint = self.root / "weights.bin"; checkpoint.write_bytes(b"\0" * 16)
        args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                  suite=self.suite_path, out=self.root / "eval", family="flex", timeout=10, eager=False, prepare_only=False)
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", return_value="1234\n"), \
             patch.object(ev, "process") as execute:
            with self.assertRaisesRegex(ValueError, "GPU busy"): ev.run(args)
            execute.assert_not_called()
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertNotIn("counts", record)
        self.assertFalse((args.out / "episodes.csv").exists())

    def test_malformed_and_nonfinite_weights_fail_before_gpu_query(self):
        for index, contents in enumerate((b"bad", bytes.fromhex("0000c07f"))):
            checkpoint = self.root / f"bad-{index}.bin"; checkpoint.write_bytes(contents)
            args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                      suite=self.suite_path, out=self.root / f"bad-eval-{index}", family="flex", timeout=10, eager=False, prepare_only=False)
            identity = ev.info(args.binary)
            with patch.object(ev, "info", return_value=identity), \
                 patch.object(ev.subprocess, "check_output") as query:
                with self.assertRaises(ValueError): ev.run(args)
                query.assert_not_called()
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def test_failed_process_keeps_partial_rows_and_monotonic_duration(self):
        checkpoint = self.root / "weights.bin"; checkpoint.write_bytes(b"\0" * 16)
        args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                  suite=self.suite_path, out=self.root / "failed", family="flex", timeout=10, eager=False, prepare_only=False)
        identity = ev.info(args.binary)
        def failure(command, cwd, log, timeout):
            log.write_text("synthetic timeout fixture, not GPU execution\n")
            log.with_suffix(".log.json").write_text(json.dumps(dict(launch_monotonic_ns=10, end_monotonic_ns=1000000010)))
            with (cwd / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS)
                writer.writeheader(); writer.writerow(self.receipts()[0])
            raise ValueError("timeout fixture")
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "test-only-hardware"]), \
             patch.object(ev, "process", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "timeout fixture"): ev.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["evaluation_process_seconds"], 1)
        self.assertNotIn("counts", record)
        self.assertTrue((args.out / "episodes.csv").exists())
        self.assertFalse((args.out / "REPORT.md").exists())

    def test_completed_process_requires_matching_summary_and_weight_size(self):
        checkpoint = self.root / "weights.bin"; checkpoint.write_bytes(b"\0" * 16)
        args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                  suite=self.suite_path, out=self.root / "completed", family="flex", timeout=10, eager=False, prepare_only=False)
        identity = ev.info(args.binary)
        def completion(command, cwd, log, timeout):
            log.write_text("synthetic audit fixture, not GPU execution\n"
                           "FLAPPY_EXACT_EVAL version=1 requested=65 completed=65 pipes=65 capped=32 params=4\n")
            data = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
            log.with_suffix(".log.json").write_text(json.dumps(data))
            with (cwd / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS)
                writer.writeheader(); writer.writerows(self.receipts())
            return data
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "test-only-hardware"]), \
             patch.object(ev, "process", side_effect=completion):
            ev.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertEqual(record["counts"]["episodes"], 65)
        self.assertEqual(record["counts"]["capped"], 32)

    def test_rehashed_manifest_config_cannot_change_native_cap_or_core(self):
        path = self.suite_path.parent / "manifest.ini"
        original = ev.read(path)
        for section, key, value in (("eval_exact", "max_steps", "8"), ("env", "gravity", "2"),
                                     ("policy", "hidden_size", "512")):
            config = copy.deepcopy(original); config[section][key] = value
            with path.open("w") as stream: config.write(stream)
            suite = dict(self.suite, manifest_ini_sha256=ev.sha(path))
            self.suite_path.write_text(json.dumps(suite))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "configuration differs"):
                ev.load_suite(self.suite_path)

    def test_legacy_config_closure_stays_explicit_and_rejects_drift(self):
        legacy = dict(self.suite)
        for key in ("configuration_receipt_version", "manifest_ini_sha256", "configuration_integrity"):
            legacy.pop(key, None)
        self.suite_path.write_text(json.dumps(legacy))
        suite, manifest = ev.load_suite(self.suite_path)
        self.assertEqual(suite["configuration_integrity"], "legacy-derived-config-no-original-manifest-hash")
        self.assertEqual(manifest, self.manifest)
        config = ev.read(self.suite_path.parent / "manifest.ini")
        config["eval_exact"]["max_steps"] = "8"
        with (self.suite_path.parent / "manifest.ini").open("w") as stream: config.write(stream)
        with self.assertRaisesRegex(ValueError, "configuration differs"): ev.load_suite(self.suite_path)
        for version in (0, 3, True):
            self.suite_path.write_text(json.dumps(dict(legacy, configuration_receipt_version=version)))
            with self.assertRaisesRegex(ValueError, "configuration receipt version"): ev.load_suite(self.suite_path)

    def run_args(self, name="run", prepare_only=False):
        checkpoint = self.root / "weights.bin"
        if not checkpoint.exists(): checkpoint.write_bytes(b"\0" * 16)
        return argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                  suite=self.suite_path, out=self.root / name, family="flex", timeout=10,
                                  eager=False, prepare_only=prepare_only)

    def test_prepare_only_preserves_24_family_core_configs_without_gpu(self):
        source = ev.read(self.config)
        cases = [(BINARY_DIR / "default", encoder, family) for encoder, family in ev.FAMILIES.items()]
        cases += [(BINARY_DIR / "default", 0, "tiny"), (BINARY_DIR / "impala", 0, "impala"),
                  (BINARY_DIR / "impoola", 0, "impoola")]
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
                self.assertEqual(effective.getint("env", "representation"), 3)
                self.assertEqual(effective.getint("env", "representation_mode"), 0)
                self.assertEqual(effective.getint("eval_exact", "max_steps"), 4096)
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if args.eager else 1)
                record = json.loads((args.out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertFalse(record["gpu_runtime_qualified"])
                self.assertEqual(record["training_appearance"]["representation_mode"], "1")
                self.assertEqual(record["suite_configuration_integrity"], "sha256-and-derived-config")
                self.assertFalse((args.out / "episodes.csv").exists()); self.assertFalse((args.out / "REPORT.md").exists())
                for name, digest in record["tooling_sha256"].items(): self.assertEqual(ev.sha(args.out / "tooling" / name), digest)
        with self.assertRaises(FileExistsError): ev.run(args)

    def test_state_preparation_and_mismatched_family_world_reject(self):
        config = ev.read(self.config); config["base"]["env_name"] = "flappy"
        config["policy"].update(encoder="5", hidden_size="256", num_layers="2")
        for key in ("representation", "representation_mode", "representation_seed"): del config["env"][key]
        path = self.root / "state.ini"
        with path.open("w") as stream: config.write(stream)
        suite_args = copy.copy(self.args)
        suite_args.binary, suite_args.config, suite_args.out, suite_args.representation = BINARY_DIR / "state", path, self.root / "state-suite", 0
        ev.create_suite(suite_args)
        args = self.run_args("state-prepared", prepare_only=True)
        args.binary, args.config, args.suite, args.family = suite_args.binary, path, suite_args.out / "suite.json", "state"
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No GPU queries")), \
             patch.object(ev, "process", side_effect=AssertionError("No policy execution")):
            ev.run(args)
        self.assertEqual(dict(ev.read(args.out / "config/default.ini")["policy"]), dict(config["policy"]))
        self.assertTrue((args.out / "config/flappy.ini").exists())
        args.out = self.root / "wrong-state-family"; args.family = "flex"
        with self.assertRaisesRegex(ValueError, "family mismatch"): ev.run(args)
        args = self.run_args("wrong-family", prepare_only=True); args.family = "nature"
        with self.assertRaisesRegex(ValueError, "construct"): ev.run(args)
        args = self.run_args("world-drift", prepare_only=True)
        config = ev.read(self.config); config["env"]["gravity"] = "2"
        with self.config.open("w") as stream: config.write(stream)
        with self.assertRaisesRegex(ValueError, "physics or native cap differs"): ev.run(args)
        self.assertFalse(args.out.exists())

    def test_reservation_empty_weights_and_query_failure_do_not_launch(self):
        identity = ev.info(BINARY_DIR / "default")
        args = self.run_args("empty"); args.checkpoint.write_bytes(b"")
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No query")), \
             patch.object(ev, "process", side_effect=AssertionError("No policy")):
            with self.assertRaises(ValueError): ev.run(args)
        self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args.checkpoint.write_bytes(b"\0"*16)
        for name, answers in (("query-failed", [ev.subprocess.CalledProcessError(1, ["smi"])]), ("no-device", ["", ""])):
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

    def fake_native(self, command, cwd, log, timeout):
        log.write_text("synthetic fixture, no policy execution\n"
                       "FLAPPY_EXACT_EVAL version=1 requested=65 completed=65 pipes=65 capped=32 params=4\n")
        timing = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
        log.with_suffix(".log.json").write_text(json.dumps(timing))
        with (cwd / "episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS); writer.writeheader(); writer.writerows(self.receipts())
        return timing

    def synthetic_run(self, args, fake):
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "synthetic-device,uuid,driver\n"]), \
             patch.object(ev, "process", side_effect=fake):
            ev.run(args)

    def test_bad_summary_quota_parameters_and_clock_fail_without_report(self):
        for label in ("summary", "duplicate", "parameters", "quota", "clock", "no-clock"):
            args = self.run_args("bad-"+label)
            def fake(command, cwd, log, timeout):
                timing = self.fake_native(command, cwd, log, timeout)
                if label == "summary": log.write_text(log.read_text().replace("pipes=65", "pipes=64"))
                elif label == "duplicate": log.write_text(log.read_text()*2)
                elif label == "parameters": log.write_text(log.read_text().replace("params=4", "params=5"))
                elif label == "quota":
                    with (cwd / "episodes.csv").open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS); writer.writeheader(); writer.writerows(self.receipts()[:-1])
                elif label == "clock":
                    timing["end_monotonic_ns"] = 1; log.with_suffix(".log.json").write_text(json.dumps(timing))
                elif label == "no-clock": log.with_suffix(".log.json").unlink()
                return timing
            with self.assertRaises(ValueError): self.synthetic_run(args, fake)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
            self.assertFalse((args.out / "REPORT.md").exists())

    def test_effective_and_copied_inputs_and_tooling_changes_fail(self):
        for label in ("checkpoint", "effective", "env-overlay", "training-copy", "suite-copy", "manifest-copy",
                      "source-copy", "manifest-ini-copy", "source-original", "tooling", "spawn", "spawn-ini"):
            args = self.run_args("changed-"+label)
            original_source = (self.suite_path.parent / "source.ini").read_bytes()
            def fake(command, cwd, log, timeout):
                result = self.fake_native(command, cwd, log, timeout)
                target = {"checkpoint": args.checkpoint, "effective": cwd / "config/default.ini",
                          "env-overlay": cwd / "config/flappycnn.ini", "training-copy": cwd / "training.ini",
                          "suite-copy": cwd / "suite/suite.json", "manifest-copy": cwd / "suite/episodes.csv",
                          "source-copy": cwd / "suite/source.ini", "manifest-ini-copy": cwd / "suite/manifest.ini",
                          "source-original": self.suite_path.parent / "source.ini",
                          "tooling": cwd / "tooling/ocean/flappycnn/exact_eval.py",
                          "spawn": cwd / "spawn-check.csv", "spawn-ini": cwd / "manifest.ini"}[label]
                with target.open("ab") as stream: stream.write(b"\0" if label == "checkpoint" else b"\n")
                return result
            with self.assertRaises(ValueError): self.synthetic_run(args, fake)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
            self.assertFalse((args.out / "REPORT.md").exists())
            args.checkpoint.write_bytes(b"\0"*16)
            (self.suite_path.parent / "source.ini").write_bytes(original_source)

    def test_audit_aggregates_are_order_independent_and_not_certification(self):
        rows = self.receipts()
        for i, reward in enumerate((ev.f32(1e20), 3, ev.f32(-1e20))): rows[i]["episode_return"] = float(reward).hex()
        forward = ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        reverse = ev.audit_csv(self.write_rows(list(reversed(rows))), self.suite, self.manifest)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["mean_return"], math.fsum(float.fromhex(r["episode_return"]) for r in rows)/65)
        self.assertFalse(forward["policy_runtime_certified"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binaries", type=Path, default=BINARY_DIR)
    args, remaining = parser.parse_known_args()
    BINARY_DIR = args.binaries.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
