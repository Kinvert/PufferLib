"""GPU-free native spawn and synthetic receipt checks; never constructs a policy."""
import argparse
import configparser
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exact_eval as ev

BINARY_DIR = ev.ROOT / "build/snakecnn/exact-preparation-20261005-v2"


class ExactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = configparser.ConfigParser(interpolation=None)
        config.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/snakecnn.ini",
                     ev.ROOT / "ocean/snakecnn/compare.ini"])
        self.config = self.root / "source.ini"
        self.write_ini(self.config, config)
        self.args = argparse.Namespace(binary=BINARY_DIR / "default", config=self.config,
                                       out=self.root / "suite", seed=2**32-1, offset=31,
                                       episodes=65, slots=64, representation=3, purpose="development")
        ev.create_suite(self.args)
        self.suite_path = self.args.out / "suite.json"
        self.suite, self.manifest = ev.load_suite(self.suite_path)

    def write_ini(self, path, config):
        with path.open("w") as stream: config.write(stream)

    def native_manifest(self, config, family="default", name="manifest"):
        path = self.root / (name+".ini"); self.write_ini(path, config)
        result = subprocess.run([str(BINARY_DIR / family), "eval_exact_manifest", str(path)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return list(csv.DictReader(result.stdout.splitlines()))

    def write_rows(self, rows):
        path = self.root / "results.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        return path

    def receipts(self):
        rows = []
        for i, start in enumerate(self.manifest):
            # Independent fixture accumulation, including food beyond growth cap.
            foods = i*3; reason = 1 if i % 2 == 0 else 2
            decisions = foods+1 if reason == 1 else int(self.suite["world"]["max_steps"])
            reward = 0.0
            for _ in range(foods): reward = ev.f32(reward+ev.f32(self.suite["world"]["reward_food"]))
            if reason == 1: reward = ev.f32(reward+ev.f32(self.suite["world"]["reward_death"]))
            score = min(foods+1, int(self.suite["world"]["max_snake_length"])-1)
            rows.append(dict(**start, version=1, block_seed=self.suite["seed"],
                             slot=(int(start["episode_id"])-self.suite["offset"]) % self.suite["slots"],
                             decisions=decisions, score=score, foods=foods, end_reason=reason,
                             perf=ev.f32(min(score/120, 1)).hex(), episode_return=reward.hex(),
                             action_hash="1234567890abcdef"))
        return rows

    def run_args(self, name="run", prepare_only=False):
        checkpoint = self.root / "weights.bin"
        if not checkpoint.exists(): checkpoint.write_bytes(b"\0" * 16)
        return argparse.Namespace(binary=BINARY_DIR / "default", config=self.config, checkpoint=checkpoint,
                                  suite=self.suite_path, out=self.root / name, family="flex", timeout=10,
                                  eager=False, prepare_only=prepare_only)

    def test_prepare_only_preserves_all_families_and_core_shapes_without_gpu(self):
        source = ev.read(self.config)
        cases = [(BINARY_DIR / "default", encoder, family) for encoder, family in ev.FAMILIES.items()]
        cases += [(BINARY_DIR / "default", 0, "tiny"), (BINARY_DIR / "impala", 0, "impala"),
                  (BINARY_DIR / "impoola", 0, "impoola")]
        for case, (binary, encoder, family) in enumerate(cases):
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                config = copy.deepcopy(source)
                config["policy"].update(encoder=str(encoder), hidden_size=str(hidden), num_layers=str(layers))
                config["env"].update(representation_mode="1", representation_seed="12345")
                path = self.root / f"config-{case}-{hidden}.ini"; self.write_ini(path, config)
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
                self.assertEqual(effective.getint("env", "representation"), 3)
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if args.eager else 1)
                self.assertEqual(effective.getint("eval_exact", "episodes"), 65)
                self.assertNotIn("max_decisions", effective["eval_exact"])
                record = json.loads((args.out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertFalse(record["gpu_runtime_qualified"])
                self.assertEqual(record["estimand"], ev.ESTIMAND)
                self.assertEqual(record["training_appearance"]["representation_mode"], "1")
                self.assertFalse((args.out / "episodes.csv").exists())
                self.assertFalse((args.out / "REPORT.md").exists())
                for name, digest in record["tooling_sha256"].items():
                    self.assertEqual(ev.sha(args.out / "tooling" / name), digest)
        with self.assertRaises(FileExistsError): ev.run(args)

    def test_state_checkpoint_preparation_and_mismatched_shape_labels(self):
        config = ev.read(self.config); config["base"]["env_name"] = "snakebench"
        config["policy"].update(encoder="5", hidden_size="256", num_layers="2")
        for key in ("representation", "representation_mode", "representation_seed"):
            del config["env"][key]
        path = self.root / "state.ini"; self.write_ini(path, config)
        suite_args = copy.copy(self.args)
        suite_args.binary, suite_args.config, suite_args.representation = BINARY_DIR / "state", path, 0
        suite_args.out = self.root / "state-suite"; ev.create_suite(suite_args)
        args = self.run_args("prepared-state", prepare_only=True)
        args.binary, args.config, args.suite, args.family = suite_args.binary, path, suite_args.out / "suite.json", "state"
        identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), \
             patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No GPU queries")), \
             patch.object(ev, "process", side_effect=AssertionError("No policy execution")):
            ev.run(args)
        self.assertEqual(dict(ev.read(args.out / "config/default.ini")["policy"]), dict(config["policy"]))
        self.assertTrue((args.out / "config/snakebench.ini").exists())
        with self.assertRaises(ValueError): ev.checkpoint_config(path, identity, "flex")
        with self.assertRaises(ValueError): ev.checkpoint_config(self.config, ev.info(BINARY_DIR / "default"), "nature")
        args = self.run_args("world-mismatch", prepare_only=True)
        config = ev.read(self.config); config["env"]["max_steps"] = "1024"
        self.write_ini(self.config, config)
        with self.assertRaisesRegex(ValueError, "game settings differ"): ev.run(args)
        self.assertFalse(args.out.exists())

    def test_bad_weights_busy_gpu_and_query_failure_never_launch_policy(self):
        identity = ev.info(BINARY_DIR / "default")
        for name, data in (("empty", b""), ("malformed", b"\0"*3), ("nonfinite", bytes.fromhex("0000c07f"))):
            args = self.run_args(name); args.checkpoint.write_bytes(data)
            with patch.object(ev, "info", return_value=identity), \
                 patch.object(ev.subprocess, "check_output", side_effect=AssertionError("Weights before GPU queries")), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(ValueError): ev.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")
        args.checkpoint.write_bytes(b"\0"*16)
        for name, answers in (("busy", ["12345\n"]), ("query-failed", [subprocess.CalledProcessError(1, ["smi"])]),
                              ("no-hardware", ["", ""])):
            args = self.run_args(name)
            with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
                 patch.object(ev.subprocess, "check_output", side_effect=answers), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy process")):
                with self.assertRaises(Exception): ev.run(args)
            self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def test_reservation_conflict_prevents_gpu_queries(self):
        args = self.run_args("reserved"); identity = ev.info(args.binary)
        directory = self.root / "build/connect4cnn"; directory.mkdir(parents=True)
        with (directory / "hardware-benchmark.lock").open("a") as lock:
            ev.fcntl.flock(lock, ev.fcntl.LOCK_EX | ev.fcntl.LOCK_NB)
            with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
                 patch.object(ev.subprocess, "check_output", side_effect=AssertionError("No queries while reserved")), \
                 patch.object(ev, "process", side_effect=AssertionError("No policy execution")):
                with self.assertRaises(BlockingIOError): ev.run(args)
        self.assertEqual(json.loads((args.out / "result.json").read_text())["status"], "failed")

    def fake_native(self, command, cwd, log, timeout):
        rows = self.receipts(); counts = ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        log.write_text("synthetic fixture, no policy execution\n"
            f"SNAKE_EXACT_EVAL version=1 requested=65 completed=65 deaths={counts['deaths']} "
            f"horizons={counts['horizon_ends']} score={counts['score']} foods={counts['foods']} params=4\n")
        with (cwd / "episodes.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        timing = dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)
        log.with_suffix(".log.json").write_text(json.dumps(timing))
        return timing

    def test_completion_requires_quota_summary_parameters_clock_and_input_closure(self):
        args = self.run_args("completed"); identity = ev.info(args.binary)
        with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(ev, "process", side_effect=self.fake_native):
            ev.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "ok"); self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertEqual(record["counts"], ev.audit_csv(self.write_rows(self.receipts()), self.suite, self.manifest))
        self.assertFalse(record["gpu_runtime_qualified"])
        self.assertIn("not original stock Snake", (args.out / "REPORT.md").read_text())
        cases = ("missing-row", "wrong-params", "wrong-deaths", "wrong-horizons", "wrong-score", "wrong-foods",
                 "duplicate-summary", "missing-summary", "changed-config", "changed-copied-suite", "changed-spawn",
                 "changed-training-copy", "changed-tool-copy", "missing-clock", "backwards-clock")
        for case in cases:
            args = self.run_args(case)
            def corrupt(command, cwd, log, timeout):
                timing = self.fake_native(command, cwd, log, timeout)
                if case == "missing-row":
                    lines = (cwd / "episodes.csv").read_text().splitlines()
                    (cwd / "episodes.csv").write_text("\n".join(lines[:-1])+"\n")
                elif case.startswith("wrong-"):
                    key = case.removeprefix("wrong-")
                    log.write_text(log.read_text().replace(key+"=", key+"=9"))
                elif case == "duplicate-summary": log.write_text(log.read_text()*2)
                elif case == "missing-summary": log.write_text("synthetic fixture without completion\n")
                elif case == "changed-config": (cwd / "config/snakecnn.ini").write_text("# tampered\n")
                elif case == "changed-copied-suite":
                    path = cwd / "suite/suite.json"; data = json.loads(path.read_text()); data["purpose"] = "heldout"
                    path.write_text(json.dumps(data))
                elif case == "changed-spawn":
                    path = cwd / "suite/episodes.csv"; path.write_text(path.read_text()+"\n")
                elif case == "changed-training-copy":
                    path = cwd / "training.ini"; path.write_text(path.read_text()+"\n")
                elif case == "changed-tool-copy":
                    path = cwd / "tooling/ocean/snakecnn/exact_eval.py"; path.write_text(path.read_text()+"\n")
                elif case == "missing-clock": log.with_suffix(".log.json").unlink()
                else:
                    timing["end_monotonic_ns"] = 0; log.with_suffix(".log.json").write_text(json.dumps(timing))
                return timing
            with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
                 patch.object(ev.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
                 patch.object(ev, "process", side_effect=corrupt):
                with self.assertRaises(Exception): ev.run(args)
            failed = json.loads((args.out / "result.json").read_text())
            self.assertEqual(failed["status"], "failed"); self.assertNotIn("counts", failed)
            self.assertFalse((args.out / "REPORT.md").exists())
            self.assertTrue((args.out / "episodes.csv").exists())

    def test_timeout_keeps_partial_receipts_and_monotonic_time(self):
        args = self.run_args("timeout"); identity = ev.info(args.binary)
        def failure(command, cwd, log, timeout):
            self.fake_native(command, cwd, log, timeout)
            lines = (cwd / "episodes.csv").read_text().splitlines()
            (cwd / "episodes.csv").write_text("\n".join(lines[:2])+"\n")
            raise subprocess.TimeoutExpired(command, timeout)
        with patch.object(ev, "info", return_value=identity), patch.object(ev, "ROOT", self.root), \
             patch.object(ev.subprocess, "check_output", side_effect=["", "synthetic-only-hardware"]), \
             patch.object(ev, "process", side_effect=failure):
            with self.assertRaises(subprocess.TimeoutExpired): ev.run(args)
        record = json.loads((args.out / "result.json").read_text())
        self.assertEqual(record["status"], "failed"); self.assertEqual(record["evaluation_process_seconds"], 2)
        self.assertNotIn("counts", record); self.assertFalse((args.out / "REPORT.md").exists())
        self.assertEqual(len((args.out / "episodes.csv").read_text().splitlines()), 2)

    def test_quota_and_integer_estimand_include_every_ending(self):
        rows = self.receipts(); result = ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        self.assertEqual(result["episodes"], 65)
        self.assertEqual(result["deaths"], 33); self.assertEqual(result["horizon_ends"], 32)
        self.assertEqual(result["score"], sum(row["score"] for row in rows))
        self.assertEqual(result["mean_score"], result["score"]/65)
        self.assertEqual(result["foods"], sum(row["foods"] for row in rows))
        self.assertGreater(rows[-1]["foods"], rows[-1]["score"])
        self.assertFalse(result["policy_runtime_certified"])
        self.assertEqual(ev.audit_csv(self.write_rows(rows[::-1]), self.suite, self.manifest), result)
        for bad in (rows[:-1], rows+[rows[0]], [rows[0]]+rows[:-1]):
            with self.assertRaises(ValueError): ev.audit_csv(self.write_rows(bad), self.suite, self.manifest)

    def test_corruptions_fail_without_dropping_episodes(self):
        rows = self.receipts()
        for key, value in (("episode_id", "9999"), ("start_rng", "0"), ("env_seed", "0"),
                           ("policy_seed", "0"), ("start_world_hash", "0"*16),
                           ("start_observation_hash", "0"*16), ("representation", "0"),
                           ("slot", 999), ("version", 2), ("block_seed", 0), ("decisions", 0),
                           ("decisions", 2049), ("score", 0), ("foods", -1), ("foods", 1),
                           ("end_reason", 0), ("end_reason", 2), ("perf", "nan"),
                           ("episode_return", "inf"), ("episode_return", "0x0p+0"),
                           ("action_hash", "bad")):
            changed = [dict(row) for row in rows]; changed[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ev.audit_csv(self.write_rows(changed), self.suite, self.manifest)

    def test_float32_running_return_not_count_times_reward(self):
        rows = self.receipts(); row = rows[0]
        row.update(foods=7, decisions=8, score=8, perf=ev.f32(8/120).hex())
        reward = 0.0
        for _ in range(7): reward = ev.f32(reward+ev.f32(.1))
        reward = ev.f32(reward-1)
        wrong = ev.f32(7*ev.f32(.1)-1)
        self.assertNotEqual(reward, wrong)
        row["episode_return"] = reward.hex()
        ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)
        row["episode_return"] = wrong.hex()
        with self.assertRaisesRegex(ValueError, "running return"):
            ev.audit_csv(self.write_rows(rows), self.suite, self.manifest)

    def test_zero_rewards_and_one_decision_game_horizon(self):
        suite = json.loads(json.dumps(self.suite))
        suite["world"].update(max_steps=1, reward_food=0, reward_death=0)
        ev.validate_suite(suite)
        rows = self.receipts()
        for i, row in enumerate(rows):
            row.update(decisions=1, foods=i % 2, score=1+i % 2, end_reason=1+i % 2,
                       perf=ev.f32((1+i % 2)/120).hex(), episode_return="0x0p+0")
        result = ev.audit_csv(self.write_rows(rows), suite, self.manifest)
        self.assertEqual(result["mean_return"], 0)
        self.assertEqual(result["mean_decisions"], 1)

    def test_all_drawings_builds_and_core_shapes_share_worlds(self):
        original = ev.read(self.suite_path.parent / "manifest.ini")
        for representation in range(6):
            original["env"]["representation"] = str(representation)
            default = self.native_manifest(original, name=f"r{representation}")
            for first, changed in zip(self.manifest, default, strict=True):
                for key in ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash"):
                    self.assertEqual(first[key], changed[key])
            for family in ("default", "impala", "impoola"):
                original["policy"].update(hidden_size="512", num_layers="4", encoder="4")
                self.assertEqual(self.native_manifest(original, family, f"r{representation}-{family}"), default)
        original["base"]["env_name"] = "snakebench"
        for key in ("representation", "representation_mode", "representation_seed"):
            del original["env"][key]
        state = self.native_manifest(original, "state", "bare-state")
        for pixel, local in zip(self.manifest, state, strict=True):
            for key in ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash"):
                self.assertEqual(pixel[key], local[key])
            self.assertEqual(local["representation"], "0")
            self.assertNotEqual(pixel["start_observation_hash"], local["start_observation_hash"])

    def test_independent_final_episode_ignores_batch_size(self):
        config = ev.read(self.suite_path.parent / "manifest.ini")
        config["eval_exact"].update(episodes="1", episode_offset="95", slots="1")
        self.assertEqual(self.native_manifest(config, name="tail"), self.manifest[-1:])

    def test_immutable_suite_closure_and_rehashed_world_tampering(self):
        source = self.suite_path.parent / "source.ini"; original = source.read_bytes()
        source.write_bytes(original+b"\n# changed\n")
        with self.assertRaises(ValueError): ev.load_suite(self.suite_path)
        source.write_bytes(original)
        changed = dict(self.suite, world=dict(self.suite["world"], max_steps=1024))
        self.suite_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "world differs"): ev.load_suite(self.suite_path)
        self.suite_path.write_text(json.dumps(self.suite))
        config = ev.read(self.suite_path.parent / "manifest.ini")
        config["policy"]["hidden_size"] = "64"
        self.write_ini(self.suite_path.parent / "manifest.ini", config)
        changed = dict(self.suite, manifest_ini_sha256=ev.sha(self.suite_path.parent / "manifest.ini"))
        self.suite_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "configuration differs"): ev.load_suite(self.suite_path)
        with self.assertRaises(FileExistsError): ev.create_suite(self.args)

    def test_invalid_world_and_ids_rejected_before_native_header(self):
        original = ev.read(self.suite_path.parent / "manifest.ini")
        cases = [("env", key, value) for key, value in
                 (("rule_version", "0"), ("num_agents", "2"), ("vision", "6"),
                  ("leave_corpse_on_death", "1"), ("width", "12"), ("height", "128.5"),
                  ("num_food", "256"), ("max_snake_length", "254"), ("max_steps", "0"),
                  ("max_steps", "16777217"), ("reward_food", "nan"), ("reward_death", "inf"),
                  ("representation_mode", "1"), ("representation", "6"))]
        cases += [("eval_exact", "slots", "0"), ("eval_exact", "seed", "4294967296"),
                  ("eval_exact", "episode_offset", "4294967295")]
        for i, (section, key, value) in enumerate(cases):
            config = configparser.ConfigParser(interpolation=None)
            config.read_dict({s: dict(original[s]) for s in original.sections()})
            config[section][key] = value
            path = self.root / f"invalid-{i}.ini"; self.write_ini(path, config)
            result = subprocess.run([str(BINARY_DIR / "default"), "eval_exact_manifest", str(path)],
                                    capture_output=True, text=True, timeout=30)
            with self.subTest(key=key, value=value):
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
            if section == "env" and key not in ("representation", "representation_mode"):
                with self.assertRaises(ValueError): ev.world(config)
        for key, value in (("slots", 0), ("offset", 2**32-1), ("episodes", True),
                           ("representation", 6), ("seed", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ev.validate_suite(dict(self.suite, **{key: value}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binary-dir", type=Path, default=BINARY_DIR)
    args, remaining = parser.parse_known_args(); BINARY_DIR = args.binary_dir.resolve()
    unittest.main(argv=[sys.argv[0]]+remaining)
