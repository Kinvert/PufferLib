"""Host manifests and synthetic receipt checks only; no neural model or GPU."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_acceptance as gate

BINARIES = {"flappycnn": gate.ROOT / "build/flappycnn/exact-launcher-20261006",
            "breakoutcnn": gate.ROOT / "build/breakoutcnn/exact-preparation-20261005",
            "pongcnn": gate.ROOT / "build/pongcnn/texture-native-20261006",
            "snakecnn": gate.ROOT / "build/snakecnn/exact-preparation-20261005-v2",
            "mazecnn": gate.ROOT / "build/mazecnn/exact-native-20261006-final"}


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_run = {task: gate.adapter(task).run for task in gate.TASKS}

    def case(self, task="flappycnn", name="quality", episodes=3, slots=2, purpose="development"):
        ev = gate.adapter(task); directory = self.root / name; directory.mkdir()
        config = ev.read(gate.ROOT / "config/default.ini")
        config.read([gate.ROOT / "config" / f"{task}.ini", gate.ROOT / "ocean" / task / "compare.ini"])
        for section in list(config):
            if section.startswith("sweep."): config.remove_section(section)
        config["base"]["env_name"] = task
        if task == "mazecnn": config["env"]["num_maps"] = "32"
        source = directory / "training.ini"
        with source.open("w") as stream: config.write(stream)
        checkpoint = directory / "synthetic-weights.bin"; checkpoint.write_bytes(b"\0"*16)
        args = argparse.Namespace(binary=BINARIES[task] / "default", config=source, out=directory / "suite",
                                  seed=56121, offset=31, episodes=episodes, slots=slots,
                                  representation=0, purpose=purpose, max_frames=16, max_decisions=8,
                                  level_offset=0, level_count=32, training_level_count=0)
        if task == "mazecnn" and purpose == "heldout":
            args.level_offset, args.level_count, args.training_level_count = 3, 29, 3
        ev.create_suite(args)
        return dict(name=name, task=task, family="flex", binary=str(args.binary), config=str(source),
                    checkpoint=str(checkpoint), suite=str(directory / "suite/suite.json"))

    def prepare(self, cases, name="packet"):
        source = self.root / (name+"-cases.json"); source.write_text(json.dumps(cases))
        args = argparse.Namespace(cases=source, out=self.root / name, timeout=10)
        gate.prepare(args); return args.out / "packet.json"

    def test_all_five_tasks_prepare_fixed_batch_partial_tail_without_gpu(self):
        cases = [self.case(task, task) for task in gate.TASKS]
        # info is the permitted host-only query; forbid all policy launch paths.
        patches = [patch.object(gate.adapter(t), "process", side_effect=AssertionError("No GPU policy")) for t in gate.TASKS]
        for context in patches: context.start(); self.addCleanup(context.stop)
        query_patches = [patch.object(gate.adapter(t).shutil, "which",
                                      side_effect=AssertionError("No GPU discovery")) for t in gate.TASKS]
        for context in query_patches: context.start(); self.addCleanup(context.stop)
        packet = self.prepare(cases); value = gate.load_packet(packet, current_inputs=True)
        gate.inspect(argparse.Namespace(packet=packet))
        self.assertEqual(len(value["cases"]), 5)
        self.assertFalse(value["policy_math_certified"])
        self.assertFalse(value["publication_confirmation_launchable"])
        for case in value["cases"]:
            ev = gate.adapter(case["task"])
            original, rows = ev.load_suite(packet.parent / case["modes"]["graph"]["suite"])
            tail, tail_rows = ev.load_suite(packet.parent / case["modes"]["tail"]["suite"])
            self.assertEqual(tail["slots"], original["slots"])
            self.assertEqual(tail["offset"], 33); self.assertEqual(tail_rows, rows[-1:])
            for mode in gate.MODES:
                out = packet.parent / case["modes"][mode]["prepared"]
                record = json.loads((out / "result.json").read_text())
                self.assertEqual(record["status"], "prepared-not-executed")
                self.assertTrue(record["prepare_only"])
                self.assertFalse((out / "episodes.csv").exists())
                effective = ev.read(out / "config/default.ini")
                self.assertEqual(effective.getint("base", "cudagraphs"), -1 if mode == "eager" else 1)
                self.assertEqual(effective.getint("eval_exact", "slots"), 2)

    def test_bad_case_paths_names_duplicate_flex2_and_fields_fail(self):
        case = self.case()
        for index, changed in enumerate((dict(case, name="../escape"), dict(case, name=".."),
                dict(case, family="flex2"), dict(case, task="connect4cnn"), dict(case, config="missing.ini"),
                dict(case, extra="bad"))):
            path = self.root / f"bad-{index}.json"; path.write_text(json.dumps([changed]))
            with self.assertRaises(ValueError): gate.read_cases(path)
        path = self.root / "duplicate.json"; path.write_text(json.dumps([case, case]))
        with self.assertRaises(ValueError): gate.read_cases(path)
        path.write_text(json.dumps([case, dict(case, name="duplicate-inputs")]))
        with self.assertRaises(ValueError): gate.read_cases(path)
        path.write_text(json.dumps([None]))
        with self.assertRaises(ValueError): gate.read_cases(path)

    def test_relative_paths_are_relative_to_case_file_not_cwd(self):
        case = self.case()
        for key in ("binary", "config", "checkpoint", "suite"):
            # Absolute binary is outside temporary root; remaining inputs are relative.
            if key != "binary": case[key] = str(Path(case[key]).relative_to(self.root))
        source = self.root / "relative.json"; source.write_text(json.dumps([case]))
        resolved = gate.read_cases(source)[0]
        self.assertEqual(Path(resolved["config"]), self.root / "quality/training.ini")
        self.assertEqual(Path(resolved["checkpoint"]), self.root / "quality/synthetic-weights.bin")

    def test_heldout_and_nonpartial_suites_keep_failed_preparation(self):
        for name, episodes, slots, purpose in (("heldout", 3, 2, "heldout"), ("one-wave", 2, 2, "development"),
                                              ("full-tail", 4, 2, "development"), ("too-many", 5, 2, "development")):
            case = self.case(name=name, episodes=episodes, slots=slots, purpose=purpose)
            with self.assertRaises(ValueError): self.prepare([case], name+"-packet")
            directory = self.root / (name+"-packet")
            self.assertTrue((directory / "preparation_failure.json").exists())
            self.assertFalse((directory / "packet.json").exists())

    def test_overwrite_and_changed_original_frozen_or_tool_inputs_rejected(self):
        case = self.case(); packet = self.prepare([case])
        with self.assertRaises(FileExistsError): self.prepare([case])
        checkpoint = Path(case["checkpoint"]); checkpoint.write_bytes(b"\0"*20)
        with self.assertRaisesRegex(ValueError, "Original input changed"): gate.load_packet(packet, current_inputs=True)
        checkpoint.write_bytes(b"\0"*16)
        tool = packet.parent / "tooling/research/eval_acceptance.py"
        with tool.open("a") as stream: stream.write("\n")
        with self.assertRaises(ValueError): gate.load_packet(packet)

    def synthetic_rows(self, ev, suite, starts):
        rows = []
        for start in starts:
            row = dict(start); row.pop("terminal_rng", None)
            row.update(version="1", block_seed=str(suite["seed"]),
                       slot=str((int(start["episode_id"])-suite["offset"]) % suite["slots"]), action_hash="0123456789abcdef")
            task = ev.RULES
            if task == "flappy-native-v1":
                row.update(decisions=str(int(suite["world"]["max_steps"])), pipes="0", perf="0x0p+0",
                           episode_return="0x0p+0", cap_reached="1")
            elif task == "breakout-native-v1":
                frames = suite["max_frames"]; skip = int(suite["world"]["frameskip"])
                row.update(decisions=str((frames+skip-1)//skip), frames=str(frames), score="0", perf="0x0p+0",
                           episode_return="0x0p+0", completed="0", capped="1")
            elif task == "pong-native-v1":
                cap = str(suite["max_decisions"])
                row.update(decisions=cap, last_rally_decisions=cap, right_points="0", left_points="0", completed="0",
                           capped="1", episode_return="0x0p+0", native_perf="", point_fraction_lower="0x0p+0",
                           point_fraction_upper="0x1p+0")
            elif task == "snake-local-episodic-v1":
                row.update(decisions=str(int(suite["world"]["max_steps"])), score="1", foods="0", end_reason="2",
                           perf=float(ev.f32(1/120)).hex(), episode_return="0x0p+0")
            else:
                row.update(decisions=start["horizon"], success="0", episode_return="0x0p+0", end_reason="2",
                           native_log_entries="1", end_rng=start["terminal_rng"])
            rows.append(row)
        return rows

    def fake_run(self, task, args, mutation=None):
        # Native host preparation is real; policy/GPU execution is explicitly fake.
        ev = gate.adapter(task); host = copy.copy(args); host.prepare_only = True
        self.original_run[task](host)
        suite, starts = ev.load_suite(args.suite); rows = self.synthetic_rows(ev, suite, starts)
        if mutation: mutation(args, rows)
        path = args.out / "episodes.csv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=ev.RESULT_FIELDS); writer.writeheader(); writer.writerows(rows)
        result = json.loads((args.out / "result.json").read_text())
        result.update(status="ok", prepare_only=False, counts=ev.audit_csv(path, suite, starts),
                      parameters=4, episodes_sha256=gate.sha(path), evaluation_process_seconds=2)
        (args.out / "result.json").write_text(json.dumps(result))
        (args.out / "native.log.json").write_text(json.dumps(dict(launch_monotonic_ns=10, end_monotonic_ns=2000000010, returncode=0)))
        (args.out / "native.log").write_text("synthetic fixture: no GPU, no policy\n")

    def execute_synthetic(self, packet, out, mutation=None):
        patches = []
        for task in gate.TASKS:
            context = patch.object(gate.adapter(task), "run", side_effect=lambda args, t=task: self.fake_run(t, args, mutation))
            context.start(); patches.append(context)
        try: gate.run(argparse.Namespace(packet=packet, out=out))
        finally:
            for context in patches: context.stop()

    def test_all_five_synthetic_comparisons_pass_without_math_certification(self):
        packet = self.prepare([self.case(task, task) for task in gate.TASKS])
        out = self.root / "synthetic-execution"; self.execute_synthetic(packet, out)
        result = json.loads((out / "acceptance.json").read_text())
        self.assertEqual(result["status"], "scoped-receipt-comparisons-passed")
        self.assertEqual(len(result["cases"]), 5)
        self.assertFalse(result["policy_math_certified"]); self.assertFalse(result["frontier_superiority_certified"])
        gate.audit(argparse.Namespace(packet=packet, execution=out))

    def test_repeat_eager_and_tail_action_drift_fails_despite_identical_scores(self):
        packet = self.prepare([self.case()])
        for mode in ("repeat", "eager", "tail"):
            out = self.root / ("drift-"+mode)
            def mutation(args, rows):
                if args.out.name == mode: rows[0]["action_hash"] = "fedcba9876543210"
            with self.assertRaisesRegex(ValueError, "differs|affected tail"): self.execute_synthetic(packet, out, mutation)
            self.assertTrue((out / "failure.json").exists()); self.assertFalse((out / "acceptance.json").exists())
            self.assertTrue((out / "quality" / mode / "episodes.csv").exists())

    def test_partial_failure_does_not_produce_campaign_acceptance(self):
        packet = self.prepare([self.case()]); out = self.root / "timeout"
        def fail(args):
            args.out.mkdir(parents=True); (args.out / "partial.txt").write_text("synthetic timeout; no policy\n")
            raise TimeoutError("synthetic native child timeout")
        with patch.object(gate.adapter("flappycnn"), "run", side_effect=fail):
            with self.assertRaises(TimeoutError): gate.run(argparse.Namespace(packet=packet, out=out))
        self.assertTrue((out / "quality/graph/partial.txt").exists())
        self.assertFalse((out / "acceptance.json").exists())
        self.assertEqual(json.loads((out / "failure.json").read_text())["status"], "failed")

    def test_rehashed_world_vectorization_and_learner_drift_rejected(self):
        packet = self.prepare([self.case()]); ev = gate.adapter("flappycnn")
        for section, key, value in (("env", "gravity", "0.75"), ("vec", "num_threads", "2"),
                                    ("train", "horizon", "2")):
            out = self.root / ("config-drift-"+section)
            def changed(args):
                self.fake_run("flappycnn", args)
                if args.out.name == "graph":
                    path = args.out / "config/default.ini"; config = ev.read(path)
                    config[section][key] = value
                    with path.open("w") as stream: config.write(stream)
                    result_path = args.out / "result.json"; result = json.loads(result_path.read_text())
                    result["effective_ini_sha256"] = gate.sha(path)
                    result_path.write_text(json.dumps(result))
            with patch.object(ev, "run", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "Prepared/executed configuration differs"):
                    gate.run(argparse.Namespace(packet=packet, out=out))
            self.assertTrue((out / "failure.json").exists())
            self.assertFalse((out / "acceptance.json").exists())

    def test_imported_adapter_source_drift_rejected(self):
        ev = gate.adapter("flappycnn")
        with patch.object(ev, "acceptance_source_sha256", "0"*64):
            with self.assertRaisesRegex(ValueError, "Imported adapter source changed"):
                gate.adapter("flappycnn")

    def test_inspect_rejects_fabricated_execution_flag(self):
        packet = self.prepare([self.case()]); value = json.loads(packet.read_text())
        prepared = value["cases"][0]["modes"]["graph"]["prepared"]
        result_path = packet.parent / prepared / "result.json"
        result = json.loads(result_path.read_text()); result["prepare_only"] = False
        result_path.write_text(json.dumps(result))
        value["files_sha256"][str(result_path.relative_to(packet.parent))] = gate.sha(result_path)
        packet.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Prepared mode was executed or changed"):
            gate.inspect(argparse.Namespace(packet=packet))

    def test_unsafe_mode_tail_and_quota_metadata_rejected(self):
        packet = self.prepare([self.case()]); original = json.loads(packet.read_text())
        for key, value in (("suite_offset", 32), ("tail_offset", 32), ("tail_episodes", 2), ("slots", 1),
                           ("name", ".."), ("family", "flex2")):
            changed = copy.deepcopy(original); changed["cases"][0][key] = value
            packet.write_text(json.dumps(changed))
            with self.assertRaises(ValueError, msg=key): gate.load_packet(packet)
        changed = copy.deepcopy(original); changed["cases"][0]["modes"]["tail"]["suite"] = "../outside.json"
        packet.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): gate.load_packet(packet)

    def test_offline_audit_rejects_modified_completed_csv(self):
        packet = self.prepare([self.case()]); out = self.root / "synthetic"
        self.execute_synthetic(packet, out)
        with (out / "quality/graph/episodes.csv").open("a") as stream: stream.write("\n")
        with self.assertRaises(ValueError): gate.audit(argparse.Namespace(packet=packet, execution=out))

    def test_relocated_unchanged_archive_can_be_audited(self):
        packet = self.prepare([self.case()]); out = self.root / "synthetic"
        self.execute_synthetic(packet, out)
        moved_packet, moved_execution = self.root / "archive-packet", self.root / "archive-execution"
        shutil.copytree(packet.parent, moved_packet); shutil.copytree(out, moved_execution)
        gate.inspect(argparse.Namespace(packet=moved_packet / "packet.json"))
        gate.audit(argparse.Namespace(packet=moved_packet / "packet.json", execution=moved_execution))


if __name__ == "__main__": unittest.main()
