"""Synthetic report/receipt fixtures only; no neural model or GPU execution."""
import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.tests import test_pixel_evaluation_plan as fixtures
import pixel_frontiers as report
import pixel_evaluation_plan as bridge
import prepare_pixel_robustness as panel


def simple_plan(tasks=("connect4cnn",), drawings=(0,), seeds=(11, 12)):
    bindings = []
    for task in tasks:
        for drawing in drawings:
            for seed in seeds:
                for model in panel.MODELS:
                    bindings.append(dict(environment=task, representation=drawing, training_seed=seed,
                        model=model, job_index=len(bindings), checkpoint_steps=[100, 200]))
    return dict(training_seeds=list(seeds), bindings=bindings)


def points(plan):
    rows = []
    for index, binding in enumerate(plan["bindings"]):
        task, treatment, drawing = report.condition(binding)
        model = panel.MODELS.index(binding["model"])
        for step in binding["checkpoint_steps"]:
            rows.append(dict(binding_index=index, steps=step, environment=task, training_appearance=treatment,
                evaluation_representation=drawing, model=binding["model"], seed=binding["training_seed"],
                seconds=step/100*(model+1), score_lower=.4+.1*model+.1*(step == 100),
                score_upper=.4+.1*model+.1*(step == 100), parameters=4, episodes=3))
    return rows


class ReductionTests(unittest.TestCase):
    def test_all_games_and_drawings_stay_separate_and_declines_remain(self):
        plan = simple_plan(tuple(panel.ENVIRONMENTS), (0, 1))
        rows = points(plan); result = report.reduce(plan, rows, [])
        self.assertEqual(result["status"], "complete_descriptive")
        self.assertEqual(len(result["conditions"]), 12)
        self.assertEqual(len(result["points"]), 192)
        self.assertEqual(len(result["means"]), 96)
        self.assertFalse(result["dominance_certified"])
        self.assertFalse(result["confidence_bands_enabled"])
        self.assertFalse(result["cross_game_score_average_enabled"])
        self.assertEqual(result["means"][0]["score_lower"], .5)
        self.assertEqual(result["means"][1]["score_lower"], .4)

    def test_missing_bad_seed_cannot_create_complete_case_front(self):
        plan = simple_plan(); rows = points(plan)
        result = report.reduce(plan, rows[:-1], [])
        self.assertEqual(len(result["missing"]), 1)
        self.assertEqual(len(result["means"]), 7)
        self.assertTrue(all(row["frontier_seconds_lower"] is None for row in result["means"]))

    def test_failed_training_suppresses_front_even_if_episodes_exist(self):
        plan = simple_plan(); result = report.reduce(plan, points(plan), [dict(job_index=0, error="timeout")])
        self.assertEqual(len(result["points"]), 16)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all(row["frontier_seconds_upper"] is None for row in result["means"]))

    def test_seed_means_never_take_best_seed(self):
        plan = simple_plan(); rows = points(plan)
        for row in rows:
            if row["model"] == "flex_quality":
                row["score_lower"] = row["score_upper"] = 1 if row["seed"] == 11 else 0
        result = report.reduce(plan, rows, [])
        self.assertTrue(all(row["score_lower"] == .5 for row in result["means"] if row["model"] == "flex_quality"))

    def test_pong_bounds_remain_intervals_not_midpoints(self):
        plan = simple_plan(("pongcnn",)); rows = points(plan)
        for row in rows: row["score_lower"], row["score_upper"] = .2, .8
        result = report.reduce(plan, rows, [])
        self.assertTrue(all(row["score_lower"] == .2 and row["score_upper"] == .8 for row in result["means"]))
        self.assertEqual(result["conditions"][0]["score_bounds_kind"], "deterministic-censoring-not-confidence")

    def test_fixed_training_sources_and_mixed_catalogs_are_distinct(self):
        binding = simple_plan()["bindings"][0]
        fixed = dict(binding, training_representation=2, evaluation_representation=0)
        mixed = dict(binding, training_representation=None, training_representation_mode=1,
                     training_representation_seed=12345, training_catalog_count=10)
        self.assertEqual(report.condition(fixed), ("connect4cnn", "fixed-r2", 0))
        self.assertEqual(report.condition(mixed), ("connect4cnn", "mixed-s12345-c10", 0))

    def test_duplicate_unexpected_nonfinite_and_rebound_points_reject(self):
        plan = simple_plan(); rows = points(plan)
        cases = [rows+[rows[0]], [dict(rows[0], steps=999)], [dict(rows[0], seconds=float("nan"))],
                 [dict(rows[0], score_lower=float("inf"))], [dict(rows[0], model="nature_cnn")]]
        for case in cases:
            with self.subTest(case=case[0]), self.assertRaises(ValueError): report.reduce(plan, case, [])


class ViewerTests(unittest.TestCase):
    def test_embedded_data_preserves_all_bounds_flags_and_declines_without_network_dependencies(self):
        plan = simple_plan(("connect4cnn", "pongcnn"))
        result = report.reduce(plan, points(plan), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.html"; report.write_viewer(path, result)
            page = path.read_text()
        embedded = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
        self.assertEqual(json.loads(embedded), result)
        self.assertNotIn('__DATA__', page); self.assertNotIn('__VIEWER_JS__', page)
        self.assertNotRegex(page, r'<script[^>]+src=|<link[^>]+href="https?://')
        self.assertNotIn('fetch(', page)

    def test_script_terminators_html_and_unicode_separators_remain_inert_data(self):
        value = dict(label='</script><script>alert("bad")</script>&>\u2028\u2029', status="incomplete")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.html"; report.write_viewer(path, value)
            page = path.read_text()
        embedded = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1)
        self.assertNotIn('<', embedded); self.assertNotIn('&', embedded)
        self.assertNotIn('\u2028', embedded); self.assertNotIn('\u2029', embedded)
        self.assertEqual(json.loads(embedded), value)

    def test_viewer_refuses_overwrite_and_nonfinite_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curves.html"; report.write_viewer(path, dict(status="incomplete"))
            with self.assertRaises(FileExistsError): report.write_viewer(path, {})
            with self.assertRaises(ValueError): report.write_viewer(Path(tmp) / "bad.html", dict(seconds=float("nan")))
            self.assertFalse((Path(tmp) / "bad.html").exists())


class MetricTests(unittest.TestCase):
    def test_units_and_negative_nonfinite_bounds(self):
        cases = [("connect4cnn", dict(wins=1, episodes=4), (.25,.25)),
                 ("flappycnn", dict(mean_pipes=37), (37,37)),
                 ("snakecnn", dict(mean_score=7), (7,7)),
                 ("breakoutcnn", dict(mean_score=450), (450,450)),
                 ("mazecnn", dict(success_fraction=.1), (.1,.1)),
                 ("pongcnn", dict(mean_final_point_fraction_lower=.3, mean_final_point_fraction_upper=.7), (.3,.7))]
        for task, counts, expected in cases: self.assertEqual(report.score(task, counts), expected)
        for low, high in ((.8,.2), (float("nan"),.3), (-1,.2), (.2,2)):
            with self.assertRaises(ValueError): report.score("pongcnn", dict(mean_final_point_fraction_lower=low, mean_final_point_fraction_upper=high))

    def test_six_native_summary_formats_and_censoring_counts(self):
        cases = [
            ("connect4cnn", "CONNECT4_EXACT_EVAL version=1 requested=3 completed=3 wins=2 params=4", dict(episodes=3,wins=2)),
            ("pongcnn", "PONG_EXACT_EVAL version=1 requested=3 evaluated=3 completed=2 capped=1 wins=1 right=21 left=15 params=4",
             dict(episodes=3,completed=2,capped=1,wins=1,right_points=21,left_points=15)),
            ("flappycnn", "FLAPPY_EXACT_EVAL version=1 requested=3 completed=3 pipes=5 capped=1 params=4", dict(episodes=3,pipes=5,capped=1)),
            ("breakoutcnn", "BREAKOUT_EXACT_EVAL version=1 requested=3 evaluated=3 completed=2 capped=1 score=50 params=4", dict(episodes=3,completed=2,capped=1,score=50)),
            ("snakecnn", "SNAKE_EXACT_EVAL version=1 requested=3 completed=3 deaths=2 horizons=1 score=20 foods=17 params=4", dict(episodes=3,deaths=2,horizon_ends=1,score=20,foods=17)),
            ("mazecnn", "MAZE_EXACT_EVAL version=1 requested=3 completed=3 successes=1 timeouts=2 params=4", dict(episodes=3,successes=1,timeouts=2))]
        for task, text, counts in cases:
            report.completion(task, text+"\n", counts, 4)
            for altered in ("", text+"\n"+text, text.replace("params=4", "params=5"), text.replace("requested=3", "requested=2")):
                with self.subTest(task=task), self.assertRaises(ValueError): report.completion(task, altered, counts, 4)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BindingTests(); self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        shutil.rmtree(self.fixture.panel)
        panel.prepare(self.fixture.panel, ["connect4cnn"], "default", [53111,53112], 65536, 32768)
        self.plan = self.fixture.prepare(); self.plan_path = self.fixture.args.out / "plan.json"
        self.collection_root = self.root / "collection"
        self.collection = report.prepare(self.plan_path, self.collection_root)
        self.collection_path = self.collection_root / "collection.json"

    def save(self): self.collection_path.write_text(json.dumps(self.collection))

    def training(self, job_index=0):
        binding = next(b for b in self.plan["bindings"] if b["job_index"] == job_index)
        job = self.fixture.panel / Path(binding["config"]).relative_to("panel")
        directory = job.parent.parent
        config = report.ini(job)
        checkpoint_root = Path(config["base"]["checkpoint_dir"]) / "connect4cnn" / "trial"
        checkpoint_root.mkdir(parents=True)
        shutil.copyfile(job, checkpoint_root / "resolved.ini")
        checkpoints = {}
        for step in binding["checkpoint_steps"]:
            weights = checkpoint_root / f"{step:016d}.bin"; weights.write_bytes(b"\0" * 16)
            checkpoints[str(step)] = str(weights)
        start = (job_index+1)*10000000000
        binary = self.plan["binaries"]["connect4cnn"][binding["binary_family"]]["path"]
        timed = dict(status="ok", returncode=0, command=[binary,"train","--headless"], cwd=str(directory),
                     launch_monotonic_ns=start, end_monotonic_ns=start+9000000000,
                     environment=dict(PUFFER_CHECKPOINT_RECEIPTS="1"))
        (directory / "train.log.json").write_text(json.dumps(timed))
        (directory / "train.log").write_text("".join(f"PUFFER_CHECKPOINT steps={step} bytes=16 monotonic_ns={start+(index+1)*1000000000}\n"
                                                  for index, step in enumerate(binding["checkpoint_steps"])))
        hardware = dict(gpu_uuid="SYNTHETIC-NO-GPU",gpu_name="fixture",host="fixture",cuda_compiler="fixture",driver="fixture")
        (directory / "hardware.json").write_text(json.dumps(hardware))
        receipt = directory / "training-receipt.json"
        receipt.write_text(json.dumps(dict(process="train.log.json",log="train.log",hardware="hardware.json",checkpoints=checkpoints)))
        self.collection["training"][job_index]["receipt"] = str(receipt)
        return receipt, binding, checkpoints

    def evaluation(self, binding_index=0, step=32768):
        binding = self.plan["bindings"][binding_index]
        receipt, _, checkpoints = self.training(binding["job_index"])
        trained = report.training(report.Inputs(), receipt, binding, self.plan, self.plan_path.parent)
        directory = self.root / "evaluation"; directory.mkdir()
        shutil.copyfile(self.plan_path.parent / binding["config"], directory / "training.ini")
        suite_path = self.plan_path.parent / binding["suite"]
        shutil.copytree(suite_path.parent, directory / "suite")
        ev = bridge.adapter("connect4cnn"); suite = ev.load_suite(suite_path)
        config = report.effective_config(ev, report.ini(directory / "training.ini"), suite, "connect4cnn", Path(checkpoints[str(step)]), directory, 1)
        (directory / "config").mkdir()
        with (directory / "config/default.ini").open("w") as stream: config.write(stream)
        (directory / "config/connect4cnn.ini").touch()
        rows = []
        for start in ev.manifest_rows(suite):
            row = {k: v for k,v in start.items() if k not in ("player_pieces", "env_pieces")}
            row.update(version=1,block_seed=suite["seed"],slot=(start["episode_id"]-suite["offset"])%suite["slots"],
                       start_player_pieces=0,start_env_pieces=0,start_rng=start["env_seed"],
                       start_observation_hash=ev.empty_observation_hash(36*44),decisions=4,score=1,win=1,invalid=0,
                       action_hash="1234567890abcdef")
            rows.append(row)
        with (directory / "episodes.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        counts=ev.audit_csv(directory / "episodes.csv",suite,"connect4cnn")
        (directory / "native.log").write_text("CONNECT4_EXACT_EVAL version=1 requested=3 completed=3 wins=3 params=4\n")
        binary=self.plan["binaries"]["connect4cnn"][binding["binary_family"]]
        measured=dict(status="ok",suite_sha256=report.sha(suite_path),training_ini_sha256=binding["config_sha256"],
            family=binding["policy_family"],binary_sha256=binary["sha256"],checkpoint=checkpoints[str(step)],
            checkpoint_sha256=report.sha(Path(checkpoints[str(step)])),parameters=4,
            command=[binary["path"],"eval_exact","--headless"],
            effective_ini_sha256=report.sha(directory / "config/default.ini"),episodes_sha256=report.sha(directory / "episodes.csv"),counts=counts,cwd=str(directory))
        path=directory / "result.json"; path.write_text(json.dumps(measured))
        self.collection["evaluations"][0]["result"]=str(path); self.save()
        return path, binding, trained

    def test_empty_real_allocation_report_is_explicitly_incomplete(self):
        with patch.object(bridge.subprocess,"run",side_effect=AssertionError("No subprocess")), \
             patch.object(bridge.subprocess,"check_output",side_effect=AssertionError("No subprocess")):
            value=report.report(self.collection_path,self.root / "empty-report")
        self.assertEqual(len(value["missing"]),16); self.assertFalse(value["points"])
        self.assertEqual(value["status"],"incomplete")
        self.assertEqual(len(value["planned_cells"]),16)
        page=(self.root / "empty-report/curves.html").read_text()
        embedded=re.search(r'<script id="data" type="application/json">(.*?)</script>',page,re.S).group(1)
        self.assertEqual(json.loads(embedded),value)
        self.assertIn(str(report.VIEWER.resolve()),value["reader_sources_sha256"])
        self.assertIn(str(report.VIEWER_JS.resolve()),value["reader_sources_sha256"])
        with self.assertRaisesRegex(ValueError,"fresh"): report.report(self.collection_path,self.root / "empty-report")

    def test_single_synthetic_receipt_audits_raw_episodes_and_monotonic_time(self):
        self.evaluation(); value=report.report(self.collection_path,self.root / "single-report")
        self.assertEqual(value["points"][0]["score_lower"],1)
        self.assertEqual(value["points"][0]["seconds"],1)
        self.assertEqual(value["accounted_training_process_seconds"],9)
        self.assertEqual(len(value["missing"]),15)
        self.assertFalse(value["means"])

    def test_episode_tampering_is_retained_as_failure_not_an_observation(self):
        path,_,_=self.evaluation()
        csv_path=path.parent / "episodes.csv"; csv_path.write_text(csv_path.read_text().replace("1234567890abcdef","bad"))
        measured=report.read(path); measured["episodes_sha256"]=report.sha(csv_path); path.write_text(json.dumps(measured))
        value=report.report(self.collection_path,self.root / "bad-report")
        self.assertFalse(value["points"]); self.assertIn("action",value["failures"][0]["error"])

    def test_rehashed_effective_policy_override_cannot_bypass_recipe_check(self):
        path,_,_=self.evaluation(); config_path=path.parent / "config/default.ini"
        config=report.ini(config_path); config["policy"]["hidden_size"]="256"
        with config_path.open("w") as stream: config.write(stream)
        measured=report.read(path); measured["effective_ini_sha256"]=report.sha(config_path); path.write_text(json.dumps(measured))
        value=report.report(self.collection_path,self.root / "override-report")
        self.assertFalse(value["points"]); self.assertIn("configuration differs",value["failures"][0]["error"])

    def test_clock_receipt_order_and_wrong_checkpoint_prefix_reject(self):
        path,binding,_=self.training(); log=path.parent / "train.log"
        log.write_text(log.read_text().replace("monotonic_ns=11000000000","monotonic_ns=10000000000"))
        with self.assertRaisesRegex(ValueError,"completion time"):
            report.training(report.Inputs(),path,binding,self.plan,self.plan_path.parent)

    def test_missing_cells_and_improper_claim_flags_cannot_be_redefined(self):
        original=copy.deepcopy(self.collection)
        for mutate in (lambda v:v["evaluations"].pop(),lambda v:v.update(gpu_execution_authorized=True),
                       lambda v:v["training"].reverse()):
            self.collection=copy.deepcopy(original); mutate(self.collection); self.save()
            with self.assertRaises(ValueError): report.report(self.collection_path,self.root / "invalid-report")

    def test_mixed_hardware_and_overlapping_training_refuse_comparison(self):
        first,_,_=self.training(0); second,_,_=self.training(1); self.save()
        hardware=second.parent / "hardware.json"; original=hardware.read_text()
        changed=report.read(hardware); changed["gpu_uuid"]="OTHER-SYNTHETIC-GPU"; hardware.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"Mixed hardware"):
            report.report(self.collection_path,self.root / "hardware-report")
        hardware.write_text(original)
        process=first.parent / "train.log.json"; changed=report.read(process); changed["end_monotonic_ns"]=100000000000
        process.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            report.report(self.collection_path,self.root / "overlap-report")


class AdapterTests(unittest.TestCase):
    def test_pending_game_adapters_produce_compatible_configs_and_audit_all_assigned_rows(self):
        # Native host manifests are genuine retained artifacts. Outcomes, weights,
        # metadata and completion summaries below are explicitly synthetic fixtures.
        packet=panel.ROOT / "research/results/flappycnn/geometric-preparation-20261006/packet"
        plan=report.checked_plan(packet / "plan.json")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for task in ("pongcnn","flappycnn","breakoutcnn","snakecnn","mazecnn"):
                drawing=6 if task == "flappycnn" else 0
                binding=next(b for b in plan["bindings"] if b["environment"] == task and b["model"] == "flex_quality" and b["representation"] == drawing)
                ev=bridge.adapter(task); suite_path=packet / binding["suite"]
                suite,manifest=ev.load_suite(suite_path)
                weights=root / f"{task}.bin"; weights.write_bytes(b"\0"*16)
                binary=plan["binaries"][task]["default"]
                for eager in (False,True):
                    directory=root / f"{task}-{'eager' if eager else 'graph'}"
                    args=SimpleNamespace(binary=Path(binary["path"]),config=packet / binding["config"],
                        checkpoint=weights,suite=suite_path,out=directory,family="flex",timeout=10,eager=eager,prepare_only=True)
                    def host_manifest(command,**kwargs):
                        if command[1] != "eval_exact_manifest": raise AssertionError("No policy/GPU command allowed")
                        kwargs["stdout"].write((suite_path.parent / "episodes.csv").read_text())
                        return SimpleNamespace(returncode=0)
                    # Actual adapter preparation is the independent producer of the
                    # effective config; only metadata/host spawn subprocesses are mocked.
                    with patch.object(ev,"info",return_value=binary["info"]), \
                         patch.object(ev.subprocess,"run",side_effect=host_manifest), redirect_stdout(io.StringIO()):
                        ev.run(args)
                    measured=report.read(directory / "result.json")
                    self.assertEqual(measured["status"],"prepared-not-executed")
                    rows=[]
                    for start in manifest:
                        row={k:v for k,v in start.items() if k != "terminal_rng"}
                        episode=int(start["episode_id"])
                        row.update(version=1,block_seed=suite["seed"],slot=(episode-suite["offset"])%suite["slots"],action_hash="1234567890abcdef")
                        if task == "pongcnn":
                            row.update(decisions=suite["max_decisions"],last_rally_decisions=suite["max_decisions"],
                                right_points=0,left_points=0,completed=0,capped=1,episode_return=0.0.hex(),
                                point_fraction_lower=0.0.hex(),point_fraction_upper=1.0.hex(),native_perf="")
                        elif task == "flappycnn":
                            row.update(decisions=1,pipes=0,perf=0.0.hex(),episode_return=(-1.0).hex(),cap_reached=0)
                        elif task == "breakoutcnn":
                            cap=suite["max_frames"]; skip=int(suite["world"]["frameskip"])
                            row.update(frames=cap,decisions=1+(cap-1)//skip,score=0,completed=0,capped=1,
                                       perf=0.0.hex(),episode_return=0.0.hex())
                        elif task == "snakecnn":
                            row.update(decisions=int(suite["world"]["max_steps"]),score=1,foods=0,end_reason=2,
                                       perf=ev.f32(1/120).hex(),episode_return=0.0.hex())
                        else:
                            row.update(decisions=int(start["horizon"]),success=0,episode_return=0.0.hex(),end_reason=2,
                                       native_log_entries=1,end_rng=start["terminal_rng"])
                        rows.append(row)
                    with (directory / "episodes.csv").open("w") as stream:
                        writer=csv.DictWriter(stream,fieldnames=ev.RESULT_FIELDS); writer.writeheader(); writer.writerows(rows)
                    counts=ev.audit_csv(directory / "episodes.csv",suite,manifest); n=counts["episodes"]
                    summary={
                        "pongcnn":f"PONG_EXACT_EVAL version=1 requested={n} evaluated={n} completed=0 capped={n} wins=0 right=0 left=0 params=4",
                        "flappycnn":f"FLAPPY_EXACT_EVAL version=1 requested={n} completed={n} pipes=0 capped=0 params=4",
                        "breakoutcnn":f"BREAKOUT_EXACT_EVAL version=1 requested={n} evaluated={n} completed=0 capped={n} score=0 params=4",
                        "snakecnn":f"SNAKE_EXACT_EVAL version=1 requested={n} completed={n} deaths=0 horizons={n} score={n} foods=0 params=4",
                        "mazecnn":f"MAZE_EXACT_EVAL version=1 requested={n} completed={n} successes=0 timeouts={n} params=4"}[task]
                    (directory / "native.log").write_text(summary+"\n")
                    measured.update(status="ok",counts=counts,parameters=4,episodes_sha256=report.sha(directory / "episodes.csv"),prepare_only=False)
                    result_path=directory / "result.json"; result_path.write_text(json.dumps(measured))
                    trained=dict(times={32768:1},checkpoints={32768:weights},parameters=4)
                    point=report.evaluation(report.Inputs(),result_path,binding,32768,trained,plan,packet)
                    self.assertEqual(point["episodes"],1000)
                    self.assertEqual(point["score_lower"],1 if task == "snakecnn" else 0)
                    self.assertEqual(point["score_upper"],1 if task in ("snakecnn","pongcnn") else 0)
                    measured["prepare_only"]=True; result_path.write_text(json.dumps(measured))
                    with self.assertRaisesRegex(ValueError,"Prepared inputs"):
                        report.evaluation(report.Inputs(),result_path,binding,32768,trained,plan,packet)


if __name__ == "__main__": unittest.main()
