"""Host/config/scalar and synthetic suite tests; no GPU or neural execution."""
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import candidate_panel as tool


def fixture_start_check(task, binary, config, suite_path, directory):
    if task == "connect4cnn": return dict(proof="declared-empty-boards-and-identity-seeds", native_manifest_executed=False)
    directory.mkdir(parents=True)
    with (directory / "manifest.ini").open("w") as stream: config.write(stream)
    path = directory / "episodes.csv"; path.write_bytes((suite_path.parent / "episodes.csv").read_bytes())
    (directory / "stderr.txt").write_text("")
    return dict(proof="native-family-world-and-raster", native_manifest_executed=True, manifest=str(path), manifest_sha256=tool.sha(path))


class SuiteFixture:
    def __init__(self, task): self.task = task

    def load_config(self, *args): return None

    def create_suite(self, args):
        args.out.mkdir(parents=True)
        data = dict(seed=args.seed, slots=args.slots, episodes=args.episodes, representation=args.representation)
        tool.save(args.out / "suite.json", data)
        with (args.out / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["episode_id", "env_seed", "policy_seed", "representation", "start_world_hash", "start_observation_hash"])
            writer.writeheader()
            writer.writerows(dict(episode_id=i, env_seed=args.seed+i, policy_seed=args.seed+100+i,
                representation=args.representation, start_world_hash="fixture-world", start_observation_hash=f"fixture-{args.representation}") for i in range(args.episodes))

    def load_suite(self, path):
        data = json.loads(path.read_text())
        return data if self.task == "connect4cnn" else (data, tool.bridge.rows(path.parent / "episodes.csv"))

    def validate_manifest(self, path, suite): return tool.bridge.rows(path)


class CandidatePanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "build/registry.json"; self.registry.parent.mkdir()
        binary = self.registry.parent / "fixture.bin"; binary.write_bytes(b"not-an-executable")
        self.binaries = {task: dict(path=str(binary), sha256=tool.sha(binary), info=dict(environment=task,
            rules=tool.panel.ENVIRONMENT_RULES[task], float32=True, receipt_version=2 if task == "connect4cnn" else 1,
            default_encoder="tiny", representation_count=count)) for task, count in tool.panel.ENVIRONMENTS.items()}
        tool.save(self.registry, dict(protocol="native-cnn-candidate-build-v1", source_sha256={}, binaries=self.binaries))
        self.args = SimpleNamespace(out=self.root / "packet", registry=self.registry,
            candidate=["happy-cat-1="+str(ROOT / "research/recipes/panel_smoke_quality.ini"),
                       "mystic-tree-2="+str(ROOT / "research/recipes/panel_smoke_small.ini")], native_campaign=[],
            seeds=[57173], appearance_seed=35173, eval_seed=67173, episodes=17, slots=16,
            steps=65536, checkpoint_steps=65536, task_budget=[], learner_recipe=[],
            pong_max_decisions=512, breakout_max_frames=2048)

    def prepare(self):
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), patch.object(tool, "check_starts", side_effect=fixture_start_check):
            return tool.prepare(self.args)

    def with_baselines(self):
        variants = {}
        for task, default in self.binaries.items():
            variants[task] = {}
            for family in ("default", "impala", "impoola"):
                variants[task][family] = {**default, "info": {**default["info"], "default_encoder": "tiny" if family == "default" else family}}
        tool.save(self.registry, dict(protocol="native-cnn-candidate-build-v2", source_sha256={}, binaries=variants))
        self.args.baselines = list(tool.BASELINES)

    def test_baselines_match_each_game_recipe_and_share_every_suite(self):
        self.with_baselines(); self.args.seeds = [57173, 57174]; self.args.checkpoint_steps = 32768
        value = self.prepare()
        self.assertEqual(value["protocol"], tool.BASELINE_PROTOCOL)
        self.assertEqual(len(value["jobs"]), 60)
        self.assertEqual(len(value["suites"]), 82)
        self.assertEqual(value["planned_evaluations"], 820)
        groups = {}
        for job in value["jobs"]:
            config = tool.panel.read(self.args.out / job["config"])
            expected = next(c for c in value["candidates"] if c["name"] == job["candidate"])
            self.assertEqual((job["binary_family"], job["policy_family"]), tool.candidate_families(expected))
            self.assertEqual(config.getint("policy", "encoder"), expected["architecture"]["encoder"])
            self.assertEqual(config.getint("policy", "hidden_size"), 128)
            if job["kind"] == "baseline": self.assertFalse(any(k.startswith("cnn_") for k in config["policy"]))
            groups.setdefault((job["environment"], job["seed"]), []).append(job["targets"])
        self.assertTrue(all(len(items) == 5 and all([t["suite"] for t in item] == [t["suite"] for t in items[0]] for item in items) for items in groups.values()))
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), patch.object(tool.subprocess, "check_output", side_effect=AssertionError("Offline inspect")):
            self.assertEqual(tool.inspect(self.args.out / "plan.json"), value)

    def test_baseline_registry_cannot_omit_a_compiled_family(self):
        self.with_baselines(); registry = json.loads(self.registry.read_text())
        registry["binaries"]["pongcnn"].pop("impoola")
        tool.save(self.registry, registry)
        with self.assertRaisesRegex(ValueError, "all baseline build families"): self.prepare()
        self.assertFalse(self.args.out.exists())

    def test_all_model_learner_rehash_cannot_bypass_frozen_recipe(self):
        self.with_baselines(); value = self.prepare()
        for job in value["jobs"]:
            if job["environment"] != "connect4cnn": continue
            path = self.args.out / job["config"]; config = tool.panel.read(path); config.set("train", "clip_coef", "0.1")
            with path.open("w") as stream: config.write(stream)
            value["files_sha256"][job["config"]] = tool.sha(path)
        tool.save(self.args.out / "plan.json", value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "frozen per-game recipe"):
            tool.inspect(self.args.out / "plan.json")

    def test_baseline_family_relabel_and_secondary_override_are_rejected(self):
        self.with_baselines(); value = self.prepare()
        baseline = next(c for c in value["candidates"] if c.get("baseline") == "impala_cnn")
        baseline["binary_family"] = "default"; tool.save(self.args.out / "plan.json", value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "Baseline architecture/build changed"):
            tool.inspect(self.args.out / "plan.json")
        baseline["binary_family"] = "impala"; job = value["jobs"][0]
        secondary = self.args.out / job["id"] / "config" / f"{job['environment']}.ini"
        secondary.write_text("[train]\nlearning_rate = 0.1\n")
        value["files_sha256"][str(secondary.relative_to(self.args.out))] = tool.sha(secondary)
        tool.save(self.args.out / "plan.json", value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "Secondary game INI"):
            tool.inspect(self.args.out / "plan.json")

    def test_changed_native_family_start_receipt_is_rejected(self):
        self.with_baselines(); value = self.prepare()
        index, job = next((i, j) for i, j in enumerate(value["jobs"]) if j["environment"] == "flappycnn" and j["kind"] == "baseline")
        target = job["targets"][0]; manifest = self.args.out / "host-checks" / f"job-{index:04d}/r0/episodes.csv"
        manifest.write_text(manifest.read_text().replace("fixture-world", "changed-world"))
        value["files_sha256"][str(manifest.relative_to(self.args.out))] = tool.sha(manifest)
        target["start_check"]["manifest_sha256"] = tool.sha(manifest); tool.save(self.args.out / "plan.json", value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "start manifest differs"):
            tool.inspect(self.args.out / "plan.json")

    def test_full_cross_product_keeps_cnn_common_and_learners_per_game(self):
        value = self.prepare()
        self.assertEqual(len(value["jobs"]), 12)
        self.assertEqual(len(value["suites"]), 41)
        self.assertEqual(value["planned_evaluations"], 82)
        self.assertFalse(value["quality_selection_allowed"])
        self.assertFalse(value["cross_game_adaptive_protein_implemented"])
        gammas = set()
        for job in value["jobs"]:
            config = tool.panel.read(self.args.out / job["config"])
            self.assertIn("sweep", config)
            self.assertTrue(config.has_option("sweep", "max_runs"))
            self.assertFalse(any(s.startswith("sweep.") for s in config.sections()))
            self.assertEqual(config.getint("env", "representation_mode"), 1)
            self.assertEqual(config.getint("env", "representation_seed"), 35173)
            self.assertEqual(config.getint("train", "total_timesteps"), 65536)
            self.assertEqual(job["checkpoint_steps"], [65536])
            self.assertEqual(config.get("base", "load_model_path"), "None")
            if job["environment"] in ("flappycnn", "pongcnn"):
                self.assertEqual(config.getint("env", "representation_mix_catalog"), 1)
            gammas.add(config.getfloat("train", "gamma"))
        self.assertGreater(len(gammas), 1)
        for task, assignment in value["assignments"].items():
            self.assertEqual(sum(assignment["counts"]), 64)
            self.assertEqual(assignment["catalog_count"], tool.panel.ENVIRONMENTS[task])
            self.assertFalse(assignment["native_vector_initialization_validated"])
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), \
             patch.object(tool.subprocess, "check_output", side_effect=AssertionError("Offline inspect must not execute/query")):
            self.assertEqual(tool.inspect(self.args.out / "plan.json"), value)

    def test_per_game_recipe_budget_and_partial_checkpoint_are_shared(self):
        recipe = self.root / "learner.ini"; recipe.write_text("[train]\nlearning_rate = 0.0003\nclip_coef = 0.15\n")
        self.args.learner_recipe = ["flappycnn="+str(recipe)]
        self.args.task_budget = ["flappycnn=98304:65536"]
        self.args.seeds = [57173, 57174]
        value = self.prepare()
        self.assertEqual(value["planned_training_jobs"], 24)
        self.assertEqual(value["planned_evaluations"], 192)
        for job in value["jobs"]:
            if job["environment"] != "flappycnn": continue
            config = tool.panel.read(self.args.out / job["config"])
            self.assertEqual(config.getfloat("train", "learning_rate"), 0.0003)
            self.assertEqual(config.getfloat("train", "clip_coef"), 0.15)
            self.assertEqual(job["checkpoint_steps"], [65536, 98304])

    def test_changed_config_and_changed_shared_learner_are_rejected(self):
        value = self.prepare(); job = next(j for j in value["jobs"] if j["environment"] == "connect4cnn")
        path = self.args.out / job["config"]; config = tool.panel.read(path)
        config.set("train", "learning_rate", "0.002")
        with path.open("w") as stream: config.write(stream)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "Changed frozen input"):
            tool.inspect(self.args.out / "plan.json")
        value["files_sha256"][job["config"]] = tool.sha(path)
        tool.save(self.args.out / "plan.json", value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "Learner/world changed"):
            tool.inspect(self.args.out / "plan.json")

    def test_reject_unpaired_budgets_and_encoder5(self):
        self.args.task_budget = ["flappycnn=65537:65536"]
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.args.out.exists())
        invalid = self.root / "encoder5.ini"; invalid.write_text("[policy]\nencoder = 5\n")
        with self.assertRaisesRegex(ValueError, "legacy encoder 4 only"): tool.architecture(invalid)

    def test_learner_cannot_change_world_or_architecture(self):
        recipe = self.root / "learner.ini"; recipe.write_text("[policy]\ncnn_channels_1 = 32\n")
        self.args.learner_recipe = ["flappycnn="+str(recipe)]
        with self.assertRaisesRegex(ValueError, "cannot alter architecture"): self.prepare()
        self.assertTrue((self.args.out / "failure.json").exists())

    def test_rotate_order_independently_of_results(self):
        records = [dict(name=str(i)) for i in range(3)]
        actual = list(tool.job_order(records, [1, 2]))
        self.assertEqual(len(actual), 36)
        self.assertEqual([r[2]["name"] for r in actual[:6]], ["0", "1", "2", "1", "2", "0"])

    def test_native_recipe_sweeps_only_cnn_and_retains_each_games_learner(self):
        from sweep import prepare
        recipe = ROOT / "research/recipes/general_cnn_discovery.ini"
        for task in tool.TASKS:
            destination = self.root / ("discovery-"+task); destination.mkdir()
            with patch.dict(tool.os.environ, {"NVCC_PREPEND_FLAGS": ""}):
                config = prepare(destination, recipe, 128, environment=task)
            dimensions = [s[6:] for s in config.sections() if s.startswith("sweep.")]
            self.assertEqual(len(dimensions), 18)
            self.assertTrue(all(k.startswith("policy.cnn_") for k in dimensions))
            expected = tool.panel.read(ROOT / "config/default.ini", ROOT / f"config/{task}.ini", ROOT / f"ocean/{task}/compare.ini")
            for key in expected["train"]:
                if key != "total_timesteps": self.assertEqual(config["train"][key], expected["train"][key])
            self.assertEqual(config.getint("policy", "hidden_size"), 128)
            self.assertEqual(config.getint("policy", "num_layers"), 1)
            self.assertEqual(config.getint("train", "total_timesteps"), 13312000)

    def test_per_game_geometry_is_shared_with_references_and_audited_offline(self):
        self.with_baselines(); self.args.baselines = ["nature_cnn"]
        self.args.per_game_learners = True
        recipe = self.root / "flappy-learner.ini"
        recipe.write_text("[vec]\ntotal_agents=128\nnum_buffers=2\nnum_threads=2\n[train]\nhorizon=64\nminibatch_size=2048\nlearning_rate=.003\n")
        self.args.learner_recipe = ["flappycnn=" + str(recipe)]
        self.args.task_budget = ["flappycnn=98304:32768"]
        value = self.prepare()
        self.assertEqual(value["protocol"], tool.LEARNER_PROTOCOL)
        self.assertEqual(value["planned_training_jobs"], 18)
        self.assertEqual(value["planned_evaluations"], 165)
        for job in value["jobs"]:
            if job["environment"] != "flappycnn": continue
            config = tool.panel.read(self.args.out / job["config"])
            self.assertEqual(config.getint("vec", "total_agents"), 128)
            self.assertEqual(config.getint("train", "horizon"), 64)
            self.assertEqual(config.getint("base", "checkpoint_interval"), 4)
            self.assertEqual(job["checkpoint_steps"], [32768,65536,98304])
        native = json.loads((self.args.out / "validation/flappycnn.json").read_text())
        self.assertEqual(native["optimizer_minibatches_per_epoch"], 4)
        self.assertEqual(native["optimizer_updates_per_rank"], 48)
        self.assertFalse(native["policy_executed"])
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), \
             patch.object(tool.subprocess, "check_output", side_effect=AssertionError("Offline only")):
            self.assertEqual(tool.inspect(self.args.out / "plan.json"), value)
        for mode in ("smoke", "mini", "research", "canary5090"):
            with self.assertRaisesRegex(ValueError, "separately scheduled"):
                tool.run_hardware(value, mode)
        self.assertEqual(tool.run_hardware(value, "development"), "5090")

    def test_general_geometry_accepts_its_rollout_multiple_and_rejects_misalignment(self):
        self.args.per_game_learners = True
        recipe = self.root / "small-learner.ini"
        recipe.write_text("[vec]\ntotal_agents=32\n[train]\nhorizon=16\nminibatch_size=512\n")
        self.args.learner_recipe = ["flappycnn=" + str(recipe)]
        self.args.task_budget = ["flappycnn=3072:1536"]
        value = self.prepare()
        job = next(j for j in value["jobs"] if j["environment"] == "flappycnn")
        self.assertEqual(job["checkpoint_steps"], [1536,3072])
        self.assertEqual(tool.panel.read(self.args.out / job["config"]).getint("base", "checkpoint_interval"), 3)
        self.args.out = self.root / "misaligned"
        self.args.task_budget = ["flappycnn=3072:1000"]
        with self.assertRaisesRegex(ValueError, "align"):
            self.prepare()
        self.assertTrue((self.args.out / "failure.json").exists())

    def test_general_learner_rehash_and_geometry_receipt_changes_are_rejected(self):
        self.args.per_game_learners = True; value = self.prepare()
        path = self.args.out / "plan.json"
        original = path.read_text()
        original_configs = {}
        for job in value["jobs"]:
            if job["environment"] != "connect4cnn": continue
            config_path = self.args.out / job["config"]; config = tool.panel.read(config_path)
            original_configs[job["config"]] = config_path.read_bytes()
            config.set("train", "learning_rate", ".01")
            with config_path.open("w") as stream: config.write(stream)
            value["files_sha256"][job["config"]] = tool.sha(config_path)
        tool.save(path, value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "frozen per-game recipe"):
            tool.inspect(path)
        path.write_text(original)
        # Restore those files before independently corrupting the geometry proof.
        value = json.loads(original)
        for job in value["jobs"]:
            if job["environment"] != "connect4cnn": continue
            config_path = self.args.out / job["config"]
            config_path.write_bytes(original_configs[job["config"]])
        receipt = self.args.out / "validation/connect4cnn.json"
        data = json.loads(receipt.read_text()); data["optimizer_updates_per_rank"] = 0
        tool.save(receipt, data)
        value["files_sha256"]["validation/connect4cnn.json"] = tool.sha(receipt)
        value["learner_geometry"]["connect4cnn"]["receipt_sha256"] = tool.sha(receipt)
        tool.save(path, value)
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), self.assertRaisesRegex(ValueError, "inconsistent native learner"):
            tool.inspect(path)

    def test_general_geometry_refuses_zero_optimizer_updates(self):
        self.args.per_game_learners = True
        recipe = self.root / "zero.ini"; recipe.write_text("[train]\nreplay_ratio=.25\n")
        self.args.learner_recipe = ["connect4cnn=" + str(recipe)]
        with self.assertRaisesRegex(ValueError, "positive updates"):
            self.prepare()
        self.assertTrue((self.args.out / "failure.json").exists())

    def test_native_import_keeps_dominated_and_duplicate_trials(self):
        campaign = self.root / "native"; (campaign / "metrics/connect4cnn").mkdir(parents=True)
        (campaign / "sweep.log").write_text("synthetic completed native observations\n")
        config = ROOT / "research/recipes/panel_smoke_quality.ini"
        for i in range(2): (campaign / f"metrics/connect4cnn/trial_{i}.ini").write_bytes(config.read_bytes())
        synthetic = [dict(run_id=f"trial_{i}", index=i, score=score, cost=2+i,
                          checkpoint_sha256="fixture-checkpoint") for i, score in enumerate((0.9, 0.0))]
        with patch.object(tool, "trials", return_value=synthetic): actual = tool.candidates([], [campaign])
        self.assertEqual(len(actual), 2)
        self.assertEqual([r["discovery_score"] for r in actual], [0.9, 0.0])
        self.assertEqual(actual[0]["architecture"], actual[1]["architecture"])

    def test_gpu_denial_and_oversized_smoke_precede_query(self):
        with patch.object(tool, "hardware", side_effect=AssertionError("No GPU query")):
            with self.assertRaisesRegex(ValueError, "requires --allow-gpu"):
                tool.run(SimpleNamespace(allow_gpu=False))
        self.args.steps = self.args.checkpoint_steps = 262144
        self.prepare()
        with patch.object(tool.bridge, "adapter", side_effect=SuiteFixture), patch.object(tool, "hardware", side_effect=AssertionError("No GPU query")), \
             self.assertRaisesRegex(ValueError, "exceeds bounded"):
            tool.run(SimpleNamespace(allow_gpu=True, plan=self.args.out / "plan.json", mode="smoke", timeout=900, train_timeout=60, eval_timeout=30))
        self.assertFalse((self.args.out / "execution").exists())


if __name__ == "__main__": unittest.main()
