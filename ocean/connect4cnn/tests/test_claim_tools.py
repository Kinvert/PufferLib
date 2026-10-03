"""Artifact/configuration/statistical checks only; no model or GPU execution."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"ocean/connect4cnn"), str(ROOT/"research")]
import claim
from claim_frontier import frontier, paired_bands, report


class ClaimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native_dir = tempfile.TemporaryDirectory()
        cls.native = Path(cls.native_dir.name)/"config-probe"
        subprocess.run(["cc", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-I", str(ROOT),
                        str(ROOT/"ocean/connect4cnn/tests/test_claim_config.c"), "-o", str(cls.native)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.native_dir.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_native_decimal_roundtrip_and_real_hyper_change(self):
        c = claim.ini(ROOT/"ocean/connect4cnn/compare.ini")
        d = copy.deepcopy(c)
        d["train"]["gamma"] = "0.80000000000000004"
        self.assertEqual(claim.normalized(c), claim.normalized(d))
        d["train"]["gamma"] = ".81"
        self.assertNotEqual(claim.common_settings(c), claim.common_settings(d))

    def test_actual_native_ini_and_uint32_rng_boundaries(self):
        text = subprocess.check_output([str(self.native), str(ROOT/"ocean/connect4cnn/compare.ini")], text=True)
        out = self.root/"native.ini"; out.write_text(text)
        self.assertEqual(claim.normalized(claim.ini(out)), claim.normalized(claim.ini(ROOT/"ocean/connect4cnn/compare.ini")))
        lines = subprocess.check_output([str(self.native)], text=True).splitlines()
        self.assertEqual(len(lines), 9)
        for line in lines:
            seed, episode, env, policy = map(int, line.split(","))
            self.assertEqual((env, policy), claim.episode_seeds(seed, episode))

    def test_checkpoint_completion_and_fail_closed(self):
        process = dict(status="ok", launch_monotonic_ns=1000, end_monotonic_ns=1000000000)
        text = "PUFFER_CHECKPOINT steps=2048 bytes=40 monotonic_ns=2000\nPUFFER_CHECKPOINT steps=4096 bytes=40 monotonic_ns=3000\n"
        self.assertEqual(claim.parse_checkpoints(text, process, [2048, 4096], 10), {2048: 1e-6, 4096: 2e-6})
        for bad in (text+text, text.replace("bytes=40", "bytes=44"), text.replace("monotonic_ns=3000", "monotonic_ns=1500"), text.splitlines()[0]):
            with self.assertRaises(ValueError):
                claim.parse_checkpoints(bad, process, [2048, 4096], 10)
        with self.assertRaises(ValueError):
            claim.parse_checkpoints(text, {**process, "status": "failed"}, [2048, 4096], 10)

    def write_episodes(self, count=5):
        block = dict(seed=1400, offset=91, episodes=count)
        rows = []
        for index in range(count):
            env_seed, policy_seed = claim.episode_seeds(block["seed"], block["offset"]+index)
            rows.append(dict(version=1, block_seed=block["seed"], episode_id=block["offset"]+index,
                             slot=index % 3, representation=0, env_seed=env_seed, policy_seed=policy_seed,
                             decisions=10, win=index % 2, score=1 if index % 2 else -1, invalid=0,
                             action_hash="0123456789abcdef"))
        return block, rows

    def test_exact_quota_ids_rng_and_terminal_checks(self):
        block, rows = self.write_episodes()
        path = self.root/"episodes.csv"
        def write(values):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(values)
        write(list(reversed(rows)))  # Completion order must not affect allocation.
        self.assertEqual(claim.episodes(path, block, 3), dict(wins=2, score=-1, episodes=5))
        corruptions = [rows[:-1], rows+[rows[0]], [rows[0]]+rows[:-1]]
        for key, value in (("env_seed", 7), ("slot", 9), ("decisions", 22), ("win", 2), ("representation", 1)):
            wrong = copy.deepcopy(rows); wrong[0][key] = value; corruptions.append(wrong)
        for wrong in corruptions:
            write(wrong)
            with self.assertRaises(ValueError):
                claim.episodes(path, block, 3)

    def test_preparation_never_launches_or_queries_gpu(self):
        out = self.root/"campaign"
        with patch.object(claim, "snapshots", return_value={}), patch.object(claim.subprocess, "check_output", return_value="revision\n"):
            p = claim.prepare(out, True, None)
        self.assertEqual(len(p["jobs"]), 8)
        self.assertFalse(p["confirmation_launch_allowed"])
        self.assertEqual(p["checkpoint_steps"], [32768, 65536])
        self.assertTrue(all(sum(b["episodes"] for b in j["evaluation"]) == 73 for j in p["jobs"]))
        self.assertEqual(p["common_settings"]["train"]["total_timesteps"], float(65536).hex())
        with self.assertRaises(ValueError):
            claim.run_canary(out, False)
        with self.assertRaises(ValueError):
            claim.prepare(out, True, None)

    def test_full_draft_cadence_and_seed_pairing(self):
        out = self.root/"draft"
        with patch.object(claim, "snapshots", return_value={}), patch.object(claim.subprocess, "check_output", return_value="revision\n"):
            p = claim.prepare(out, False, [81, 82, 83, 84])
        self.assertEqual(len(p["checkpoint_steps"]), 51)
        self.assertEqual(p["checkpoint_steps"][-1], 13312000)
        self.assertEqual(len(p["jobs"]), 16)
        for seed in p["training_seeds"]:
            self.assertEqual(len({json.dumps(j["evaluation"]) for j in p["jobs"] if j["seed"] == seed}), 1)
        for pos in range(4):
            self.assertEqual({p["jobs"][4*block+pos]["model"] for block in range(4)}, set(claim.MODELS))
        with self.assertRaises(ValueError):
            claim.run_canary(out, True)

    def test_training_timeout_preserves_failure(self):
        log = self.root/"failed.log"
        with self.assertRaises(ValueError):
            claim.process([sys.executable, "-c", "raise SystemExit(7)"], self.root, log, 5)
        record = json.loads(log.with_suffix(".log.json").read_text())
        self.assertEqual((record["status"], record["returncode"]), ("failed", 7))
        self.assertLess(record["launch_monotonic_ns"], record["end_monotonic_ns"])
        with self.assertRaises(ValueError):
            claim.process(["unused"], self.root, log, 5)
        with self.assertRaises(subprocess.TimeoutExpired):
            claim.process([sys.executable, "-c", "import time; time.sleep(30)"], self.root, self.root/"timeout.log", .05)
        self.assertEqual(json.loads((self.root/"timeout.log.json").read_text())["status"], "failed")

    def test_frontier_keeps_ties_and_declines(self):
        np.testing.assert_array_equal(frontier([1, 2, 3, 2, 4], [.1, .6, .5, .6, .8]), [True, True, False, True, True])

    def test_joint_resampling_and_degenerate_intervals(self):
        values = np.array([[[1+i*.1, .1+i*.02], [2+i*.2, .3+i*.01]] for i in range(8)])
        first = paired_bands(values, draws=100)
        second = paired_bands(values, draws=100)
        np.testing.assert_array_equal(first["low"], second["low"])
        np.testing.assert_array_equal(first["high"], second["high"])
        changed = paired_bands(values[::-1], draws=100)
        self.assertEqual(first["mean"].shape, changed["mean"].shape)
        constant = paired_bands(np.tile([[[1., 1.]]], (5, 1, 1)), draws=20)
        self.assertEqual(constant["low"].tolist(), [[0, 0]])
        self.assertTrue(np.isinf(constant["high"][0, 0]))
        self.assertEqual(constant["high"][0, 1], 1)

    def test_missing_seed_never_becomes_a_complete_case_mean(self):
        p = dict(models=["flex_quality", "nature_cnn"], training_seeds=[1, 2], checkpoint_steps=[10, 20],
                 purpose="synthetic_artifact_test", timing="synthetic")
        points = [dict(model=m, seed=s, steps=k, seconds=k/10, win_rate=.5, games=17, parameters=10,
                       process_sps=10, native_sps=11, train_seconds=2) for m in p["models"] for s in [1, 2] for k in [10, 20]]
        points.pop()
        data = report(self.root, p, points, [{"error": "synthetic missing result"}])
        self.assertEqual(data["status"], "incomplete")
        self.assertEqual(len(data["points"]), 7)
        self.assertEqual(len(data["means"]), 3)
        self.assertFalse(any(m["model"] == "nature_cnn" and m["steps"] == 20 for m in data["means"]))
        self.assertTrue((self.root/"curves.html").exists())

    def test_complete_synthetic_archive_then_corrupted_episode(self):
        out = self.root/"synthetic"
        with patch.object(claim, "snapshots", return_value={}), patch.object(claim.subprocess, "check_output", return_value="revision\n"):
            p = claim.prepare(out, True, None)
        (out/"binaries").mkdir()
        binaries = {}
        for model in p["models"]:
            path = out/"binaries"/model
            path.write_bytes(b"Synthetic test fixture, never an executable")
            binaries[model] = claim.sha(path)
        claim.save(out/"builds.json", binaries)
        execution = dict(status="ok", jobs=[], failures=[])
        for job in p["jobs"]:
            directory = out/job["id"]
            config = claim.ini(directory/"config/default.ini")
            config["metrics"] = {"uptime": "0.1", "agent_steps": str(p["steps"])}
            (directory/"metrics/connect4cnn").mkdir(parents=True)
            with (directory/"metrics/connect4cnn/trial.ini").open("w") as handle:
                config.write(handle)
            process = dict(status="ok", command=[str(out/"binaries"/job["model"]), "train", "--headless"],
                           cwd=str(directory), environment={"PUFFER_CHECKPOINT_RECEIPTS": "1"},
                           launch_monotonic_ns=1000, end_monotonic_ns=1000000000)
            claim.save(directory/"train.log.json", process)
            (directory/"train.log").write_text("".join(f"PUFFER_CHECKPOINT steps={step} bytes={job['parameters']*4} monotonic_ns={2000+i*1000}\n"
                                                      for i, step in enumerate(p["checkpoint_steps"])))
            record = dict(id=job["id"], status="ok", checkpoints={}, evaluations={}, repeats={})
            for step in p["checkpoint_steps"]:
                checkpoint = directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                np.zeros(job["parameters"], dtype=np.float32).tofile(checkpoint)
                record["checkpoints"][str(step)] = claim.sha(checkpoint)
                for index, block in enumerate(job["evaluation"]):
                    stem = directory/f"eval-{step:016d}-b{index}"
                    _, rows = self.write_episodes(block["episodes"])
                    for i, row in enumerate(rows):
                        env, policy = claim.episode_seeds(block["seed"], block["offset"]+i)
                        row.update(block_seed=block["seed"], episode_id=block["offset"]+i, slot=i % p["evaluation_slots"],
                                   env_seed=env, policy_seed=policy)
                    with Path(str(stem)+".csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
                    Path(str(stem)+".log").write_text(f"CONNECT4_EXACT_EVAL version=1 requested={len(rows)} completed={len(rows)} wins={sum(r['win'] for r in rows)} params={job['parameters']}\n")
                    claim.save(Path(str(stem)+".log.json"), dict(status="ok", cwd=str(directory),
                               command=claim.evaluation_command(out, job, step, index, block, p["evaluation_slots"])))
                    record["evaluations"][stem.name] = {suffix: claim.sha(Path(str(stem)+suffix)) for suffix in (".csv", ".log", ".log.json")}
            for label in ("repeat", "eager", "worker1"):
                step = p["checkpoint_steps"][-1]
                original = directory/f"eval-{step:016d}-b0"
                stem = directory/f"eval-{step:016d}-{label}"
                for suffix in (".csv", ".log", ".log.json"):
                    Path(str(stem)+suffix).write_bytes(Path(str(original)+suffix).read_bytes())
                claim.save(Path(str(stem)+".log.json"), dict(status="ok", cwd=str(directory),
                           command=claim.repeat_command(out, job, step, label, p["evaluation_slots"])))
                record["repeats"][label] = {suffix: claim.sha(Path(str(stem)+suffix)) for suffix in (".csv", ".log", ".log.json")}
            execution["jobs"].append(record)
        claim.save(out/"execution.json", execution)
        result = json.loads((claim.audit(out)/"analysis.json").read_text())
        self.assertEqual(result["status"], "complete_descriptive")
        self.assertEqual(len(result["points"]), 16)
        bad = out/p["jobs"][0]["id"]/"eval-0000000000032768-b0.csv"
        bad.write_text(bad.read_text().replace("0123456789abcdef", "fedcba9876543210"))
        result = json.loads((claim.audit(out)/"analysis.json").read_text())
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["points"]), 15)
        self.assertEqual(len(result["missing"]), 1)


if __name__ == "__main__":
    unittest.main()
