"""Suite/config/receipt checks; these do not run a neural model on CPU."""
import argparse
import configparser
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deterministic_eval as ev


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        args = argparse.Namespace(out=self.root/"suite", seed=2**32-1, episodes=65,
                                  offset=31, slots=64, representation=3, purpose="development")
        ev.create_suite(args)
        self.path = args.out/"suite.json"
        self.suite = ev.load_suite(self.path)

    def test_manifest_tamper(self):
        with (self.path.parent/"episodes.csv").open("a") as stream:
            stream.write("bad\n")
        with self.assertRaises(ValueError):
            ev.load_suite(self.path)

    def test_manifest_mapping_not_just_hash(self):
        manifest = self.path.parent/"episodes.csv"
        manifest.write_text(manifest.read_text().replace("31,", "32,", 1))
        self.suite["episodes_sha256"] = ev.sha(manifest)
        self.path.write_text(json.dumps(self.suite))
        with self.assertRaises(ValueError):
            ev.load_suite(self.path)

    def test_boundary_and_no_overwrite(self):
        self.suite["offset"] = 2**32-1
        with self.assertRaises(ValueError):
            ev.validate_suite(self.suite)
        with self.assertRaises(FileExistsError):
            ev.save(self.path, {})

    def config(self, encoder, hidden, layers):
        c = configparser.ConfigParser()
        for section in ("base", "vec", "selfplay", "env", "policy", "train"):
            c[section] = {}
        c["base"]["env_name"] = "connect4cnn"
        c["policy"] = dict(encoder=str(encoder), hidden_size=str(hidden), num_layers=str(layers))
        path = self.root/"policy.ini"
        with path.open("w") as stream:
            c.write(stream)
        return path

    def test_all_architectures_and_core_sizes_preserved(self):
        info = dict(environment="connect4cnn", default_encoder="tiny")
        for encoder, family in {0: "tiny", **ev.FAMILIES}.items():
            for hidden, layers in ((32, 1), (128, 2), (512, 4)):
                c = ev.load_config(self.config(encoder, hidden, layers), info, family)
                self.assertEqual(c.getint("policy", "hidden_size"), hidden)
                self.assertEqual(c.getint("policy", "num_layers"), layers)
        for family in ("nature", "impala", "impoola"):
            info["default_encoder"] = family
            ev.load_config(self.config(0, 128, 1), info, family)

    def test_reject_wrong_build(self):
        path = self.config(0, 128, 1)
        info = dict(environment="connect4cnn", default_encoder="impoola")
        with self.assertRaises(ValueError):
            ev.load_config(path, info, "impala")
        info["environment"] = "connect4"
        with self.assertRaises(ValueError):
            ev.load_config(path, info, "state")

    def test_legacy_compiled_default(self):
        path = self.config(0, 128, 1)
        path.write_text(path.read_text().replace("encoder = 0\n", ""))
        ev.load_config(path, dict(environment="connect4cnn", default_encoder="tiny"), "tiny")

    def receipts(self):
        rows = []
        for item in ev.manifest_rows(self.suite):
            episode = item["episode_id"]
            rows.append(dict(version=1, block_seed=self.suite["seed"], episode_id=episode,
                             slot=(episode-self.suite["offset"]) % 64,
                             representation=3, env_seed=item["env_seed"], policy_seed=item["policy_seed"],
                             decisions=8, win=1, score=1, invalid=0, action_hash="1234567890abcdef",
                             start_observation_hash=ev.empty_observation_hash(36*44),
                             start_player_pieces=0, start_env_pieces=0, start_rng=item["env_seed"]))
        return rows

    def write_receipts(self, rows):
        path = self.root/"receipts.csv"
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_exact_quota_and_corruptions(self):
        rows = self.receipts()
        self.assertEqual(ev.audit_csv(self.write_receipts(rows), self.suite, "connect4cnn")["episodes"], 65)
        for corrupted in (rows[:-1], rows+[rows[0]], [rows[0]]+rows[:-1]):
            with self.assertRaises(ValueError):
                ev.audit_csv(self.write_receipts(corrupted), self.suite, "connect4cnn")
        for key, value in (("start_rng", 0), ("start_player_pieces", 1),
                           ("start_observation_hash", "0"*16), ("policy_seed", 0)):
            corrupted = [dict(row) for row in rows]
            corrupted[0][key] = value
            with self.assertRaises(ValueError):
                ev.audit_csv(self.write_receipts(corrupted), self.suite, "connect4cnn")

    def test_sampling_independent_of_slots_and_policy(self):
        original = list(ev.manifest_rows(self.suite))
        self.suite["slots"] = 1
        self.assertEqual(original, list(ev.manifest_rows(self.suite)))
        self.suite["offset"] += 64
        self.suite["episodes"] = 1
        self.assertEqual(original[-1:], list(ev.manifest_rows(self.suite)))


if __name__ == "__main__":
    unittest.main()
