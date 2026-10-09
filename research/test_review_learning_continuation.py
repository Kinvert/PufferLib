"""Host-only rejection checks for completed-prefix replay receipts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("prefix_review", Path(__file__).with_name("review_learning_continuation.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def require(value, message):
    if not value:
        raise ValueError(message)


class PrefixTests(unittest.TestCase):
    def test_prefix_requires_failed_allocation_and_proper_trial_count(self):
        c = SimpleNamespace(require=require)
        result = dict(status="failed", inherited_observations=12, trials=[{}] * 33)
        self.assertEqual(r.prefix_boundary(c, result, dict(additional_trials=36), 12), 45)
        for override in (dict(status="ok"), dict(trials=[]), dict(trials=[{}] * 36), dict(inherited_observations=0)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                r.prefix_boundary(c, {**result, **override}, dict(additional_trials=36), 12)

    def test_proposal_requires_exact_ledger_and_all_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "trial-0045"
            directory.mkdir()
            history = "PUFFER_CROSS_GAME_V1 1\n" + "0 0.25 1 0.5\n" * 45
            (directory / "history.tsv").write_text(history)
            proposal = dict(protocol="native-cross-game-protein-v1", trial=45, observations_replayed=45,
                success_observations=45, failure_observations=0, gp_observations=45)
            timing = dict(status="ok", returncode=0, cwd=str(root), timeout=120,
                command=["/worker", "research_protein_propose", str(root / "recipe.ini"),
                         str(directory / "history.tsv"), str(directory / "proposal.json")],
                launch_monotonic_ns=100, end_monotonic_ns=200)
            c = SimpleNamespace(require=require, read=lambda p: json.loads(p.read_text()))
            config = SimpleNamespace(getint=lambda s, k: 120)
            def verify(p, t):
                (directory / "proposal.json").write_text(json.dumps(p))
                (directory / "native.txt.json").write_text(json.dumps(t))
                return r.verify_proposal(c, directory, history, p, t, root, dict(optimizer="/worker"), config, 45, 90)
            self.assertEqual(verify(proposal, timing), 200)
            for override in (dict(observations_replayed=44), dict(trial=44), dict(gp_observations=0), dict(failure_observations=1)):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    verify({**proposal, **override}, timing)
            for override in (dict(returncode=1), dict(launch_monotonic_ns=80), dict(timeout=121), dict(cwd="/foreign")):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    verify(proposal, {**timing, **override})
            (directory / "history.tsv").write_text(history + "0 0 0 0\n")
            with self.assertRaisesRegex(ValueError, "replay receipt"):
                verify(proposal, timing)


if __name__ == "__main__":
    unittest.main()
