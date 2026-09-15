"""CPU safeguards for the frozen-model Pong search and shared log parser."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import report
from wandb_sidecar import architecture, read_ini


class SweepTest(unittest.TestCase):
    def fixture(self, root, variant="flex_quality"):
        ini = read_ini(ROOT / "config/default.ini")
        for section in list(ini.sections()):
            if section.startswith("sweep."):
                ini.remove_section(section)
        ini.read([HERE / "compare.ini", HERE / "sweep.ini"])
        ini.set("policy", "encoder", str(report.ENCODERS[variant]))
        (root / "config").mkdir()
        (root / "variant.txt").write_text(variant)
        (root / "config/pongcnn.ini").write_text("# isolated override\n")
        self.save(root, ini)
        return ini

    def save(self, root, ini):
        with (root / "config/default.ini").open("w") as f:
            ini.write(f)

    def test_all_families_same_search_space_and_core(self):
        spaces = []
        for variant in report.ENCODERS:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, variant)
                ini, actual = report.validate(root)
                self.assertEqual(actual, variant)
                spaces.append({s: dict(ini[s]) for s in ini if s.startswith("sweep.")})
                self.assertEqual(ini.getint("policy", "hidden_size"), 128)
                self.assertEqual(ini.getint("policy", "num_layers"), 1)
        self.assertTrue(all(space == spaces[0] for space in spaces))

    def test_reject_architecture_environment_or_search_drift(self):
        for section, key, value in (("policy", "cnn_kernel_1", "3"), ("policy", "hidden_size", "64"),
                                    ("env", "frameskip", "4"), ("sweep.policy.cnn_depth", "min", "1")):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ini = self.fixture(root)
                if not ini.has_section(section):
                    ini.add_section(section)
                ini.set(section, key, value)
                self.save(root, ini)
                with self.assertRaises(AssertionError):
                    report.validate(root)

    def test_compiled_reference_identity_is_explicit(self):
        config = {"policy.encoder": 0, "policy.hidden_size": 128, "policy.num_layers": 1}
        with self.assertRaises(ValueError):
            architecture(config)
        self.assertNotEqual(architecture(config, "impala_cnn"), architecture(config, "impoola_cnn"))

    def test_default_within_every_search_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = self.fixture(root)
            ini.set("train", "learning_rate", "1")
            self.save(root, ini)
            with self.assertRaises(AssertionError):
                report.validate(root)


if __name__ == "__main__":
    unittest.main()
