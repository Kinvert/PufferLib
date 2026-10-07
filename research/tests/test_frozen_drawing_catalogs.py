"""Captured-catalog and suite-domain regression checks; no GPU/policy."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.tests import test_pixel_evaluation_plan as fixtures
import audit_pixel_panel as audit
import pixel_evaluation_plan as bridge
import prepare_pixel_robustness as panel


class CatalogTests(unittest.TestCase):
    def test_current_inventory_comes_from_hashed_native_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "panel"
            value = panel.prepare(root, list(panel.ENVIRONMENTS), "all", [53161, 53162], 65536, 32768)
            self.assertEqual(audit.catalog_counts(root, value)["flappycnn"], 7)
            self.assertEqual(len(value["jobs"]), 328)
            with patch.dict(panel.ENVIRONMENTS, flappycnn=99):
                self.assertEqual(audit.load(root / "protocol.json"), value)
                specs = list(bridge.binding_specs(value, "all", audit.catalog_counts(root, value)))
                self.assertEqual(len([s for s in specs if s[1]["environment"] == "flappycnn"]), 392)

    def test_retained_four_drawing_flappy_packets_stay_four_drawing(self):
        roots = ("pixel-evaluation-binding-20261006", "task-budgets-20261006",
                 "pixel-transfer-bindings-20261006", "pixel-mixed-20261006")
        for name in roots:
            root = panel.ROOT / "research/results" / name / "packet"
            with self.subTest(packet=name):
                value = audit.load(root / "panel/protocol.json")
                self.assertEqual(audit.catalog_counts(root / "panel", value)["flappycnn"], 4)
                with patch.object(bridge.subprocess, "run", side_effect=AssertionError("Must be offline")), \
                     patch.object(bridge.subprocess, "check_output", side_effect=AssertionError("Must be offline")):
                    inspected = bridge.inspect(root / "plan.json")
                self.assertEqual({b["representation"] for b in inspected["bindings"] if b["environment"] == "flappycnn"}, {0,1,2,3})

    def test_native_registry_rejects_missing_or_small_expanded_catalog(self):
        identity = dict(environment="flappycnn", rules=panel.ENVIRONMENT_RULES["flappycnn"],
                        float32=True, receipt_version=1, default_encoder="tiny")
        bridge.validate_identity("flappycnn", "default", identity, 4)
        with self.assertRaisesRegex(ValueError, "Stale native"): bridge.validate_identity("flappycnn", "default", identity, 7)
        bridge.validate_identity("flappycnn", "default", dict(identity, representation_count=7), 7)
        for count in (4, 7.0, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                bridge.validate_identity("flappycnn", "default", dict(identity, representation_count=count), 7)

    def test_flappy_v1_domain_is_preserved_and_v2_is_explicit(self):
        ev = bridge.adapter("flappycnn")
        suite = dict(protocol="flappy-fixed-suite-v1", rules=ev.RULES, environment="flappycnn",
                     seed=56123, offset=0, episodes=1000, slots=64, representation=3,
                     purpose="development", sampling="native-philox-categorical-v1",
                     world=ev.world(ev.read(panel.ROOT / "config/flappycnn.ini")),
                     cap_semantics="native-terminal-with-crash-reward")
        ev.validate_suite(suite)
        for altered in (dict(suite, representation=4), dict(suite, representation_count=7)):
            with self.assertRaises(ValueError): ev.validate_suite(altered)
        expanded = dict(suite, protocol="flappy-fixed-suite-v2", representation_count=7, representation=6)
        ev.validate_suite(expanded)
        for key, value in (("representation_count", 4), ("representation", 7)):
            altered = copy.deepcopy(expanded); altered[key] = value
            with self.assertRaises(ValueError): ev.validate_suite(altered)


if __name__ == "__main__": unittest.main()
