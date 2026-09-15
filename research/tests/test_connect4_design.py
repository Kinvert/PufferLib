"""Guard against score-based checkpoint selection and missing-data optimism."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from design_connect4_confirmation import latest_available


class SelectionTest(unittest.TestCase):
    def test_preserves_decline_and_excludes_future(self):
        points = [dict(steps=1, seconds=10, wins=.9),
                  dict(steps=2, seconds=20, wins=.4),
                  dict(steps=3, seconds=21, wins=1.)]
        self.assertEqual(latest_available(points, 20), points[1])

    def test_missing_is_not_zero(self):
        self.assertIsNone(latest_available([dict(steps=1, seconds=61, wins=.8)], 60))
        self.assertEqual(latest_available([dict(steps=1, seconds=60, wins=0.)], 60)["wins"], 0.)

    def test_bad_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            latest_available([dict(steps=1, seconds=20, wins=.4),
                              dict(steps=2, seconds=10, wins=.8)], 60)

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            latest_available([dict(steps=1, seconds=1, wins=float("nan"))], 60)


if __name__ == "__main__":
    unittest.main()
