"""Verify coverage reporting for both flexible encoder result schemas."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from audit_sweep_coverage import summarize


class CoverageTest(unittest.TestCase):
    def test_old_and_expanded_families(self):
        architectures = [
            {"policy.cnn_depth": 1, "policy.cnn_pool_1": 0, "policy.cnn_global_pool": 1},
            {"policy.cnn_depth": 2, "policy.cnn_pool_1": 0, "policy.cnn_pool_2": 2,
             "policy.cnn_dilation_1": 1, "policy.cnn_dilation_2": 3,
             "policy.cnn_activation_1": 0, "policy.cnn_activation_2": 4,
             "policy.cnn_readout": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("architecture_json", "architecture_sha256", "steps"))
                writer.writeheader()
                for index, architecture in enumerate(architectures):
                    writer.writerow({"architecture_json": json.dumps(architecture),
                                     "architecture_sha256": str(index), "steps": 13312000})
            report = summarize(path, 8000000)
        self.assertEqual(report["distinct_active_architectures"], 2)
        self.assertEqual(report["depth"]["1"]["global_pool"], {"1": 1})
        self.assertEqual(report["depth"]["2"]["pooling"], {"0": 1, "2": 1})
        self.assertEqual(report["depth"]["2"]["dilation"], {"1": 1, "3": 1})
        self.assertEqual(report["depth"]["2"]["activation"], {"0": 1, "4": 1})
        self.assertEqual(report["depth"]["2"]["readout"], {"2": 1})
        self.assertEqual(report["depth"]["2"]["controls"]["policy.cnn_dilation_2"], {"3": 1})


if __name__ == "__main__":
    unittest.main()
