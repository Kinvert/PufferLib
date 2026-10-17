"""CPU checks for portable GPU inspection and safe fixed-model dispatch."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import compare


class HardwareTests(unittest.TestCase):
    def test_gpu_tool_prefers_path_then_wsl(self):
        with patch.object(compare.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            self.assertEqual(compare.find_nvidia_smi(), "/usr/bin/nvidia-smi")
        with patch.object(compare.shutil, "which", return_value=None), patch.object(compare.Path, "is_file", return_value=True):
            self.assertEqual(compare.find_nvidia_smi(), "/usr/lib/wsl/lib/nvidia-smi")
        with patch.object(compare.shutil, "which", return_value=None), patch.object(compare.Path, "is_file", return_value=False):
            with self.assertRaises(RuntimeError):
                compare.find_nvidia_smi()

    def test_busy_gpu_is_rejected(self):
        with patch.object(compare.subprocess, "check_output", return_value="12345\n"):
            with self.assertRaises(RuntimeError):
                compare.require_idle_gpu("nvidia-smi")
        with patch.object(compare.subprocess, "check_output", return_value=""):
            compare.require_idle_gpu("nvidia-smi")

    def test_wrapper_selects_frozen_full_budget(self):
        command = subprocess.check_output(["bash", str(HERE / "hardware_compare.sh"), "--print-command"], text=True)
        for text in ("flex_quality nature_cnn impala_cnn impoola_cnn", "--steps 13312000", "--seeds 173", "--checkpoints 13", "--require-idle-gpu"):
            self.assertIn(text, command)


if __name__ == "__main__":
    unittest.main()
