"""CPU-only checks: portable tool discovery and fail-closed GPU guards."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class RuntimeTest(unittest.TestCase):
    def run_guard(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "include").mkdir()
            (root / "include/nccl.h").touch()
            tool = root / "nvidia-smi"
            tool.write_text("#!/bin/bash\n" + body + "\n")
            tool.chmod(0o755)
            env = dict(os.environ, NCCL_ROOT=tmp, PATH=tmp + ":" + os.environ["PATH"])
            result = subprocess.run([
                "/bin/bash", "-c",
                'set -eu; source ocean/connect4cnn/runtime_env.sh; '
                'tool=$(puffer_find_nvidia_smi); echo "$tool"; '
                'puffer_require_idle_gpu "$tool"',
            ], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.stdout.splitlines()[0], str(tool))
            return result

    def test_path_tool_and_idle(self):
        self.assertEqual(self.run_guard("printf '  \\n'").returncode, 0)

    def test_busy_gpu_rejected(self):
        result = self.run_guard("echo 1234")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU busy", result.stderr)

    def test_query_failure_is_not_idle(self):
        self.assertEqual(self.run_guard("exit 9").returncode, 9)


if __name__ == "__main__":
    unittest.main()
