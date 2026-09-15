"""CPU regressions for query failures and incomplete evaluation evidence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'ocean/pongcnn'))
import eval_profile_audit as audit


class AuditTests(unittest.TestCase):
    def test_failed_empty_gpu_query_is_not_idle(self):
        result = subprocess.run(['bash','-c', 'source ocean/pongcnn/gpu_env.sh; PONG_SMI=false; if pong_gpu_idle; then exit 99; else exit 0; fi'], cwd=ROOT)
        self.assertEqual(result.returncode, 0)

    def test_successful_empty_gpu_query_is_idle(self):
        result = subprocess.run(['bash','-c', 'source ocean/pongcnn/gpu_env.sh; PONG_SMI=true; pong_gpu_idle'], cwd=ROOT)
        self.assertEqual(result.returncode, 0)

    def test_path_gpu_discovery_precedes_wsl(self):
        with tempfile.TemporaryDirectory() as tmp:
            smi = Path(tmp)/'nvidia-smi'
            smi.write_text('#!/bin/sh\nexit 0\n')
            smi.chmod(0o755)
            value = subprocess.check_output(['bash','-c', 'source ocean/pongcnn/gpu_env.sh; PATH="$1:$PATH"; pong_find_smi', 'test', tmp], cwd=ROOT,text=True)
            self.assertEqual(value.strip(), str(smi))

    def test_progress_and_partial_score_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root/'weights.bin'
            checkpoint.write_bytes(b'1234')
            (root/'output.log').write_text('CUDA_EVAL_PROGRESS steps=204800 games=13 requested=512 elapsed=5.001\n')
            (root/'exit-code.txt').write_text('124\n')
            a = dict(name='.',group='panel',checkpoint=str(checkpoint),checkpoint_sha256=audit.sha(checkpoint),games=512,model='flex_quality')
            r = audit.result(root,a)
            self.assertFalse(r['success'])
            self.assertEqual(r['final'],[])
            self.assertEqual(r['progress'][-1]['games'],13)
            (root/'exit-code.txt').unlink()
            self.assertEqual(audit.result(root,a)['exit_code'],'incomplete')


if __name__ == '__main__': unittest.main()
