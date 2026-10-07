"""Scalar descriptor checks only; no CPU network, GPU query or policy."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("alias_audit", Path(__file__).parents[1] / "audit_dense_patch_alias.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class AliasMetadataTests(unittest.TestCase):
    def fixture(self, model="flex_quality", hidden=128, batch=64):
        base = dict(schema=1, model=model, hidden=hidden, batch=batch,
                    cuda_executed=False, policy_executed=False,
                    encoder_parameters=123, train_payload_bytes=456)
        layers, width = 0, 0
        if model == "flex_quality":
            layers, width = (1, 1584) if hidden == 64 else (2, 1648)
        elif model == "nature_cnn":
            layers, width = 1, 128
        candidate = dict(base, dense_patch_alias_candidate=True, dense_patch_alias_active=bool(layers),
                         dense_patch_alias_layers=layers, skipped_forward_patch_payload_bytes=width * batch * 4)
        return base, candidate

    def test_all_48_declared_shapes_and_unmodified_controls(self):
        for model in a.FAMILIES:
            for hidden in (64, 128, 256):
                for batch in (1, 64, 2048, 16384):
                    with self.subTest(model=model, hidden=hidden, batch=batch):
                        a.compare(*self.fixture(model, hidden, batch))

    def test_changed_registration_rejected(self):
        for key in ("encoder_parameters", "train_payload_bytes", "batch", "hidden"):
            base, candidate = self.fixture()
            candidate[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "registration"):
                a.compare(base, candidate)

    def test_missing_extra_and_wrong_typed_variant_fields_rejected(self):
        for key in a.ALIAS_KEYS:
            base, candidate = self.fixture()
            del candidate[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.compare(base, candidate)
        for key, wrong in (("dense_patch_alias_candidate", 1), ("dense_patch_alias_active", 1),
                           ("dense_patch_alias_layers", 2.0), ("skipped_forward_patch_payload_bytes", 421889)):
            base, candidate = self.fixture()
            candidate[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.compare(base, candidate)
        base, candidate = self.fixture()
        base["dense_patch_alias_candidate"] = False
        with self.assertRaises(ValueError): a.compare(base, candidate)

    def test_executed_policy_rejected(self):
        for key in ("cuda_executed", "policy_executed"):
            base, candidate = self.fixture()
            base[key] = candidate[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "host-only"):
                a.compare(base, candidate)

    def test_inactive_impala_cannot_claim_a_copy_reduction(self):
        base, candidate = self.fixture("impala_cnn")
        candidate["dense_patch_alias_layers"] = 1
        with self.assertRaises(ValueError): a.compare(base, candidate)

    def test_source_closure_rejects_changed_unsafe_and_duplicate_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "file.cu"
            source.write_text("scalar test fixture\n")
            receipt = root / "source.sha256"
            line = f"{a.sha(source)}  file.cu\n"
            receipt.write_text(line)
            self.assertEqual(a.manifest(receipt, root), {"file.cu": a.sha(source)})
            receipt.write_text(line * 2)
            with self.assertRaises(ValueError): a.manifest(receipt, root)
            receipt.write_text(line.replace("file.cu", "../file.cu"))
            with self.assertRaises(ValueError): a.manifest(receipt, root)
            receipt.write_text(line)
            source.write_text("changed fixture\n")
            with self.assertRaisesRegex(ValueError, "changed"): a.manifest(receipt, root)

    def test_candidate_build_rejects_native_arch_wrong_target_and_overwrite(self):
        root = Path(__file__).resolve().parents[2]
        script = root / "ocean/connect4cnn/tests/build_encoder_test.sh"
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "original.so"
            existing.write_bytes(b"retained fixture; not a model library")
            cases = [("native", ["test_nature", str(Path(tmp) / "new.so")], "explicit NVCC_ARCH"),
                     ("sm_120", ["test_nature"], "fresh explicit output"),
                     ("sm_120", ["test_nature", str(existing)], "fresh explicit output"),
                     ("sm_120", ["test_encoder", str(Path(tmp) / "new.so")], "requires test_nature")]
            for arch, arguments, message in cases:
                env = dict(os.environ, C4_DENSE_PATCH_ALIAS="1", NVCC_ARCH=arch)
                result = subprocess.run(["bash", str(script), *arguments], cwd=root,
                                        env=env, capture_output=True, text=True, timeout=30)
                with self.subTest(arguments=arguments, arch=arch):
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(existing.read_bytes(), b"retained fixture; not a model library")
                    self.assertFalse((Path(tmp) / "new.so").exists())


if __name__ == "__main__": unittest.main()
