"""Native host registration/scalar/artifact checks, never CPU/GPU neural tests."""
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/results/flappycnn/geometric-preparation-20261006/packet"
MODELS = {"flex_quality": "default", "nature_cnn": "default", "impala_cnn": "impala", "impoola_cnn": "impoola"}
HEADERS = {name: ROOT / f"ocean/{name}/{name}.h" for name in
    ("connect4cnn", "pongcnn", "flappycnn", "breakoutcnn", "snakecnn", "mazecnn")}
HEADERS["snakecnn"] = ROOT / "ocean/snakecnn/game.h"


class NativePolicyMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import configparser
        cls.binaries = Path(os.environ.get("POLICY_METADATA_DIR", ROOT / "build/policy-metadata/native-20261006")).resolve()
        cls.costs = Path(os.environ.get("POLICY_METADATA_COSTS", ROOT / "build/policy-metadata/costs-20261006")).resolve()
        for path in (cls.costs, *(cls.binaries / name for name in set(MODELS.values()))):
            if not path.is_file(): raise RuntimeError(f"Compile host metadata/scalar targets first: {path}")
        cls.plan = json.loads((PACKET / "plan.json").read_text())
        cls.actions = {}
        cls.header_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in HEADERS.values()}
        for task, path in HEADERS.items():
            match = re.search(r"^#define ACT_SIZES \{([0-9]+)\}$", path.read_text(), re.M)
            if match is None: raise ValueError(f"Unsupported action declaration: {path}")
            cls.actions[task] = int(match[1])
        cls.ConfigParser = configparser.ConfigParser
        cls.capture = Path(os.environ["POLICY_METADATA_CHECKS"]).resolve() if "POLICY_METADATA_CHECKS" in os.environ else None
        if cls.capture is not None: cls.capture.mkdir(parents=True, exist_ok=False)
        cls.calls = []
        cls.scalar_cache = {}
        cls.binary_hashes = {name: hashlib.sha256((cls.binaries / name).read_bytes()).hexdigest() for name in set(MODELS.values())}
        cls.scalar_hash = hashlib.sha256(cls.costs.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        for path, digest in cls.header_hashes.items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Action header changed during host checks: {path}")
        for name, digest in cls.binary_hashes.items():
            if hashlib.sha256((cls.binaries / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Metadata binary changed during checks: {name}")
        if hashlib.sha256(cls.costs.read_bytes()).hexdigest() != cls.scalar_hash:
            raise ValueError("Scalar count binary changed during checks")
        if cls.capture is not None:
            (cls.capture / "calls.json").write_text(json.dumps(dict(
                native_host_calls=cls.calls, gpu_executed=False, policy_executed=False,
                numerical_acceptance=False, checkpoint_contents_validated=False,
                action_headers_sha256=cls.header_hashes, binaries_sha256=cls.binary_hashes,
                scalar_binary_sha256=cls.scalar_hash, independent_scalar_rows=[dict(hidden=h,layers=l,actions=a,rows=rows)
                    for (h,l,a),rows in sorted(cls.scalar_cache.items())]), indent=2) + "\n")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, model="flex_quality", task="connect4cnn", hidden=128, layers=1):
        binding = next(b for b in self.plan["bindings"] if b["environment"] == task and b["model"] == model)
        config = self.ConfigParser(interpolation=None); config.read(PACKET / binding["config"])
        config["policy"].update(hidden_size=str(hidden), num_layers=str(layers))
        return config

    def call(self, config, family="default", actions=7, checkpoint=None, ok=True):
        path = self.root / f"config-{len(self.calls)}.ini"
        with path.open("x") as stream: config.write(stream)
        command = [str(self.binaries / family), "--describe" if checkpoint is None else "--check-bytes", str(path), str(actions)]
        if checkpoint is not None: command.append(str(checkpoint))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
        record = dict(test=self._testMethodName, command=command, cwd=str(ROOT), returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, config=dict((section,dict(config[section])) for section in config.sections()))
        self.calls.append(record)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        if not ok: return result.stderr
        data = json.loads(result.stdout)
        for key in ("cuda_executed", "gpu_queried", "weights_initialized", "policy_executed",
                    "checkpoint_contents_validated", "numerical_acceptance"):
            self.assertIs(data[key], False)
        return data

    def scalar(self, hidden, layers, actions):
        key = (hidden,layers,actions)
        if key not in self.scalar_cache:
            result = subprocess.run([str(self.costs),*(str(n) for n in key)], cwd=ROOT,
                text=True,capture_output=True,check=True,timeout=30)
            self.scalar_cache[key] = {row["model"]: row for row in csv.DictReader(io.StringIO(result.stdout))}
        return self.scalar_cache[key]

    def test_six_game_four_family_full_policy_grid_matches_independent_c_arithmetic(self):
        for task, actions in self.actions.items():
            for model, family in MODELS.items():
                for hidden in (64,128,256):
                    for layers in (1,2):
                        with self.subTest(task=task,model=model,hidden=hidden,layers=layers):
                            data = self.call(self.config(model,task,hidden,layers),family,actions)
                            expected = self.scalar(hidden,layers,actions)[model]
                            for actual, column in (("encoder_parameters","encoder_parameters"),
                                ("decoder_parameters","head_parameters"),("core_parameters","core_parameters"),
                                ("total_parameters","total_parameters")):
                                self.assertEqual(data[actual],int(expected[column]))
                            self.assertEqual(data["model"],model)
                            self.assertEqual(data["observation_shape"],[1,36,44])
                            self.assertEqual(data["discrete_actions"],actions)
                            self.assertTrue(data["raw_checkpoint_contiguous"])
                            self.assertEqual(data["parameter_payload_bytes"],data["total_parameters"]*4)
                            self.assertIsNone(data["checkpoint_bytes"])
                            tensors = data["parameter_tensors"]
                            self.assertEqual([p["index"] for p in tensors],list(range(len(tensors))))
                            offset = 0
                            for tensor in tensors:
                                self.assertEqual(math.prod(tensor["shape"]),tensor["elements"])
                                self.assertEqual(tensor["offset_bytes"],offset)
                                offset += tensor["elements"]*4
                            self.assertEqual(offset,data["parameter_allocator_bytes"])
                            for group, field in (("encoder","encoder_parameters"),("decoder","decoder_parameters"),("core","core_parameters")):
                                self.assertEqual(sum(t["elements"] for t in tensors if t["group"] == group),data[field])

    def test_quality_h64_projection_branch_and_actual_parameter_order(self):
        small = self.call(self.config(hidden=64)); large = self.call(self.config(hidden=128))
        self.assertEqual(sum(t["group"] == "encoder" for t in small["parameter_tensors"]),4)
        self.assertEqual(sum(t["group"] == "encoder" for t in large["parameter_tensors"]),6)
        for data in (small,large):
            self.assertEqual([t["group"] for t in data["parameter_tensors"]][-2:],["decoder","core"])
        self.assertEqual(large["total_parameters"],160736)

    def test_checkpoint_size_is_checked_before_any_policy_execution_but_not_contents(self):
        config = self.config(); data = self.call(config)
        artifact = self.root / "synthetic-size-only.bin"
        # Sparse synthetic artifact, not initialized weights or a trained policy.
        with artifact.open("wb") as stream:
            stream.truncate(data["parameter_payload_bytes"])
            stream.seek(0); stream.write(struct.pack("<f",float("nan")))
        checked = self.call(config,checkpoint=artifact)
        self.assertEqual(checked["checkpoint_bytes"],data["parameter_payload_bytes"])
        self.assertFalse(checked["checkpoint_contents_validated"])
        for difference in (-4,4):
            with artifact.open("wb") as stream: stream.truncate(data["parameter_payload_bytes"]+difference)
            self.assertIn("byte count differs",self.call(config,checkpoint=artifact,ok=False))
        with artifact.open("wb") as stream: stream.truncate(data["parameter_payload_bytes"])
        self.assertIn("byte count differs",self.call(self.config(hidden=64),checkpoint=artifact,ok=False))
        self.assertIn("byte count differs",self.call(config,actions=2,checkpoint=artifact,ok=False))
        self.assertIn("regular checkpoint",self.call(config,checkpoint=self.root,ok=False))
        self.assertIn("regular checkpoint",self.call(config,checkpoint=self.root / "missing.bin",ok=False))

    def test_invalid_or_unqualified_graph_core_and_action_values_reject(self):
        for section,key,value in (("policy","hidden_size","0"),("policy","hidden_size","127"),
            ("policy","hidden_size","128.5"),("policy","num_layers","0"),("policy","num_layers","17"),
            ("train","horizon","0"),("policy","encoder","5"),("policy","encoder","2.5"),
            ("policy","cnn_kernel_1","3"),("policy","cnn_channels_1","99999999")):
            with self.subTest(section=section,key=key,value=value):
                config = self.config(); config[section][key] = value; self.call(config,ok=False)
        for actions in ("0","65","3.5","nan","7junk"):
            with self.subTest(actions=actions): self.call(self.config(),actions=actions,ok=False)
        self.call(self.config("nature_cnn"),family="impala",ok=False)
        self.call(self.config("impala_cnn"),family="default",ok=False)

    def test_horizon_and_game_name_cannot_turn_size_acceptance_into_recipe_acceptance(self):
        config = self.config(); first = self.call(config)
        changed = copy.deepcopy(config); changed["train"]["horizon"] = "64"
        changed["train"]["learning_rate"] = "0.123"
        changed["base"]["env_name"] = "flappycnn"
        second = self.call(changed)
        self.assertEqual(first["parameter_tensors"],second["parameter_tensors"])
        self.assertEqual(first["total_parameters"],second["total_parameters"])
        self.assertEqual(second["horizon"],64)
        self.assertNotEqual(first["horizon"],second["horizon"])

    def test_host_descriptor_repeats_identically(self):
        config = self.config("nature_cnn",hidden=256,layers=2)
        self.assertEqual(self.call(config),self.call(config))

    def test_build_guards_reject_discovery_override_and_existing_destination(self):
        script = ROOT / "research/build_policy_metadata.sh"
        fresh = self.root / "fresh-build"
        for changes in (dict(NVCC_ARCH="native"),dict(NVCC_ARCH="compute_120"),
                        dict(NVCC_APPEND_FLAGS="-DNDEBUG"),dict(NVCC_PREPEND_FLAGS="-arch=native")):
            environment = dict(os.environ)
            environment.update(NVCC_ARCH="sm_120",NVCC_PREPEND_FLAGS="",NVCC_APPEND_FLAGS="")
            environment.update(changes)
            result = subprocess.run(["bash",str(script),str(fresh)],cwd=ROOT,env=environment,text=True,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0); self.assertFalse(fresh.exists())
        fresh.mkdir(); sentinel = fresh / "sentinel"; sentinel.write_text("keep")
        result = subprocess.run(["bash",str(script),str(fresh)],cwd=ROOT,text=True,capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0); self.assertEqual(sentinel.read_text(),"keep")


if __name__ == "__main__": unittest.main()
