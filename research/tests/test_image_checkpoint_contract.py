"""Descriptor and schema checks only; synthetic artifacts, no neural execution."""
import configparser
import copy
from pathlib import Path
import sys
import unittest
import json
import os
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"ocean/connect4cnn"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import checkpoint_preflight as gate
import prepare_image_inputs as images


class ImageCheckpointContractTests(unittest.TestCase):
    def setUp(self):
        self.config=configparser.ConfigParser(); self.config.read_dict(dict(
            policy=dict(hidden_size="128",num_layers="1"),train=dict(horizon="16")))
        self.contract=dict(version="human-geometry-v1",shape_hwc=[640,420,3],
                           layout="NHWC",dtype="float32",preprocessing="none")
        self.value=dict(schema="native-policy-metadata-v2",environment="flappycnn",image_contract=self.contract,
                        observation_shape=[3,640,420],discrete_actions=2,hidden=128,layers=1,horizon=16,
                        model="flex",raw_checkpoint_contiguous=True,checkpoint_bytes=160,
                        parameter_payload_bytes=160,parameter_allocator_bytes=160,total_parameters=40,
                        encoder_parameters=16,decoder_parameters=8,core_parameters=16,
                        parameter_tensors=[dict(index=i,group=group,offset_bytes=offset,elements=n,shape=shape)
                            for i,(group,offset,n,shape) in enumerate((
                                ("encoder",0,16,[4,4]),("decoder",64,8,[2,4]),("core",96,16,[4,4])))],
                        **{key:False for key in gate.FALSE_FLAGS})

    def check(self,value):
        gate.validate_descriptor(value,self.config,"flex",2,160,self.contract,"flappycnn")

    def test_rgb_requires_exact_frozen_contract_and_game(self):
        self.check(self.value)
        for key, value in (("environment","pongcnn"),("observation_shape",[3,420,640]),
                           ("schema","native-policy-metadata-v1"),("gpu_queried",True)):
            changed=copy.deepcopy(self.value); changed[key]=value
            with self.assertRaises(ValueError): self.check(changed)

    def test_image_descriptor_cannot_pass_legacy_validator(self):
        with self.assertRaises(ValueError):
            gate.validate_descriptor(self.value,self.config,"flex",2,160)

    def test_legacy_shape_keeps_original_contract(self):
        legacy={**self.value,"schema":"native-policy-metadata-v1","observation_shape":[1,36,44]}
        legacy.pop("image_contract"); legacy.pop("environment")
        gate.validate_descriptor(legacy,self.config,"flex",2,160)
        with self.assertRaises(ValueError): self.check(legacy)

    def test_rehashed_layout_preprocessing_and_dtype_do_not_pass(self):
        for key, value in (("preprocessing","resize-36x44"),("layout","NCHW"),("dtype","uint8"),
                           ("shape_hwc",[640,420,1]),("shape_hwc",[640,True,3])):
            changed=copy.deepcopy(self.value); changed["image_contract"][key]=value
            with self.assertRaises(ValueError): self.check(changed)

    def test_byte_length_and_tensor_offsets_remain_strict(self):
        changed=copy.deepcopy(self.value); changed["parameter_tensors"][2]["offset_bytes"]=100
        with self.assertRaises(ValueError): self.check(changed)
        changed=copy.deepcopy(self.value); changed["checkpoint_bytes"]=164
        with self.assertRaises(ValueError): self.check(changed)

    def test_per_game_binary_selection_does_not_reuse_connect4_metadata(self):
        image=dict(schema=gate.IMAGE_SCHEMA)
        self.assertEqual(gate.binary_key(image,"flappycnn","nature"),"flappycnn/default")
        self.assertEqual(gate.binary_key(image,"pongcnn","impoola"),"pongcnn/impoola")
        self.assertEqual(gate.binary_key({},"flappycnn","nature"),"default")


@unittest.skipUnless(os.environ.get("IMAGE_METADATA_REGISTRY"),"Explicit frozen native image registry needed")
class RetainedImageRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.original=Path(os.environ["IMAGE_METADATA_REGISTRY"]).resolve()
        self.root=Path(self.temp.name)/"copy"
        gate.freeze(self.original,self.root)

    def test_actual_registry_relocation_is_artifact_only(self):
        self.assertEqual(gate.inspect_registry(self.original),gate.inspect_registry(self.root))
        self.assertEqual(set(gate.inspect_registry(self.root)["image_contracts"]),set(gate.HEADERS))

    def test_rehashed_shape_identity_cannot_replace_compiler_geometry(self):
        value=json.loads((self.root/"registry.json").read_text())
        value["image_contracts"]["flappycnn"]["shape_hwc"]=[420,640,3]
        for family in ("default","impala","impoola"):
            key="flappycnn/"+family
            if key not in value["builds"]: continue
            path=self.root/"builds"/key/"metadata-info.json"; info=json.loads(path.read_text())
            info["image_contract"]=value["image_contracts"]["flappycnn"]; path.write_text(json.dumps(info))
            value["builds"][key]["receipts"]["metadata-info.json"]=gate.sha(path)
        (self.root/"registry.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"compiler flags"):
            gate.inspect_registry(self.root)

    def test_rehashed_binary_manifest_cannot_replace_registered_tool(self):
        value=json.loads((self.root/"registry.json").read_text()); key="flappycnn/default"
        path=self.root/"builds"/key/"binaries.sha256"; text=path.read_text()
        text=text.replace(value["binaries"][key],"f"*64); path.write_text(text)
        value["builds"][key]["receipts"]["binaries.sha256"]=gate.sha(path)
        (self.root/"registry.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"binary provenance"):
            gate.inspect_registry(self.root)


@unittest.skipUnless(os.environ.get("LEGACY_IMAGE_METADATA_DIR"),"Fresh legacy host tools needed")
class LegacyNativeConstructorTests(unittest.TestCase):
    def test_actual_legacy_layouts_keep_historical_counts_and_schema(self):
        tools=Path(os.environ["LEGACY_IMAGE_METADATA_DIR"]).resolve()
        out=Path(os.environ["LEGACY_IMAGE_METADATA_EVIDENCE"]).resolve()
        out.mkdir(parents=True,exist_ok=False)
        cases=(("flex","default",images.CONTROLS["quality"],160736),
               ("nature","default",images.CONTROLS["nature"],138528),
               ("impala","impala",dict(encoder=0),270496),
               ("impoola","impoola",dict(encoder=0),151712))
        records=[]
        for family,binary,arch,expected in cases:
            config,_=images.config("connect4cnn",arch); path=out/(family+".ini")
            with path.open("x") as stream: config.write(stream)
            gate.process([str(tools/binary),"--describe-flex",str(path),"7"],gate.ROOT,out/(family+".txt"),30)
            value=json.loads((out/(family+".txt")).read_text())
            gate.validate_registration(value,config,family,7)
            self.assertEqual(value["total_parameters"],expected)
            self.assertNotIn("image_contract",value)
            records.append(dict(family=family,parameters=expected,descriptor_sha256=gate.sha(out/(family+".txt"))))
        gate.save(out/"REPORT.json",dict(status="passed",legacy_schema="native-policy-metadata-v1",cases=records,
                                       gpu_queried=False,cuda_executed=False,numerical_acceptance=False))


if __name__ == "__main__": unittest.main()
