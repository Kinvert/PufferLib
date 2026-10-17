"""Evidence binding/control-flow checks; CUDA and array audits are mocked."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_validation_collection as tool


class CollectionTests(unittest.TestCase):
    def exercise(self, change=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); catalog=root/"catalog"; catalog.mkdir()
            config=root/"config.ini"; config.write_text("frozen")
            value=dict(shapes={"h36-w44-c3":dict(shape_hwc=[36,44,3],bundle="a/bundle",bundle_sha256="a"*64),
                               "h44-w36-c3":dict(shape_hwc=[44,36,3],bundle="b/bundle",bundle_sha256="b"*64)},
                       batches=[1,4,16],games=dict(one="h36-w44-c3",two="h44-w36-c3",three="h36-w44-c3"))
            (catalog/"catalog.json").write_text(json.dumps(value))
            allocations={}; reports={}
            for shape,entry in value["shapes"].items():
                path=root/shape; path.mkdir(); allocations[shape]=path
                (path/"request.json").write_text(json.dumps(dict(bundle=str(catalog/entry["bundle"]),
                    bundle_sha256=entry["bundle_sha256"],hardware="5060")))
                reports[path]=dict(status="passed",schema=tool.graphs.IMAGE_SCHEMA,shape_hwc=entry["shape_hwc"],
                    batches=[1,4,16],config_sha256=tool.sha(config),hardware={"gpu_name":"RTX 5060","gpu_uuid":"paired"},cases=18,native_calls=72)
                (path/"REPORT.json").write_text(json.dumps(reports[path]))
            if change: change(value,allocations,reports)
            for path,report in reports.items(): (path/"REPORT.json").write_text(json.dumps(report))
            with patch.object(tool.catalogs,"inspect",return_value=value),patch.object(tool.graphs,"audit",side_effect=lambda p:{k:v for k,v in reports[p].items() if k!="hardware"}):
                result=tool.collect(catalog,config,allocations,"nature",root/"collection")
            manifest=json.loads((root/"collection/evidence.json").read_text())
            self.assertEqual(manifest["candidates"]["nature"]["one"],manifest["candidates"]["nature"]["three"])
            self.assertEqual(result["total_cases"],36)
            self.assertEqual(result["total_native_calls"],144)
            self.assertFalse(result["cuda_executed_in_collection"])

    def test_complete_shape_collection_shares_equal_shape_proof(self): self.exercise()

    def test_missing_or_failed_shapes_and_changed_inputs_are_refused(self):
        def missing(value,paths,reports): paths.pop(next(iter(paths)))
        def report_edit(field,new):
            def edit(value,paths,reports): reports[list(paths.values())[-1]][field]=new
            return edit
        def wrong_bundle(value,paths,reports):
            path=next(iter(paths.values()))/"request.json"
            request=json.loads(path.read_text());request["bundle_sha256"]="c"*64;path.write_text(json.dumps(request))
        for change in (missing,report_edit("status","failed"),report_edit("shape_hwc",[36,36,3]),
                       report_edit("batches",[1,4]),report_edit("config_sha256","d"*64),
                       report_edit("hardware",{"gpu_uuid":"other"}),wrong_bundle):
            with self.subTest(change=change),self.assertRaises(ValueError): self.exercise(change)


if __name__=="__main__": unittest.main()
