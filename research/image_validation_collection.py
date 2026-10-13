#!/usr/bin/env python3
"""Collect complete per-shape CUDA allocations without launching or rerunning them.

Every allocation is independently audited against one frozen catalog and one
configuration. A partial/failed allocation cannot contribute a successful shape.
Raw arrays remain in their original allocations; this is an evidence index.
"""
import argparse
import json
import os
from pathlib import Path

import cnn_graph_acceptance as graphs
import image_encoder_checks as evidence
import image_validation_catalog as catalogs
from claim import require, save, sha


def collect(catalog, config, allocations, name, out):
    catalog, config, out = catalog.resolve(), config.resolve(), out.resolve()
    value = catalogs.inspect(catalog, current=True)
    require(set(allocations) == set(value["shapes"]), "Need every declared image shape exactly once")
    require(name and name.replace("-", "").isalnum(), "Use a simple candidate name")
    out.mkdir(parents=True, exist_ok=False)
    checks, device = {}, None
    try:
        for shape, entry in value["shapes"].items():
            original = allocations[shape].resolve()
            report = graphs.audit(original)
            stored = json.loads((original/"REPORT.json").read_text())
            request = json.loads((original/"request.json").read_text())
            require(all(stored.get(key)==number for key,number in report.items()), "Changed audited shape report")
            require(report["status"] == "passed" and report["schema"] == graphs.IMAGE_SCHEMA
                    and report["shape_hwc"] == entry["shape_hwc"]
                    and report["batches"] == value["batches"]
                    and report["config_sha256"] == sha(config), "Failed or mismatched shape allocation")
            require(Path(request["bundle"]).resolve() == catalog/entry["bundle"]
                    and request["bundle_sha256"] == entry["bundle_sha256"],
                    "Shape allocation uses a different frozen catalog")
            hardware=stored.get("hardware")
            require(isinstance(hardware,dict) and isinstance(hardware.get("gpu_name"),str)
                    and request["hardware"] in hardware["gpu_name"] and hardware.get("gpu_uuid"),
                    "Missing or mismatched supervisor hardware receipt")
            require(device is None or hardware == device, "Mixed CUDA hardware in collection")
            device = hardware
            checks[shape] = dict(path=os.path.relpath(original, out),
                report_sha256=sha(original/"REPORT.json"), cases=report["cases"],
                native_calls=report["native_calls"])
        catalogs.inspect(catalog, current=True)
        manifest = dict(schema=evidence.SCHEMA,
            candidates={name:{task:checks[shape]["path"] for task,shape in value["games"].items()}})
        save(out/"evidence.json", manifest, exclusive=True)
        result = dict(status="passed", schema="native-image-validation-collection-v1",
            hardware=device, config_sha256=sha(config), catalog_sha256=sha(catalog/"catalog.json"),
            checks=checks, total_cases=sum(c["cases"] for c in checks.values()),
            total_native_calls=sum(c["native_calls"] for c in checks.values()),
            cuda_executed_in_collection=False, cpu_neural_reference=False,
            full_policy_qualified=False, learning_or_frontier_claim=False)
        save(out/"REPORT.json", result, exclusive=True)
        return result
    except BaseException as error:
        save(out/"failure.json", dict(status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("catalog", "config", "out"): parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("--allocation",action="append",required=True,help="SHAPE=completed-allocation-directory")
    args=parser.parse_args(); pairs=[item.split("=",1) for item in args.allocation]
    require(all(len(item)==2 for item in pairs) and len({item[0] for item in pairs})==len(pairs),
            "Duplicate or malformed shape allocation")
    result=collect(args.catalog,args.config,{key:Path(path) for key,path in pairs},args.name,args.out)
    print(json.dumps(dict(status=result["status"],cases=result["total_cases"],native_calls=result["total_native_calls"])))
