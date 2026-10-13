#!/usr/bin/env python3
"""Build/run independent CUDA encoder checks for an all-game image panel.

Build performs no GPU query. Run uses the existing reserved CUDA graph gate,
one fresh allocation per distinct image shape, retaining every result/failure.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

import candidate_panel as panels
import cnn_graph_acceptance as graphs
import image_encoder_checks as evidence
from claim import process, require, save, sha

SCHEMA="native-image-validation-catalog-v1"


def build(args):
    registry=json.loads(args.registry.read_text()); panels.registry_builds(registry)
    require(registry["protocol"]==panels.IMAGE_BUILD,"Need real image build identities")
    require(re.fullmatch(r"sm_[0-9]+[a-z]?",os.environ.get("NVCC_ARCH","")),"Explicit compile architecture required")
    require(not any(os.environ.get(k) for k in ("NVCC_EXTRA","NVCC_APPEND_FLAGS","NVCC_PREPEND_FLAGS")),"Unrecorded compiler overrides refused")
    batches=args.batches
    require(batches==sorted(set(batches)) and batches[0]==1 and all(type(b) is int and 1<=b<=64 for b in batches),"Invalid image check batches")
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    shapes={}; games={}
    reuse=getattr(args,"libraries_from",None)
    previous=inspect(reuse.resolve()) if reuse else None
    try:
        for task,contract in registry["image_contracts"].items():
            h,w,c=contract["shape_hwc"]; name=f"h{h}-w{w}-c{c}"; games[task]=name
            if name in shapes: continue
            directory=out/name; directory.mkdir()
            if reuse:
                require(name in previous["shapes"],"Retained libraries do not cover this input shape")
                retained=reuse.resolve()/previous["shapes"][name]["bundle"]
                for kind in ("native","reference"):
                    shutil.copyfile(retained/(kind+".so"),directory/(kind+".so"))
                    shutil.copytree(retained/(kind+".build"),directory/(kind+".so.build"))
                libraries=directory
            else:
                env=dict(os.environ,CNN_TEST_IMAGE=f"{h},{w},{c}",C4_DENSE_PATCH_ALIAS="0")
                process(["bash","ocean/connect4cnn/tests/build_encoder_test.sh","test_nature",str(directory/"native.so")],
                        panels.ROOT,directory/"native-build.txt",600,env)
                process(["bash","research/build_native_flex_reference.sh",str(directory/"reference.so")],
                        panels.ROOT,directory/"reference-build.txt",600,env)
                libraries=directory
            # bundle() checks every retained native source and binary against
            # current production, then freezes today's supervisor source. No
            # old CUDA result is carried into this fresh, unexecuted catalog.
            graphs.bundle(argparse.Namespace(native=libraries/"native.so",reference=libraries/"reference.so",
                out=directory/"bundle",image_shape=[h,w,c],batches=batches))
            shapes[name]=dict(shape_hwc=[h,w,c],bundle=str((directory/"bundle").relative_to(out)),
                              bundle_sha256=sha(directory/"bundle/bundle.json"))
        value=dict(schema=SCHEMA,games=games,shapes=shapes,batches=batches,registry_sha256=sha(args.registry),
                   gpu_queried=False,cuda_executed=False,numerical_acceptance=False)
        if reuse: value["retained_libraries_catalog_sha256"]=sha(reuse.resolve()/"catalog.json")
        save(out/"catalog.json",value); inspect(out,current=True); return value
    except BaseException as error:
        save(out/"failure.json",dict(status="failed",error=str(error))); raise


def inspect(root,current=False):
    value=json.loads((root/"catalog.json").read_text())
    require(value["schema"]==SCHEMA and set(value["games"])==set(panels.TASKS)
            and set(value["games"].values())==set(value["shapes"]),"Incomplete image shape validation catalog")
    for name,record in value["shapes"].items():
        h,w,c=record["shape_hwc"]
        require(name==f"h{h}-w{w}-c{c}" and record["bundle"]==name+"/bundle","Malformed image shape/bundle path")
        bundle=root/record["bundle"]; descriptor=graphs.inspect_bundle(bundle,current=current)
        require(sha(bundle/"bundle.json")==record["bundle_sha256"] and descriptor["schema"]==graphs.IMAGE_SCHEMA
                and descriptor["shape_hwc"]==record["shape_hwc"] and descriptor["batches"]==value["batches"],
                "Changed image numerical bundle")
    return value


def run(args):
    catalog=args.catalog.resolve(); value=inspect(catalog,current=True)
    require(re.fullmatch(r"[a-z][a-z0-9-]{0,63}",args.name),"Use a simple candidate name")
    require(0<args.timeout<=120,"Bounded per-worker numerical deadline required")
    deadline=getattr(args,"deadline",None)
    graphs.worker_timeout(args.timeout,deadline)
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    try:
        hardware=None; checks={}; original=sha(args.config)
        for name,record in value["shapes"].items():
            graphs.worker_timeout(args.timeout,deadline)
            report=graphs.run(catalog/record["bundle"],args.config.resolve(),out/name,args.hardware,args.timeout,deadline=deadline)
            if hardware is None: hardware=report["hardware"]
            require(report["hardware"]==hardware,"Mixed image check hardware")
            checks[name]=dict(path=name,report_sha256=sha(out/name/"REPORT.json"),cases=report["cases"],native_calls=report["native_calls"])
        require(sha(args.config)==original,"Candidate configuration changed")
        manifest=dict(schema=evidence.SCHEMA,candidates={args.name:{task:value["games"][task] for task in panels.TASKS}})
        save(out/"evidence.json",manifest)
        result=dict(status="passed",hardware=hardware,config_sha256=original,catalog_sha256=sha(catalog/"catalog.json"),
                    checks=checks,total_native_calls=sum(c["native_calls"] for c in checks.values()),
                    cpu_neural_reference=False,full_policy_qualified=False)
        save(out/"REPORT.json",result); return result
    except BaseException as error:
        save(out/"failure.json",dict(status="failed",error=str(error))); raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build"); b.add_argument("--registry",type=Path,required=True); b.add_argument("--batches",type=int,nargs="+",default=[1,4,16])
    b.add_argument("--libraries-from",type=Path,help="Recheck unchanged owned native sources/binaries and freeze fresh tooling; never reuse numerical results")
    r=sub.add_parser("run"); r.add_argument("--catalog",type=Path,required=True); r.add_argument("--name",required=True)
    r.add_argument("--config",type=Path,required=True); r.add_argument("--hardware",choices=("5060","5090"),required=True)
    r.add_argument("--timeout",type=int,default=120)
    for p in (b,r): p.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); result=build(args) if args.command=="build" else run(args)
    print(json.dumps(dict(status=result.get("status","built-not-executed"),cpu_neural_reference=False)))
