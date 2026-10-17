#!/usr/bin/env python3
"""Register retained native image trainers for the shared candidate panel.

This freezes build/source/compiler identities and calls host-only native info.
It neither recompiles nor queries a GPU. Input JSON maps all six games to
default and optional impala/impoola build directories, relative to that JSON.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import candidate_panel as panels
from claim import require, save, sha


def register(args):
    path=args.builds.resolve(); declared=json.loads(path.read_text())
    require(set(declared)==set(panels.TASKS),"Need exactly the six image games")
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    try:
        sources=panels.capture(out); records={}; contracts={}
        for task, families in declared.items():
            require(isinstance(families,dict) and "default" in families
                    and set(families)<={"default","impala","impoola"},"Unsupported/incomplete image build families")
            records[task]={}
            for family,name in families.items():
                require(isinstance(name,str) and name,"Missing image build path")
                directory=(path.parent/name).resolve(); binary=directory/"puffer"
                owned=panels.policy_layout.source_hashes(directory/"source.sha256")
                native={n:h for n,h in owned.items() if n.startswith("src/") or
                        (n.startswith("ocean/") and Path(n).suffix in (".h",".cu",".c") and "/tests/" not in n)}
                required={"src/pufferl.cu","src/ocean.cu","ocean/connect4cnn/image.h",
                          f"ocean/{task}/image_v1.h",panels.policy_layout.HEADERS[task]}
                require(required<=set(native) and all(sources.get(n)==h for n,h in native.items()),
                        "Image trainer production sources differ; build fresh")
                hashes={}
                for line in (directory/"binaries.sha256").read_text().splitlines():
                    digest,filename=line.split(maxsplit=1); artifact=Path(filename.lstrip("*"))
                    require(artifact.parent==directory and artifact.name in ("puffer","metadata")
                            and artifact.name not in hashes and re.fullmatch(r"[0-9a-f]{64}",digest),
                            "Invalid original image binary manifest")
                    hashes[artifact.name]=digest
                require(hashes.get("puffer")==sha(binary),"Native image trainer differs from build receipt")
                identity=json.loads(subprocess.check_output([str(binary),"eval_exact_info"],text=True,timeout=30))
                panels.bridge.validate_identity(task,family,identity,panels.panel.ENVIRONMENTS[task])
                require("image_contract" in identity,"Legacy trainer cannot be registered as a realistic image build")
                contract=identity["image_contract"]
                if task in contracts: require(contracts[task]==contract,"Game's families have different image inputs")
                contracts[task]=contract
                h,w,c=contract["shape_hwc"]
                flags=["-DCNN_IMAGE_V1",f"-DCNN_IMAGE_HEIGHT={h}",f"-DCNN_IMAGE_WIDTH={w}",f"-DCNN_IMAGE_CHANNELS={c}"]
                if family!="default": flags.append("-DC4_IMPALA_CNN" if family=="impala" else "-DC4_IMPOOLA_CNN")
                require((directory/"image-flags.txt").read_text().split()==flags,"Image trainer flags differ")
                receipt=out/"builds"/task/family; receipt.mkdir(parents=True)
                copied={}
                for file in ("source.sha256","source-files.txt","binaries.sha256","compiler.txt","environment.txt",
                             "revision.txt","worktree.txt","source-check.txt","train-build.txt","image-flags.txt"):
                    shutil.copyfile(directory/file,receipt/file); copied[file]=sha(receipt/file)
                require(sha(binary)==hashes["puffer"],"Trainer changed during host info query")
                records[task][family]=dict(path=str(binary),sha256=hashes["puffer"],info=identity,
                    original_build=str(directory),native_source_sha256=native,receipt_sha256=copied)
        panels.verify_sources(out,sources,current=True)
        value=dict(protocol=panels.IMAGE_BUILD,binaries=records,image_contracts=contracts,source_sha256=sources,
                   gpu_queried=False,policy_executed=False,vendor_system_link_closure_complete=False)
        save(out/"registry.json",value); panels.registry_builds(value)
        return value
    except BaseException as error:
        save(out/"failure.json",dict(status="failed",error=str(error),policy_executed=False)); raise


def inspect(root, value, current=False):
    panels.registry_builds(value); panels.verify_sources(root,value["source_sha256"],current)
    for task,families in value["binaries"].items():
        for family,record in families.items():
            panels.bridge.validate_identity(task,family,record["info"],panels.panel.ENVIRONMENTS[task])
            require(all(value["source_sha256"].get(n)==h for n,h in record["native_source_sha256"].items()),
                    "Image registry native source differs")
            for name,digest in record["receipt_sha256"].items():
                require(sha(root/"builds"/task/family/name)==digest,"Image build receipt changed")
            if current: require(sha(Path(record["path"]))==record["sha256"],"Image trainer changed")
    return value


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--builds",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    value=register(args); inspect(args.out.resolve(),value,current=True)
    print(json.dumps(dict(status="host-image-builds-registered",games=len(value["binaries"]),gpu_queried=False)))
