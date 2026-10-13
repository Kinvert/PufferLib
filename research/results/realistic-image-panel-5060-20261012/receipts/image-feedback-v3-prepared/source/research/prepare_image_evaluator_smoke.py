#!/usr/bin/env python3
"""Bind one new-image checkpoint to existing exact evaluator acceptance.

Preparation runs native host identity/manifest/stat tools only. A separate
explicit eval_acceptance.py run command schedules the four GPU comparisons.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ocean/connect4cnn"))
import checkpoint_preflight as layouts
import eval_acceptance as acceptance
from claim import require, save, sha


def prepare(args):
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    try:
        require(args.task in acceptance.TASKS, "This acceptance adapter covers five games; Connect4 uses its dedicated versioned checker")
        metadata=args.metadata.resolve(); registry=layouts.inspect_registry(metadata,current=True)
        require(registry.get("schema")==layouts.IMAGE_SCHEMA,"Need per-game image metadata registry")
        build=args.build.resolve(); sources=layouts.source_hashes(build/"source.sha256")
        # Match actual constructor/action/encoder/raster sources. Auxiliary
        # historical tests and Python supervisors are not compiled dependencies.
        for name,digest in registry["sources"].items():
            if name in sources and (name.startswith("src/") or
                    (name.startswith("ocean/") and Path(name).suffix in (".h",".cu",".c") and "/tests/" not in name)):
                require(sources[name]==digest,"Trainer and metadata production sources differ: "+name)
        required={"src/ocean.cu","src/pufferl.cu","ocean/connect4cnn/image.h",layouts.HEADERS[args.task],
                  f"ocean/{args.task}/image_v1.h"}
        require(required<=set(sources) and all(sources[n]==registry["sources"][n] for n in required),
                "Missing image trainer provenance")
        ev=acceptance.adapter(args.task); binary=build/"puffer"
        identity=ev.info(binary)
        require(identity["environment"]==args.task and identity["image_contract"]==registry["image_contracts"][args.task],
                "Trainer and host descriptor image contracts differ")
        config=args.config.resolve(); checkpoint=args.checkpoint.resolve()
        require(type(args.eval_seed) is int and 0<=args.eval_seed<2**32
                and args.eval_seed != ev.read(config).getint("base","seed"), "Explicit separate evaluation seed required")
        # Five exact episodes in four slots exercise the independent second-wave
        # tail. No outcome/architecture selection or training happens here.
        ev.create_suite(SimpleNamespace(binary=binary,config=config,out=root/"suite",seed=args.eval_seed,
            episodes=5,slots=4,offset=0,representation=0,purpose="development"))
        cases=[dict(name="image-checkpoint",task=args.task,family="flex",binary=str(binary),config=str(config),
                    checkpoint=str(checkpoint),suite=str(root/"suite/suite.json"))]
        save(root/"cases.json",cases)
        acceptance.prepare(SimpleNamespace(cases=root/"cases.json",out=root/"packet",timeout=30,policy_metadata=metadata))
        acceptance.load_packet(root/"packet/packet.json",current_inputs=True)
        result=dict(schema="native-image-evaluator-smoke-preparation-v1",status="prepared-not-executed",
                    task=args.task,image_contract=identity["image_contract"],checkpoint_sha256=sha(checkpoint),
                    binary_sha256=sha(binary),packet_sha256=sha(root/"packet/packet.json"),
                    gpu_executed=False,learning_or_frontier_claim=False,seed_role="development-only")
        result["evaluation_seed"]=args.eval_seed
        save(root/"REPORT.json",result); return result
    except BaseException as error:
        save(root/"failure.json",dict(status="failed",error=str(error))); raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=acceptance.TASKS,required=True)
    parser.add_argument("--eval-seed",type=int,required=True)
    for name in ("build","metadata","config","checkpoint","out"): parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(prepare(args),indent=2))
