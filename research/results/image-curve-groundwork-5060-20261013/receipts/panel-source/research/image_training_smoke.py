#!/usr/bin/env python3
"""Bounded native image training/checkpoint smoke after matching CUDA checks.

This is launch/audit glue. It does not train a Python model, certify learning or
claim a frontier result. Full cross-game campaign integration is separate.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import candidate_panel as panels
import cnn_graph_acceptance as graphs
import eval_acceptance

require, sha, save, process = panels.require, panels.sha, panels.save, panels.process


def run(args):
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    try:
        proof=graphs.audit(args.graph_check.resolve())
        require(proof["schema"]==graphs.IMAGE_SCHEMA and proof["status"]=="passed"
                and proof["config_sha256"]==sha(args.config), "Need matching portable-image CUDA evidence")
        request=json.loads((args.graph_check/"request.json").read_text())
        graphs.inspect_bundle(Path(request["bundle"]),current=True)
        config=panels.panel.read(args.config)
        initial=getattr(args,"include_initial",False)
        require(type(initial) is bool,"Invalid initial checkpoint option")
        task=config.get("base","env_name")
        require(task in eval_acceptance.TASKS, "Bounded initial smoke excludes Connect4's legacy empty-board checker")
        actors=config.getint("vec","total_agents"); batch=config.getint("train","minibatch_size")
        steps=config.getint("train","total_timesteps"); horizon=config.getint("train","horizon")
        require(actors in proof["batches"] and batch in proof["batches"] and 0<steps<=1024,
                "Smoke geometry leaves accepted actor/learner batch coverage")
        require(config.getint("train","gpus")==1 and config.getint("base","async")==0
                and config.getint("selfplay","enabled")==0 and config.getint("vec","num_policies")==1,
                "Smoke requires synchronous single-GPU/single-policy geometry")
        binary=args.build.resolve()/"puffer"; metadata=args.build.resolve()/"metadata"
        build_sources=panels.policy_layout.source_hashes(args.build.resolve()/"source.sha256")
        require({"src/pufferl.cu",f"ocean/{task}/{task}.h","ocean/connect4cnn/image.h"}<=set(build_sources)
                and all(sha(panels.ROOT/name)==digest for name,digest in build_sources.items()),
                "Trainer build sources differ; compile fresh before smoke")
        build_hashes={Path(line.split(maxsplit=1)[1].lstrip("*")).resolve():line.split()[0]
                      for line in (args.build.resolve()/"binaries.sha256").read_text().splitlines()}
        require(build_hashes.get(binary)==sha(binary) and build_hashes.get(metadata)==sha(metadata),
                "Trainer or metadata differs from the recorded build")
        action_header=panels.ROOT/f"ocean/{task}/{task}.h"
        actions={"pongcnn":3,"flappycnn":2,"breakoutcnn":3,"snakecnn":4,"mazecnn":5}[task]
        inputs={str(p.resolve()):sha(p) for p in (binary,metadata,args.config,args.graph_check/"REPORT.json",action_header,
                args.build/"source.sha256",args.build/"binaries.sha256")}
        process([str(metadata),"--storage",str(args.config.resolve())],panels.ROOT,root/"storage.log",30)
        storage=json.loads((root/"storage.log").read_text())
        require(storage["shape_hwc"]==proof["shape_hwc"] and storage["storage_floor_bytes"]<4*2**30,
                "Smoke input shape/storage exceeds bounded allocation")
        process([str(metadata),"--describe-flex",str(args.config.resolve()),str(actions)],panels.ROOT,root/"layout.log",30)
        layout=json.loads((root/"layout.log").read_text())
        require(layout["raw_checkpoint_contiguous"] and layout["parameter_payload_bytes"]==layout["total_parameters"]*4,
                "Raw native checkpoint layout is not contiguous")
        config["base"].update(run_id="happy-cat-1",checkpoint_dir=str(root/"checkpoints"),log_dir=str(root/"metrics"))
        (root/"config").mkdir()
        config_path=root/"config/default.ini"
        with config_path.open("x") as stream: config.write(stream)
        (root/"config"/(task+".ini")).touch()
        lock_path=panels.ROOT/"build/connect4cnn/hardware-benchmark.lock"; lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hardware=panels.hardware(args.hardware)
            environment=dict(os.environ,PUFFER_CHECKPOINT_RECEIPTS="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            environment.pop("PUFFER_SAVE_INITIAL_CHECKPOINT",None)
            if initial: environment["PUFFER_SAVE_INITIAL_CHECKPOINT"]="1"
            timed=process([str(binary),"train","--headless"],root,root/"train.log",30,environment)
            require(panels.hardware(args.hardware)==hardware, "Hardware changed or a competing GPU job appeared")
        cadence=config.getint("base","checkpoint_interval")*actors*horizon
        scheduled=list(range(cadence,steps,cadence))+[steps]
        if initial: scheduled.insert(0,0)
        times=panels.parse_checkpoints((root/"train.log").read_text(),timed,scheduled,layout["total_parameters"])
        checkpoint_dir=root/"checkpoints"/task/"happy-cat-1"
        weights=[]
        for step in scheduled:
            path=checkpoint_dir/f"{step:016d}.bin"
            require(path.stat().st_size==layout["parameter_payload_bytes"] and np.isfinite(np.fromfile(path,dtype="<f4")).all(),
                    "Invalid checkpoint size or nonfinite weights")
            weights.append(dict(step=step,sha256=sha(path),train_seconds=times[step]))
        require(panels.normalized(panels.panel.read(checkpoint_dir/"resolved.ini"))==panels.normalized(config),
                "Native resolved recipe differs from frozen inputs")
        require(all(sha(Path(p))==h for p,h in inputs.items()), "Smoke inputs changed")
        seconds=(timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9
        result=dict(status="passed",hardware=hardware,shape_hwc=proof["shape_hwc"],steps=steps,training_seed=config.getint("base","seed"),
                    learner_batch=batch,actor_batch=actors,checkpoints=weights,process_seconds=seconds,process_sps=steps/seconds,
                    inputs_sha256=inputs,parameters=layout["total_parameters"],
                    learning_or_frontier_claim=False,full_policy_numerical_qualification=False,evaluation_executed=False)
        result["initial_checkpoint_executed"]=initial
        save(root/"REPORT.json",result); return result
    except BaseException as error:
        save(root/"failure.json",dict(status="failed",error=str(error))); raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","build","graph-check","out"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--hardware",choices=("5060","5090"),required=True)
    p.add_argument("--include-initial",action="store_true",help="Check the native zero-step save before training")
    print(json.dumps(run(p.parse_args()),indent=2))
