#!/usr/bin/env python3
"""Bounded inference timing/repeat probe on an existing image checkpoint.

This is a new allocation, never a continuation/retry of a failed panel. Inputs,
episode assignments and the native horizon stay unchanged. No training occurs.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import candidate_panel as panels
from claim import require, save, sha


def run(args):
    plan_path=args.plan.resolve(); root=plan_path.parent
    plan=panels.inspect(plan_path)
    require(panels.image_protocol(plan["protocol"]),"Need a retained image panel")
    require(args.hardware=="5060" and 0<args.timeout<=90 and 0<args.campaign_timeout<=240,
            "Probe is limited to a short local 5060 inference allocation")
    require(0<=args.job<len(plan["jobs"]),"Unknown job")
    job=plan["jobs"][args.job]; task=job["environment"]
    require(args.step in job["checkpoint_steps"],"Checkpoint was not scheduled")
    target=next((t for t in job["targets"] if t["representation"]==args.drawing),None)
    require(target is not None and plan["episodes"]<=5 and plan["slots"]==4,
            "Probe requires an existing small exact-assignment suite")
    checkpoint=root/job["id"]/"checkpoints"/task/job["candidate"]/f"{args.step:016d}.bin"
    binary=Path(job["command"][0]); config=root/job["config"]; suite=root/target["suite"]
    require(sha(binary)==plan["native_registry"]["binaries"][task][job.get("binary_family","default")]["sha256"],
            "Native trainer has changed")
    panels.checkpoint_size(job,checkpoint)
    inputs={str(p):sha(p) for p in (plan_path,binary,config,suite,checkpoint)}
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+args.campaign_timeout
    record=dict(schema="native-image-evaluation-probe-v1",status="running",inputs_sha256=inputs,
                job_index=args.job,environment=task,step=args.step,drawing=args.drawing,
                original_panel_status=json.loads((root/"execution/result.json").read_text())["status"],
                checkpoint_sha256=sha(checkpoint),episodes=plan["episodes"],slots=plan["slots"],
                restarted_panel=False,training_executed=False,frontier_claim_qualified=False,
                started_monotonic_ns=time.monotonic_ns(),processes=[])
    try:
        reference=None; hardware=None
        for mode in ("graph","repeat","eager"):
            remaining=deadline-time.monotonic(); require(remaining>0,"Probe deadline reached")
            device=panels.hardware(args.hardware)
            require(hardware is None or hardware==device,"Probe hardware changed")
            hardware=device; record["hardware"]=device
            destination=out/mode
            kwargs=dict(binary=binary,config=config,checkpoint=checkpoint,suite=suite,out=destination,
                        family=job.get("policy_family","flex"),timeout=min(args.timeout,remaining),
                        eager=mode=="eager",prepare_only=False)
            with contextlib.redirect_stdout(io.StringIO()): panels.bridge.adapter(task).run(SimpleNamespace(**kwargs))
            result=json.loads((destination/"result.json").read_text())
            require(result["status"]=="ok" and result["counts"]["episodes"]==plan["episodes"],
                    "Incomplete exact-assignment probe")
            data=(destination/"episodes.csv").read_bytes()
            require(reference is None or reference==data,"Graph/repeat/eager rows differ")
            reference=data
            record["processes"].append(dict(mode=mode,result_sha256=sha(destination/"result.json"),
                episodes_sha256=sha(destination/"episodes.csv"),seconds=result["evaluation_process_seconds"],
                counts=result["counts"]))
            save(out/"progress.json",record)
        require(all(sha(Path(p))==h for p,h in inputs.items()),"Probe input changed")
        record["status"]="passed"
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_monotonic_ns"]=time.monotonic_ns()
        save(out/"REPORT.json",record)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True); p.add_argument("--job",type=int,required=True)
    p.add_argument("--step",type=int,required=True); p.add_argument("--drawing",type=int,default=0)
    p.add_argument("--hardware",choices=("5060",),required=True)
    p.add_argument("--timeout",type=float,default=60); p.add_argument("--campaign-timeout",type=float,default=200)
    p.add_argument("--out",type=Path,required=True)
    r=run(p.parse_args()); print(json.dumps(dict(status=r["status"],processes=len(r["processes"]))))
