"""Bounded CUDA-only finite-difference sensitivity diagnostic, not acceptance."""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import secrets
import sys

import numpy as np
import candidate_panel as panels
import cnn_graph_acceptance as checks
from claim import require, save, sha


def worker(out):
    request=json.loads((out/"request.json").read_text())
    require(request["parent"]==os.getppid() and request["token"]==os.environ.get("PUFFER_LOSS_PROBE_TOKEN"),"Scheduled diagnostic parent required")
    bundle=Path(request["bundle"]); checks.inspect_bundle(bundle)
    require(sha(bundle/"bundle.json")==request["bundle_sha256"],"Changed retained oracle")
    case=Path(request["case"])
    for name,digest in request["files_sha256"].items(): require(sha(case/name)==digest,"Changed failed-case diagnostic input")
    model,settings=checks.settings(panels.panel.read(request["config"]))
    values=np.array(settings,dtype=np.int32)
    descriptor=checks.inspect_bundle(bundle)
    rows,count=checks.topology(model,settings,shape=descriptor["shape_hwc"])
    require(model==1 and case.name=="b1-smooth","Bounded Nature B1 smooth-loss diagnostic only")
    arrays=[np.fromfile(case/(name+".f32"),dtype=np.float32).astype(np.float64) for name in ("input","parameters","upstream")]
    gradient=np.fromfile(case/"full-gradient.f64",dtype=np.float64)
    require(len(gradient)==len(arrays[1])==count and len(arrays[0])==np.prod(descriptor["shape_hwc"])
            and len(arrays[2])==128,"Diagnostic shape differs from captured GPU arrays")
    index=request["index"]; require(0<=index<count,"Unassigned diagnostic parameter")
    oracle=ctypes.CDLL(str(bundle/"reference.so")); doubles=ctypes.POINTER(ctypes.c_double)
    oracle.flexref_run.argtypes=[ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS"),ctypes.c_int]+[doubles]*7
    pointer=lambda a:a.ctypes.data_as(doubles) if a is not None else doubles()
    probes=[]
    for epsilon in (1e-6,1e-4):
        losses=[]
        for sign in (1,-1):
            changed=arrays[1].copy();changed[index]+=sign*epsilon
            loss=np.empty(1,dtype=np.float64)
            require(oracle.flexref_run(model,128,values,1,*(pointer(a) for a in (arrays[0],changed,arrays[2],None,None,None,loss)))==0,"CUDA loss diagnostic failed")
            losses.append(float(loss[0]))
        derivative=(losses[0]-losses[1])/(2*epsilon); analytical=float(gradient[index])
        probes.append(dict(epsilon=epsilon,losses=losses,derivative=derivative,analytical=analytical,
                           absolute_error=abs(derivative-analytical),within_original_tolerance=abs(derivative-analytical)<=1e-8+2e-5*abs(analytical)))
    save(out/"REPORT.json",dict(status="diagnostic-completed",probes=probes,reference_calls=4,
        numerical_acceptance=False,publication_claim_qualified=False,cpu_neural_reference=False))


def run(args):
    bundle=args.bundle.resolve(); checks.inspect_bundle(bundle)
    case=args.case.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    request=dict(bundle=str(bundle),bundle_sha256=sha(bundle/"bundle.json"),case=str(case),config=str(args.config.resolve()),
        index=args.index,parent=os.getpid(),token=secrets.token_hex(16),files_sha256={n:sha(case/n) for n in
            ("input.f32","parameters.f32","upstream.f32","full-gradient.f64")})
    save(out/"request.json",request)
    with checks.fixtures.LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hardware=panels.hardware("5060")
        receipt=panels.process([sys.executable,str(Path(__file__).resolve()),"_worker","--out",str(out)],
            panels.ROOT,out/"worker.log",60,dict(os.environ,PUFFER_LOSS_PROBE_TOKEN=request["token"],OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
        require(panels.hardware("5060")==hardware,"Diagnostic hardware changed")
    result=json.loads((out/"REPORT.json").read_text());result.update(hardware=hardware,process=receipt)
    save(out/"REPORT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    for name in ("bundle","case","config"): r.add_argument("--"+name,type=Path,required=True)
    r.add_argument("--index",type=int,required=True); sub.add_parser("_worker")
    for command in sub.choices.values(): command.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.command=="_worker": worker(args.out.resolve())
    else: print(json.dumps(run(args)))
