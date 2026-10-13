#!/usr/bin/env python3
"""Per-proposal native CUDA checks. No CPU neural reference or training.

The direct float64 CUDA reference checks every local forward layer on its
float32 input, then all gradients using the actual float32 ReLU branches and
independently selected max-pool winners. A separate smooth fixture also checks
the entire float64 graph and selected forward-loss finite differences. This
does not relabel the historical unconditional H256 comparison as passed.
"""
import argparse
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import time

import numpy as np
import candidate_panel as panels
import dense_alias_acceptance as fixtures

ROOT = panels.ROOT
SCHEMA = "native-cnn-graph-check-v1"
IMAGE_SCHEMA = "native-cnn-image-graph-check-v2"
MODES = ("eager-0", "eager-1", "graph-0", "graph-1")
RTOL, ATOL = 8e-4, 6e-5
TOLERANCES = {"flex": [RTOL, ATOL], "nature": [3e-4, 3e-5]}
TOOLS = ("research/cnn_graph_acceptance.py", "research/candidate_panel.py",
         "research/dense_alias_acceptance.py", "ocean/connect4cnn/claim.py",
         "ocean/connect4cnn/encoder_profile.py")
require, sha, save = fixtures.require, fixtures.sha, fixtures.save


def bundle(args):
    root = args.out.resolve(); root.mkdir(parents=True, exist_ok=False)
    shape = getattr(args, "image_shape", None)
    batches = getattr(args, "batches", None)
    if shape is None:
        require(batches is None, "Custom batches require an explicit image contract")
    else:
        require(len(shape) == 3 and all(type(v) is int for v in shape)
                and all(36 <= v <= 1024 for v in shape[:2]) and shape[2] in (1,3), "Invalid image shape")
        batches = batches or [1,4,16]
        require(batches == sorted(set(batches)) and batches[0] == 1
                and all(type(v) is int and 1 <= v <= 64 for v in batches), "Invalid bounded image batches")
    inputs = {}
    for kind in ("native", "reference"):
        lib = getattr(args, kind).resolve(); receipts = Path(str(lib)+".build")
        sources = fixtures.sources(receipts / "source.sha256", ROOT)
        command = (receipts / "command.txt").read_text()
        environment = (receipts / "environment.txt").read_text()
        if shape is not None:
            require("CNN_TEST_IMAGE=" + ",".join(map(str, shape)) + "\n" in environment,
                    "Native/reference compile image differs")
        require("NVCC_ARCH=native" not in environment, "Explicit compile architecture required")
        if kind == "native":
            require("test_nature.cu" in command and "C4_DENSE_PATCH_ALIAS=0" in environment,
                    "Unchanged production callbacks required")
        else:
            require("native_flex_reference.cu" in command and "-fmad=false" in command,
                    "Independent CUDA graph reference required")
        require(sha(lib) == (receipts / "binary.sha256").read_text().split()[0], "Changed library")
        shutil.copyfile(lib, root / (kind+".so"))
        shutil.copytree(receipts, root / (kind+".build"))
        inputs.update({str(ROOT / n): h for n,h in sources.items()})
        inputs[str(lib)] = sha(lib)
    inputs.update({str(ROOT/n):sha(ROOT/n) for n in TOOLS})
    files = {str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file()}
    value = dict(schema=IMAGE_SCHEMA if shape is not None else SCHEMA, originals_sha256=inputs, files_sha256=files,
                 rtol=RTOL, atol=ATOL, tolerances=TOLERANCES, cpu_neural_reference=False)
    if shape is not None: value.update(shape_hwc=shape, batches=batches, layout="NHWC", dtype="float32")
    save(root / "bundle.json", value); inspect_bundle(root, current=True)
    return value


def inspect_bundle(root, current=False):
    value = json.loads((root / "bundle.json").read_text())
    require(value["schema"] in (SCHEMA, IMAGE_SCHEMA) and value["rtol"] == RTOL and value["atol"] == ATOL
            and value["cpu_neural_reference"] is False, "Changed numerical protocol")
    if value["schema"] == IMAGE_SCHEMA:
        shape, batches = value["shape_hwc"], value["batches"]
        require(len(shape) == 3 and all(type(v) is int for v in shape)
                and all(36 <= v <= 1024 for v in shape[:2]) and shape[2] in (1,3)
                and batches == sorted(set(batches)) and batches[0] == 1
                and all(type(v) is int and 1 <= v <= 64 for v in batches)
                and value["layout"] == "NHWC" and value["dtype"] == "float32", "Invalid image-check contract")
    if "tolerances" in value: require(value["tolerances"] == TOLERANCES, "Changed family tolerances")
    fixtures.verify_files(root, value["files_sha256"])
    if current:
        for n,h in value["originals_sha256"].items(): require(sha(n)==h, "Changed graph-check source: "+n)
    return value


def settings(config):
    family = config.getint("policy", "encoder")
    require(config.getint("policy", "hidden_size") == 128 and family in (2,4), "Graph gate requires encoder2/4 H128")
    if family==2: return 1, [1,64,0]+[8,3,1,0,0]*3
    values = [config.getint("policy", k) for k in ("cnn_depth", "cnn_projection", "cnn_global_pool")]
    for stage in range(1,4):
        values += [config.getint("policy", f"cnn_{n}_{stage}", fallback=d)
                   for n,d in zip(("channels","kernel","stride","pool","residual"),(8,3,1,0,0))]
    return 0, values


def cases(contract=None):
    if contract and contract["schema"] == IMAGE_SCHEMA:
        return [(b,s) for b in contract["batches"] for s in ("signed","smooth","zero")]
    return [(b,s) for b in (1,64) for s in ("signed","smooth","zero")] + [(2048,"signed")]


def topology(model, values, shape=(36,44,1)):
    # Independent scalar shape/layout arithmetic only; no images/weights/math.
    result=[]; h,w,ci=shape; offset=0
    def conv(co,k,s,same=True,skip=False):
        nonlocal h,w,ci,offset
        oh=(h+s-1)//s if same else (h-k)//s+1
        ow=(w+s-1)//s if same else (w-k)//s+1
        ph=max((oh-1)*s+k-h,0)//2 if same else 0
        pw=max((ow-1)*s+k-w,0)//2 if same else 0
        result.append([h,w,ci,oh,ow,co,k,s,ph,pw,offset,0,int(skip)])
        offset += co*(ci*k*k+1); h,w,ci=oh,ow,co
    if model:
        for co,k,s in ((32,8,4),(64,4,2),(64,3,1)): conv(co,k,s,False)
        ci*=h*w; h=w=1; conv(128,1,1,False)
    else:
        depth,projection,gap=values[:3]
        require(1<=depth<=3 and projection in (16,32,64,128) and gap in (0,1), "Invalid scalar topology")
        for stage in range(depth):
            co,k,s,pool,skip=values[3+stage*5:8+stage*5]
            require(co in (8,16,32) and 1<=k<=(8 if stage==0 else 5)
                    and s in ((1,2,4,8) if stage==0 else (1,2,4)) and pool in (0,1,2) and skip in (0,1), "Invalid stage")
            conv(co,k,s)
            if skip: conv(ci,3,1,skip=True)
            if pool:
                oh,ow=(h+1)//2,(w+1)//2
                result.append([h,w,ci,oh,ow,ci,3,2,h%2,w%2,offset,pool,0]); h,w=oh,ow
        if gap: result.append([h,w,ci,1,1,ci,1,1,0,0,offset,3,0]); h=w=1
        ci*=h*w; h=w=1; conv(projection,1,1)
        if projection!=128: conv(128,1,1)
    return result,offset


def fixture(rows, count, batch, state):
    rng=np.random.default_rng(73173+batch)
    x=rng.uniform(0,1,(batch,math.prod(rows[0][:3]))).astype(np.float32)
    weights=np.empty(count,dtype=np.float32)
    for row in rows:
        ci,co,k,offset,kind=row[2],row[5],row[6],row[10],row[11]
        if kind: continue
        n=co*k*k*ci
        weights[offset:offset+n] = (rng.uniform(0,.2/math.sqrt(k*k*ci),n) if state=="smooth"
                                    else rng.normal(0,.3/math.sqrt(k*k*ci),n))
        weights[offset+n:offset+n+co] = .2 if state=="smooth" else rng.choice([-.2,.2],co)
    upstream=rng.normal(0,.1,(batch,128)).astype(np.float32)
    if state=="zero": x.fill(0); weights.fill(0)
    return x,weights,upstream


def worker(out):
    packet=json.loads((out/"request.json").read_text()); root=Path(packet["bundle"])
    bundle_value=inspect_bundle(root,current=True)
    require(packet["parent"]==os.getppid() and packet["token"]==os.environ.get("PUFFER_GRAPH_CHECK_TOKEN"), "Scheduled parent required")
    model, values=settings(panels.panel.read(out/"config.ini")); spec=np.array(values,dtype=np.int32)
    rtol,atol=bundle_value.get("tolerances",{}).get("nature" if model else "flex",[RTOL,ATOL])
    rows,count=topology(model,values,bundle_value.get("shape_hwc",(36,44,1)))
    native=ctypes.CDLL(str(root/"native.so")); reference=ctypes.CDLL(str(root/"reference.so"))
    floats=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    ints=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    doubles=ctypes.POINTER(ctypes.c_double)
    pointer=lambda a:a.ctypes.data_as(doubles) if a is not None else doubles()
    native.flextest_init.argtypes=[ctypes.c_int,ctypes.c_int,ints]
    native.naturetest_init.argtypes=[ctypes.c_int,ctypes.c_int]
    native.naturetest_run.argtypes=[floats]*5+[ctypes.c_int]
    native.naturetest_read_params.argtypes=[floats]; native.naturetest_read_trace.argtypes=[floats]
    reference.flexref_layout.argtypes=[ctypes.c_int,ctypes.c_int,ints,ints]
    reference.flexref_run.argtypes=[ctypes.c_int,ctypes.c_int,ints,ctypes.c_int]+[doubles]*7
    if bundle_value["schema"] == IMAGE_SCHEMA:
        for lib, name in ((native,"naturetest_image_shape"),(reference,"flexref_image_shape")):
            function = getattr(lib,name); function.argtypes = [ints]
            compiled = np.empty(3,dtype=np.int32); function(compiled)
            require(compiled.tolist() == bundle_value["shape_hwc"], "Library compiled image contract differs")
    actual_layout=np.empty(1+13*12,dtype=np.int32)
    require(reference.flexref_layout(model,128,spec,actual_layout)==count
            and actual_layout[:1+13*len(rows)].tolist()==[len(rows)]+sum(rows,[]), "Independent native/scalar layout differs")
    require(reference.cnnref_selftest()==0, "Direct CUDA reference dyadic self-test failed")
    arrays=out/"arrays"; arrays.mkdir()
    def call(data,batch,trace,out_trace,grad,loss=None):
        require(reference.flexref_run(model,128,spec,batch,*(pointer(a) for a in (*data,trace,out_trace,grad,loss)))==0, "CUDA graph oracle failed")
    receipts=[]
    for batch,state in cases(bundle_value):
        directory=arrays/f"b{batch}-{state}"; directory.mkdir()
        data=fixture(rows,count,batch,state)
        for name,a in zip(("input","parameters","upstream"),data): a.tofile(directory/(name+".f32"))
        trace_size=sum(batch*r[3]*r[4]*r[5] for r in rows)
        init=native.naturetest_init(batch,128) if model else native.flextest_init(batch,128,spec)
        require(init==count and native.naturetest_trace_elems()==trace_size, "Production layout differs")
        first=None; receipt=dict(batch=batch,state=state,finite_differences=[])
        try:
            expected_trace=np.empty(trace_size,dtype=np.float64); expected_grad=np.empty(count,dtype=np.float64)
            precise=tuple(a.astype(np.float64) for a in data)
            for mode in MODES:
                output=np.empty((batch,128),dtype=np.float32); grad=np.empty(count,dtype=np.float32)
                trace=np.empty(trace_size,dtype=np.float32); params=np.empty(count,dtype=np.float32)
                native.naturetest_run(*data,output,grad,int(mode.startswith("graph")))
                native.naturetest_read_trace(trace); native.naturetest_read_params(params)
                require(params.tobytes()==data[1].tobytes(), "Device parameters changed")
                signature=(output.tobytes(),grad.tobytes(),trace.tobytes())
                require(first is None or first==signature, "Eager/graph/repeat bytes differ")
                if first is None:
                    call(precise,batch,trace.astype(np.float64),expected_trace,expected_grad)
                    expected_trace.tofile(directory/"local-trace.f64"); expected_grad.tofile(directory/"conditional-gradient.f64")
                    np.testing.assert_allclose(trace,expected_trace,rtol=rtol,atol=atol)
                    np.testing.assert_allclose(grad,expected_grad,rtol=rtol,atol=atol)
                    if state=="smooth":
                        strict_trace=np.empty_like(expected_trace); strict_grad=np.empty_like(expected_grad)
                        call(precise,batch,None,strict_trace,strict_grad)
                        np.testing.assert_allclose(trace,strict_trace,rtol=rtol,atol=atol)
                        np.testing.assert_allclose(grad,strict_grad,rtol=rtol,atol=atol)
                        strict_trace.tofile(directory/"full-trace.f64"); strict_grad.tofile(directory/"full-gradient.f64")
                        if batch==1:
                            for row in rows:
                                if row[11]: continue
                                start=row[10]; length=row[5]*(row[6]*row[6]*row[2]+1)
                                index=start+int(np.argmax(np.abs(strict_grad[start:start+length])))
                                losses=[]
                                for sign in (1,-1):
                                    changed=precise[1].copy(); changed[index]+=sign*1e-6
                                    loss=np.empty(1,dtype=np.float64)
                                    call((precise[0],changed,precise[2]),batch,None,None,None,loss)
                                    losses.append(float(loss[0]))
                                derivative=(losses[0]-losses[1])/2e-6
                                np.testing.assert_allclose(derivative,strict_grad[index],rtol=2e-5,atol=1e-8)
                                receipt["finite_differences"].append(dict(index=index,losses=losses,epsilon=1e-6,analytic=float(strict_grad[index])))
                for name,a in zip(("output","gradient","trace","device-parameters"),(output,grad,trace,params)):
                    a.tofile(directory/(mode+"-"+name+".f32"))
                first=signature
        finally: native.naturetest_close()
        save(directory/"case.json",receipt); receipts.append(receipt)
        print(f"PASS b{batch}-{state}",flush=True)
    save(out/"worker.json",dict(status="passed",cases=receipts,
         arrays_sha256={str(p.relative_to(arrays)):sha(p) for p in arrays.rglob("*") if p.is_file()}))


def worker_timeout(timeout, deadline=None):
    require(math.isfinite(timeout) and 0<timeout<=120, "Bounded graph worker deadline required")
    if deadline is None: return timeout
    require(math.isfinite(deadline), "Invalid campaign deadline")
    remaining=deadline-time.monotonic()
    require(remaining>0, "Campaign deadline before numerical worker")
    return min(timeout,remaining)


def run(bundle_root, config, out, expected="5090", timeout=120, deadline=None):
    bundle_root=bundle_root.resolve(); contract=inspect_bundle(bundle_root,current=True)
    worker_timeout(timeout,deadline)
    out=out.resolve(); out.mkdir(parents=True,exist_ok=False); shutil.copyfile(config,out/"config.ini")
    settings(panels.panel.read(out/"config.ini"))
    token=secrets.token_hex(16)
    request=dict(schema=contract["schema"],bundle=str(bundle_root),bundle_sha256=sha(bundle_root/"bundle.json"),
        config_sha256=sha(out/"config.ini"),parent=os.getpid(),token=token,hardware=expected,timeout=timeout)
    if deadline is not None: request["campaign_deadline_monotonic_ns"]=int(deadline*1e9)
    save(out/"request.json",request)
    try:
        with fixtures.LOCK.open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hardware=panels.hardware(expected)
            for index in range(2):
                directory=out/f"worker-{index}"; directory.mkdir()
                shutil.copyfile(out/"config.ini",directory/"config.ini"); shutil.copyfile(out/"request.json",directory/"request.json")
                require(panels.hardware(expected)==hardware, "Hardware changed or GPU busy")
                command=[sys.executable,str(Path(__file__).resolve()),"_worker","--out",str(directory)]
                panels.process(command,directory,directory/"worker.log",worker_timeout(timeout,deadline),
                    dict(os.environ,PUFFER_GRAPH_CHECK_TOKEN=token,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1"))
            require(panels.hardware(expected)==hardware,"Hardware changed at completion")
        inspect_bundle(bundle_root,current=True)
        result=audit(out,completing=True); result["hardware"]=hardware
        save(out/"REPORT.json",result); return result
    except BaseException as error:
        save(out/"failure.json",dict(status="failed",error=str(error))); raise


def audit(out, completing=False):
    request=json.loads((out/"request.json").read_text())
    require(request["schema"] in (SCHEMA,IMAGE_SCHEMA) and sha(out/"config.ini")==request["config_sha256"],"Changed graph request")
    require(sha(Path(request["bundle"])/"bundle.json")==request["bundle_sha256"],"Changed graph bundle")
    bundle_value=inspect_bundle(Path(request["bundle"]))
    require(request["schema"] == bundle_value["schema"], "Graph request/bundle version differs")
    model,values=settings(panels.panel.read(out/"config.ini")); rows,count=topology(model,values,bundle_value.get("shape_hwc",(36,44,1)))
    rtol,atol=bundle_value.get("tolerances",{}).get("nature" if model else "flex",[RTOL,ATOL])
    prior=None
    for index in range(2):
        directory=out/f"worker-{index}"; arrays=directory/"arrays"
        require((directory/"request.json").read_bytes()==(out/"request.json").read_bytes()
                and sha(directory/"config.ini")==request["config_sha256"],"Worker input changed")
        record=json.loads((directory/"worker.json").read_text())
        require(record["status"]=="passed" and [(c["batch"],c["state"]) for c in record["cases"]]==cases(bundle_value),"Incomplete graph panel")
        fixtures.verify_files(arrays,record["arrays_sha256"])
        require(record["arrays_sha256"]=={str(p.relative_to(arrays)):sha(p) for p in arrays.rglob("*") if p.is_file()},"Changed array inventory")
        require(prior is None or prior==record["arrays_sha256"],"Independent process arrays differ"); prior=record["arrays_sha256"]
        for batch,state in cases(bundle_value):
            folder=arrays/f"b{batch}-{state}"; data=fixture(rows,count,batch,state)
            for name,a in zip(("input","parameters","upstream"),data): require((folder/(name+".f32")).read_bytes()==a.tobytes(),"Changed fixture")
            for mode in MODES:
                require((folder/(mode+"-device-parameters.f32")).read_bytes()==data[1].tobytes(),"Parameter mutation")
                trace_bytes=(folder/(mode+"-trace.f32")).read_bytes()
                require((folder/(mode+"-output.f32")).read_bytes()==trace_bytes[-batch*128*4:],"Output differs from final activation")
                for name,reference in (("trace","local-trace"),("gradient","conditional-gradient")):
                    actual=np.fromfile(folder/(mode+"-"+name+".f32"),dtype="<f4")
                    oracle=np.fromfile(folder/(reference+".f64"),dtype="<f8")
                    require(actual.size==oracle.size and np.isfinite(actual).all() and np.isfinite(oracle).all(),"Invalid graph arrays")
                    np.testing.assert_allclose(actual,oracle,rtol=rtol,atol=atol)
                    if state=="smooth": np.testing.assert_allclose(actual,np.fromfile(folder/("full-"+name+".f64"),dtype="<f8"),rtol=rtol,atol=atol)
                if mode!="eager-0":
                    for name in ("output","gradient","trace"):
                        require((folder/(mode+"-"+name+".f32")).read_bytes()==(folder/("eager-0-"+name+".f32")).read_bytes(),"Changed mode bytes")
            receipt=json.loads((folder/"case.json").read_text())
            require(receipt==next(c for c in record["cases"] if (c["batch"],c["state"])==(batch,state)),"Changed case receipt")
            expected_probes=sum(r[11]==0 for r in rows) if (batch,state)==(1,"smooth") else 0
            require(len(receipt["finite_differences"])==expected_probes,"Missing GPU finite differences")
            for probe in receipt["finite_differences"]:
                strict=np.fromfile(folder/"full-gradient.f64",dtype="<f8")
                require(probe["epsilon"]==1e-6 and probe["analytic"]==strict[probe["index"]],"Changed finite difference binding")
                derivative=(probe["losses"][0]-probe["losses"][1])/2e-6
                np.testing.assert_allclose(derivative,probe["analytic"],rtol=2e-5,atol=1e-8)
        timing=json.loads((directory/"worker.log.json").read_text())
        command=[sys.executable,str(Path(__file__).resolve()),"_worker","--out",str(directory)]
        require(timing["returncode"]==0 and timing["status"]=="ok" and timing["command"]==command
                and timing["cwd"]==str(directory) and 0<timing["launch_monotonic_ns"]<timing["end_monotonic_ns"], "Failed/changed worker")
        require(0<timing["timeout"]<=request["timeout"], "Worker exceeds declared numerical timeout")
        if "campaign_deadline_monotonic_ns" in request:
            require(timing["launch_monotonic_ns"]<request["campaign_deadline_monotonic_ns"],
                    "Numerical worker launched after campaign deadline")
        if index: require(previous_end<=timing["launch_monotonic_ns"],"Workers overlap")
        previous_end=timing["end_monotonic_ns"]
    result=dict(status="passed",schema=request["schema"],config_sha256=request["config_sha256"],cases=2*len(cases(bundle_value)),native_calls=8*len(cases(bundle_value)),
        batches=sorted({b for b,s in cases(bundle_value)}),hidden_size=128,cpu_neural_reference=False,full_policy_qualified=False,
        backward_contract="independent CUDA on actual float32 activations/branches",publication_claim_qualified=False)
    if request["schema"] == IMAGE_SCHEMA: result.update(shape_hwc=bundle_value["shape_hwc"],layout="NHWC",dtype="float32")
    if not completing:
        report=json.loads((out/"REPORT.json").read_text()); require({k:report[k] for k in result}==result,"Changed graph report")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    b=sub.add_parser("bundle"); b.add_argument("--native",type=Path,required=True); b.add_argument("--reference",type=Path,required=True)
    b.add_argument("--image-shape",type=int,nargs=3); b.add_argument("--batches",type=int,nargs="+")
    r=sub.add_parser("run"); r.add_argument("--bundle",type=Path,required=True); r.add_argument("--config",type=Path,required=True)
    r.add_argument("--hardware",choices=("5060","5090"),required=True); r.add_argument("--timeout",type=int,default=120)
    sub.add_parser("audit"); sub.add_parser("_worker")
    for parser in sub.choices.values(): parser.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.command=="_worker": worker(args.out.resolve()); return
    result=bundle(args) if args.command=="bundle" else run(args.bundle,args.config,args.out,args.hardware,args.timeout) if args.command=="run" else audit(args.out.resolve())
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
