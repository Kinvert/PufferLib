"""Bind completed independent CUDA checks to every image panel policy.

No model executes here. Scalar topology comparisons and retained CUDA-array
audits use the existing graph acceptance implementation without new tolerances.
"""
import json
import configparser
from pathlib import Path
import shutil

from claim import require, save, sha

SCHEMA="native-image-panel-encoder-checks-v1"


def freeze(manifest, output, jobs, contracts, panel_root, evaluation_slots=None):
    import cnn_graph_acceptance as graphs
    value=json.loads(manifest.read_text())
    require(value["schema"]==SCHEMA and set(value["candidates"])=={j["candidate"] for j in jobs},
            "Need exactly the panel's architecture checks")
    output.mkdir(parents=True,exist_ok=False); checks={}; proofs={}
    for job in jobs:
        candidate,task=job["candidate"],job["environment"]
        require(set(value["candidates"][candidate])==set(contracts),"Missing per-game encoder check")
        original=(manifest.parent/value["candidates"][candidate][task]).resolve()
        key=str(original)
        if key not in proofs:
            report=graphs.audit(original)
            request=json.loads((original/"request.json").read_text())
            graphs.inspect_bundle(Path(request["bundle"]),current=True)
            name=f"proof-{len(proofs):04d}"; target=output/name; target.mkdir()
            files={}
            for filename in ["REPORT.json","request.json","config.ini"] + [f"worker-{i}/worker.json" for i in range(report.get("workers",2))]:
                dest=target/filename; dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original/filename,dest); files[filename]=sha(dest)
            proofs[key]=dict(path=name,original=str(original),files_sha256=files,report=report)
        checks.setdefault(candidate,{})[task]=proofs[key]["path"]
    recorded=dict(schema=SCHEMA,checks=checks,proofs=list(proofs.values()),
                  manifest_sha256=sha(manifest),raw_arrays_copied=False,full_policy_qualified=False)
    save(output/"checks.json",recorded)
    inspect(output,jobs,contracts,panel_root,current=True,evaluation_slots=evaluation_slots)
    return recorded


def inspect(output, jobs, contracts, panel_root, current=False, evaluation_slots=None):
    import cnn_graph_acceptance as graphs
    value=json.loads((output/"checks.json").read_text())
    require(value["schema"]==SCHEMA and value["raw_arrays_copied"] is False and value["full_policy_qualified"] is False,
            "Wrong encoder evidence scope")
    require(set(value["checks"])=={j["candidate"] for j in jobs},"Wrong checked architecture catalog")
    proofs={p["path"]:p for p in value["proofs"]}; require(len(proofs)==len(value["proofs"]),"Duplicate encoder proof IDs")
    for name,proof in proofs.items():
        require(Path(name).name==name and name.startswith("proof-"),"Unsafe encoder proof path")
        workers=proof["report"].get("workers",2)
        require(type(workers) is int and 2<=workers<=128 and workers%2==0,"Invalid numerical worker coverage")
        require(set(proof["files_sha256"])==set(["REPORT.json","request.json","config.ini"] + [f"worker-{i}/worker.json" for i in range(workers)]),
                "Incomplete encoder check receipts")
        for file,digest in proof["files_sha256"].items(): require(sha(output/name/file)==digest,"Changed encoder receipt")
        report=json.loads((output/name/"REPORT.json").read_text())
        require(all(report[k]==v for k,v in proof["report"].items()) and report["schema"]==graphs.IMAGE_SCHEMA
                and report["status"]=="passed" and report["cpu_neural_reference"] is False
                and report["full_policy_qualified"] is False and report["hidden_size"]==128,
                "Failed/wrong encoder report")
        require(sha(output/name/"config.ini")==report["config_sha256"],"Encoder graph config differs")
        request=json.loads((output/name/"request.json").read_text())
        if current:
            original=Path(proof["original"])
            require(all(sha(original/f)==h for f,h in proof["files_sha256"].items()),"Original encoder evidence changed")
            graphs.inspect_bundle(Path(request["bundle"]),current=True)
            require(sha(Path(request["bundle"])/"bundle.json")==request["bundle_sha256"],"Changed encoder bundle")
    used=set()
    for job in jobs:
        candidate,task=job["candidate"],job["environment"]
        require(set(value["checks"][candidate])==set(contracts),"Missing game numerical coverage")
        name=value["checks"][candidate][task]; require(name in proofs,"Missing encoder proof")
        used.add(name); report=json.loads((output/name/"REPORT.json").read_text())
        require(report["shape_hwc"]==contracts[task]["shape_hwc"] and report["layout"]==contracts[task]["layout"]
                and report["dtype"]==contracts[task]["dtype"],"Encoder check uses wrong image shape/layout/dtype")
        config=configparser.ConfigParser(interpolation=None); require(config.read(panel_root/job["config"]),"Missing policy recipe")
        checked=configparser.ConfigParser(interpolation=None); checked.read(output/name/"config.ini")
        require(config.getint("vec","total_agents")//config.getint("vec","num_buffers") in report["batches"]
                and config.getint("train","minibatch_size") in report["batches"],"Actor/learner batch leaves CUDA coverage")
        if evaluation_slots is not None:
            require(type(evaluation_slots) is int and evaluation_slots in report["batches"],
                    "Evaluator batch leaves independent CUDA coverage")
        require(graphs.settings(config)[0]==graphs.settings(checked)[0]
                and graphs.topology(*graphs.settings(config),shape=tuple(report["shape_hwc"]))
                ==graphs.topology(*graphs.settings(checked),shape=tuple(report["shape_hwc"])),
                "CUDA check covers a different effective architecture")
    require(used==set(proofs),"Unassigned encoder numerical evidence")
    return value


def hardware(output):
    value=json.loads((output/"checks.json").read_text())
    devices=[json.loads((output/p["path"]/"REPORT.json").read_text())["hardware"] for p in value["proofs"]]
    require(devices and all(device==devices[0] for device in devices),"Mixed numerical-check hardware")
    return devices[0]
