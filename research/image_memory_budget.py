"""Explicit 5090 VRAM budget; host registration is not peak-memory acceptance.

Legacy packets keep their original 4GiB default/12GiB ceiling. The opt-in
5090 policy allows a 24GiB storage floor and requires a retained idle-device
memory query with 4GiB of additional free headroom before native proposals.
No device allocation, library installation or neural computation occurs here.
"""
import csv
import json
from pathlib import Path
import shutil

from claim import require, save, sha, process

LEGACY = "legacy-fixed-v1"
POLICY = "5090-vram-v1"
HEADROOM_MIB = 4096
MAXIMUM_MIB = 24576
SCHEMA = "image-5090-memory-receipt-v1"


def budget(config):
    policy=config.get("search","image_memory_policy",fallback=LEGACY)
    require(policy in (LEGACY,POLICY),"Unknown image memory policy")
    if policy==POLICY:
        require(config.has_option("search","image_storage_limit_mib"),
                "5090 memory policy needs an explicit storage limit")
    limit=config.getint("search","image_storage_limit_mib",fallback=4096)
    require(1<=limit<=(MAXIMUM_MIB if policy==POLICY else 12288),
            "Invalid image proposal storage admission bound")
    return policy,limit


def measurement(text,hardware,config):
    policy,limit=budget(config)
    require(policy==POLICY and "5090" in hardware["gpu_name"],
            "The enlarged image memory budget is only for the 5090")
    rows=list(csv.reader(text.splitlines()))
    require(len(rows)==1 and len(rows[0])==4,"Need one device memory query")
    name,uuid,total,free=(v.strip() for v in rows[0])
    require(name==hardware["gpu_name"] and uuid==hardware["gpu_uuid"],
            "Memory query belongs to a different GPU")
    require(total.isdigit() and free.isdigit(),"Need integral nvidia-smi MiB values")
    total,free=int(total),int(free)
    require(0<free<=total and limit+HEADROOM_MIB<=free,
            "5090 has insufficient free VRAM for the declared storage budget plus headroom")
    return dict(schema=SCHEMA,policy=policy,gpu_name=name,gpu_uuid=uuid,
        total_mib=total,free_mib=free,storage_limit_mib=limit,
        required_headroom_mib=HEADROOM_MIB,peak_memory_certified=False,
        cuda_executed=False,policy_executed=False,gpu_queried=True)


def probe(config,hardware,out,repository,timeout):
    require(budget(config)[0]==POLICY,"Only explicit 5090 allocations query VRAM")
    out.mkdir(parents=True,exist_ok=False)
    smi=shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    command=[smi,"--query-gpu=name,uuid,memory.total,memory.free","--format=csv,noheader,nounits"]
    timing=process(command,repository,out/"query.txt",timeout)
    receipt=measurement((out/"query.txt").read_text(),hardware,config)
    receipt.update(query_sha256=sha(out/"query.txt"),process_sha256=sha(out/"query.txt.json"),
        command=command,repository=str(repository),timing=timing)
    save(out/"receipt.json",receipt,exclusive=True)
    return receipt


def inspect(config,hardware,out,before_ns,after_ns):
    value=json.loads((out/"receipt.json").read_text())
    require(sha(out/"query.txt")==value["query_sha256"]
            and sha(out/"query.txt.json")==value["process_sha256"],"Changed memory query receipt")
    rebuilt=measurement((out/"query.txt").read_text(),hardware,config)
    require(all(value[key]==number for key,number in rebuilt.items()),"Changed memory admission values")
    timing=json.loads((out/"query.txt.json").read_text())
    require(timing==value["timing"] and timing["status"]=="ok" and timing["returncode"]==0
            and timing["cwd"]==value["repository"] and timing["command"]==value["command"]
            and Path(timing["command"][0]).name=="nvidia-smi"
            and timing["command"][1:]==["--query-gpu=name,uuid,memory.total,memory.free","--format=csv,noheader,nounits"]
            and before_ns<=timing["launch_monotonic_ns"]<timing["end_monotonic_ns"]<=after_ns,
            "Wrong memory query command or proposal chronology")
    return value
