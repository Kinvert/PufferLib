#!/usr/bin/env python3
"""Owner-authorized retirement of three fixed historical local evidence trees.

Preserve/verify compact metadata before removing raw tensors and binaries.
Never clean arbitrary paths, touch system/CUDA, compact a WSL disk, or restart
any allocation. This script does not replace the retired raw numerical audit.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
TREES=("build/image-panels-20261012","build/image-panels-20261013","build/image-portability-20261012")
SUFFIXES={".json",".jsonl",".ini",".csv",".tsv",".txt",".log",".sha256",".md",".html",
    ".h",".c",".cu",".cuh",".cpp",".cc",".py",".sh",".js",".svg",".png",".jpg",".jpeg"}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def check_idle(targets):
    own=set();pid=os.getpid()
    while pid>1 and pid not in own:
        own.add(pid)
        try:pid=int((Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()[1])
        except (OSError,ValueError):break
    busy=[]
    needles=[str(target) for target in targets]+list(TREES)
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) in own:continue
        references=[]
        for name in ("cmdline","maps"):
            try:references.append((folder/name).read_bytes().decode(errors="replace"))
            except OSError:pass
        for location in (folder/"cwd",folder/"exe"):
            try:references.append(os.readlink(location))
            except OSError:pass
        try:
            for descriptor in (folder/"fd").iterdir():
                try:references.append(os.readlink(descriptor))
                except OSError:pass
        except OSError:pass
        if any(needle in reference for reference in references for needle in needles):busy.append(int(folder.name))
    require(not busy,"Live processes reference retirement targets: "+str(busy))


def retire(delete):
    require(delete,"Explicit --delete-authorized is required")
    targets=[ROOT/name for name in TREES]
    require(all(path.is_dir() and not path.is_symlink() and path.resolve()==path for path in targets),
            "All three exact authorized directories must exist; no symlinks or partial restart")
    tracked=subprocess.check_output(["git","ls-files","--",*TREES],cwd=ROOT,text=True)
    require(not tracked.strip(),"Refuse to retire tracked files")
    check_idle(targets)
    storage=ROOT/"build/hardware-artifacts";storage.mkdir(parents=True,exist_ok=True)
    directory=Path(tempfile.mkdtemp(prefix="retired-local-images-20261014.",dir=storage))
    inventory=directory/"inventory.jsonl";files=[];total=0
    with inventory.open("x") as stream:
        for target in targets:
            for base,dirs,names in os.walk(target,followlinks=False):
                for name in names:
                    path=Path(base)/name;info=path.lstat();relative=str(path.relative_to(ROOT))
                    selected=stat.S_ISREG(info.st_mode) and path.suffix in SUFFIXES
                    row=dict(path=relative,bytes=info.st_size,blocks=info.st_blocks,
                        inode=info.st_ino,device=info.st_dev,links=info.st_nlink,metadata_preserved=selected)
                    if path.is_symlink():row["symlink_target"]=os.readlink(path)
                    stream.write(json.dumps(row)+"\n")
                    if selected:files.append(path);total+=info.st_size
    require(total<2**30,"Compact metadata exceeds1GiB; stop without deletion")
    manifest=dict(schema="retired-local-image-evidence-v1",root=str(ROOT),trees=list(TREES),
        owner_authorized=True,raw_tensors_weights_binaries_preserved=False,
        independent_raw_replay_available=False,inventory_sha256=sha(inventory),
        files_sha256={str(path.relative_to(ROOT)):sha(path) for path in files},
        metadata_bytes=total,metadata_files=len(files))
    manifest_path=directory/"MANIFEST.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    archive=directory/"receipts.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        tar.add(inventory,arcname="inventory.jsonl",recursive=False)
        tar.add(manifest_path,arcname="MANIFEST.json",recursive=False)
        for path in files:tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    with tarfile.open(archive,"r:gz") as tar:
        expected=set(manifest["files_sha256"])|{"inventory.jsonl","MANIFEST.json"}
        require(set(tar.getnames())==expected,"Archive membership differs")
        require(json.load(tar.extractfile("MANIFEST.json"))==manifest,"Archive manifest differs")
        require(hashlib.file_digest(tar.extractfile("inventory.jsonl"),"sha256").hexdigest()==manifest["inventory_sha256"],
                "Archive inventory differs")
        for name,digest in manifest["files_sha256"].items():
            require(hashlib.file_digest(tar.extractfile(name),"sha256").hexdigest()==digest,"Archive bytes differ: "+name)
    require(all(sha(path)==manifest["files_sha256"][str(path.relative_to(ROOT))] for path in files),
            "Metadata changed during retirement preparation")
    check_idle(targets)
    digest=sha(archive);(directory/"receipts.tar.gz.sha256").write_text(digest+"  receipts.tar.gz\n")
    receipt=dict(status="archive-verified",archive=str(archive),sha256=digest,archive_bytes=archive.stat().st_size,
        metadata_files=len(files),metadata_bytes=total,trees=list(TREES),removed=[],started_unix=time.time(),
        independent_raw_replay_available=False,system_cuda_changed=False)
    result=directory/"RETIREMENT.json";result.write_text(json.dumps(receipt,indent=2)+"\n")
    for target in targets:
        require(target.resolve()==ROOT/str(target.relative_to(ROOT)) and not target.is_symlink(),"Retirement path changed")
        shutil.rmtree(target)
        receipt["removed"].append(str(target.relative_to(ROOT)))
        result.write_text(json.dumps(receipt,indent=2)+"\n")
    receipt.update(status="retired",ended_unix=time.time())
    result.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete-authorized",action="store_true")
    args=parser.parse_args();print(json.dumps(retire(args.delete_authorized),indent=2))
