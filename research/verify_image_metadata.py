#!/usr/bin/env python3
"""Actual native host layout/storage checks; no CUDA allocation or CPU CNN.

Freeze six freshly compiled per-game image tools, register every declared
validation architecture, and exercise explicit kernel-index/encoder rejection.
This is construction evidence, not numerical or peak-memory qualification.
"""
import argparse
import configparser
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"ocean/connect4cnn"))

import checkpoint_preflight as layouts
import cnn_graph_acceptance as graphs
from claim import process, require, save, sha


def read(path):
    value = configparser.ConfigParser(interpolation=None)
    require(value.read(path), "Missing full native recipe")
    return value


def run(args):
    root = args.out.resolve(); root.mkdir(parents=True, exist_ok=False)
    try:
        source = args.inputs.resolve(); inputs = json.loads((source/"inputs.json").read_text())
        require(inputs["schema"] == "realistic-image-validation-inputs-v1" and inputs["seed_sweep_allowed"] is False,
                "Need unswept-seed image validation inputs")
        builds = {task:{"default":args.builds.resolve()/f"{task}-metadata-{args.build_suffix}"} for task in layouts.HEADERS}
        for family in ("impala", "impoola"):
            path = args.builds.resolve()/f"flappycnn-{family}-metadata-{args.build_suffix}"
            if path.is_dir(): builds["flappycnn"][family] = path
        registry = layouts.freeze_images(builds, root/"metadata")
        successes=[]; failures=[]
        for index, entry in enumerate(inputs["entries"]):
            path = source/entry["config"]; require(sha(path) == entry["sha256"], "Changed declared recipe")
            config = read(path); task = entry["game"]
            require(registry["image_contracts"][task]["shape_hwc"] == entry["shape_hwc"], "Wrong per-game image")
            target = root/"configs"/entry["config"]; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
            family = "nature" if config.getint("policy","encoder") == 2 else "flex"
            record = layouts.describe(root/"metadata", target, task, family, root/"layouts"/f"case-{index:03d}",30)
            _, encoder = graphs.topology(*graphs.settings(config),shape=tuple(entry["shape_hwc"]))
            hidden = config.getint("policy","hidden_size"); layers = config.getint("policy","num_layers")
            actions = registry["actions"][task]["count"]
            expected = encoder+(actions+1)*hidden+layers*3*hidden*hidden
            require(record["parameters"] == expected, "Native layout differs from independent scalar counts")
            binary = root/"metadata"/layouts.binary_key(registry,task,family)
            process([str(binary),"--storage",str(target)],layouts.ROOT,root/"layouts"/f"case-{index:03d}"/"storage.txt",30)
            storage = json.loads((root/"layouts"/f"case-{index:03d}"/"storage.txt").read_text())
            require(storage["parameters"] == expected and storage["shape_hwc"] == entry["shape_hwc"]
                    and storage["cuda_executed"] is False and storage["gpu_queried"] is False
                    and storage["peak_memory_certified"] is False, "Wrong storage/qualification result")
            successes.append(dict(**entry,parameters=expected,storage_floor_bytes=storage["storage_floor_bytes"],
                                  descriptor_sha256=record["descriptor_sha256"]))
        # Each game's actual host target rejects huge float-frame batches before
        # activation registration or device work; these deliberate failures stay.
        for task in layouts.HEADERS:
            original=root/"configs"/task/"stride4-gap.ini"
            for label, section, key, value, message in (
                ("index-overflow","train","minibatch_size","65536","32-bit kernel indexing"),
                ("encoder5","policy","encoder","5","Invalid integer policy/core setting"),
            ):
                config=read(original); config.set(section,key,value)
                directory=root/"rejects"/task/label; directory.mkdir(parents=True)
                path=directory/"config.ini"
                with path.open("x") as stream: config.write(stream)
                command=[str(root/"metadata"/f"{task}/default"),"--storage",str(path)]
                try: process(command,layouts.ROOT,directory/"native.txt",30)
                except ValueError:
                    timing=json.loads((directory/"native.txt.json").read_text())
                    require(timing["returncode"] != 0 and message in (directory/"native.txt").read_text(),
                            "Negative native guard failed for an unrelated reason")
                    failures.append(dict(task=task,case=label,expected_rejection=True,timing=timing))
                else: raise ValueError("Unsafe native configuration was accepted")
        baselines=[]
        for family in ("impala","impoola"):
            if family not in builds["flappycnn"]: continue
            config=read(root/"configs/flappycnn/nature.ini"); config.set("policy","encoder","0")
            path=root/"configs/flappycnn"/(family+".ini")
            with path.open("x") as stream: config.write(stream)
            record=layouts.describe(root/"metadata",path,"flappycnn",family,root/"baseline-layouts"/family,30)
            process([str(root/"metadata"/f"flappycnn/{family}"),"--storage",str(path)],layouts.ROOT,
                    root/"baseline-layouts"/family/"storage.txt",30)
            baselines.append(dict(family=family,parameters=record["parameters"],numerical_acceptance=False))
        layouts.inspect_registry(root/"metadata",current=True)
        result=dict(schema="native-image-metadata-verification-v1",status="passed",layouts=successes,
                    negative_cases=failures,baseline_construction=baselines,
                    gpu_queried=False,cuda_executed=False,cpu_neural_reference=False,
                    numerical_acceptance=False,peak_memory_certified=False,
                    inputs_sha256=sha(source/"inputs.json"))
        save(root/"REPORT.json",result)
        return result
    except BaseException as error:
        save(root/"failure.json",dict(status="failed",error=str(error))); raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","builds","out"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--build-suffix",default="v2",choices=("v2","v3"))
    args=parser.parse_args(); result=run(args)
    print(json.dumps(dict(status=result["status"],layouts=len(result["layouts"]),
                         negative_cases=len(result["negative_cases"]),baselines=len(result["baseline_construction"]))))
