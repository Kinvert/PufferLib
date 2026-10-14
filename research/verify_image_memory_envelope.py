#!/usr/bin/env python3
"""Targeted native host registration at the new 16-actor image geometry.

Checks C8/K8/GAP/projection64/H128/L1 at all six RGB shapes/first strides1/2/4/8.
No CPU CNN, CUDA allocation/query or policy execution. This does not qualify
the broad architecture envelope, actual peak VRAM or numerical correctness.
"""
import argparse
import json
from pathlib import Path

import candidate_panel as panels
import prepare_image_inputs as inputs
import cnn_graph_acceptance as graphs
import image_memory_budget as memory
from claim import require, save, sha


def run(metadata,out):
    metadata,out=metadata.resolve(),out.resolve()
    registry=panels.policy_layout.inspect_registry(metadata,current=True)
    require(registry["schema"]==panels.policy_layout.IMAGE_SCHEMA,"Use the current native RGB metadata registry")
    out.mkdir(parents=True,exist_ok=False)
    record=dict(schema="targeted-image-storage-envelope-v1",status="running",cases=[],
        metadata_registry_sha256=sha(metadata/"registry.json"),metadata_root=str(metadata),
        sources=panels.capture(out),gpu_queried=False,cuda_executed=False,policy_executed=False,
        peak_memory_certified=False,broad_search_envelope_qualified=False)
    try:
        for task in panels.TASKS:
            for stride in (1,2,4,8):
                folder=out/task/("stride%d"%stride);folder.mkdir(parents=True)
                config,_=inputs.config(task,inputs.architecture(stride=stride,gap=1))
                config["vec"].update(total_agents="16",num_buffers="1",num_threads="2")
                config["train"].update(horizon="16",minibatch_size="16")
                path=folder/"config.ini"
                with path.open("x") as stream:config.write(stream)
                layout=panels.policy_layout.describe(metadata,path,task,"flex",folder/"layout",60)
                storage=panels.policy_layout.storage(metadata,path,task,"flex",folder/"storage",60,layout["parameters"])
                descriptor=json.loads((folder/"layout/descriptor.json").read_text())
                counts=json.loads((folder/"storage/descriptor.json").read_text())
                shape=registry["image_contracts"][task]["shape_hwc"]
                _,encoder_parameters=graphs.topology(*graphs.settings(config),shape=shape)
                require(descriptor["encoder_parameters"]==encoder_parameters,"Native encoder parameter count differs from scalar topology")
                require(counts["agents"]==counts["learner_batch"]==counts["horizon"]==16
                        and counts["buffers"]==1 and not counts["gpu_queried"] and not counts["cuda_executed"],
                        "Wrong registration geometry or scope")
                require(storage["storage_floor_bytes"]<memory.MAXIMUM_MIB*2**20,
                        "Targeted candidate exceeds the explicit 5090 storage floor budget")
                record["cases"].append(dict(environment=task,stride=stride,shape_hwc=shape,
                    config_sha256=sha(path),layout=layout,storage=storage))
                save(out/"progress.json",record)
        record["status"]="passed"
        return record
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:save(out/"REPORT.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();value=run(args.metadata,args.out)
    print(json.dumps(dict(status=value["status"],cases=len(value["cases"]),
        maximum_storage_floor_gib=max(c["storage"]["storage_floor_bytes"] for c in value["cases"])/2**30,
        cuda_executed=False,peak_memory_certified=False,broad_search_envelope_qualified=False),indent=2))
