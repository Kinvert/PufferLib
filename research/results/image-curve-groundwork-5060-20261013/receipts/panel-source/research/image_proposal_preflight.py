"""Reject unsafe image proposals using native host registration before CUDA.

These are tensor-size/index checks, not a CPU neural reference or a measured
peak-memory certificate. An unsafe proposal stops with its retained receipts.
"""
import json
from pathlib import Path
import time

import candidate_panel as panels
from claim import require, save, sha, normalized

SCHEMA="native-image-proposal-preflight-v1"


def configuration(root, value, recipe, candidate_path, task, output):
    import cross_game_feedback as feedback
    budgets=feedback.task_budgets(recipe)
    learner=next(root/r["file"] for r in value["learner_recipes"] if r["environment"]==task)
    config=panels.base_config(root/"source",task,budgets[task],recipe.getint("search","appearance_seed"),learner,True)
    candidate=dict(name="proposal-preflight",architecture=panels.architecture(candidate_path))
    seed=int(recipe.get("search","training_seeds").split(",")[0])
    panels.apply_job(config,candidate,seed,output/task)
    return config


def run(root, value, recipe, candidate_path, output, deadline):
    limit=recipe.getint("search","image_storage_limit_mib",fallback=4096)*2**20
    require(0<limit<=12288*2**20,"Explicit bounded image storage floor required")
    metadata=root/"policy-metadata"
    registry=panels.policy_layout.inspect_registry(metadata,current=True)
    require(registry.get("schema")==panels.policy_layout.IMAGE_SCHEMA
            and registry["image_contracts"]==value["image_contracts"],"Preflight uses different image metadata")
    output.mkdir(parents=True,exist_ok=False)
    record=dict(schema=SCHEMA,status="running",candidate_sha256=sha(candidate_path),storage_limit_bytes=limit,
                image_contracts=value["image_contracts"],cases=[],cuda_executed=False,gpu_queried=False,
                peak_memory_certified=False,preparation_root=str(output),metadata_root=str(metadata))
    def remaining():
        seconds=deadline-time.monotonic(); require(seconds>0,"Campaign deadline before image preflight")
        return min(60,seconds)
    try:
        for task in panels.TASKS:
            directory=output/task; directory.mkdir()
            config=configuration(root,value,recipe,candidate_path,task,output)
            path=directory/"config.ini"
            with path.open("x") as stream: config.write(stream)
            layout=panels.policy_layout.describe(metadata,path,task,"flex",directory/"layout",remaining())
            storage=panels.policy_layout.storage(metadata,path,task,"flex",directory/"storage",remaining(),layout["parameters"])
            record["cases"].append(dict(environment=task,config_sha256=sha(path),layout=layout,storage=storage))
            save(output/"progress.json",record)
            require(storage["storage_floor_bytes"]<limit,"Image proposal storage floor exceeds admission bound: "+task)
        record["status"]="passed"
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally: save(output/"REPORT.json",record)
    return record


def inspect(root, value, recipe, candidate_path, output):
    record=json.loads((output/"REPORT.json").read_text())
    require(record["schema"]==SCHEMA and record["status"]=="passed"
            and record["candidate_sha256"]==sha(candidate_path)
            and record["image_contracts"]==value["image_contracts"]
            and record["cuda_executed"] is False and record["gpu_queried"] is False
            and record["peak_memory_certified"] is False,"Incomplete/wrong pre-CUDA registration")
    require(record["storage_limit_bytes"]==recipe.getint("search","image_storage_limit_mib",fallback=4096)*2**20,
            "Changed image admission bound")
    require([c["environment"] for c in record["cases"]]==list(panels.TASKS),"Missing/duplicate pre-CUDA game")
    metadata=root/"policy-metadata"; original=Path(record["preparation_root"])
    for case in record["cases"]:
        task=case["environment"]; path=output/task/"config.ini"
        require(sha(path)==case["config_sha256"]
                and normalized(panels.read_ini(path))==normalized(configuration(root,value,recipe,candidate_path,task,original)),
                "Preflight altered a learner, graph, world, seed or budget")
        proof=panels.policy_layout.inspect_description(metadata,path,task,"flex",output/task/"layout",
            Path(record["metadata_root"]),original/task/"config.ini",panels.ROOT,timeout=case["layout"]["timing"]["timeout"])
        storage=panels.policy_layout.inspect_storage(metadata,path,task,"flex",output/task/"storage",
            Path(record["metadata_root"]),original/task/"config.ini",panels.ROOT,proof["parameters"],
            timeout=case["storage"]["timing"]["timeout"])
        require(proof==case["layout"] and storage==case["storage"]
                and storage["storage_floor_bytes"]<record["storage_limit_bytes"],"Changed image admission receipt")
    return record
