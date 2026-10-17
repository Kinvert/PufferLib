#!/usr/bin/env python3
"""Fixed three-model 5090 calibration; native training/evaluation, external glue.

Preparation is GPU-free. Explicit run performs hardware-matched CUDA checks,
an audited timing probe, then admits a complete learning panel if it fits.
It never proposes architectures, retries a failed allocation, or tunes seeds.
"""
import argparse
import configparser
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import tarfile
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

import candidate_panel as panels
import image_validation_catalog as catalog
import image_encoder_checks as checks
import image_seed_roles as roles
import prepare_image_inputs as inputs
from claim import process, require, save, sha

SCHEMA="realistic-image-5090-calibration-v1"
MODELS=("quality-reference","spatial-control","nature-cnn")
PROBE_STEPS=65536
MIN_STEPS=2097152
MAX_STEPS=13631488
ROLLOUT=256
QUANTUM=8*ROLLOUT
TOTAL_SECONDS=12*3600
RESERVE_SECONDS=1800
EPISODES=5
SLOTS=16
NORMALIZATION={t:dict(offset=1. if t=="snakecnn" else 0.,
    target=100. if t=="flappycnn" else 20. if t in ("snakecnn","breakoutcnn") else 1.) for t in panels.TASKS}


def write_ini(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream: value.write(stream)


def prepare(args):
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    try:
        require(sys.version_info[:2]==(3,12),"Use the existing uv Python 3.12 venv")
        require(os.environ.get("NVCC_ARCH")=="sm_120","Set NVCC_ARCH=sm_120")
        require(not any(os.environ.get(key) for key in
            ("NVCC_EXTRA","NVCC_FLAGS","NVCC_APPEND_FLAGS","NVCC_PREPEND_FLAGS")),"Unset compiler overrides")
        shutil.copyfile(panels.ROOT/"research/recipes/image_seed_roles.json",root/"seed-roles.json")
        records=[]
        for name,source in (("quality-reference","panel_smoke_quality.ini"),
                            ("spatial-control","image_fast_spatial_deep.ini")):
            target=root/"models"/(name+".ini"); target.parent.mkdir(exist_ok=True)
            shutil.copyfile(panels.ROOT/"research/recipes"/source,target)
            records.append(dict(name=name,config=str(target.relative_to(root)),architecture=panels.architecture(target)))
        nature=configparser.ConfigParser(interpolation=None)
        nature["policy"]=dict(encoder="2",hidden_size="128",num_layers="1")
        write_ini(root/"models/nature-cnn.ini",nature)
        records.append(dict(name="nature-cnn",config="models/nature-cnn.ini",architecture=dict(encoder=2)))
        learners={}
        for task in panels.TASKS:
            value,paths=inputs.config(task,inputs.CONTROLS["quality"])
            # Explicit common geometry, smaller than historical tiny-image batches.
            # Keep the compare.ini scalar optimizer settings; no learner sweep.
            value["vec"].update(total_agents="16",num_buffers="1",num_threads="2")
            value["train"].update(horizon="16",minibatch_size="16")
            allowed={"base":("async","cudagraphs","reset_every_horizon"),
                "vec":("total_agents","num_buffers","num_threads"),
                "train":tuple(key for key in value["train"] if key not in ("gpus","total_timesteps"))}
            learner=configparser.ConfigParser(interpolation=None)
            learner.read_dict({section:{key:value[section][key] for key in keys} for section,keys in allowed.items()})
            destination=root/"learners"/(task+".ini");write_ini(destination,learner)
            learners[task]=dict(file=str(destination.relative_to(root)),sha256=sha(destination),
                source_sha256={str(path.relative_to(panels.ROOT)):sha(path) for path in paths})
        environment=dict(os.environ,IMAGE_PYTHON=sys.executable)
        if getattr(args,"native_builds",None) is None:
            process(["bash","research/build_image_delivery.sh",str(root/"games")],
                panels.ROOT,root/"build.txt",3600,environment)
        else:
            # Development-only host reuse, still checking every owned native byte.
            import import_image_builds
            builds=args.native_builds.resolve()
            import_image_builds.register(SimpleNamespace(builds=builds,out=root/"games/registry"))
            mapping=json.loads(builds.read_text())
            panels.policy_layout.freeze_images({task:{family:(builds.parent/path).resolve()
                for family,path in groups.items()} for task,groups in mapping.items()},root/"games/metadata")
        bundle=catalog.build(SimpleNamespace(registry=root/"games/registry/registry.json",
            batches=[1,4,16],libraries_from=getattr(args,"libraries_from",None),out=root/"catalog",
            compact_mode_exports=True,partition_batches=True))
        sources=panels.capture(root)
        value=dict(schema=SCHEMA,status="prepared-not-executed",models=records,learners=learners,
            source_sha256=sources,normalization=NORMALIZATION,probe_steps=PROBE_STEPS,
            min_steps=MIN_STEPS,max_steps=MAX_STEPS,episodes=EPISODES,slots=SLOTS,
            overall_seconds=TOTAL_SECONDS,report_archive_reserve_seconds=RESERVE_SECONDS,
            trained_checkpoint_count=8,seed_roles_sha256=sha(root/"seed-roles.json"),
            catalog_sha256=sha(root/"catalog/catalog.json"),registry_sha256=sha(root/"games/registry/registry.json"),
            model_sha256={r["name"]:sha(root/r["config"]) for r in records},
            gpu_queried=False,policy_executed=False,learning_calibrated=False,
            publication_claim_qualified=False,automatically_starts_protein=False)
        save(root/"CAMPAIGN.json",value,exclusive=True);inspect(root,current=True)
        return value
    except BaseException as error:
        save(root/"preparation-failure.json",dict(status="failed",error=f"{type(error).__name__}: {error}",
            gpu_queried=False,policy_executed=False),exclusive=True);raise


def inspect(root,current=False):
    value=json.loads((root/"CAMPAIGN.json").read_text())
    require(value["schema"]==SCHEMA and tuple(r["name"] for r in value["models"])==MODELS,
            "Wrong calibration delivery or model list")
    expected=dict(normalization=NORMALIZATION,probe_steps=PROBE_STEPS,min_steps=MIN_STEPS,max_steps=MAX_STEPS,
        episodes=EPISODES,slots=SLOTS,overall_seconds=TOTAL_SECONDS,
        report_archive_reserve_seconds=RESERVE_SECONDS,trained_checkpoint_count=8,
        automatically_starts_protein=False)
    require(all(value[key]==number for key,number in expected.items()),"Changed declared calibration allocation")
    panels.verify_sources(root,value["source_sha256"],current)
    require(sha(root/"seed-roles.json")==value["seed_roles_sha256"],"Changed seed roles")
    role=json.loads((root/"seed-roles.json").read_text());roles.validate(role)
    require(role==roles.DEFAULT,"Use the committed development/confirmation seed assignments")
    for record in value["models"]:
        require(sha(root/record["config"])==value["model_sha256"][record["name"]],"Changed fixed control")
        if record["name"]!="nature-cnn": require(panels.architecture(root/record["config"])==record["architecture"],"Changed control graph")
    require(set(value["learners"])==set(panels.TASKS),"Missing per-game learner")
    for record in value["learners"].values(): require(sha(root/record["file"])==record["sha256"],"Changed per-game learner")
    require(sha(root/"catalog/catalog.json")==value["catalog_sha256"]
            and sha(root/"games/registry/registry.json")==value["registry_sha256"],"Changed numerical/training registry")
    catalog.inspect(root/"catalog",current=current)
    import import_image_builds
    import_image_builds.inspect(root/"games/registry",json.loads((root/"games/registry/registry.json").read_text()),current=current)
    return value


def arguments(root,value,output,math_file,budgets,calibration):
    role=json.loads((root/"seed-roles.json").read_text())
    return SimpleNamespace(out=output,registry=root/"games/registry/registry.json",
        policy_metadata=root/"games/metadata",image_math=math_file,
        candidate=[r["name"]+"="+str(root/r["config"]) for r in value["models"] if r["name"]!="nature-cnn"],
        baselines=["nature_cnn"],native_campaign=[],per_game_learners=True,
        learner_recipe=[task+"="+str(root/record["file"]) for task,record in value["learners"].items()],
        seeds=role["development_training"] if calibration else [role["development_training"][0]],
        evaluation_seeds=role["development_evaluation"] if calibration else [role["development_evaluation"][0]],
        seed_roles=root/"seed-roles.json" if calibration else None,seed_role="development",
        include_initial=calibration,appearance_seed=role["appearance_seed"],eval_seed=role["development_evaluation"][0],
        steps=PROBE_STEPS,checkpoint_steps=PROBE_STEPS,
        task_budget=[f"{task}={b['steps']}:{b['checkpoint_steps']}" for task,b in budgets.items()],
        episodes=EPISODES,slots=SLOTS,pong_max_decisions=8192,breakout_max_frames=16384)


def choose_budgets(plan,analysis,result,report_seconds,remaining_seconds):
    require(plan["protocol"]==panels.IMAGE_PROTOCOL and analysis["status"]==result["status"]=="ok"
            and tuple(c["name"] for c in plan["candidates"])==MODELS
            and plan["seeds"]==[7] and plan["evaluation_seeds"]==[7007]
            and plan["episodes"]==EPISODES and plan["slots"]==SLOTS
            and analysis["audited_jobs"]==18 and not analysis["missing_evaluations"]
            and analysis["audited_evaluations"]==123,"Use a complete paired timing probe")
    require(math.isfinite(report_seconds) and report_seconds>0 and math.isfinite(remaining_seconds),"Invalid measured stage clocks")
    speed={};seen=set()
    for row in analysis["training"]:
        key=row["environment"],row["candidate"],row["seed"]
        require(key not in seen and key[0] in panels.TASKS and key[1] in MODELS and key[2]==7,
                "Duplicate or unknown probe job")
        seen.add(key);duration=row["train_seconds"]
        require(math.isfinite(duration) and duration>0 and row["process_sps"]==PROBE_STEPS/duration,
                "Probe SPS differs from audited process time")
        speed[key[0]]=max(speed.get(key[0],0.),duration/PROBE_STEPS)
    require(len(seen)==18 and set(speed)==set(panels.TASKS)
            and all(b==dict(steps=PROBE_STEPS,checkpoint_steps=PROBE_STEPS) for b in plan["task_budgets"].values()),
            "Probe changes declared decision budgets")
    wall=(result["ended_monotonic_ns"]-result["started_monotonic_ns"])/1e9
    training=math.fsum(r["train_seconds"] for r in analysis["training"])
    require(wall>=training>0,"Probe process clocks overlap or leave its supervision window")
    # Same episodes/slots/styles; initial+8 stages and three replicas replace one.
    # Retain a factor-two margin. Learned episodes can still be longer: the actual
    # outer deadline remains authoritative and partial panels never qualify.
    fixed=2.*27.*(wall-training+report_seconds)+RESERVE_SECONDS
    train_min=2.*3.*3.*MIN_STEPS*math.fsum(speed.values())
    if remaining_seconds<fixed+train_min:
        return dict(status="insufficient-time-for-minimum-learning-panel",remaining_seconds=remaining_seconds,
            fixed_evaluation_reporting_reserve_seconds=fixed,minimum_training_reserve_seconds=train_min,
            required_seconds=fixed+train_min,slowest_seconds_per_decision=speed,budgets=None,
            empirical_estimate=True,hard_runtime_guarantee=False)
    share=(remaining_seconds-fixed)/(2.*3.*3.*len(panels.TASKS))
    budgets={task:dict(steps=max(MIN_STEPS,min(MAX_STEPS,int(share/rate)//QUANTUM*QUANTUM)))
             for task,rate in speed.items()}
    for b in budgets.values():b["checkpoint_steps"]=b["steps"]//8
    predicted=fixed+2.*3.*3.*math.fsum(speed[t]*b["steps"] for t,b in budgets.items())
    require(predicted<=remaining_seconds+1e-6,"Admitted budgets exceed measured reserve")
    return dict(status="admit-fixed-learning-panel",remaining_seconds=remaining_seconds,
        fixed_evaluation_reporting_reserve_seconds=fixed,estimated_total_seconds=predicted,
        slowest_seconds_per_decision=speed,budgets=budgets,empirical_estimate=True,hard_runtime_guarantee=False)


def report_panel(root,plan,out,deadline):
    remaining=deadline-time.monotonic();require(remaining>0,"No time remains for the reserved report")
    return process([sys.executable,"research/report_image_feedback.py","--panel","--plan",str(plan),"--out",str(out)],
        panels.ROOT,out.parent/(out.name+".txt"),min(remaining,3600))


def read_report(plan_path,report_root,device):
    summary=json.loads((report_root/"REPORT.json").read_text())
    audit_path=report_root/"review/analysis.json"
    analysis=json.loads(audit_path.read_text())
    result_path=plan_path.parent/"execution/result.json"
    require(summary["status"]=="ok" and summary["hardware"]==analysis["hardware"]==device
            and summary["plan_sha256"]==analysis["plan_sha256"]==sha(plan_path)
            and summary["result_sha256"]==analysis["result_sha256"]==sha(result_path)
            and summary["files_sha256"]["review/analysis.json"]==sha(audit_path),
            "Report differs from the audited panel, result or hardware")
    return analysis,json.loads(result_path.read_text())


def run(args):
    require(args.allow_gpu,"Explicit --allow-gpu is required; this is only for the idle 5090")
    root=args.prepared.resolve();value=inspect(root,current=True)
    execution=root/"execution";execution.mkdir(exist_ok=False)
    started=time.monotonic_ns();deadline=time.monotonic()+TOTAL_SECONDS
    record=dict(schema=SCHEMA,status="running",started_monotonic_ns=started,stages=[],
        campaign_sha256=sha(root/"CAMPAIGN.json"),publication_claim_qualified=False,protein_started=False)
    def receipt(stage):
        record["stages"].append(stage);save(execution/"progress.json",record)
    try:
        math_mapping={};device=None
        for model in value["models"]:
            out=execution/"math"/model["name"]
            report=catalog.run(SimpleNamespace(catalog=root/"catalog",config=root/model["config"],
                name=model["name"],out=out,hardware="5090",timeout=120,deadline=deadline-RESERVE_SECONDS))
            require(device is None or device==report["hardware"],"Changed numerical execution hardware")
            device=report["hardware"]
            record["hardware"]=device
            manifest=json.loads((out/"evidence.json").read_text())
            math_mapping[model["name"]]={task:os.path.relpath((out/path).resolve(),execution)
                for task,path in manifest["candidates"][model["name"]].items()}
            receipt(dict(name="math-"+model["name"],report=str(out/"REPORT.json"),sha256=sha(out/"REPORT.json")))
        save(execution/"math.json",dict(schema=checks.SCHEMA,candidates=math_mapping),exclusive=True)
        probe_budget={t:dict(steps=PROBE_STEPS,checkpoint_steps=PROBE_STEPS) for t in panels.TASKS}
        with redirect_stdout(io.StringIO()):
            probe=panels.prepare(arguments(root,value,execution/"probe",execution/"math.json",probe_budget,False))
            panels.run(SimpleNamespace(plan=execution/"probe/plan.json",mode="development",allow_gpu=True,
                timeout=min(3*3600,deadline-time.monotonic()-RESERVE_SECONDS),train_timeout=300,eval_timeout=300))
        timing=report_panel(root,execution/"probe/plan.json",execution/"probe-report",deadline-RESERVE_SECONDS)
        audit,probe_result=read_report(execution/"probe/plan.json",execution/"probe-report",device)
        receipt(dict(name="probe",report=str(execution/"probe-report/REPORT.json"),
            sha256=sha(execution/"probe-report/REPORT.json")))
        admission=choose_budgets(probe,audit,probe_result,
            (timing["end_monotonic_ns"]-timing["launch_monotonic_ns"])/1e9,deadline-time.monotonic())
        save(execution/"ADMISSION.json",admission,exclusive=True)
        if admission["budgets"] is None:
            record.update(status="stopped-insufficient-time",admission=admission)
            return record
        with redirect_stdout(io.StringIO()):
            calibration=panels.prepare(arguments(root,value,execution/"calibration",execution/"math.json",admission["budgets"],True))
            panels.run(SimpleNamespace(plan=execution/"calibration/plan.json",mode="development",allow_gpu=True,
                timeout=deadline-time.monotonic()-RESERVE_SECONDS,train_timeout=3600,eval_timeout=900))
        report_panel(root,execution/"calibration/plan.json",execution/"calibration-report",deadline)
        import image_learning_gate as learning
        import image_curve_feedback as curves
        audit_path=execution/"calibration-report/review/analysis.json"
        analysis,_=read_report(execution/"calibration/plan.json",execution/"calibration-report",device)
        gate=learning.qualify(calibration,analysis,NORMALIZATION)
        save(execution/"learning-eligibility.json",gate,exclusive=True)
        if gate["status"]=="eligible-development-search":
            curves.freeze(calibration,analysis,execution/"calibration",NORMALIZATION,
                execution/"nature-reference.json",analysis_sha256=sha(audit_path))
        receipt(dict(name="calibration",report=str(execution/"calibration-report/REPORT.json"),
            sha256=sha(execution/"calibration-report/REPORT.json")))
        record.update(status="calibration-complete" if gate["status"]=="eligible-development-search"
            else "stopped-unlearned-games",learning_eligibility=gate,admission=admission)
        return record
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}")
        # Native panel files remain intact. Export a descriptive failed prefix if
        # one exists; never turn partial games/replicas into learning eligibility.
        for stage in ("probe","calibration"):
            plan=execution/stage/"plan.json"
            if plan.is_file() and (plan.parent/"execution/result.json").is_file() and not (execution/(stage+"-failure-review")).exists():
                try:
                    remaining=deadline-time.monotonic()
                    if remaining>0:
                        process([sys.executable,"research/candidate_viewer.py","report","--plan",str(plan),
                            "--out",str(execution/(stage+"-failure-review"))],panels.ROOT,
                            execution/(stage+"-failure-review.txt"),min(remaining,1800))
                except BaseException as review_error:record.setdefault("review_errors",[]).append(str(review_error))
        raise
    finally:
        record["ended_monotonic_ns"]=time.monotonic_ns()
        record["overall_seconds"]=(record["ended_monotonic_ns"]-started)/1e9
        record["declared_window_exceeded"]=record["overall_seconds"]>TOTAL_SECONDS
        save(execution/"RESULT.json",record)


def handoff(root,archive_directory):
    """Compact evidence summary; original arrays/weights/binaries stay on 5090."""
    root,archive_directory=root.resolve(),archive_directory.resolve()
    require((root/"CAMPAIGN.json").is_file(),"Missing prepared campaign")
    suffixes={".json",".ini",".csv",".txt",".md",".html",".tsv",".sha256"}
    files=[p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
        and p.suffix in suffixes and "source" not in p.relative_to(root).parts]
    require(sum(p.stat().st_size for p in files)<2**30,"Compact metadata archive exceeds1GiB; retain raw tree and report this")
    archive_directory.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix="image-calibration-"+root.name+".",suffix=".tar.gz",dir=archive_directory)
    os.close(fd);path=Path(name)
    declaration=dict(schema="image-calibration-summary-archive-v1",self_contained_raw_audit=False,
        weights_arrays_binaries_included=False,original_tree=str(root),
        files_sha256={str(p.relative_to(root)):sha(p) for p in files})
    encoded=(json.dumps(declaration,indent=2)+"\n").encode()
    with tarfile.open(path,"w:gz") as archive:
        info=tarfile.TarInfo(root.name+"/ARCHIVE.json");info.size=len(encoded)
        archive.addfile(info,io.BytesIO(encoded))
        for file in files:archive.add(file,arcname=root.name+"/"+str(file.relative_to(root)),recursive=False)
    digest=sha(path);path.with_suffix(path.suffix+".sha256").write_text(digest+"  "+path.name+"\n")
    return dict(archive=str(path),sha256=digest,bytes=path.stat().st_size,
        raw_tree_required_for_independent_numerical_replay=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--out",type=Path,required=True)
    prep.add_argument("--native-builds",type=Path,help="Host development reuse only; verify owned native build receipts")
    prep.add_argument("--libraries-from",type=Path,help="Host development reuse only; freeze fresh tools, never reuse CUDA results")
    inspect_cmd=sub.add_parser("inspect");inspect_cmd.add_argument("--prepared",type=Path,required=True)
    execute=sub.add_parser("run");execute.add_argument("--prepared",type=Path,required=True);execute.add_argument("--allow-gpu",action="store_true")
    pack=sub.add_parser("handoff");pack.add_argument("--prepared",type=Path,required=True)
    pack.add_argument("--out",type=Path,default=panels.ROOT/"build/hardware-artifacts")
    args=parser.parse_args()
    result=prepare(args) if args.command=="prepare" else inspect(args.prepared.resolve(),True) if args.command=="inspect" \
        else handoff(args.prepared,args.out) if args.command=="handoff" else run(args)
    print(json.dumps(result,indent=2))
    if result.get("status") in ("stopped-insufficient-time","stopped-unlearned-games"):raise SystemExit(2)
