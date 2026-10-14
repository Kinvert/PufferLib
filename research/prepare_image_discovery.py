#!/usr/bin/env python3
"""No-edit 5090 discovery handoff from actual eligible image calibration.

Preparation is host/artifact work. Explicit pilot runs three native PROTEIN
observations; explicit sweep requires its independently audited measured
complete-panel reserve. No phase retries, tunes seeds or changes learners.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import candidate_panel as panels
import cross_game_feedback as feedback
import image_calibration_campaign as calibration
import image_campaign_budget as budgets
import image_curve_bindings as bindings
import image_learning_gate as learning
import image_seed_roles as roles
from claim import process, require, save, sha

SCHEMA="realistic-image-discovery-delivery-v1"
TEMPLATE=panels.ROOT/"research/recipes/image_cnn_discovery_5090.ini"


def calibrated_inputs(root):
    value=calibration.inspect(root,current=False)
    require((root/"execution/RESULT.json").is_file(),"Run the fixed-model calibration first; preparation alone is not learning evidence")
    result=json.loads((root/"execution/RESULT.json").read_text())
    require(result["status"]=="calibration-complete" and result["campaign_sha256"]==sha(root/"CAMPAIGN.json")
            and result["protein_started"] is False,"Use a completed, eligible fixed-model calibration")
    panel=root/"execution/calibration/plan.json"
    analysis,_=calibration.read_report(panel,root/"execution/calibration-report",result["hardware"])
    plan=panels.inspect(panel,current=False)
    gate=learning.qualify(plan,analysis,value["normalization"])
    require(gate["status"]=="eligible-development-search" and gate==result["learning_eligibility"]
            and gate==json.loads((root/"execution/learning-eligibility.json").read_text()),
            "Missing or altered all-game learning eligibility")
    reference=root/"execution/nature-reference.json"
    require(reference.is_file(),"Missing audited Nature curve anchors")
    return value,plan,panel,root/"execution/calibration-report/review/analysis.json",reference


def fixed_recipe(template,value,plan,phase):
    config=panels.read_ini(template)
    require(phase in ("pilot","sweep"),"Unknown discovery phase")
    require(set(plan["task_budgets"])==set(value["normalization"])==set(panels.TASKS),
            "Discovery must retain every calibrated game")
    config["search"].update(max_runs="3" if phase=="pilot" else "128",
        campaign_timeout="14400" if phase=="pilot" else "43200",
        training_seeds=",".join(map(str,plan["seeds"])),
        evaluation_seeds=",".join(map(str,plan["evaluation_seeds"])),
        seed_role="development",appearance_seed=str(plan["appearance_seed"]),
        eval_seed=str(plan["eval_seed"]),episodes=str(plan["episodes"]),slots=str(plan["slots"]))
    for key,number in plan["caps"].items():config["search"][key]=str(number)
    for task,budget in plan["task_budgets"].items():
        require(budget["steps"]>=calibration.MIN_STEPS,"Do not turn a tiny smoke into learning calibration")
        panels.checkpoint_schedule(budget,True)
        config["task."+task]={key:str(number) for key,number in budget.items()}
    for task,anchors in value["normalization"].items():
        config["objective."+task]={key:str(number) for key,number in anchors.items()}
    bindings.check_roles(config,roles.DEFAULT)
    return config


def completed_pilot(root):
    require((root/"RUN.json").is_file(),"The measured pilot launcher must finish before sweep preparation")
    value=json.loads((root/"RUN.json").read_text())
    result_path=root/"feedback/execution/result.json"
    require(value.get("phase")=="pilot" and value.get("status")==value.get("native_status")=="ok"
            and value.get("observations")==value.get("final_observations_replayed")==3
            and value.get("inherited_observations")==0
            and value.get("native_result_sha256")==sha(result_path),
            "Use a successful complete three-observation pilot, not a failed or partial allocation")
    start,end=value["started_monotonic_ns"],value["ended_monotonic_ns"]
    require(type(start) is int and type(end) is int and 0<start<end
            and value["overall_seconds"]==(end-start)/1e9
            and value["overall_seconds"]<=14400 and value["declared_window_exceeded"] is False,
            "Pilot execution/report window was exceeded or its clocks changed")
    return dict(delivery=str(root),run_sha256=sha(root/"RUN.json"))


def prepare(args):
    phase=args.phase; out=args.out.resolve(); root=args.calibration.resolve()
    value,plan,panel,audit,reference=calibrated_inputs(root)
    require(phase=="pilot" or args.pilot is not None,"Sweep preparation requires the measured pilot delivery")
    config=fixed_recipe(TEMPLATE,value,plan,phase)
    out.mkdir(parents=True,exist_ok=False)
    try:
        recipe=out/"recipe.ini"
        with recipe.open("x") as stream:config.write(stream)
        feedback.recipe(recipe)
        runtime=None;pilot_receipt=None
        if phase=="sweep":
            pilot=inspect(args.pilot.resolve(),current=False)
            require(pilot["phase"]=="pilot" and pilot["calibration"]==str(root)
                    and pilot["calibration_campaign_sha256"]==sha(root/"CAMPAIGN.json"),
                    "Runtime pilot came from a different fixed calibration")
            pilot_receipt=completed_pilot(args.pilot.resolve())
            runtime=args.pilot.resolve()/"runtime-review/runtime-calibration.json"
            measured=json.loads(runtime.read_text());budgets.verify_pilot(measured)
            # Only count/deadline may change between pilot and search.
            pilot_recipe,_=feedback.recipe(args.pilot.resolve()/"recipe.ini")
            for section in config.sections():
                left,right=dict(config[section]),dict(pilot_recipe[section])
                if section=="search":
                    for key in ("max_runs","campaign_timeout"):left.pop(key);right.pop(key)
                require(left==right,"Pilot/search architecture envelope or fixed conditions differ")
        process(["bash","research/build_protein_feedback.sh",str(out/"optimizer")],
            panels.ROOT,out/"optimizer-build.txt",600)
        arguments=SimpleNamespace(recipe=recipe,registry=root/"games/registry/registry.json",
            optimizer_build=out/"optimizer",policy_metadata=root/"games/metadata",out=out/"feedback",
            learner_recipe=[task+"="+str(root/record["file"]) for task,record in value["learners"].items()],
            encoder_validation=None,image_validation_catalog=root/"catalog",
            nature_reference=reference,nature_panel=panel,nature_audit=audit,
            seed_roles=root/"seed-roles.json",runtime_calibration=runtime,
            initial_history_plan=args.pilot.resolve()/"feedback/plan.json" if phase=="sweep" else None)
        with redirect_stdout(io.StringIO()):packet=feedback.prepare(arguments)
        receipt=dict(schema=SCHEMA,phase=phase,status="prepared-not-executed",calibration=str(root),
            calibration_campaign_sha256=sha(root/"CAMPAIGN.json"),calibration_plan_sha256=sha(panel),
            template_sha256=sha(TEMPLATE),recipe_sha256=sha(recipe),
            feedback_plan_sha256=sha(out/"feedback/plan.json"),dimensions=packet["dimensions"],
            storage_limit_mib=config.getint("search","image_storage_limit_mib"),
            gpu_queried=False,policy_executed=False,publication_claim_qualified=False)
        require(receipt["dimensions"]==18,"Expected all18 native CNN coordinates")
        if pilot_receipt is not None:receipt["completed_pilot"]=pilot_receipt
        save(out/"CAMPAIGN.json",receipt,exclusive=True)
        inspect(out,current=True)
        return receipt
    except BaseException as error:
        save(out/"preparation-failure.json",dict(status="failed",error=f"{type(error).__name__}: {error}",
            gpu_queried=False,policy_executed=False),exclusive=True)
        raise


def inspect(root,current=False):
    value=json.loads((root/"CAMPAIGN.json").read_text())
    require(value["schema"]==SCHEMA and value["phase"] in ("pilot","sweep")
            and value["dimensions"]==18 and value["storage_limit_mib"]==24576,
            "Wrong discovery delivery")
    require(sha(root/"recipe.ini")==value["recipe_sha256"]
            and sha(root/"feedback/plan.json")==value["feedback_plan_sha256"],"Changed discovery delivery inputs")
    if current:require(sha(TEMPLATE)==value["template_sha256"],"Changed discovery template")
    source=Path(value["calibration"])
    require(sha(source/"CAMPAIGN.json")==value["calibration_campaign_sha256"]
            and sha(source/"execution/calibration/plan.json")==value["calibration_plan_sha256"],
            "Changed fixed learning calibration")
    feedback.inspect(root/"feedback",current=current)
    if value["phase"]=="sweep":
        require(completed_pilot(Path(value["completed_pilot"]["delivery"]))==value["completed_pilot"],
                "Changed completed pilot launcher receipt")
    return value


def run(args):
    require(args.allow_gpu,"Explicit --allow-gpu is required; only run on the idle5090")
    root=args.delivery.resolve();value=inspect(root,current=True)
    require(not (root/"RUN.json").exists() and not (root/"feedback/execution").exists(),
            "Preserve every started allocation; no restart or automatic continuation")
    record=dict(status="running",phase=value["phase"],publication_claim_qualified=False)
    record["started_monotonic_ns"]=time.monotonic_ns()
    seconds=14400 if value["phase"]=="pilot" else 43200
    deadline=time.monotonic()+seconds
    save(root/"RUN.json",record,exclusive=True)
    try:
        with redirect_stdout(io.StringIO()):
            result=feedback.run(SimpleNamespace(plan=root/"feedback/plan.json",allow_gpu=True,
                mode="curve-pilot5090" if value["phase"]=="pilot" else "development"))
        if value["phase"]=="pilot":
            require(deadline>time.monotonic(),"Pilot exhausted its report reserve; preserve failed allocation")
            process([sys.executable,"research/image_campaign_budget.py","--pilot-plan",str(root/"feedback/plan.json"),
                "--out",str(root/"runtime-review")],panels.ROOT,root/"runtime-review.txt",deadline-time.monotonic())
            reserve=json.loads((root/"runtime-review/runtime-calibration.json").read_text())
            record["next_panel_reserve_seconds"]=reserve["next_panel_seconds"]
        else:
            require(deadline>time.monotonic(),"Sweep exhausted its report reserve; preserve failed allocation")
            process([sys.executable,"research/report_image_feedback.py","--plan",str(root/"feedback/plan.json"),
                "--out",str(root/"report")],panels.ROOT,root/"report.txt",min(3600,deadline-time.monotonic()))
        record.update(status=result["status"],observations=len(result["trials"]),
            native_result_sha256=sha(root/"feedback/execution/result.json"))
        return record
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}")
        # No partial game averages or invented PROTEIN observations. Native
        # executions/math/preflight files remain intact, including the last job.
        for plan in sorted((root/"feedback/execution").glob("trial-*/panel/plan.json")):
            if (plan.parent/"execution/result.json").is_file() and deadline>time.monotonic():
                try:
                    process([sys.executable,"research/candidate_viewer.py","report","--plan",str(plan),
                        "--out",str(plan.parent.parent/"failed-prefix-review")],panels.ROOT,
                        plan.parent.parent/"failed-prefix-review.txt",min(1800,deadline-time.monotonic()))
                except BaseException as review_error:record.setdefault("review_errors",[]).append(str(review_error))
        raise
    finally:
        native=root/"feedback/execution/result.json"
        if native.is_file():
            try:
                result=json.loads(native.read_text())
                record.update(native_result_sha256=sha(native),native_status=result["status"],
                    observations=len(result["trials"]),inherited_observations=result.get("inherited_observations",0),
                    final_observations_replayed=result.get("final_observations_replayed"))
            except (OSError,ValueError,KeyError,TypeError) as error:
                record.update(status="failed",native_receipt_error=str(error))
        record["ended_monotonic_ns"]=time.monotonic_ns()
        record["overall_seconds"]=(record["ended_monotonic_ns"]-record["started_monotonic_ns"])/1e9
        record["declared_window_exceeded"]=record["overall_seconds"]>seconds
        save(root/"RUN.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--phase",choices=("pilot","sweep"),required=True)
    prep.add_argument("--calibration",type=Path,required=True);prep.add_argument("--pilot",type=Path)
    prep.add_argument("--out",type=Path,required=True)
    for name in ("inspect","run","handoff"):
        command=sub.add_parser(name);command.add_argument("--delivery",type=Path,required=True)
        if name=="run":command.add_argument("--allow-gpu",action="store_true")
    args=parser.parse_args()
    result=prepare(args) if args.command=="prepare" else inspect(args.delivery.resolve(),True) if args.command=="inspect" \
        else calibration.handoff(args.delivery,panels.ROOT/"build/hardware-artifacts") if args.command=="handoff" else run(args)
    print(json.dumps(result,indent=2))
    if result.get("status")=="failed":raise SystemExit(1)
