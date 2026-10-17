#!/usr/bin/env python3
"""Learning controls -> frozen recipes -> native cross-game CNN discovery.

Only native CUDA trains/optimizes/tests models. This is preparation, process,
scalar eligibility and artifact/report glue. No partial-game feedback, retries,
best checkpoints, or long G240 run. All failures leave their original packet.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import candidate_panel as panels
import candidate_viewer as viewer
import cnn_graph_acceptance as graph_checks
import cross_game_feedback as feedback
import pixel_frontiers as reporting

ROOT=panels.ROOT
SCHEMA="learning-cross-game-campaign-v1"
require,sha,save=panels.require,panels.sha,panels.save


def study(path):
    config=panels.panel.read(path)
    require(set(config.sections())=={"campaign"}|{"task."+t for t in panels.TASKS},"Incomplete campaign/tasks")
    c=config["campaign"]
    seeds={name:[int(v.strip()) for v in c[name].split(",")] for name in ("calibration_seeds","search_seeds")}
    require(all(len(v)==2 and len(set(v))==2 and all(0<=s<2**32 for s in v) for v in seeds.values())
            and not set(seeds["calibration_seeds"])&set(seeds["search_seeds"]),"Use two disjoint paired seeds per phase")
    require(5<=c.getint("max_runs")<=64 and 0<=c.getint("random_warmup")<c.getint("max_runs"),"Real campaign needs bounded GP/exploration")
    require(0<float(c["seed_floor"])<=float(c["learning_floor"])<1,"Positive fixed learning thresholds required")
    for k,hi in (("calibration_timeout",172800),("search_timeout",172800),("reference_timeout",172800),
                 ("train_timeout",3600),("eval_timeout",3600),("proposal_timeout",3600),
                 ("episodes",1000000),("slots",1024),("pong_max_decisions",16777216),("breakout_max_frames",16777216)):
        require(0<c.getint(k)<=hi,"Invalid campaign bound: "+k)
    allocated=[]
    for phase in ("calibration","search"):
        base=c.getint(phase+"_eval_seed")
        require(0<=base<=2**32-2,"Invalid evaluation seeds")
        allocated+=list(range(base,base+2))
    require(len(set(allocated))==4 and not set(allocated)&set(sum(seeds.values(),[])),"Evaluation/training seed overlap")
    budgets={t:{k:config.getint("task."+t,k) for k in ("steps","checkpoint_steps")} for t in panels.TASKS}
    for task,b in budgets.items():
        require(b["steps"]>0 and 0<b["checkpoint_steps"]<=b["steps"]
                and b["steps"]%b["checkpoint_steps"]==0 and b["steps"]//b["checkpoint_steps"]<=4,
                "One to four exact checkpoint stages required: "+task)
    return config,seeds,budgets


def panel_args(root, value, phase, out):
    config,seeds,budgets=study(root/"study.ini"); c=config["campaign"]
    return SimpleNamespace(out=out,registry=Path(value["registry"]),policy_metadata=Path(value["metadata"]),
        candidate=["quality-reference="+str(root/"quality.ini")],native_campaign=[],baselines=["nature_cnn"],
        seeds=seeds["calibration_seeds" if phase=="calibration" else "search_seeds"],
        steps=budgets["connect4cnn"]["steps"],checkpoint_steps=budgets["connect4cnn"]["checkpoint_steps"],
        task_budget=[f"{t}={b['steps']}:{b['checkpoint_steps']}" for t,b in budgets.items()],
        per_game_learners=True,learner_recipe=[t+"="+str(root/p) for t,p in value["learners"].items()],
        appearance_seed=c.getint("appearance_seed"),eval_seed=c.getint(phase+"_eval_seed"),
        episodes=c.getint("episodes"),slots=c.getint("slots"),pong_max_decisions=c.getint("pong_max_decisions"),
        breakout_max_frames=c.getint("breakout_max_frames"))


def prepare(args):
    config,seeds,budgets=study(args.study)
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    try:
        for origin,name in ((args.study,"study.ini"),(ROOT/"research/recipes/cross_game_feedback_prepare.ini","search-space.ini"),
                            (ROOT/"research/recipes/panel_smoke_quality.ini","quality.ini"),
                            (ROOT/"research/recipes/panel_nature.ini","nature.ini")):
            shutil.copyfile(origin,root/name)
        learners={}
        for task in panels.TASKS:
            filename=config.get("task."+task,"learner",fallback=None)
            if filename:
                source=(args.study.parent/filename).resolve()
                require(source.parent==args.study.parent.resolve(),"Learner must be a neighboring recipe")
                (root/"learners").mkdir(exist_ok=True)
                target=root/"learners"/(task+".ini"); shutil.copyfile(source,target)
                learners[task]=str(target.relative_to(root))
        graph_checks.inspect_bundle(args.validation,current=True)
        feedback.optimizer_receipts(args.optimizer,current=True)
        value=dict(schema=SCHEMA,root=str(root),registry=str(args.registry.resolve()),metadata=str(args.metadata.resolve()),
            optimizer=str(args.optimizer.resolve()),validation=str(args.validation.resolve()),learners=learners,
            validation_sha256=sha(args.validation/"bundle.json"),publication_claim_qualified=False)
        with redirect_stdout(io.StringIO()):
            controls=panels.prepare(panel_args(root,value,"calibration",root/"calibration"))
            panels.prepare(panel_args(root,value,"search",root/"references"))
        # The validation currently covers actor batch64 and learner batch2048.
        for j in controls["jobs"]:
            cfg=panels.panel.read(root/"calibration"/j["config"])
            require(cfg.getint("vec","total_agents")==64 and cfg.getint("train","minibatch_size")==2048,
                    "Campaign recipe leaves the declared graph-check batch coverage")
        value["files_sha256"]={str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file()}
        save(root/"campaign.json",value); inspect(root,current=True)
        return value
    except BaseException as error:
        save(root/"preparation_failure.json",dict(status="failed",error=str(error))); raise


def inspect(root,current=False):
    value=json.loads((root/"campaign.json").read_text())
    require(value["schema"]==SCHEMA and (not current or value["root"]==str(root.resolve())),"Wrong/relocated campaign")
    for n,h in value["files_sha256"].items():
        require(not Path(n).is_absolute() and ".." not in Path(n).parts and sha(root/n)==h,"Changed campaign input: "+n)
    graph_checks.inspect_bundle(Path(value["validation"]),current=current)
    require(sha(Path(value["validation"])/"bundle.json")==value["validation_sha256"],"Changed campaign validation")
    panels.inspect(root/"calibration/plan.json",current=current)
    panels.inspect(root/"references/plan.json",current=current)
    study(root/"study.ini")
    return value


def eligibility(plan,analysis,recipe,mean_floor,seed_floor):
    require(analysis["status"]=="ok" and not analysis["missing_evaluations"]
            and analysis["audited_jobs"]==len(plan["jobs"]),"Incomplete calibration cannot start search")
    controls=[]
    for candidate in plan["candidates"]:
        name=candidate["name"]; jobs=[j for j in plan["jobs"] if j["candidate"]==name]
        rows=[r for r in analysis["observations"] if r["candidate"]==name]
        subset={**plan,"candidates":[candidate],"jobs":jobs,"planned_evaluations":len(rows)}
        evidence={**analysis,"observations":rows,"audited_jobs":len(jobs)}
        result=feedback.aggregate(subset,evidence,recipe)
        for game in result["games"]:
            task=game["environment"]; lo,hi=game["offset"],game["target"]
            seed_means=[]
            for seed in plan["seeds"]:
                selected=[r for r in rows if r["environment"]==task and r["seed"]==seed
                          and r["decisions"]==plan["task_budgets"][task]["steps"]]
                require(len(selected)==panels.panel.ENVIRONMENTS[task],"Missing calibration seed/drawing")
                seed_means.append(math.fsum(max(0,min(1,(r["score_lower"]-lo)/(hi-lo))) for r in selected)/len(selected))
            controls.append(dict(environment=task,candidate=name,mean=game["normalized_lower"],seed_means=seed_means,
                eligible=game["normalized_lower"]>=mean_floor and min(seed_means)>=seed_floor))
    unresolved=[t for t in panels.TASKS if not any(r["eligible"] for r in controls if r["environment"]==t)]
    return dict(status="eligible-development-search" if not unresolved else "blocked-learning-calibration",
        unresolved_games=unresolved,controls=controls,learning_floor=mean_floor,seed_floor=seed_floor,
        task_budgets=plan["task_budgets"],learner_recipes=plan["learner_recipes"],
        selection="fixed final budgets; no checkpoint/seed/architecture winner selection",publication_claim_qualified=False)


def search_recipe(root,gate):
    cfg=panels.panel.read(root/"search-space.ini"); study_cfg,seeds,_=study(root/"study.ini"); c=study_cfg["campaign"]
    for s in list(cfg.sections()):
        if s.startswith("task."): cfg.remove_section(s)
    cfg["search"].update(max_runs=c["max_runs"],training_seeds=",".join(map(str,seeds["search_seeds"])),
        appearance_seed=c["appearance_seed"],eval_seed=c["search_eval_seed"],episodes=c["episodes"],slots=c["slots"],
        pong_max_decisions=c["pong_max_decisions"],breakout_max_frames=c["breakout_max_frames"],
        campaign_timeout=c["search_timeout"],train_timeout=c["train_timeout"],eval_timeout=c["eval_timeout"],proposal_timeout=c["proposal_timeout"])
    cfg["protein"].update(seed=c["protein_seed"],num_random_samples=c["random_warmup"])
    for task,b in gate["task_budgets"].items(): cfg["task."+task]={k:str(v) for k,v in b.items()}
    destination=root/"execution/search.ini"
    with destination.open("x") as stream: cfg.write(stream)
    feedback.recipe(destination)
    return destination


def combined_view(root,value):
    """Reaudit all panels; show complete paired curves without pooled game scores."""
    search_root=root/"execution/search"
    outer=json.loads((search_root/"execution/result.json").read_text())
    plans=[]; observations=[]; training=[]; hardware=None
    for entry in outer["trials"]:
        panel_root=search_root/"execution"/f"trial-{entry['index']:04d}"/"panel"
        plans.append(panels.inspect(panel_root/"plan.json"))
        with redirect_stdout(io.StringIO()): analysis=panels.audit(panel_root/"plan.json",root/"review"/f"curves-trial-{entry['index']:04d}")
        require(hardware is None or hardware==analysis["hardware"],"Mixed hardware in comparison")
        hardware=analysis["hardware"]
        observations+=analysis["observations"]; training+=analysis["training"]
    plans.append(panels.inspect(root/"references/plan.json"))
    with redirect_stdout(io.StringIO()): analysis=panels.audit(root/"references/plan.json",root/"review/curves-references")
    require(hardware==analysis["hardware"],"Reference hardware differs from search")
    observations+=analysis["observations"]; training+=analysis["training"]
    merged={**plans[0],"candidates":sum((p["candidates"] for p in plans),[]),"jobs":sum((p["jobs"] for p in plans),[]),
            "planned_evaluations":sum(p["planned_evaluations"] for p in plans)}
    require(all(p["seeds"]==merged["seeds"] and p["task_budgets"]==merged["task_budgets"] for p in plans),"Comparison budgets/seeds differ")
    curves=viewer.assemble(merged,dict(status="ok",mode="development",observations=observations,training=training,hardware=hardware))
    curves["publication_claim_qualified"]=False
    curves["selection_warning"]="Adaptive discovery seeds, not held-out confirmation; frontier marks are descriptive only."
    folder=root/"review/combined"; folder.mkdir()
    reporting.write_viewer(folder/"curves.html",curves)
    reporting.write_csv(folder/"training.csv",training,list(training[0]))
    panels.write_frontiers(folder,curves)
    return dict(curves=str(folder/"curves.html"),observations=len(observations),training_jobs=len(training))


def run(args):
    root=args.out.resolve(); require(args.allow_gpu,"Explicit scheduled GPU execution required")
    value=inspect(root,current=True); cfg,_,_=study(root/"study.ini"); c=cfg["campaign"]
    execution=root/"execution"; execution.mkdir(exist_ok=False)
    record=dict(status="running",campaign_sha256=sha(root/"campaign.json"),stages=[],
        started_monotonic_ns=time.monotonic_ns(),publication_claim_qualified=False)
    try:
        for model in ("quality","nature"):
            graph_checks.run(Path(value["validation"]),root/(model+".ini"),execution/(model+"-encoder-check"),"5090",120)
        with redirect_stdout(io.StringIO()):
            panels.run(SimpleNamespace(plan=root/"calibration/plan.json",allow_gpu=True,mode="development",
                timeout=c.getint("calibration_timeout"),train_timeout=c.getint("train_timeout"),eval_timeout=c.getint("eval_timeout")))
            analysis=panels.audit(root/"calibration/plan.json",root/"calibration-review")
        plan=panels.inspect(root/"calibration/plan.json")
        gate=eligibility(plan,analysis,panels.panel.read(root/"search-space.ini"),c.getfloat("learning_floor"),c.getfloat("seed_floor"))
        gate.update(calibration_result_sha256=analysis["result_sha256"],calibration_plan_sha256=sha(root/"calibration/plan.json"))
        save(execution/"calibration-gate.json",gate); record["stages"].append("calibration-audited")
        require(gate["status"]=="eligible-development-search","Learning calibration failed for: "+", ".join(gate["unresolved_games"]))
        recipe=search_recipe(root,gate)
        with redirect_stdout(io.StringIO()):
            feedback.prepare(SimpleNamespace(recipe=recipe,registry=Path(value["registry"]),optimizer_build=Path(value["optimizer"]),
                policy_metadata=Path(value["metadata"]),encoder_validation=Path(value["validation"]),
                learner_recipe=[t+"="+str(root/p) for t,p in value["learners"].items()],out=execution/"search"))
            feedback.run(SimpleNamespace(plan=execution/"search/plan.json",allow_gpu=True,mode="development"))
            feedback.audit(SimpleNamespace(plan=execution/"search/plan.json",out=root/"search-review"))
        record["stages"].append("native-search-audited")
        with redirect_stdout(io.StringIO()):
            panels.run(SimpleNamespace(plan=root/"references/plan.json",allow_gpu=True,mode="development",
                timeout=c.getint("reference_timeout"),train_timeout=c.getint("train_timeout"),eval_timeout=c.getint("eval_timeout")))
            panels.audit(root/"references/plan.json",root/"reference-review")
        record["stages"].append("paired-references-audited")
        (root/"review").mkdir(); record["curves"]=combined_view(root,value)
        record["status"]="ok"
        return record
    except BaseException as error:
        record.update(status="failed",error=f"{type(error).__name__}: {error}"); raise
    finally:
        record["ended_monotonic_ns"]=time.monotonic_ns()
        record["process_seconds"]=(record["ended_monotonic_ns"]-record["started_monotonic_ns"])/1e9
        save(execution/"result.json",record)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    for name in ("study","registry","metadata","optimizer","validation"): prep.add_argument("--"+name,type=Path,required=True)
    sub.add_parser("inspect")
    execute=sub.add_parser("run"); execute.add_argument("--allow-gpu",action="store_true")
    for parser in sub.choices.values(): parser.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    value=prepare(args) if args.command=="prepare" else inspect(args.out) if args.command=="inspect" else run(args)
    print(json.dumps(value,indent=2))


if __name__=="__main__": main()
