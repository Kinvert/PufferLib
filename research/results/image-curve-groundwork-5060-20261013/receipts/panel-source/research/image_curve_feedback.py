#!/usr/bin/env python3
"""Scalar Nature-time curve feedback from complete audited native panels.

No training, optimizer, policy, GPU query or CPU neural computation happens
here. The calibration panel/individual episode audit remain separate gates.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path

import candidate_panel as panels
from claim import common_settings, require, save, sha

SCHEMA="fixed-nature-image-curve-v1"
OBJECTIVE="balanced-eight-nature-time-anchors-v1"
EPSILON=0.05
CHECKPOINTS=8


def balanced(values):
    require(len(values)==len(panels.TASKS) and all(math.isfinite(v) and 0<=v<=1 for v in values),
            "Need six finite normalized game scores")
    return max(0.0,min(1.0,math.exp(math.fsum(math.log(EPSILON+v) for v in values)/len(values))-EPSILON))


def conditions(plan, root):
    require(panels.image_protocol(plan["protocol"]) and plan.get("per_game_learners") is True,
            "Curve feedback requires an explicit image/per-game learner panel")
    require(set(plan["image_contracts"])==set(panels.TASKS) and set(plan["learner_recipes"])==set(panels.TASKS),
            "Incomplete image/learner conditions")
    learners={}; games={}
    for job in plan["jobs"]:
        task=job["environment"]; require(task in panels.TASKS,"Unexpected curve game")
        config=panels.panel.read(root/job["config"])
        fixed=common_settings(config)
        require(task not in games or games[task]==fixed,"Learner/world/core differs inside a game")
        games[task]=fixed
    for task,record in plan["learner_recipes"].items(): learners[task]=record["sha256"]
    result = dict(image_contracts=plan["image_contracts"],seeds=plan["seeds"],task_budgets=plan["task_budgets"],
        learner_sha256=learners,game_settings=games,appearance_seed=plan["appearance_seed"],
        eval_seed=plan["eval_seed"],episodes=plan["episodes"],slots=plan["slots"],caps=plan["caps"],
        native_sources={n:h for n,h in plan["source_sha256"].items()
                       if n.startswith(("src/","config/")) or n.startswith("ocean/") and Path(n).suffix in (".h",".cu",".c")})
    if "evaluation_seeds" in plan: result["evaluation_seeds"] = plan["evaluation_seeds"]
    if "seed_roles" in plan: result["seed_roles"] = plan["seed_roles"]
    if "initial_checkpoint" in plan: result["initial_checkpoint"] = plan["initial_checkpoint"]
    return result


def panel_rows(plan, analysis, name=None):
    require(analysis["status"]=="ok" and not analysis["missing_evaluations"]
            and analysis["audited_jobs"]==len(plan["jobs"])
            and analysis["audited_evaluations"]==plan["planned_evaluations"]==len(analysis["observations"]),
            "Incomplete panel cannot supply curve feedback")
    names={c["name"] for c in plan["candidates"]}
    if name is None:
        require(len(names)==1,"One architecture per feedback observation")
        name=next(iter(names))
    require(name in names,"Unknown curve architecture")
    jobs=[j for j in plan["jobs"] if j["candidate"]==name]
    require(len(jobs)==len(panels.TASKS)*len(plan["seeds"]),"Incomplete game/replica jobs")
    expected={}; schedules={}; costs={}
    for job in jobs:
        task,seed=job["environment"],job["seed"]; key=(task,seed)
        require(task in panels.TASKS and seed in plan["seeds"] and key not in schedules,"Duplicate/unassigned curve job")
        assigned=job["checkpoint_steps"]
        initial = plan["protocol"] == panels.IMAGE_CURVE_PROTOCOL
        if initial:
            require(assigned[0] == 0 and plan["initial_checkpoint"] == "same-policy-before-first-update-v1",
                    "Curve panel lacks its own measured initial checkpoint")
        steps=assigned[1:] if initial else assigned
        require(len(steps)==CHECKPOINTS and steps==sorted(set(steps)) and all(type(s) is int and s>0 for s in steps)
                and steps[-1]==plan["task_budgets"][task]["steps"],"Exactly eight chronological checkpoint stages required")
        schedules[key]=steps
        for step in assigned:
            for drawing in range(panels.panel.ENVIRONMENTS[task]): expected[(task,seed,step,drawing)]=None
    rows=[r for r in analysis["observations"] if r["candidate"]==name]; indexed={}
    for row in rows:
        key=tuple(row[k] for k in ("environment","seed","decisions","representation"))
        require(key in expected and key not in indexed,"Duplicate/extra/unassigned curve cell")
        require(all(math.isfinite(row[k]) for k in ("train_seconds","score_lower","score_upper"))
                and row["train_seconds"]>0 and row["score_lower"]<=row["score_upper"],"Invalid curve score/time bounds")
        indexed[key]=row
        costkey=key[:3]
        require(costkey not in costs or costs[costkey]==row["train_seconds"],"Drawing changes/duplicates checkpoint training cost")
        costs[costkey]=row["train_seconds"]
    require(set(indexed)==set(expected),"Missing curve cells")
    for (task,seed),steps in schedules.items():
        times=[costs[(task,seed,s)] for s in ([0]+steps if plan["protocol"] == panels.IMAGE_CURVE_PROTOCOL else steps)]
        require(all(a<b for a,b in zip(times,times[1:])),"Checkpoint clocks are not strictly chronological")
    return name,indexed,schedules,costs


def freeze(plan, analysis, root, anchors, output, analysis_sha256=None):
    """Freeze actual Nature curves after the caller's native artifact audit."""
    name,rows,schedules,costs=panel_rows(plan,analysis,"nature-cnn")
    nature=next(c for c in plan["candidates"] if c["name"]==name)
    require(nature.get("kind")=="baseline" and nature.get("baseline")=="nature_cnn"
            and nature["architecture"]=={"encoder":2},"Anchor policy is not the declared native Nature reference")
    require(set(anchors)==set(panels.TASKS),"Need fixed normalization anchors for every game")
    for task,pair in anchors.items():
        require(set(pair)=={"offset","target"} and all(math.isfinite(v) for v in pair.values())
                and pair["target"]>pair["offset"],"Invalid normalization anchors")
    games={}
    for task in panels.TASKS:
        games[task]={str(seed):[dict(decisions=step,seconds=costs[(task,seed,step)],
            drawings=[dict(representation=d,score_lower=rows[(task,seed,step,d)]["score_lower"],
                           score_upper=rows[(task,seed,step,d)]["score_upper"])
                      for d in range(panels.panel.ENVIRONMENTS[task])])
            for step in schedules[(task,seed)]] for seed in plan["seeds"]}
    value=dict(schema=SCHEMA,objective=OBJECTIVE,epsilon=EPSILON,checkpoints=CHECKPOINTS,
               conditions=conditions(plan,root),normalization=anchors,nature=games,
               audited_result_sha256=analysis["result_sha256"],source_analysis_sha256=analysis_sha256,
               publication_claim_qualified=False,hardware=analysis["hardware"])
    save(output,value,exclusive=True); return value


def validate_reference(reference):
    require(reference["schema"]==SCHEMA and reference["objective"]==OBJECTIVE
            and reference["epsilon"]==EPSILON and reference["checkpoints"]==CHECKPOINTS
            and reference["publication_claim_qualified"] is False,"Changed curve objective contract")
    require(set(reference["nature"])==set(reference["normalization"])==set(panels.TASKS),"Incomplete Nature reference")
    seeds=reference["conditions"]["seeds"]
    for task in panels.TASKS:
        offset,target=(reference["normalization"][task][k] for k in ("offset","target"))
        require(math.isfinite(offset) and math.isfinite(target) and target>offset,"Invalid reference score anchors")
        require(set(reference["nature"][task])=={str(s) for s in seeds},"Wrong reference replicas")
        for seed in seeds:
            points=reference["nature"][task][str(seed)]
            require(len(points)==CHECKPOINTS,"Wrong Nature curve resolution")
            budget=reference["conditions"]["task_budgets"][task]
            require([p["decisions"] for p in points]==list(range(budget["checkpoint_steps"],budget["steps"]+1,budget["checkpoint_steps"])),
                    "Nature anchor schedule differs from frozen decision budget")
            require(all(type(p["decisions"]) is int and p["decisions"]>0 and math.isfinite(p["seconds"])
                        and p["seconds"]>0 for p in points),"Invalid Nature checkpoint")
            require(all(a["seconds"]<b["seconds"] and a["decisions"]<b["decisions"] for a,b in zip(points,points[1:])),
                    "Nonchronological Nature anchors")
            for point in points:
                require([d["representation"] for d in point["drawings"]]==list(range(panels.panel.ENVIRONMENTS[task])),
                        "Missing/duplicate reference drawing")
                require(all(math.isfinite(d["score_lower"]) and math.isfinite(d["score_upper"])
                            and d["score_lower"]<=d["score_upper"] for d in point["drawings"]),"Invalid Nature bounds")


def aggregate(plan, analysis, root, reference, initial=None):
    validate_reference(reference)
    require(conditions(plan,root)==reference["conditions"] and analysis["hardware"]==reference["hardware"],
            "Candidate/Nature pixels, learner, seeds, world, caps or hardware differ")
    name,rows,schedules,costs=panel_rows(plan,analysis)
    if plan["protocol"] == panels.IMAGE_CURVE_PROTOCOL:
        require(initial is None, "Image v5 initial controls must come from this complete audited panel")
        initial = {(task, seed): [rows[(task, seed, 0, d)] for d in range(panels.panel.ENVIRONMENTS[task])]
                   for task, seed in schedules}
    initial=initial or {}; points=[]
    def normalize(task,value):
        offset,target=(reference["normalization"][task][k] for k in ("offset","target"))
        return max(0.0,min(1.0,(value-offset)/(target-offset)))
    for index in range(CHECKPOINTS):
        games=[]
        for task in panels.TASKS:
            lower=[]; upper=[]; nature_lower=[]; selections=[]
            for seed in plan["seeds"]:
                anchor=reference["nature"][task][str(seed)][index]
                affordable=[s for s in schedules[(task,seed)] if costs[(task,seed,s)]<=anchor["seconds"]]
                if affordable:
                    step=affordable[-1]
                    selected=[rows[(task,seed,step,d)] for d in range(panels.panel.ENVIRONMENTS[task])]
                else:
                    # A measured random-initial policy is architecture-specific;
                    # never substitute Nature's initial return or an invented0.
                    selected=initial.get((task,seed),[]); step=0
                    require(len(selected)==panels.panel.ENVIRONMENTS[task]
                            and [r["representation"] for r in selected]==list(range(panels.panel.ENVIRONMENTS[task]))
                            and all(r["candidate"]==name and r["environment"]==task and r["seed"]==seed
                                    and r["decisions"]==0 and math.isfinite(r["train_seconds"])
                                    and 0<=r["train_seconds"]<=anchor["seconds"]
                                    and math.isfinite(r["score_lower"]) and math.isfinite(r["score_upper"])
                                    and r["score_lower"]<=r["score_upper"] for r in selected),
                            "No affordable checkpoint: require this policy's audited measured untrained control")
                lower.append(math.fsum(normalize(task,r["score_lower"]) for r in selected)/len(selected))
                upper.append(math.fsum(normalize(task,r["score_upper"]) for r in selected)/len(selected))
                nature_lower.append(math.fsum(normalize(task,r["score_lower"]) for r in anchor["drawings"])/len(anchor["drawings"]))
                selections.append(dict(seed=seed,nature_seconds=anchor["seconds"],selected_decisions=step))
            lo=math.fsum(lower)/len(lower); hi=math.fsum(upper)/len(upper); baseline=math.fsum(nature_lower)/len(nature_lower)
            games.append(dict(environment=task,normalized_lower=lo,normalized_upper=hi,nature_lower=baseline,
                              delta_lower=lo-baseline,selections=selections))
        points.append(dict(anchor_index=index,games=games,
            balanced_lower=balanced([g["normalized_lower"] for g in games]),
            balanced_upper=balanced([g["normalized_upper"] for g in games]),
            arithmetic_lower=math.fsum(g["normalized_lower"] for g in games)/len(games),
            coverage={label:sum((g["delta_lower"]>0 if label=="wins" else g["delta_lower"]<0 if label=="losses" else g["delta_lower"]==0)
                                for g in games) for label in ("wins","ties","losses")}))
    return dict(objective=OBJECTIVE,score=math.fsum(p["balanced_lower"] for p in points)/CHECKPOINTS,
        score_upper=math.fsum(p["balanced_upper"] for p in points)/CHECKPOINTS,
        cost=math.fsum(costs[(task,seed,steps[-1])] for (task,seed),steps in schedules.items()),
        training_jobs_charged=len(schedules),points=points,conservative_pong=True,
        publication_claim_qualified=False,initial_policy_required_when_no_affordable_checkpoint=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--anchors",type=Path,required=True)
    for command in (f,sub.add_parser("aggregate")):
        command.add_argument("--plan",type=Path,required=True); command.add_argument("--audit",type=Path,required=True)
        command.add_argument("--out",type=Path,required=True)
        command.add_argument("--reaudit-out",type=Path,required=True,
                             help="Fresh directory for mandatory raw native artifact replay; never a GPU execution")
    sub.choices["aggregate"].add_argument("--reference",type=Path,required=True)
    args=parser.parse_args(); plan=panels.inspect(args.plan.resolve())
    analysis=json.loads(args.audit.read_text())
    require(analysis["plan_sha256"]==sha(args.plan),"Analysis belongs to a different panel")
    with redirect_stdout(io.StringIO()): replay=panels.audit(args.plan,args.reaudit_out)
    require(all(replay[k]==analysis[k] for k in ("status","mode","plan_sha256","result_sha256","audited_jobs",
                "audited_evaluations","missing_evaluations","training","observations","hardware")),
            "Curve input differs from raw native receipt replay")
    root=args.plan.resolve().parent
    if args.command=="freeze":
        value=freeze(plan,analysis,root,json.loads(args.anchors.read_text()),args.out,sha(args.audit))
    else:
        value=aggregate(plan,analysis,root,json.loads(args.reference.read_text())); save(args.out,value,exclusive=True)
    print(json.dumps(dict(objective=value.get("objective"),publication_claim_qualified=False)))


if __name__=="__main__": main()
