#!/usr/bin/env python3
"""Prepare matched quality/Nature image controls, without GPU execution.

Budgets and learners are explicit development inputs, not estimates from old
tiny-image SPS. Successful preparation is not successful learning calibration.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import candidate_panel as panels
import cross_game_feedback as feedback
import image_curve_bindings as bindings
import image_encoder_checks as checks
import image_seed_roles as seed_roles
from claim import require, save, sha


def panel_arguments(args, output, recipe, roles, manifest):
    require(feedback.recipe_protocol(recipe) == feedback.CURVE_PROTOCOL,
            "Use an explicit realistic-image curve recipe; old plumbing recipes are not learning budgets")
    bindings.check_roles(recipe, roles)
    require(len(roles["development_training"]) >= 3, "Learning controls need all fixed development replicas")
    learners = {}
    for raw in args.learner_recipe:
        task, separator, filename = raw.partition("=")
        require(separator and task in panels.TASKS and task not in learners,
                "Invalid or duplicate per-game learner")
        learners[task] = Path(filename).resolve()
    require(set(learners) == set(panels.TASKS), "Every game needs an explicit frozen learner")
    budgets = feedback.task_budgets(recipe)
    for budget in budgets.values(): panels.checkpoint_schedule(budget, True)
    panels.architecture(args.quality_config)
    return SimpleNamespace(out=output/"panel", registry=args.registry,
        candidate=["quality-reference="+str(args.quality_config.resolve())], native_campaign=[],
        baselines=["nature_cnn"], policy_metadata=args.policy_metadata,
        image_math=manifest, seeds=roles["development_training"],
        evaluation_seeds=roles["development_evaluation"], seed_roles=args.seed_roles,
        seed_role="development", include_initial=True,
        appearance_seed=roles["appearance_seed"], eval_seed=roles["development_evaluation"][0],
        per_game_learners=True, learner_recipe=[task+"="+str(learners[task]) for task in panels.TASKS],
        steps=recipe.getint("search","steps"), checkpoint_steps=recipe.getint("search","checkpoint_steps"),
        task_budget=[f"{task}={budget['steps']}:{budget['checkpoint_steps']}" for task,budget in budgets.items()],
        episodes=recipe.getint("search","episodes"), slots=recipe.getint("search","slots"),
        pong_max_decisions=recipe.getint("search","pong_max_decisions"),
        breakout_max_frames=recipe.getint("search","breakout_max_frames"))


def math_manifest(quality, nature, output):
    merged = {}
    for name, path in (("quality-reference",quality),("nature-cnn",nature)):
        value = json.loads(path.read_text())
        require(value["schema"] == checks.SCHEMA and set(value["candidates"]) == {name}
                and set(value["candidates"][name]) == set(panels.TASKS),
                "Need separately audited all-game numerical evidence for each declared control")
        merged[name] = {task:os.path.relpath((path.parent/entry).resolve(),output.parent)
                        for task,entry in value["candidates"][name].items()}
    save(output,dict(schema=checks.SCHEMA,candidates=merged),exclusive=True)


def prepare(args):
    recipe,_ = feedback.recipe(args.recipe)
    roles = json.loads(args.seed_roles.read_text()); seed_roles.validate(roles)
    out = args.out.resolve()
    # Validate fixed assignments before creating an allocation or spawning host tools.
    panel_arguments(args,out,recipe,roles,out/"math.json")
    out.mkdir(parents=True,exist_ok=False)
    try:
        shutil.copyfile(args.recipe,out/"recipe.ini")
        math_manifest(args.quality_math,args.nature_math,out/"math.json")
        arguments = panel_arguments(args,out,recipe,roles,out/"math.json")
        plan = panels.prepare(arguments)
        require([c["name"] for c in plan["candidates"]] == ["quality-reference","nature-cnn"]
                and plan["protocol"] == panels.IMAGE_CURVE_PROTOCOL,
                "Calibration preparation changed control identities or checkpoint contract")
        allocation = dict(schema="realistic-image-calibration-allocation-v1",status="prepared-not-executed",
            panel_plan_sha256=sha(out/"panel/plan.json"), recipe_sha256=sha(out/"recipe.ini"),
            seed_roles_sha256=sha(args.seed_roles), normalization=bindings.anchors(recipe),
            training_jobs=plan["planned_training_jobs"], evaluations=plan["planned_evaluations"],
            assigned_episode_executions=plan["planned_evaluations"]*plan["episodes"],
            declared_decisions=sum(plan["task_budgets"][j["environment"]]["steps"] for j in plan["jobs"]),
            quality_architecture=plan["candidates"][0]["architecture"],
            learning_calibrated=False,gpu_queried=False,policy_executed=False,publication_claim_qualified=False)
        save(out/"ALLOCATION.json",allocation,exclusive=True)
        return allocation
    except BaseException as error:
        save(out/"failure.json",dict(status="failed-preparation",error=f"{type(error).__name__}: {error}",
            gpu_queried=False,policy_executed=False),exclusive=True)
        raise


def qualify(allocation, out):
    import image_learning_gate as learning
    import image_curve_feedback as curves
    import report_image_feedback as reporting
    allocation, out = allocation.resolve(), out.resolve()
    value = json.loads((allocation/"ALLOCATION.json").read_text())
    require(value["schema"] == "realistic-image-calibration-allocation-v1"
            and sha(allocation/"panel/plan.json") == value["panel_plan_sha256"]
            and sha(allocation/"recipe.ini") == value["recipe_sha256"], "Changed calibration allocation")
    out.mkdir(parents=True,exist_ok=False)
    try:
        reporting.report(allocation/"panel/plan.json",out/"report",panel_only=True)
        plan = panels.inspect(allocation/"panel/plan.json")
        audit_path = out/"report/review/analysis.json"
        analysis = json.loads(audit_path.read_text())
        recipe,_ = feedback.recipe(allocation/"recipe.ini")
        require(bindings.anchors(recipe) == value["normalization"], "Changed calibration score normalization")
        require(plan["task_budgets"] == feedback.task_budgets(recipe)
                and plan["episodes"] == recipe.getint("search","episodes")
                and plan["slots"] == recipe.getint("search","slots")
                and plan["caps"] == {key:recipe.getint("search",key)
                    for key in ("pong_max_decisions","breakout_max_frames")},
                "Calibration panel differs from declared budgets/evaluation regime")
        roles=json.loads((allocation/"panel"/plan["seed_roles"]["file"]).read_text())
        require(sha(allocation/"panel"/plan["seed_roles"]["file"]) == value["seed_roles_sha256"],
                "Calibration changed its declared seed-role document")
        bindings.check_roles(recipe,roles)
        gate = learning.qualify(plan,analysis,value["normalization"])
        save(out/"learning-eligibility.json",gate,exclusive=True)
        # Keep every completed control and the unresolved game report on failure.
        # Unlearned controls never become reference anchors for a runtime pilot.
        if gate["status"] == "eligible-development-search":
            curves.freeze(plan,analysis,allocation/"panel",value["normalization"],
                out/"nature-reference.json",analysis_sha256=sha(audit_path))
        return gate
    except BaseException as error:
        save(out/"failure.json",dict(status="failed-qualification",error=f"{type(error).__name__}: {error}",
            gpu_queried=False,policy_executed=False),exclusive=True)
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prep=commands.add_parser("prepare",help="Native host registration/artifact preparation only")
    for key in ("recipe","registry","policy-metadata","quality-config","quality-math","nature-math","out"):
        prep.add_argument("--"+key,type=Path,required=True)
    prep.add_argument("--seed-roles",type=Path,default=panels.ROOT/"research/recipes/image_seed_roles.json")
    prep.add_argument("--learner-recipe",action="append",required=True,help="ENV=INI; all six games, fixed across both controls")
    review=commands.add_parser("qualify",help="Offline receipt replay and learning gate; no policy execution")
    review.add_argument("--allocation",type=Path,required=True); review.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args) if args.command=="prepare" else qualify(args.allocation,args.out)
    print(json.dumps(result,indent=2))
    if result["status"] == "blocked-learning-calibration": raise SystemExit(2)
