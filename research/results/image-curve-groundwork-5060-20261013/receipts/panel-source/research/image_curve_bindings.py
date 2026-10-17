"""Bind measured Nature curves and immutable development/confirmation roles.

Preparation replays the native artifact audit; no policy is executed here.
The Nature panel must remain available for the final independent campaign audit.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil

import candidate_panel as panels
import image_curve_feedback as curves
import image_seed_roles as seeds
from claim import require, save, sha, common_settings


def evaluation(recipe):
    training = [int(v) for v in recipe.get("search", "training_seeds").split(",")]
    explicit = [int(v) for v in recipe.get("search", "evaluation_seeds").split(",")]
    return panels.evaluation_seeds(training, recipe.getint("search", "eval_seed"), explicit)


def anchors(recipe):
    return {t: {k: recipe.getfloat("objective."+t, k) for k in ("offset", "target")}
            for t in panels.TASKS}


def check_roles(recipe, roles):
    training = [int(v) for v in recipe.get("search", "training_seeds").split(",")]
    seeds.check_search(roles, recipe.get("search", "seed_role"), training,
        evaluation(recipe), recipe.getint("search", "appearance_seed"), recipe.getint("protein", "seed"))


def prepare(args, root, recipe):
    require(all(getattr(args, k, None) is not None for k in
                ("nature_reference", "nature_panel", "nature_audit", "seed_roles")),
            "Curve feedback needs audited Nature panel/reference and fixed seed roles")
    roles = json.loads(args.seed_roles.read_text()); check_roles(recipe, roles)
    panel_path = args.nature_panel.resolve()
    plan = panels.inspect(panel_path)
    require(plan["protocol"] == panels.IMAGE_CURVE_PROTOCOL and plan.get("seed_roles", {}).get("role") == "development",
            "Nature curve must use measured initial controls and frozen development seeds")
    with redirect_stdout(io.StringIO()):
        audit = panels.audit(panel_path, root / "nature-replay")
    original = json.loads(args.nature_audit.read_text())
    require(all(audit[k] == original[k] for k in ("status", "plan_sha256", "result_sha256", "audited_jobs",
            "audited_evaluations", "missing_evaluations", "training", "observations", "hardware")),
            "Nature calibration differs from raw receipt replay")
    shutil.copyfile(panel_path,root/"nature-calibration-plan.json")
    import image_learning_gate as learning
    gate=learning.qualify(plan,audit,anchors(recipe))
    save(root/"learning-eligibility.json",gate,exclusive=True)
    require(gate["status"]=="eligible-development-search",
            "Unlearned full-image games cannot start architecture search: "+", ".join(gate["unresolved_games"]))
    expected = curves.freeze(plan, audit, panel_path.parent, anchors(recipe), root / "nature-reference.json",
                             analysis_sha256=sha(args.nature_audit))
    require(expected == json.loads(args.nature_reference.read_text()), "Nature reference differs from measured calibration")
    require(roles == json.loads((panel_path.parent / plan["seed_roles"]["file"]).read_text()),
            "Nature and discovery have different seed-role contracts")
    shutil.copyfile(args.seed_roles, root / "seed-roles.json")
    declaration = dict(nature_panel=str(panel_path), nature_plan_sha256=sha(panel_path),
        nature_audit=str(args.nature_audit.resolve()), nature_audit_sha256=sha(args.nature_audit),
        nature_reference_sha256=sha(root / "nature-reference.json"), seed_roles_sha256=sha(root / "seed-roles.json"),
        replay_sha256=sha(root / "nature-replay/analysis.json"),learning_eligibility_sha256=sha(root/"learning-eligibility.json"),
        calibration_plan_sha256=sha(root/"nature-calibration-plan.json"))
    save(root / "curve-bindings.json", declaration, exclusive=True)
    inspect(root, recipe)
    return declaration


def inspect(root, recipe):
    value = json.loads((root / "curve-bindings.json").read_text())
    require(sha(root / "nature-reference.json") == value["nature_reference_sha256"]
            and sha(root / "seed-roles.json") == value["seed_roles_sha256"]
            and sha(root / "nature-replay/analysis.json") == value["replay_sha256"], "Changed frozen curve/role/replay inputs")
    require(sha(root/"learning-eligibility.json")==value["learning_eligibility_sha256"],"Changed full-image learning eligibility")
    gate=json.loads((root/"learning-eligibility.json").read_text())
    require(gate["status"]=="eligible-development-search" and not gate["unresolved_games"],
            "Unlearned calibration cannot qualify a runtime pilot or overnight search")
    require(sha(root/"nature-calibration-plan.json")==value["calibration_plan_sha256"]==value["nature_plan_sha256"],
            "Changed captured learning calibration plan")
    import image_learning_gate as learning
    calibration_plan=json.loads((root/"nature-calibration-plan.json").read_text())
    calibration_audit=json.loads((root/"nature-replay/analysis.json").read_text())
    require(calibration_audit["plan_sha256"]==value["nature_plan_sha256"]
            and learning.qualify(calibration_plan,calibration_audit,anchors(recipe))==gate,
            "Learning eligibility differs from captured native calibration")
    reference = json.loads((root / "nature-reference.json").read_text())
    roles = json.loads((root / "seed-roles.json").read_text())
    curves.validate_reference(reference); check_roles(recipe, roles)
    require(reference["normalization"] == anchors(recipe), "Feedback changes calibrated normalization anchors")
    require(reference["conditions"]["seeds"] == roles["development_training"]
            and reference["conditions"]["evaluation_seeds"] == roles["development_evaluation"]
            and reference["conditions"]["appearance_seed"] == roles["appearance_seed"],
            "Nature curves differ from fixed development seed assignments")
    return reference


def feedback_conditions(root, value, recipe, bases):
    """Reject mismatched pixels/learners/budgets before any proposal CUDA work."""
    reference = inspect(root, recipe); fixed = reference["conditions"]
    require(fixed["image_contracts"] == value["image_contracts"]
            and fixed["task_budgets"] == value["task_budgets"]
            and fixed["episodes"] == recipe.getint("search", "episodes")
            and fixed["slots"] == recipe.getint("search", "slots")
            and fixed["caps"] == {k:recipe.getint("search", k) for k in ("pong_max_decisions", "breakout_max_frames")},
            "Discovery changes Nature pixels/budgets/evaluation assignments")
    require(fixed["game_settings"] == {t:common_settings(c) for t,c in bases.items()}
            and fixed["learner_sha256"] == {r["environment"]:sha(root/r["file"]) for r in value["learner_recipes"]},
            "Discovery changes calibrated per-game learners/world/core")
    native = {n:h for n,h in value["source_sha256"].items()
              if n.startswith(("src/", "config/")) or n.startswith("ocean/") and Path(n).suffix in (".h", ".cu", ".c")}
    require(fixed["native_sources"] == native, "Discovery changes calibrated native source contract")


def audit_reference(root, recipe, out):
    reference = inspect(root, recipe)
    binding = json.loads((root / "curve-bindings.json").read_text())
    path = Path(binding["nature_panel"])
    require(sha(path) == binding["nature_plan_sha256"]
            and sha(binding["nature_audit"]) == binding["nature_audit_sha256"], "Changed retained Nature panel/audit")
    plan = panels.inspect(path)
    with redirect_stdout(io.StringIO()): analysis = panels.audit(path, out / "panel")
    import image_learning_gate as learning
    require(learning.qualify(plan,analysis,anchors(recipe))==json.loads((root/"learning-eligibility.json").read_text()),
            "Final native audit changes full-image learning eligibility")
    expected = curves.freeze(plan, analysis, path.parent, anchors(recipe), out / "nature-reference.json",
                             analysis_sha256=binding["nature_audit_sha256"])
    require(expected == reference, "Final campaign audit changes measured Nature curve")
    return reference
