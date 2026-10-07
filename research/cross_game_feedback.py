#!/usr/bin/env python3
"""Temporary multi-game trial orchestration; PROTEIN/search/training stay native.

Native proposal processes replay the fixed observation ledger and exit before
any policy runs. Python only prepares existing panels, audits exact outcomes,
reduces scalar objectives and supervises processes. This is not PR delivery code.
"""
import argparse
import contextlib
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import candidate_panel as panels
from claim import common_settings

ROOT = panels.ROOT
PROTOCOL = "cross-game-native-protein-research-v1"
require, save, sha = panels.require, panels.save, panels.sha


def recipe(path):
    config = panels.read_ini(path)
    required = {"search", "protein", "policy"} | {"objective." + task for task in panels.TASKS}
    require(required <= set(config.sections()), "Incomplete research recipe/objectives")
    require(config.getint("policy", "encoder") == 4 and config.getint("policy", "hidden_size") == 128
            and config.getint("policy", "num_layers") == 1, "Research requires encoder4/H128/L1")
    # Reuse the existing grammar/constructor contract; no learner/core/world search.
    panels.architecture(path)
    keys = [s for s in config.sections() if s.startswith("sweep.")]
    require(keys and all(s.startswith("sweep.policy.cnn_") for s in keys), "Only CNN coordinates may vary")
    require(set(config.sections()) == required | set(keys), "Unsupported recipe section; use fixed learner overlays")
    for section in keys:
        coordinate = section[6:]
        require(coordinate in panels.LIMITS[4], "Unsupported CNN coordinate")
        distribution, values = panels.LIMITS[4][coordinate]
        require(config.get(section, "distribution") == distribution, "Wrong coordinate distribution")
        lo, hi = config.getint(section, "min"), config.getint(section, "max")
        require(lo in values and hi in values and lo < hi, "Bad CNN range")
        require(config.has_option("policy", coordinate[7:]), "Missing CNN coordinate default")
        require(lo <= config.getint("policy", coordinate[7:]) <= hi, "Default outside range")
        if config.getint("policy", "cnn_depth") == 1 and coordinate[-1:].isdigit():
            require(coordinate.endswith("_1"), "Inactive stage must not consume search dimensions")
    seeds = [int(v.strip()) for v in config.get("search", "training_seeds").split(",")]
    require(seeds and len(seeds) == len(set(seeds)) and all(0 <= s < 2**32 for s in seeds), "Bad training seeds")
    for task in panels.TASKS:
        section = "objective." + task
        offset, target = config.getfloat(section, "offset"), config.getfloat(section, "target")
        require(math.isfinite(offset) and math.isfinite(target) and target > offset, "Invalid fixed objective anchors")
    require(1 <= config.getint("search", "max_runs") <= 512, "Invalid proposal count")
    for key in ("steps", "checkpoint_steps", "episodes", "slots", "pong_max_decisions",
                "breakout_max_frames", "campaign_timeout", "train_timeout", "eval_timeout", "proposal_timeout"):
        require(config.getint("search", key) > 0, "Invalid positive search setting: " + key)
    for key in ("appearance_seed", "eval_seed"):
        require(0 <= config.getint("search", key) < 2**32, "Invalid panel seed")
    require(config.getint("search", "eval_seed") <= 2**32-len(seeds), "Evaluation seed allocation overflows")
    require(not set(range(config.getint("search", "eval_seed"), config.getint("search", "eval_seed")+len(seeds))) & set(seeds), "Evaluation seed overlaps training")
    require(config.getint("search", "steps") % 2048 == 0
        and config.getint("search", "checkpoint_steps") % 2048 == 0, "Decisions/cadence must be 2048 multiples")
    require(0 <= config.getint("protein", "seed") < 2**32
        and 0 <= config.getint("protein", "num_random_samples") <= 100
        and 1 <= config.getint("protein", "gp_training_iter") <= 100, "Invalid optimizer controls")
    return config, seeds


def optimizer_receipts(directory, current=False):
    directory = directory.resolve()
    for ledger, base in (("source.sha256", directory / "source"), ("binary.sha256", directory)):
        for line in (directory / ledger).read_text().splitlines():
            digest, name = line.split(None, 1); name = name.lstrip(" *")
            path = Path(name)
            if ledger == "binary.sha256": path = directory / path.name
            else:
                require(not path.is_absolute() and ".." not in path.parts, "Unsafe build source receipt")
                path = base / path
                if current: require(sha(ROOT / name) == digest, "Stale optimizer build: " + name)
            require(sha(path) == digest, "Changed optimizer receipt: " + name)
    require("PUFFER_RESEARCH_PROTEIN_FEEDBACK" in (directory / "environment.txt").read_text(), "Wrong optimizer build")
    return {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*") if p.is_file()}


def inspect(root, current=False):
    value = json.loads((root / "plan.json").read_text())
    require(value["protocol"] == PROTOCOL, "Wrong research feedback protocol")
    require(not current or str(root.resolve()) == value["preparation_root"], "Cannot run relocated research packet")
    for name, digest in value["files_sha256"].items():
        require(sha(root / name) == digest, "Changed research input: " + name)
    panels.verify_sources(root, value["source_sha256"], current)
    if current:
        require(sha(Path(value["optimizer"])) == value["optimizer_sha256"], "Changed native optimizer")
        require(optimizer_receipts(Path(value["optimizer_build"]), current=True) == value["optimizer_receipts"], "Changed native build packet")
    require(optimizer_receipts(root / "optimizer-build") == value["optimizer_receipts"], "Changed frozen optimizer build")
    recipe(root / "recipe.ini")
    return value


def prepare(args):
    config, seeds = recipe(args.recipe)
    registry = json.loads(args.registry.read_text())
    require(registry["protocol"] == "native-cnn-candidate-build-v1", "Use fresh default-only game builds")
    panels.verify_sources(args.registry.parent, registry["source_sha256"], current=True)
    optimizer = args.optimizer_build.resolve() / "worker"
    require(optimizer.is_file() and (args.optimizer_build / "source-check.txt").is_file(), "Missing native optimizer receipts")
    receipts = optimizer_receipts(args.optimizer_build, current=True)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.recipe, out / "recipe.ini")
    shutil.copyfile(args.registry, out / "registry.json")
    shutil.copytree(args.optimizer_build, out / "optimizer-build")
    learners = []
    for raw in getattr(args, "learner_recipe", []):
        task, separator, filename = raw.partition("=")
        require(separator and task in panels.TASKS and task not in [r['environment'] for r in learners], "Invalid/duplicate per-game learner")
        destination = out / "learners" / (task + ".ini"); destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(filename, destination)
        learners.append(dict(environment=task, file=str(destination.relative_to(out))))
    save(out / "registry-location.json", dict(original=str(args.registry.resolve())), exclusive=True)
    sources = panels.capture(out)
    with contextlib.redirect_stdout(io.StringIO()):
        layout = panels.policy_layout.freeze(args.policy_metadata, out / "policy-metadata")
    panels.metadata_source_binding(layout, registry, sources)
    command = [str(optimizer), "research_protein_describe", str(out / "recipe.ini")]
    panels.process(command, out, out / "describe.txt", 30)
    description = json.loads((out / "describe.txt").read_text())
    require(description["gpu_queried"] is False and description["policy_executed"] is False, "Descriptor ran GPU/model")
    value = dict(protocol=PROTOCOL, preparation_root=str(out), optimizer=str(optimizer),
        optimizer_sha256=sha(optimizer), optimizer_build=str(args.optimizer_build.resolve()), optimizer_receipts=receipts,
        policy_metadata=str(args.policy_metadata.resolve()), seeds=seeds,
        learner_recipes=learners,
        dimensions=description["dimensions"], description=description,
        source_sha256=sources, files_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        normalized_objective="equal-game mean of clipped fixed-anchor exact final-checkpoint scores; Pong lower bound",
        optimizer_cost="sum final monotonic checkpoint training seconds once per game/seed",
        temporary_research_core=True, native_optimizer=True, cross_game_feedback_implemented=True,
        publication_claim_qualified=False, broader_encoder_math_qualified=False)
    save(out / "plan.json", value, exclusive=True)
    inspect(out, current=True)
    return value


def aggregate(plan, analysis, config):
    require(analysis["status"] == "ok" and not analysis["missing_evaluations"]
            and analysis["audited_jobs"] == len(plan["jobs"]), "Incomplete panel cannot supply successful feedback")
    require(len(plan["candidates"]) == 1, "One architecture per feedback observation")
    rows = analysis["observations"]
    require(len(rows) == plan["planned_evaluations"], "Incomplete objective cells")
    games, costs = [], []
    identities = set()
    for task in panels.TASKS:
        offset = config.getfloat("objective." + task, "offset")
        target = config.getfloat("objective." + task, "target")
        normalize = lambda score: max(0.0, min(1.0, (score - offset) / (target - offset)))
        selected = [r for r in rows if r["environment"] == task and r["decisions"] == plan["task_budgets"][task]["steps"]]
        wanted = {(seed, drawing) for seed in plan["seeds"] for drawing in range(panels.panel.ENVIRONMENTS[task])}
        require(len(selected) == len(wanted) and {(r["seed"], r["representation"]) for r in selected} == wanted,
                "Missing/duplicate game-seed-drawing feedback cell")
        require(all(math.isfinite(r[k]) for r in selected for k in ("score_lower", "score_upper", "train_seconds")), "Nonfinite objective")
        require(all(r["score_lower"] <= r["score_upper"] for r in selected), "Reversed objective bounds")
        for seed in plan["seeds"]:
            shared = {r["train_seconds"] for r in selected if r["seed"] == seed}
            require(len(shared) == 1 and next(iter(shared)) > 0, "Drawing duplicated or changed training cost")
            identity = task, seed; require(identity not in identities, "Duplicate training charge")
            identities.add(identity); costs.append(next(iter(shared)))
        games.append(dict(environment=task, cells=len(selected), offset=offset, target=target,
            normalized_lower=math.fsum(normalize(r["score_lower"]) for r in selected) / len(selected),
            normalized_upper=math.fsum(normalize(r["score_upper"]) for r in selected) / len(selected)))
    return dict(score=math.fsum(g["normalized_lower"] for g in games)/len(games),
        score_upper=math.fsum(g["normalized_upper"] for g in games)/len(games),
        cost=math.fsum(costs), training_jobs_charged=len(costs), games=games,
        conservative_pong=True, publication_claim_qualified=False)


def run(args):
    require(args.allow_gpu, "Scheduled feedback run requires --allow-gpu")
    root = args.plan.resolve().parent; value = inspect(root, current=True)
    config, seeds = recipe(root / "recipe.ini"); settings = config["search"]
    count = settings.getint("max_runs"); timeout = settings.getint("campaign_timeout")
    expected = "5060" if args.mode == "smoke" else "5090"
    if args.mode in ("smoke", "canary5090"):
        require(count <= 3 and len(seeds) == 1 and settings.getint("steps") <= 131072
            and settings.getint("checkpoint_steps") == settings.getint("steps")
            and settings.getint("episodes") <= 17 and settings.getint("slots") <= 16
            and 0 < timeout <= 600, "Feedback canary exceeds bounded allocation")
    limit = 30 if args.mode in ("smoke", "canary5090") else 3600
    require(0 < timeout <= 172800 and all(0 < settings.getint(k) <= limit for k in
            ("train_timeout", "eval_timeout", "proposal_timeout")), "Unbounded research deadlines")
    execution = root / "execution"; execution.mkdir(exist_ok=False)
    history = "PUFFER_CROSS_GAME_V1 " + str(value["dimensions"]) + "\n"
    record = dict(status="running", mode=args.mode, trials=[], started_monotonic_ns=time.monotonic_ns(),
        learning_quality_claim=False, plan_sha256=sha(args.plan), history=history)
    deadline = time.monotonic() + timeout
    def remaining(limit):
        require(deadline > time.monotonic(), "Research campaign deadline reached")
        return min(limit, deadline - time.monotonic())
    def propose(directory):
        inspect(root, current=True)
        directory.mkdir(exist_ok=False)
        ledger = directory / "history.tsv"; ledger.write_text(history)
        before = sha(ledger)
        output = directory / "proposal.json"
        command = [value["optimizer"], "research_protein_propose", str(root / "recipe.ini"), str(ledger), str(output)]
        lock_path = ROOT / "build/connect4cnn/hardware-benchmark.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hardware = panels.hardware(expected)
            if "hardware" in record: require(hardware == record["hardware"], "Changed research hardware")
            else: record["hardware"] = hardware
            timing = panels.process(command, root, directory / "native.txt", remaining(settings.getint("proposal_timeout")))
        require(sha(ledger) == before, "Native proposal changed fixed history")
        proposal = json.loads(output.read_text())
        require(proposal["observations_replayed"] == len(record["trials"])
                and len(proposal["normalized"]) == value["dimensions"], "Native feedback replay incomplete")
        return proposal, timing
    try:
        for index in range(count):
            trial = execution / f"trial-{index:04d}"
            proposal, timing = propose(trial)
            name = ("happy-cat-1", "mystic-tree-2", "swift-fox-3")[index] if index < 3 else f"quiet-owl-{index+1}"
            candidate = panels.panel.read(); candidate["policy"] = dict(config["policy"])
            require(set(proposal["policy"]) == set(value["description"]["coordinates"]), "Native coordinate set changed")
            for key, number in proposal["policy"].items():
                require(type(number) in (int, float) and math.isfinite(number) and number == int(number), "Invalid discrete native coordinate")
                candidate["policy"][key] = str(int(number))
            candidate_path = trial / "candidate.ini"
            with candidate_path.open("x") as stream: candidate.write(stream)
            panels.architecture(candidate_path)
            spec = panels.architecture(candidate_path)
            duplicate = next((r["index"] for r in record["trials"] if r["architecture"] == spec), None)
            prep = SimpleNamespace(out=trial / "panel", registry=root / "registry.json",
                candidate=[name + "=" + str(candidate_path)], native_campaign=[], baselines=[],
                policy_metadata=Path(value["policy_metadata"]), seeds=seeds,
                steps=settings.getint("steps"), checkpoint_steps=settings.getint("checkpoint_steps"),
                task_budget=[], learner_recipe=[r["environment"] + "=" + str(root / r["file"]) for r in value["learner_recipes"]], appearance_seed=settings.getint("appearance_seed"),
                eval_seed=settings.getint("eval_seed"), episodes=settings.getint("episodes"), slots=settings.getint("slots"),
                pong_max_decisions=settings.getint("pong_max_decisions"), breakout_max_frames=settings.getint("breakout_max_frames"))
            # Registry source snapshots live in the original build directory.
            prep.registry = Path(json.loads((root / "registry-location.json").read_text())["original"])
            with contextlib.redirect_stdout(io.StringIO()):
                plan = panels.prepare(prep)
                panels.run(SimpleNamespace(plan=prep.out / "plan.json", allow_gpu=True,
                    mode={"smoke": "research", "canary5090": "canary5090", "development": "development"}[args.mode],
                    timeout=remaining(timeout), train_timeout=settings.getint("train_timeout"), eval_timeout=settings.getint("eval_timeout")))
                analysis = panels.audit(prep.out / "plan.json", trial / "review")
            feedback = aggregate(plan, analysis, config)
            save(trial / "feedback.json", feedback, exclusive=True)
            fields = ["0", format(feedback["score"], ".17g"), format(feedback["cost"], ".17g")]
            fields.extend(format(x, ".9g") for x in proposal["normalized"])
            history += " ".join(fields) + "\n"
            record["trials"].append(dict(index=index, name=name, proposal=proposal, optimizer_process=timing,
                architecture=spec, duplicate_of=duplicate,
                feedback=feedback, feedback_sha256=sha(trial / "feedback.json"), candidate_sha256=sha(candidate_path),
                audit_sha256=sha(trial / "review/analysis.json")))
            record["history"] = history
            save(execution / "progress.json", record)
        # Explicit unused final proposal confirms the last panel reached native observe.
        final, timing = propose(execution / "final-feedback")
        record.update(status="ok", final_observations_replayed=final["observations_replayed"],
            final_unused_proposal=final, final_optimizer_process=timing)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_monotonic_ns"] = time.monotonic_ns()
        save(execution / "result.json", record)
    return record


def audit(args):
    """Recompute scalar feedback from native panel audits; no query or policy."""
    root = args.plan.resolve().parent; value = inspect(root)
    config, seeds = recipe(root / "recipe.ini")
    result_path = root / "execution/result.json"
    result = json.loads(result_path.read_text())
    require(result["status"] == "ok" and result["plan_sha256"] == sha(args.plan), "Incomplete or foreign feedback run")
    require(len(result["trials"]) == config.getint("search", "max_runs"), "Missing feedback trial")
    args.out.mkdir(parents=True, exist_ok=False)
    history = f"PUFFER_CROSS_GAME_V1 {value['dimensions']}\n"
    previous_end = result["started_monotonic_ns"]
    learners, architectures, summaries = {}, [], []
    def proposal(directory, expected, timing):
        nonlocal previous_end
        require((directory / "history.tsv").read_text() == history, "Changed feedback history")
        require(json.loads((directory / "proposal.json").read_text()) == expected, "Changed native proposal")
        require(json.loads((directory / "native.txt.json").read_text()) == timing, "Changed proposal timing")
        require(timing["status"] == "ok" and timing["returncode"] == 0
            and timing["cwd"] == str(root) and timing["command"] == [value["optimizer"], "research_protein_propose",
                str(root / "recipe.ini"), str(directory / "history.tsv"), str(directory / "proposal.json")]
            and 0 < timing["timeout"] <= config.getint("search", "proposal_timeout")
            and previous_end <= timing["launch_monotonic_ns"] < timing["end_monotonic_ns"], "Invalid proposal command/clock")
        require(expected["protocol"] == "native-cross-game-protein-v1" and expected["trial"] == len(summaries)
            and expected["observations_replayed"] == len(summaries)
            and (expected["success_observations"] == 0 if not summaries
                 else 1 <= expected["success_observations"] <= len(summaries))
            and expected["failure_observations"] == 0,
            "Incomplete native observation replay")
        previous_end = timing["end_monotonic_ns"]
    for index, entry in enumerate(result["trials"]):
        require(entry["index"] == index, "Unassigned/reordered trial")
        trial = root / "execution" / f"trial-{index:04d}"
        proposal(trial, entry["proposal"], entry["optimizer_process"])
        for name, key in (("feedback.json", "feedback_sha256"), ("candidate.ini", "candidate_sha256"),
                          ("review/analysis.json", "audit_sha256")):
            require(sha(trial / name) == entry[key], "Changed trial receipt: " + name)
        spec = panels.architecture(trial / "candidate.ini")
        require(spec == entry["architecture"], "Changed trial architecture")
        require(entry["duplicate_of"] == next((i for i, a in enumerate(architectures) if a == spec), None), "Hidden architecture duplicate")
        for key, number in entry["proposal"]["policy"].items():
            require(spec[key] == number, "Candidate does not use native proposal")
        architectures.append(spec)
        panel_root = trial / "panel"; plan = panels.inspect(panel_root / "plan.json")
        require(plan["seeds"] == seeds and len(plan["candidates"]) == 1
            and plan["candidates"][0]["architecture"] == spec, "Panel changed assigned architecture/seeds")
        for job in plan["jobs"]:
            key = job["environment"], job["seed"]
            fixed = common_settings(panels.panel.read(panel_root / job["config"]))
            if key in learners: require(learners[key] == fixed, "Learner/world/budget changed across candidates")
            else: learners[key] = fixed
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = panels.audit(panel_root / "plan.json", args.out / f"trial-{index:04d}")
        feedback = aggregate(plan, analysis, config)
        require(feedback == entry["feedback"] == json.loads((trial / "feedback.json").read_text()), "Feedback does not match exact outcomes")
        panel_result = json.loads((panel_root / "execution/result.json").read_text())
        require(previous_end <= panel_result["started_monotonic_ns"] < panel_result["ended_monotonic_ns"], "Optimizer overlaps panel")
        previous_end = panel_result["ended_monotonic_ns"]
        fields = ["0", format(feedback["score"], ".17g"), format(feedback["cost"], ".17g")]
        fields.extend(format(x, ".9g") for x in entry["proposal"]["normalized"])
        history += " ".join(fields) + "\n"
        summaries.append(dict(index=index, feedback=feedback, duplicate_of=entry["duplicate_of"]))
    proposal(root / "execution/final-feedback", result["final_unused_proposal"], result["final_optimizer_process"])
    require(result["history"] == history and result["final_observations_replayed"] == len(summaries)
        and previous_end <= result["ended_monotonic_ns"], "Final feedback was not observed")
    if config.getint("protein", "num_random_samples") == 0:
        require(result["final_unused_proposal"]["gp_observations"] > 0, "GP feedback path did not run")
    report = dict(status="ok", protocol=PROTOCOL, trials=summaries,
        result_sha256=sha(result_path), policy_execution_in_audit=False, native_feedback_receipts_audited=True,
        publication_claim_qualified=False, broader_encoder_math_qualified=False,
        optimizer_process_seconds=math.fsum((r["optimizer_process"]["end_monotonic_ns"]-r["optimizer_process"]["launch_monotonic_ns"])/1e9
            for r in result["trials"]) + (result["final_optimizer_process"]["end_monotonic_ns"]-result["final_optimizer_process"]["launch_monotonic_ns"])/1e9)
    save(args.out / "analysis.json", report, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("recipe", "registry", "optimizer-build", "policy-metadata", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--learner-recipe", action="append", default=[], help="ENV=INI overlay fixed across every candidate")
    check = sub.add_parser("inspect"); check.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("run"); execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--mode", choices=("smoke", "canary5090", "development"), required=True)
    execute.add_argument("--allow-gpu", action="store_true")
    review = sub.add_parser("audit")
    review.add_argument("--plan", type=Path, required=True); review.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args)
    elif args.command == "inspect": value = inspect(args.out)
    elif args.command == "audit": value = audit(args)
    else: value = run(args)
    print(json.dumps(value, indent=2))


if __name__ == "__main__": main()
