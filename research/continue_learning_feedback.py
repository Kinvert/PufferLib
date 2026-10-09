#!/usr/bin/env python3
"""Append native PROTEIN observations in a fresh, parent-linked research packet.

Loads the unchanged campaign modules from their retained source worktree. No
Python optimizer or learner; calibration/references are audited and reused.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SCHEMA = "native-learning-continuation-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def modules(source):
    sys.path.insert(0, str(Path(source).resolve() / "research"))
    fb = importlib.import_module("cross_game_feedback")
    require(fb.ROOT.resolve() == Path(source).resolve(), "Wrong retained campaign modules")
    return fb, importlib.import_module("feedback_campaign")


def history_row(entry):
    return " ".join(["0", format(entry["feedback"]["score"], ".17g"),
                     format(entry["feedback"]["cost"], ".17g")]
                    + [format(x, ".9g") for x in entry["proposal"]["normalized"]]) + "\n"


def check_budget(additional, hours, inherited):
    require(type(additional) is int and 1 <= additional <= 64, "Allocate 1 to 64 additional trials")
    require(math.isfinite(hours) and 0 < hours <= 48, "Allocate a positive cap through 48 hours")
    require(inherited + additional <= 512, "Native history exceeds its 512-row limit")


def parent_records(parent, fb, current=False):
    """Resolve immutable ancestry, including subsequent continuation packets."""
    if (parent / "continuation.json").is_file():
        spec = inspect(parent, fb, current)
        entries, history, final, hardware, campaign = parent_records(Path(spec["parent"]), fb, current)
        result = read(parent / "execution/result.json")
        require(result["status"] == "ok" and result["plan_sha256"] == sha(parent / "continuation.json"),
                "Parent continuation did not finish successfully")
        require(result["inherited_observations"] == len(entries), "Changed continuation ancestry")
        for entry in result["trials"]:
            require(entry["index"] == len(entries), "Reordered inherited trial")
            history += history_row(entry)
            entries.append((parent / "search", entry))
        require(history == result["history"] and result["final_observations_replayed"] == len(entries),
                "Changed continuation history")
        return entries, history, result["final_unused_proposal"], result["hardware"], campaign
    # The initial parent is the prepared directory of a completed learning campaign.
    _, campaign_module = modules(fb.ROOT)
    campaign_module.inspect(parent, current=current)
    require(read(parent / "execution/result.json")["status"] == "ok", "Initial campaign is incomplete")
    require(read(parent / "execution/calibration-gate.json")["status"] == "eligible-development-search",
            "Initial controls did not qualify")
    search = parent / "execution/search"
    plan = fb.inspect(search, current=current)
    result = read(search / "execution/result.json")
    require(result["status"] == "ok" and result["plan_sha256"] == sha(search / "plan.json"), "Incomplete native history")
    entries = [(search, e) for e in result["trials"]]
    require(all(e["index"] == i for i, (_, e) in enumerate(entries)), "Reordered native history")
    history = f"PUFFER_CROSS_GAME_V1 {plan['dimensions']}\n" + "".join(history_row(e) for _, e in entries)
    require(history == result["history"] and result["final_observations_replayed"] == len(entries), "Incomplete final replay")
    return entries, history, result["final_unused_proposal"], result["hardware"], parent


def inspect(root, fb, current=False):
    spec = read(root / "continuation.json")
    require(spec["schema"] == SCHEMA and (not current or spec["root"] == str(root.resolve())), "Wrong/relocated continuation")
    require(sha(spec["supervisor"]) == spec["supervisor_sha256"], "Continuation supervisor changed")
    for filename, digest in spec["parent_receipts"].items():
        require(sha(filename) == digest, "Parent receipt changed: " + filename)
    fb.inspect(root / "search", current=current)
    require(sha(root / "search/recipe.ini") == spec["recipe_sha256"], "Frozen search recipe changed")
    check_budget(spec["additional_trials"], spec["hours"], spec["inherited_observations"])
    return spec


def audit_ancestry(parent, fb, out):
    if (parent / "continuation.json").is_file():
        return audit(parent, fb, out)
    with contextlib.redirect_stdout(io.StringIO()):
        fb.audit(SimpleNamespace(plan=parent / "execution/search/plan.json", out=out / "search"))
        fb.panels.audit(parent / "calibration/plan.json", out / "calibration")
        fb.panels.audit(parent / "references/plan.json", out / "references")


def prepare(args, fb):
    parent = args.parent.resolve()
    if (parent / "prepared/campaign.json").is_file():
        parent /= "prepared"
    entries, history, final, hardware, campaign = parent_records(parent, fb, current=True)
    check_budget(args.additional_trials, args.hours, len(entries))
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    audit_ancestry(parent, fb, root / "parent-review")
    previous = entries[-1][0]
    value = fb.inspect(previous, current=True)
    registry = read(previous / "registry-location.json")["original"]
    require("encoder_validation" in value, "Continuation requires the inherited numerical gate")
    with contextlib.redirect_stdout(io.StringIO()):
        fb.prepare(SimpleNamespace(recipe=previous / "recipe.ini", registry=Path(registry),
            optimizer_build=Path(value["optimizer_build"]), policy_metadata=Path(value["policy_metadata"]),
            encoder_validation=Path(value["encoder_validation"]),
            learner_recipe=[r["environment"] + "=" + str(previous / r["file"]) for r in value["learner_recipes"]],
            out=root / "search"))
    receipts = [parent / ("continuation.json" if (parent / "continuation.json").is_file() else "campaign.json"),
                parent / "execution/result.json", previous / "plan.json", previous / "execution/result.json"]
    spec = dict(schema=SCHEMA, root=str(root), source=str(fb.ROOT), parent=str(parent), campaign=str(campaign),
        additional_trials=args.additional_trials, hours=args.hours, inherited_observations=len(entries),
        recipe_sha256=sha(previous / "recipe.ini"), parent_receipts={str(p): sha(p) for p in receipts},
        inherited_history_sha256=hashlib.sha256(history.encode()).hexdigest(), hardware=hardware,
        supervisor=str(Path(__file__).resolve()), supervisor_sha256=sha(__file__),
        native_optimizer=True, calibration_reused=True, references_reused=True, publication_claim_qualified=False)
    save(root / "continuation.json", spec)
    inspect(root, fb, current=True)
    return spec


def run(args, fb):
    require(args.allow_gpu, "Scheduled GPU execution requires --allow-gpu")
    root = args.out.resolve()
    spec = inspect(root, fb, current=True)
    entries, history, expected_next, hardware, campaign = parent_records(Path(spec["parent"]), fb, current=True)
    require(len(entries) == spec["inherited_observations"] and
            hashlib.sha256(history.encode()).hexdigest() == spec["inherited_history_sha256"], "Changed imported history")
    search = root / "search"
    value = fb.inspect(search, current=True)
    config, seeds = fb.recipe(search / "recipe.ini")
    settings = config["search"]
    panels = fb.panels
    execution = search / "execution"
    execution.mkdir(exist_ok=False)
    (root / "execution").mkdir(exist_ok=False)
    record = dict(status="running", plan_sha256=sha(root / "continuation.json"), trials=[], history=history,
        inherited_observations=len(entries), hardware=hardware, started_monotonic_ns=time.monotonic_ns(),
        requested_additional_trials=spec["additional_trials"], publication_claim_qualified=False)
    deadline = time.monotonic() + spec["hours"] * 3600
    def remaining(limit, reserve=0):
        amount = deadline - time.monotonic() - reserve
        require(amount > 0, "Continuation deadline reached")
        return min(limit, amount)
    def propose(directory):
        inspect(root, fb, current=True)
        directory.mkdir(exist_ok=False)
        ledger = directory / "history.tsv"
        ledger.write_text(history)
        digest = sha(ledger)
        output = directory / "proposal.json"
        command = [value["optimizer"], "research_protein_propose", str(search / "recipe.ini"), str(ledger), str(output)]
        lock_path = fb.ROOT / "build/connect4cnn/hardware-benchmark.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(panels.hardware("5090") == hardware, "Changed continuation hardware")
            timing = panels.process(command, search, directory / "native.txt", remaining(settings.getint("proposal_timeout")))
        proposal = read(output)
        require(sha(ledger) == digest and proposal["observations_replayed"] == len(entries) + len(record["trials"])
                and len(proposal["normalized"]) == value["dimensions"], "Incomplete native continuation replay")
        return proposal, timing
    try:
        for local in range(spec["additional_trials"]):
            # A completed prefix is valid; reserve time for the final native observe.
            if deadline - time.monotonic() <= settings.getint("proposal_timeout") + 120:
                record["stop_reason"] = "time-cap-between-trials"
                break
            index = len(entries) + local
            trial = execution / f"trial-{index:04d}"
            proposal, timing = propose(trial)
            if local == 0:
                require(proposal["normalized"] == expected_next["normalized"] and proposal["policy"] == expected_next["policy"],
                        "First continued proposal differs from the parent's final unused proposal")
            candidate = panels.panel.read()
            candidate["policy"] = dict(config["policy"])
            require(set(proposal["policy"]) == set(value["description"]["coordinates"]), "Changed native coordinates")
            for key, number in proposal["policy"].items():
                require(type(number) in (int, float) and math.isfinite(number) and number == int(number), "Invalid native coordinate")
                candidate["policy"][key] = str(int(number))
            candidate_path = trial / "candidate.ini"
            with candidate_path.open("x") as stream:
                candidate.write(stream)
            architecture = panels.architecture(candidate_path)
            name = f"quiet-owl-{index + 1}"
            previous = [e for _, e in entries] + record["trials"]
            duplicate = next((e["index"] for e in previous if e["architecture"] == architecture), None)
            checks = importlib.import_module("cnn_graph_acceptance")
            math_path = trial / "encoder-check"
            proof = checks.run(Path(value["encoder_validation"]), candidate_path, math_path, "5090", remaining(120))
            require(proof["hardware"] == hardware, "Changed numerical-check hardware")
            prep = SimpleNamespace(out=trial / "panel", registry=Path(read(search / "registry-location.json")["original"]),
                candidate=[name + "=" + str(candidate_path)], native_campaign=[], baselines=[],
                policy_metadata=Path(value["policy_metadata"]), seeds=seeds,
                steps=settings.getint("steps"), checkpoint_steps=settings.getint("checkpoint_steps"),
                task_budget=[f"{t}={b['steps']}:{b['checkpoint_steps']}" for t, b in fb.task_budgets(config).items()],
                per_game_learners=fb.per_game_learners(config),
                learner_recipe=[r["environment"] + "=" + str(search / r["file"]) for r in value["learner_recipes"]],
                appearance_seed=settings.getint("appearance_seed"), eval_seed=settings.getint("eval_seed"),
                episodes=settings.getint("episodes"), slots=settings.getint("slots"),
                pong_max_decisions=settings.getint("pong_max_decisions"), breakout_max_frames=settings.getint("breakout_max_frames"))
            with contextlib.redirect_stdout(io.StringIO()):
                plan = panels.prepare(prep)
                panels.run(SimpleNamespace(plan=prep.out / "plan.json", allow_gpu=True, mode="development",
                    timeout=remaining(spec["hours"] * 3600, settings.getint("proposal_timeout")),
                    train_timeout=settings.getint("train_timeout"), eval_timeout=settings.getint("eval_timeout")))
                analysis = panels.audit(prep.out / "plan.json", trial / "review")
            outcome = fb.aggregate(plan, analysis, config)
            save(trial / "feedback.json", outcome)
            entry = dict(index=index, name=name, proposal=proposal, optimizer_process=timing,
                architecture=architecture, duplicate_of=duplicate, feedback=outcome,
                feedback_sha256=sha(trial / "feedback.json"), candidate_sha256=sha(candidate_path),
                audit_sha256=sha(trial / "review/analysis.json"),
                encoder_check=dict(report=str(math_path / "REPORT.json"), sha256=sha(math_path / "REPORT.json")))
            history += history_row(entry)
            record["trials"].append(entry)
            record["history"] = history
            save(root / "execution/progress.json", record)
            save(execution / "progress.json", record)
        final, timing = propose(execution / "final-feedback")
        record.update(status="ok", final_observations_replayed=final["observations_replayed"],
            final_unused_proposal=final, final_optimizer_process=timing)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_monotonic_ns"] = time.monotonic_ns()
        record["process_seconds"] = (record["ended_monotonic_ns"] - record["started_monotonic_ns"]) / 1e9
        save(root / "execution/result.json", record)
        save(execution / "result.json", record)
    audit(root, fb, root / "review")
    return record


def audit(root, fb, out):
    """Reaudit ancestry and new exact outcomes without executing GPU work."""
    spec = inspect(root, fb)
    result = read(root / "execution/result.json")
    require(result["status"] == "ok" and result["plan_sha256"] == sha(root / "continuation.json"), "Incomplete continuation")
    out.mkdir(parents=True, exist_ok=False)
    audit_ancestry(Path(spec["parent"]), fb, out / "parent")
    entries, history, expected_next, hardware, campaign = parent_records(Path(spec["parent"]), fb)
    require(result["inherited_observations"] == len(entries) and result["hardware"] == hardware, "Changed ancestry/hardware")
    require(len(result["trials"]) <= spec["additional_trials"], "Exceeded allocated trials")
    search = root / "search"
    value = fb.inspect(search)
    config, seeds = fb.recipe(search / "recipe.ini")
    previous_end = result["started_monotonic_ns"]
    def proposal(directory, expected, timing, count):
        nonlocal previous_end
        require((directory / "history.tsv").read_text() == history and read(directory / "proposal.json") == expected
                and read(directory / "native.txt.json") == timing, "Changed native replay receipt")
        require(timing["status"] == "ok" and timing["returncode"] == 0 and timing["cwd"] == str(search)
                and timing["command"] == [value["optimizer"], "research_protein_propose", str(search / "recipe.ini"),
                    str(directory / "history.tsv"), str(directory / "proposal.json")]
                and 0 < timing["timeout"] <= config.getint("search", "proposal_timeout")
                and previous_end <= timing["launch_monotonic_ns"] < timing["end_monotonic_ns"], "Invalid optimizer process receipt")
        require(expected["protocol"] == "native-cross-game-protein-v1" and expected["trial"] == count
                and expected["observations_replayed"] == count and expected["failure_observations"] == 0
                and 1 <= expected["success_observations"] <= count and expected["gp_observations"] > 0,
                "Native continuation did not replay feedback")
        previous_end = timing["end_monotonic_ns"]
    all_entries = [e for _, e in entries]
    observations, training, plans, learners = [], [], [], {}
    # Reaudit each retained candidate once for the combined Pareto collection.
    for search_root, entry in entries:
        panel_root = search_root / "execution" / f"trial-{entry['index']:04d}" / "panel"
        plan = fb.panels.inspect(panel_root / "plan.json")
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = fb.panels.audit(panel_root / "plan.json", out / f"curves-{entry['index']:04d}")
        require(analysis["hardware"] == hardware, "Mixed hardware in inherited curves")
        plans.append(plan); observations += analysis["observations"]; training += analysis["training"]
        for job in plan["jobs"]:
            key = job["environment"], job["seed"]
            fixed = fb.common_settings(fb.panels.panel.read(panel_root / job["config"]))
            require(key not in learners or learners[key] == fixed, "Inherited learners differ")
            learners[key] = fixed
    for entry in result["trials"]:
        index = len(all_entries)
        require(entry["index"] == index, "Reordered continuation trial")
        trial = search / "execution" / f"trial-{index:04d}"
        proposal(trial, entry["proposal"], entry["optimizer_process"], index)
        if not result["trials"].index(entry):
            require(entry["proposal"]["normalized"] == expected_next["normalized"]
                    and entry["proposal"]["policy"] == expected_next["policy"], "Changed first continuation proposal")
        for filename, key in (("feedback.json", "feedback_sha256"), ("candidate.ini", "candidate_sha256"),
                              ("review/analysis.json", "audit_sha256")):
            require(sha(trial / filename) == entry[key], "Changed continuation trial receipt")
        architecture = fb.panels.architecture(trial / "candidate.ini")
        require(architecture == entry["architecture"] and entry["duplicate_of"] ==
                next((e["index"] for e in all_entries if e["architecture"] == architecture), None), "Changed architecture/duplicate")
        assigned = fb.panels.panel.read(trial / "candidate.ini")
        require(all(assigned.getfloat("policy", k) == v for k, v in entry["proposal"]["policy"].items()), "Candidate changed native proposal")
        math_path = trial / "encoder-check"
        require(entry["encoder_check"] == dict(report=str(math_path / "REPORT.json"), sha256=sha(math_path / "REPORT.json"))
                and sha(math_path / "config.ini") == sha(trial / "candidate.ini"), "Changed numerical gate")
        importlib.import_module("cnn_graph_acceptance").audit(math_path)
        panel_root = trial / "panel"
        plan = fb.panels.inspect(panel_root / "plan.json")
        require(plan["seeds"] == seeds and plan["task_budgets"] == value["task_budgets"]
                and len(plan["candidates"]) == 1 and plan["candidates"][0]["architecture"] == architecture
                and plan.get("per_game_learners") is True, "Changed continuation allocation")
        for job in plan["jobs"]:
            require(learners[job["environment"], job["seed"]] == fb.common_settings(
                fb.panels.panel.read(panel_root / job["config"])), "Learner changed across continuation")
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = fb.panels.audit(panel_root / "plan.json", out / f"curves-{index:04d}")
        require(analysis["hardware"] == hardware, "Mixed continuation hardware")
        require(fb.aggregate(plan, analysis, config) == entry["feedback"] == read(trial / "feedback.json"), "Feedback differs from exact outcomes")
        panel_result = read(panel_root / "execution/result.json")
        require(previous_end <= panel_result["started_monotonic_ns"] < panel_result["ended_monotonic_ns"], "Overlapping optimizer/trainer")
        previous_end = panel_result["ended_monotonic_ns"]
        history += history_row(entry)
        all_entries.append(entry)
        plans.append(plan); observations += analysis["observations"]; training += analysis["training"]
    proposal(search / "execution/final-feedback", result["final_unused_proposal"], result["final_optimizer_process"], len(all_entries))
    require(history == result["history"] and result["final_observations_replayed"] == len(all_entries)
            and previous_end <= result["ended_monotonic_ns"]
            and result["process_seconds"] <= spec["hours"] * 3600 + 5, "Changed final replay/time cap")
    ref = campaign / "references"
    plans.append(fb.panels.inspect(ref / "plan.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        analysis = fb.panels.audit(ref / "plan.json", out / "curves-references")
    require(analysis["hardware"] == hardware, "Mixed reference hardware")
    observations += analysis["observations"]; training += analysis["training"]
    merged = {**plans[0], "candidates": sum((p["candidates"] for p in plans), []),
              "jobs": sum((p["jobs"] for p in plans), []),
              "planned_evaluations": sum(p["planned_evaluations"] for p in plans)}
    require(all(p["seeds"] == merged["seeds"] and p["task_budgets"] == merged["task_budgets"] for p in plans), "Mixed paired allocation")
    viewer = importlib.import_module("candidate_viewer")
    reporting = importlib.import_module("pixel_frontiers")
    curves = viewer.assemble(merged, dict(status="ok", mode="development", observations=observations, training=training, hardware=hardware))
    curves["publication_claim_qualified"] = False
    curves["selection_warning"] = "Adaptive discovery seeds, not held-out confirmation; frontier marks are descriptive only."
    folder = out / "combined"
    folder.mkdir()
    reporting.write_viewer(folder / "curves.html", curves)
    reporting.write_csv(folder / "training.csv", training, list(training[0]))
    fb.panels.write_frontiers(folder, curves)
    report = dict(status="ok", inherited_observations=len(entries), new_observations=len(result["trials"]),
        total_observations=len(all_entries), result_sha256=sha(root / "execution/result.json"),
        curves=str(folder / "curves.html"), training_jobs=len(training), observations=len(observations),
        native_history_audited=True, policy_execution_in_audit=False, publication_claim_qualified=False)
    save(out / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "inspect", "run", "audit"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--additional-trials", type=int)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    fb, _ = modules(args.source)
    if args.command == "prepare":
        require(args.parent is not None and args.additional_trials is not None and args.hours is not None, "Preparation needs parent and bounded allocation")
        value = prepare(args, fb)
    elif args.command == "inspect":
        value = inspect(args.out.resolve(), fb, current=True)
    elif args.command == "audit":
        require(args.review is not None, "Offline audit needs a fresh --review directory")
        value = audit(args.out.resolve(), fb, args.review.resolve())
    else:
        value = run(args, fb)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
