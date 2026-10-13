#!/usr/bin/env python3
"""Offline arbitrary-candidate curves; preparation or full native receipt audit."""
import argparse
import json
import math
from pathlib import Path

import candidate_panel as candidates
import pixel_frontiers as reporting
from claim import require, save, sha


def assemble(plan, analysis=None):
    """Display allocation/scalars only. CLI report obtains analysis via audit."""
    mode = analysis["mode"] if analysis is not None else "preparation"
    result = candidates.candidate_frontiers(plan, analysis["observations"] if analysis else [], mode)
    result["model_catalog"] = [candidate["name"] for candidate in plan["candidates"]]
    require(len(set(result["model_catalog"])) == len(result["model_catalog"]), "Duplicate declared candidate")
    result["planned_cells"] = []
    jobs = {(job["environment"], job["candidate"], job["seed"]): index for index, job in enumerate(plan["jobs"])}
    require(len(jobs) == len(plan["jobs"]), "Duplicate training job identity")
    binding = 0
    for index, job in enumerate(plan["jobs"]):
        task = job["environment"]
        treatment = f"mixed-s{plan['appearance_seed']}-c{plan['assignments'][task]['catalog_count']}"
        for target in job["targets"]:
            for step in job["checkpoint_steps"]:
                result["planned_cells"].append(dict(binding_index=binding, job_index=index, steps=step,
                    environment=task, training_appearance=treatment, evaluation_representation=target["representation"],
                    model=job["candidate"], seed=job["seed"]))
            binding += 1
    require(len(result["planned_cells"]) == plan["planned_evaluations"], "Incomplete viewer allocation")
    actual = analysis["training"] if analysis else []
    trained = {(row["environment"], row["candidate"], row["seed"]): row for row in actual}
    require(len(trained) == len(actual) and set(trained) <= set(jobs), "Unknown/duplicate audited training cost")
    costs = []
    for key, row in trained.items():
        require(type(row["train_seconds"]) in (int, float) and math.isfinite(row["train_seconds"])
                and row["train_seconds"] > 0, "Invalid audited process time")
        job = plan["jobs"][jobs[key]]
        require(row["process_sps"] == plan["task_budgets"][job["environment"]]["steps"] / row["train_seconds"],
                "Process SPS differs from audited training duration")
        costs.append(dict(job_index=jobs[key], status="ok", process_seconds=row["train_seconds"],
            process_sps=row["process_sps"], native_sps=None, native_sps_status="not collected by this reader",
            checkpoint_decisions=job["checkpoint_steps"], hardware=analysis["hardware"]))
    observations = {(row["environment"], row["candidate"], row["seed"], row["representation"], row["decisions"]): row
                    for row in analysis["observations"]} if analysis else {}
    require(not analysis or len(observations) == len(analysis["observations"]), "Duplicate displayed observation")
    for point in result["points"]:
        row = trained[(point["environment"], point["model"], point["seed"])]
        point["parameters"] = row["parameters"]
        original = observations[(point["environment"], point["model"], point["seed"],
                                 point["evaluation_representation"], point["steps"])]
        point["episodes"] = original["counts"]["episodes"]
    result.update(training_job_costs=costs, training_processes=len(costs),
        accounted_training_process_seconds=math.fsum(cost["process_seconds"] for cost in costs),
        training_cost_accounting_complete=len(costs) == len(plan["jobs"]),
        qualification="Development display only; independent math/runtime, learning calibration, selection and frontier inference remain separate gates.",
        timing="Audited native monotonic checkpoint completion minus training process launch; startup/writes included; evaluation excluded.",
        execution_status=analysis["status"] if analysis else "prepared-not-executed")
    if mode in ("smoke", "mini"):
        result["qualification"] = "Plumbing smoke only; short budgets/caps do not establish learned-quality frontiers. " + result["qualification"]
    elif mode == "preparation":
        result["qualification"] = "Allocation only: no policy observations or measured training costs. " + result["qualification"]
    # A failed campaign cannot acquire front markers merely by retaining scores.
    if analysis is not None and analysis["status"] != "ok":
        for row in result["means"]:
            row["frontier_seconds_lower"] = row["frontier_seconds_upper"] = None
    return result


def render(plan_path, out, executed=False):
    plan_path = plan_path.resolve(); out = out.resolve()
    require(not out.exists(), "Use a fresh viewer output directory")
    plan_hash = sha(plan_path); plan = candidates.inspect(plan_path)
    sources = {"research/candidate_viewer.py": Path(__file__),
               "research/candidate_panel.py": Path(candidates.__file__),
               "research/pixel_frontiers.py": Path(reporting.__file__),
               "research/pixel_frontier.js": reporting.VIEWER_JS,
               "research/pixel_frontier.html": reporting.VIEWER}
    hashes = {name: sha(path) for name, path in sources.items()}
    out.mkdir(parents=True, exist_ok=False)
    try:
        analysis = candidates.audit(plan_path, out / "audit") if executed else None
        result = assemble(plan, analysis)
        result.update(plan_sha256=plan_hash, collection_sha256=analysis["result_sha256"] if analysis else None,
            reporter_sha256=hashes["research/candidate_viewer.py"], viewer_sources_sha256=hashes,
            inputs_sha256=analysis["inputs_sha256"] if analysis else {str(plan_path): plan_hash},
            observations_origin="audited native assigned-episode receipts" if analysis else "none; allocation only",
            gpu_queried=False, policy_executed_by_viewer=False)
        if analysis:
            binding_jobs = {cell["binding_index"]: cell["job_index"] for cell in result["planned_cells"]}
            for point in result["points"]:
                job = plan["jobs"][binding_jobs[point["binding_index"]]]
                checkpoint = plan_path.parent / job["id"] / "checkpoints" / job["environment"] / job["candidate"] / f"{point['steps']:016d}.bin"
                require(str(checkpoint) in analysis["inputs_sha256"], "Checkpoint absent from audit proof")
                point["checkpoint_sha256"] = analysis["inputs_sha256"][str(checkpoint)]
        save(out / "analysis.json", result, exclusive=True)
        reporting.write_viewer(out / "curves.html", result)
        fields = ["environment", "training_appearance", "evaluation_representation", "model", "seed", "steps",
                  "seconds", "score_lower", "score_upper", "parameters", "episodes", "checkpoint_sha256"]
        reporting.write_csv(out / "observations.csv", result["points"], fields)
        reporting.write_csv(out / "means.csv", result["means"], fields[:4]+["steps", "seeds", "metric", "seconds",
            "score_lower", "score_upper", "frontier_seconds_lower", "frontier_seconds_upper"])
        reporting.write_csv(out / "missing.csv", result["missing"], ["binding_index", "steps"])
        lines = ["# CNN candidate curves — offline development report", "",
            f"Execution: {result['execution_status']}; coverage: {len(result['points'])}/{len(result['planned_cells'])} observations.", "",
            "[Full interactive curves](curves.html) · [All data and hashes](analysis.json) · [Every observation](observations.csv) · [Paired means](means.csv) · [Missing cells](missing.csv)", "",
            "Every declared candidate/reference stays in the model catalog, including unobserved or losing models. Games/drawings and Pong bounds remain separate. Stored frontier marks are never recomputed when hiding models. Smoke/preparation/failed campaigns have no frontier marks.", "",
            "Training process time is charged once per job; native SPS is not collected here. This report executes no native binary/GPU/model and cannot certify learning superiority or SOTA.", ""]
        (out / "REPORT.md").write_text("\n".join(lines))
        require(sha(plan_path) == plan_hash and all(sha(sources[name]) == digest for name, digest in hashes.items()),
                "Plan or display source changed during rendering")
        candidates.inspect(plan_path)
        return result
    except BaseException as error:
        save(out / "failure.json", dict(error=f"{type(error).__name__}: {error}", gpu_queried=False), exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "report"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True); command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.plan, args.out, executed=args.command == "report")
    print(json.dumps(dict(status=result["execution_status"], models=len(result["model_catalog"]),
        observations=len(result["points"]), allocated=len(result["planned_cells"]), gpu_queried=False), indent=2))


if __name__ == "__main__": main()
