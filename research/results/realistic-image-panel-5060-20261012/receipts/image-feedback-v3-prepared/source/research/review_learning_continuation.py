#!/usr/bin/env python3
"""Offline audit of a completed native-feedback prefix in a failed allocation.

Preserves the failed result and partial trial. Its final successful proposal must
prove every completed observation reached native observe. Never queries a GPU.
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace


def load_supervisor(packet):
    manifest = json.loads((packet / "continuation.json").read_text())
    spec = importlib.util.spec_from_file_location("retained_continuation", manifest["supervisor"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fb, _ = module.modules(Path(manifest["source"]))
    module.inspect(packet, fb)
    return module, fb


def verify_proposal(c, directory, history, expected, timing, search, value, config, count, previous_end):
    c.require((directory / "history.tsv").read_text() == history and c.read(directory / "proposal.json") == expected
              and c.read(directory / "native.txt.json") == timing, "Changed native replay receipt")
    c.require(timing["status"] == "ok" and timing["returncode"] == 0 and timing["cwd"] == str(search)
              and timing["command"] == [value["optimizer"], "research_protein_propose", str(search / "recipe.ini"),
                  str(directory / "history.tsv"), str(directory / "proposal.json")]
              and 0 < timing["timeout"] <= config.getint("search", "proposal_timeout")
              and previous_end <= timing["launch_monotonic_ns"] < timing["end_monotonic_ns"], "Invalid native process receipt")
    c.require(expected["protocol"] == "native-cross-game-protein-v1" and expected["trial"] == count
              and expected["observations_replayed"] == count and expected["failure_observations"] == 0
              and 1 <= expected["success_observations"] <= count and expected["gp_observations"] > 0,
              "Native replay did not observe the completed prefix")
    return timing["end_monotonic_ns"]


def prefix_boundary(c, result, spec, inherited):
    c.require(result["status"] == "failed" and result["inherited_observations"] == inherited,
              "This reader requires a failed allocation with unchanged ancestry")
    c.require(0 < len(result["trials"]) < spec["additional_trials"], "Missing or invalid completed prefix")
    return inherited + len(result["trials"])


def review(packet, out):
    c, fb = load_supervisor(packet)
    spec = c.inspect(packet, fb)
    result_path = packet / "execution/result.json"
    result = c.read(result_path)
    c.require(result["plan_sha256"] == c.sha(packet / "continuation.json"), "Changed continuation plan")
    entries, history, expected_next, hardware, campaign = c.parent_records(Path(spec["parent"]), fb)
    count = prefix_boundary(c, result, spec, len(entries))
    c.require(result["hardware"] == hardware, "Changed continuation hardware")
    out.mkdir(parents=True, exist_ok=False)
    c.audit_ancestry(Path(spec["parent"]), fb, out / "parent-audit")
    search = packet / "search"
    value = fb.inspect(search)
    config, seeds = fb.recipe(search / "recipe.ini")
    previous_end = result["started_monotonic_ns"]
    all_entries = [e for _, e in entries]
    observations, training, plans, learners = [], [], [], {}
    def add_panel(panel_root, index):
        plan = fb.panels.inspect(panel_root / "plan.json")
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = fb.panels.audit(panel_root / "plan.json", out / f"panel-{index:04d}")
        c.require(analysis["hardware"] == hardware and plan["seeds"] == seeds
                  and plan["task_budgets"] == value["task_budgets"] and plan.get("per_game_learners") is True,
                  "Mixed panel hardware/learners/seeds/budget")
        for job in plan["jobs"]:
            key = job["environment"], job["seed"]
            fixed = fb.common_settings(fb.panels.panel.read(panel_root / job["config"]))
            c.require(key not in learners or learners[key] == fixed, "Learner changed across candidates")
            learners[key] = fixed
        plans.append(plan)
        observations.extend(analysis["observations"])
        training.extend(analysis["training"])
        return plan, analysis
    for parent_search, entry in entries:
        add_panel(parent_search / "execution" / f"trial-{entry['index']:04d}" / "panel", entry["index"])
    for local, entry in enumerate(result["trials"]):
        index = len(all_entries)
        c.require(entry["index"] == index, "Reordered completed prefix")
        trial = search / "execution" / f"trial-{index:04d}"
        previous_end = verify_proposal(c, trial, history, entry["proposal"], entry["optimizer_process"],
                                       search, value, config, index, previous_end)
        if local == 0:
            c.require(entry["proposal"]["normalized"] == expected_next["normalized"]
                      and entry["proposal"]["policy"] == expected_next["policy"], "Changed first continued suggestion")
        for filename, key in (("feedback.json", "feedback_sha256"), ("candidate.ini", "candidate_sha256"),
                              ("review/analysis.json", "audit_sha256")):
            c.require(c.sha(trial / filename) == entry[key], "Changed completed trial receipt: " + filename)
        architecture = fb.panels.architecture(trial / "candidate.ini")
        c.require(architecture == entry["architecture"] and entry["duplicate_of"] ==
                  next((e["index"] for e in all_entries if e["architecture"] == architecture), None),
                  "Changed architecture/duplicate identity")
        assigned = fb.panels.panel.read(trial / "candidate.ini")
        c.require(all(assigned.getfloat("policy", k) == v for k, v in entry["proposal"]["policy"].items()),
                  "Candidate differs from native suggestion")
        math_path = trial / "encoder-check"
        c.require(entry["encoder_check"] == dict(report=str(math_path / "REPORT.json"), sha256=c.sha(math_path / "REPORT.json"))
                  and c.sha(math_path / "config.ini") == c.sha(trial / "candidate.ini"), "Changed numerical gate")
        importlib.import_module("cnn_graph_acceptance").audit(math_path)
        plan, analysis = add_panel(trial / "panel", index)
        c.require(len(plan["candidates"]) == 1 and plan["candidates"][0]["architecture"] == architecture,
                  "Panel architecture changed")
        c.require(fb.aggregate(plan, analysis, config) == entry["feedback"] == c.read(trial / "feedback.json"),
                  "Saved feedback differs from audited native outcomes")
        panel_result = c.read(trial / "panel/execution/result.json")
        c.require(previous_end <= panel_result["started_monotonic_ns"] < panel_result["ended_monotonic_ns"],
                  "Overlapping proposal/training")
        previous_end = panel_result["ended_monotonic_ns"]
        history += c.history_row(entry)
        all_entries.append(entry)
    c.require(history == result["history"], "Changed completed feedback ledger")
    # The next proposal already replayed the last completed feedback. Its later
    # partial training contributes neither a score nor a successful observation.
    failed_trial = search / "execution" / f"trial-{count:04d}"
    final_proposal = c.read(failed_trial / "proposal.json")
    final_timing = c.read(failed_trial / "native.txt.json")
    previous_end = verify_proposal(c, failed_trial, history, final_proposal, final_timing,
                                   search, value, config, count, previous_end)
    failed_panel = c.read(failed_trial / "panel/execution/result.json")
    c.require(failed_panel["status"] == "failed" and previous_end <= failed_panel["started_monotonic_ns"]
              < failed_panel["ended_monotonic_ns"] <= result["ended_monotonic_ns"], "Missing partial failure receipt")
    c.require(result["process_seconds"] <= spec["hours"] * 3600 + 5, "Exceeded execution cap")
    reference = campaign / "references"
    ref_plan = fb.panels.inspect(reference / "plan.json")
    with contextlib.redirect_stdout(io.StringIO()):
        ref_analysis = fb.panels.audit(reference / "plan.json", out / "references")
    c.require(ref_analysis["hardware"] == hardware and ref_plan["seeds"] == seeds
              and ref_plan["task_budgets"] == value["task_budgets"], "Changed paired references")
    plans.append(ref_plan); observations.extend(ref_analysis["observations"]); training.extend(ref_analysis["training"])
    merged = {**plans[0], "candidates": sum((p["candidates"] for p in plans), []),
              "jobs": sum((p["jobs"] for p in plans), []),
              "planned_evaluations": sum(p["planned_evaluations"] for p in plans)}
    viewer = importlib.import_module("candidate_viewer")
    reporting = importlib.import_module("pixel_frontiers")
    curves = viewer.assemble(merged, dict(status="ok", mode="development", observations=observations, training=training, hardware=hardware))
    curves["publication_claim_qualified"] = False
    curves["allocation_status"] = "failed"
    curves["excluded_partial_trial_one_based"] = count + 1
    curves["selection_warning"] = ("Audited completed prefix of a failed allocation; partial trial excluded and preserved. "
                                    "Adaptive discovery seeds, not held-out confirmation; fronts are descriptive only.")
    folder = out / "review/combined"
    folder.mkdir(parents=True)
    reporting.write_viewer(folder / "curves.html", curves)
    reporting.write_csv(folder / "training.csv", training, list(training[0]))
    fb.panels.write_frontiers(folder, curves)
    # Use the external scalar exporter with the original frozen objective anchors.
    exporter_spec = importlib.util.spec_from_file_location("offline_sps_report", Path(__file__).with_name("report_learning_progress.py"))
    exporter = importlib.util.module_from_spec(exporter_spec)
    exporter_spec.loader.exec_module(exporter)
    exports = out / "exports"
    exports.mkdir()
    exporter.environment_tables(out, exports, campaign / "search-space.ini")
    report = dict(status="audited-completed-prefix", allocation_status="failed", allocation_error=result["error"],
        inherited_observations=len(entries), completed_new_observations=len(result["trials"]), total_observations=count,
        final_observations_replayed=final_proposal["observations_replayed"], gp_observations=final_proposal["gp_observations"],
        failed_trial_one_based=count + 1, partial_feedback_submitted=False,
        result_sha256=c.sha(result_path), continuation_manifest_sha256=c.sha(packet / "continuation.json"),
        reader_sha256=c.sha(__file__), frozen_supervisor_sha256=spec["supervisor_sha256"],
        completed_comparison_training_jobs=len(training), completed_comparison_observations=len(observations),
        execution_seconds=result["process_seconds"], policy_execution_in_audit=False, gpu_query_in_audit=False,
        publication_claim_qualified=False, curves=str(folder / "curves.html"),
        partial_training_jobs_excluded=len(failed_panel["jobs"]),
        successful_prefix_feedback=sorted([dict(trial=e["index"] + 1, name=e["name"], **e["feedback"],
                                                duplicate_of=e["duplicate_of"]) for e in all_entries], key=lambda e: e["score"], reverse=True))
    report["files_sha256"] = {str(p.relative_to(out)): c.sha(p) for p in out.rglob("*") if p.is_file()}
    c.save(out / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = review(args.packet.resolve(), args.out.resolve())
    print(json.dumps({k: v for k, v in value.items() if k not in ("files_sha256", "successful_prefix_feedback")}, indent=2))


if __name__ == "__main__":
    main()
