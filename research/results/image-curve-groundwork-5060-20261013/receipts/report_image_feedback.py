#!/usr/bin/env python3
"""Audited image-panel/feedback exports; no native execution or GPU query.

Process SPS, monotonic checkpoint costs and raw native metric bins have distinct
columns. Initial checkpoints are observations, not a zero-duration training run.
"""
import argparse
import configparser
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import candidate_panel as panels
import candidate_viewer as viewer
import cross_game_feedback as feedback
import pixel_frontiers as frontiers
from claim import require, save, sha


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    require(all(tuple(row) == tuple(rows[0]) for row in rows), "Inconsistent export columns")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def panel_exports(stage, plan, analysis, execution):
    require(panels.image_protocol(plan["protocol"]) and analysis["status"] == "ok"
            and analysis["audited_jobs"] == len(plan["jobs"])
            and not analysis["missing_evaluations"], "Use a complete audited image panel")
    trained = {(row["environment"], row["candidate"], row["seed"]): row for row in analysis["training"]}
    require(len(trained) == len(plan["jobs"]) == len(analysis["training"])
            and len(execution["jobs"]) == len(plan["jobs"]), "Missing or duplicate audited training jobs")
    training, checkpoints, native = [], [], []
    indices = set()
    for entry in execution["jobs"]:
        index = entry["job_index"]
        require(type(index) is int and 0 <= index < len(plan["jobs"]) and index not in indices,
                "Duplicate or unassigned training export")
        indices.add(index); job = plan["jobs"][index]
        task, name, seed = job["environment"], job["candidate"], job["seed"]
        row = trained[(task, name, seed)]
        decisions = plan["task_budgets"][task]["steps"]
        duration = entry["process_seconds"]
        require(math.isfinite(duration) and duration > 0 and duration == row["train_seconds"]
                and entry["parameters"] == row["parameters"]
                and row["process_sps"] == entry["process_sps"] == decisions/duration,
                "Training export differs from audited time/layout")
        identity = dict(stage=stage, environment=task, candidate=name, seed=seed)
        metric_path = Path(row["native_metrics_ini"])
        training.append(dict(**identity, decisions=decisions, parameters=row["parameters"],
            process_seconds=duration, process_sps=row["process_sps"],
            native_avg_sps=None, native_metrics_ini=str(metric_path), native_metrics_sha256=sha(metric_path)))
        previous_step, previous_time = None, None
        for step in job["checkpoint_steps"]:
            seconds = entry["checkpoint_seconds"][str(step)]
            require(type(step) is int and step >= 0 and math.isfinite(seconds) and seconds > 0
                    and (previous_step is None or step > previous_step and seconds > previous_time),
                    "Nonchronological checkpoint export")
            interval_steps = None if previous_step is None else step-previous_step
            interval_seconds = None if previous_time is None else seconds-previous_time
            checkpoints.append(dict(**identity, decisions=step, checkpoint_seconds=seconds,
                cumulative_checkpoint_sps=None if step == 0 else step/seconds,
                interval_decisions=interval_steps, interval_seconds=interval_seconds,
                interval_checkpoint_sps=None if interval_steps is None else interval_steps/interval_seconds,
                checkpoint_sha256=entry["checkpoints"][str(step)]["sha256"]))
            previous_step, previous_time = step, seconds
        metrics = configparser.ConfigParser(interpolation=None); metrics.optionxform = str
        require(metrics.read(metric_path) and metrics.has_section("metrics"), "Missing native metric arrays")
        arrays = {key:[float(v) for v in metrics["metrics"][key].split(",")]
                  for key in ("agent_steps", "SPS", "uptime")}
        require(len({len(v) for v in arrays.values()}) == 1
                and all(math.isfinite(v) for values in arrays.values() for v in values), "Invalid native metric bins")
        for bin_index, values in enumerate(zip(arrays["agent_steps"], arrays["SPS"], arrays["uptime"])):
            native.append(dict(**identity, bin_index=bin_index, mean_agent_steps=values[0],
                native_bin_mean_sps=values[1], native_bin_mean_uptime=values[2],
                uptime_exceeds_process_seconds=values[2] > duration,
                native_metrics_sha256=training[-1]["native_metrics_sha256"]))
    return training, checkpoints, native


def report(plan_path, out, panel_only=False):
    plan_path, out = plan_path.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        roots = []
        with redirect_stdout(io.StringIO()):
            if panel_only:
                plan = panels.inspect(plan_path)
                audit = panels.audit(plan_path, out/"review")
                roots.append(("panel", plan_path.parent, plan, audit))
                result = json.loads((plan_path.parent/"execution/result.json").read_text())
                anchor_rows = []; selection_rows = []
            else:
                plan = feedback.inspect(plan_path.parent)
                require(plan["protocol"] == feedback.CURVE_PROTOCOL, "Use image curve-feedback v4, not historical campaigns")
                audit = feedback.audit(SimpleNamespace(plan=plan_path, out=out/"review"))
                result = json.loads((plan_path.parent/"execution/result.json").read_text())
                binding = json.loads((plan_path.parent/"curve-bindings.json").read_text())
                controls = Path(binding["nature_panel"])
                roots.append(("calibration", controls.parent, panels.inspect(controls),
                    json.loads((out/"review/nature-reference/panel/analysis.json").read_text())))
                anchor_rows = []; selection_rows = []
                for entry in result["trials"]:
                    stage = f"trial-{entry['index']:04d}"
                    root = plan_path.parent/"execution"/stage/"panel"
                    roots.append((stage, root, panels.inspect(root/"plan.json"),
                        json.loads((out/"review"/stage/"analysis.json").read_text())))
                    for point in entry["feedback"]["points"]:
                        for game in point["games"]:
                            anchor_rows.append(dict(stage=stage, candidate=entry["name"],
                                anchor_index=point["anchor_index"], environment=game["environment"],
                                normalized_lower=game["normalized_lower"], normalized_upper=game["normalized_upper"],
                                nature_lower=game["nature_lower"], delta_lower=game["delta_lower"],
                                balanced_lower=point["balanced_lower"], arithmetic_lower=point["arithmetic_lower"]))
                            for selection in game["selections"]:
                                selection_rows.append(dict(stage=stage, candidate=entry["name"],
                                    environment=game["environment"], anchor_index=point["anchor_index"], **selection))
        training, checkpoints, native, links = [], [], [], []
        for stage, root, panel, panel_audit in roots:
            execution = json.loads((root/"execution/result.json").read_text())
            rows = panel_exports(stage, panel, panel_audit, execution)
            training.extend(rows[0]); checkpoints.extend(rows[1]); native.extend(rows[2])
            display = viewer.assemble(panel, panel_audit)
            # Each panel has its own descriptive curves; discovery is not fresh confirmation.
            display["publication_claim_qualified"] = False
            curves = out/"panels"/stage; curves.mkdir(parents=True)
            save(curves/"data.json", display, exclusive=True)
            frontiers.write_viewer(curves/"curves.html", display)
            links.append(f"- [{stage}: all games/drawings/checkpoints](panels/{stage}/curves.html)")
        for name, rows in (("training",training),("checkpoints",checkpoints),("native-bins",native),
                           ("anchors",anchor_rows),("anchor-selections",selection_rows)):
            write_csv(out/(name+".csv"), rows)
        summary = dict(schema="audited-image-feedback-report-v1", status="ok", termination=result["status"],
            plan_sha256=sha(plan_path), result_sha256=audit["result_sha256"], hardware=roots[0][3]["hardware"],
            panels=len(roots), training_jobs=len(training), checkpoints=len(checkpoints), native_bins=len(native),
            training_process_seconds=math.fsum(row["process_seconds"] for row in training),
            policy_executed=False, gpu_queried=False, publication_claim_qualified=False,
            files_sha256={str(path.relative_to(out)):sha(path) for path in out.rglob("*") if path.is_file()})
        (out/"REPORT.md").write_text("\n".join([
            "# Audited realistic-image results", "", f"Termination: {result['status']}. "
            f"{len(training)} training jobs; {len(checkpoints)} checkpoints. All declared games/drawings remain.", "",
            "[Process times/SPS](training.csv) · [Monotonic checkpoint costs](checkpoints.csv) · "
            "[Raw native SPS/uptime bins](native-bins.csv) · [Every anchor](anchors.csv) · "
            "[Selected affordable checkpoints](anchor-selections.csv)", "", *links, "",
            "Process SPS includes startup and checkpoint writes. Checkpoint intervals include writes. "
            "Native bins retain the trainer's own clock and are not averaged into process SPS. "
            "An uptime exceeding process duration is retained and flagged, not corrected.", "",
            "The initial policy has a measured positive startup/checkpoint cost and no invented training SPS. "
            "Discovery curves are descriptive; fresh paired confirmation and statistical inference remain required. "
            "No superiority, generalization or SOTA claim follows from this report.", ""]))
        summary["files_sha256"]["REPORT.md"] = sha(out/"REPORT.md")
        save(out/"REPORT.json",summary,exclusive=True)
        return summary
    except BaseException as error:
        save(out/"failure.json",dict(status="failed",error=f"{type(error).__name__}: {error}",
            policy_executed=False,gpu_queried=False),exclusive=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--panel",action="store_true",help="Report one audited image panel instead of curve feedback")
    args=parser.parse_args(); print(json.dumps(report(args.plan,args.out,args.panel),indent=2))
