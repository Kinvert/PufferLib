#!/usr/bin/env python3
"""Offline SPS snapshot from a learning campaign; never runs policy/GPU work."""
import argparse
import configparser
import csv
import collections
import datetime
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_tables(prepared, output, anchors=None):
    path = prepared / "review/combined/curves.json"
    if not path.exists():
        return
    curves = read_json(path)
    assert not curves["missing"] and not curves["failures"]
    groups = collections.defaultdict(list)
    for row in curves["means"]:
        groups[(row["environment"], row["model"], row["steps"])].append(row)
    rows = []
    for (environment, model, steps), values in sorted(groups.items()):
        drawings = {c["evaluation_representation"] for c in curves["conditions"]
                    if c["environment"] == environment}
        assert {r["evaluation_representation"] for r in values} == drawings
        assert len(values) == len(drawings) and {r["seeds"] for r in values} == {2}
        assert len({r["seconds"] for r in values}) == 1
        rows.append(dict(environment=environment, candidate=model, decisions=steps,
            drawings=len(drawings), training_seeds=2, metric=values[0]["metric"],
            mean_checkpoint_seconds=values[0]["seconds"],
            score_lower=math.fsum(r["score_lower"] for r in values) / len(values),
            score_upper=math.fsum(r["score_upper"] for r in values) / len(values)))
    final_steps = {(r["environment"], r["candidate"]): r["decisions"] for r in rows}
    finals = [r for r in rows if r["decisions"] == final_steps[(r["environment"], r["candidate"])]]
    write_csv(output / "per-environment-checkpoints.csv", rows)
    write_csv(output / "per-environment-final.csv", finals)
    config = configparser.ConfigParser(interpolation=None)
    config.read(anchors or prepared / "search-space.ini")
    normalized = []
    for model in curves["model_catalog"]:
        game_scores = []
        for row in (r for r in finals if r["candidate"] == model):
            environment = row["environment"]
            anchor = config["objective." + environment]
            offset, target = float(anchor["offset"]), float(anchor["target"])
            points = [p for p in curves["points"] if p["model"] == model
                      and p["environment"] == environment and p["steps"] == row["decisions"]]
            assert len(points) == row["drawings"] * row["training_seeds"]
            game_scores.append(math.fsum(max(0, min(1, (p["score_lower"] - offset) / (target - offset)))
                                         for p in points) / len(points))
        assert len(game_scores) == 6
        normalized.append(dict(candidate=model, equal_game_normalized_final_score=math.fsum(game_scores) / 6,
            total_checkpoint_training_seconds=2 * math.fsum(r["mean_checkpoint_seconds"] for r in finals
                                                          if r["candidate"] == model)))
    write_csv(output / "normalized-final-feedback.csv", normalized)


def panel_paths(campaign):
    if (campaign / "continuation.json").is_file():
        spec = read_json(campaign / "continuation.json")
        parent = Path(spec["parent"])
        inherited = panel_paths(parent if (parent / "continuation.json").exists() else parent.parent)
        return inherited + [(p.parent.name, p) for p in sorted((campaign / "search/execution").glob("trial-*/panel"))]
    prepared = campaign / "prepared"
    panels = [("calibration", prepared / "calibration"),
              ("references", prepared / "references")]
    panels += [(p.parent.name, p) for p in sorted(
        (prepared / "execution/search/execution").glob("trial-*/panel"))]
    return panels


def snapshot(campaign, output):
    continuation = (campaign / "continuation.json").is_file()
    prepared = campaign if continuation else campaign / "prepared"
    panels = panel_paths(campaign)
    training, checkpoints, native, statuses = [], [], [], []
    for stage, root in panels:
        result = root / "execution/result.json"
        progress = root / "execution/progress.json"
        if not result.exists() and not progress.exists():
            continue
        record = read_json(result if result.exists() else progress)
        plan = read_json(root / "plan.json")
        statuses.append(dict(stage=stage, status=record["status"],
            completed_training_jobs=len(record["jobs"]),
            planned_training_jobs=len(plan["jobs"]),
            completed_drawing_evaluations=len(record["evaluations"]),
            planned_drawing_evaluations=plan["planned_evaluations"],
            completed_repeat_eager_checks=len(record["repeat_checks"])))
        for entry in record["jobs"]:
            job = plan["jobs"][entry["job_index"]]
            steps = plan["task_budgets"][job["environment"]]["steps"]
            wall = entry["process_seconds"]
            assert wall > 0 and math.isclose(entry["process_sps"], steps / wall)
            identity = dict(stage=stage, environment=job["environment"],
                candidate=job["candidate"], seed=job["seed"])
            metrics = configparser.ConfigParser(interpolation=None)
            metrics.optionxform = str
            metrics.read(root / job["config"])
            metric_file = Path(metrics["base"]["log_dir"]) / job["environment"] / (job["candidate"] + ".ini")
            training.append(dict(**identity, decisions=steps,
                parameters=entry["parameters"], process_wall_seconds=wall,
                process_sps=steps / wall, native_metric_file=str(metric_file)))
            previous_step, previous_time = 0, 0.0
            for step in job["checkpoint_steps"]:
                seconds = entry["checkpoint_seconds"][str(step)]
                assert seconds > previous_time and step > previous_step
                checkpoints.append(dict(**identity, decisions=step,
                    checkpoint_seconds=seconds, cumulative_checkpoint_sps=step / seconds,
                    interval_decisions=step - previous_step,
                    interval_seconds=seconds - previous_time,
                    interval_checkpoint_sps=(step - previous_step) / (seconds - previous_time),
                    checkpoint_sha256=entry["checkpoints"][str(step)]["sha256"]))
                previous_step, previous_time = step, seconds
            metrics.read(metric_file)
            arrays = {key: [float(v) for v in metrics["metrics"][key].split(",")]
                      for key in ("agent_steps", "SPS", "uptime")}
            assert len({len(v) for v in arrays.values()}) == 1
            assert all(math.isfinite(v) for values in arrays.values() for v in values)
            for index, values in enumerate(zip(arrays["agent_steps"], arrays["SPS"], arrays["uptime"])):
                native.append(dict(**identity, bin_index=index,
                    mean_agent_steps=values[0], native_bin_mean_sps=values[1],
                    native_bin_mean_uptime=values[2], native_metric_file=str(metric_file)))
    summary = dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=str(campaign), panels=statuses, completed_training_jobs=len(training),
        completed_checkpoints=len(checkpoints), native_metric_bins=len(native),
        timing="Checkpoint CLOCK_MONOTONIC minus process launch; startup/writes included; evaluation excluded.",
        native_metric_semantics="Original step-bin means, not instantaneous samples or qualified native-average SPS.",
        audit_status="Snapshot consistency checks only; final campaign audit remains separate.",
        score_or_frontier_claim=False, policy_execution=False)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "training-sps.csv", training)
    write_csv(output / "checkpoint-sps.csv", checkpoints)
    write_csv(output / "native-bin-mean-sps.csv", native)
    anchors = Path(read_json(campaign / "continuation.json")["campaign"]) / "search-space.ini" if continuation else None
    environment_tables(prepared, output, anchors)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.campaign.resolve(), args.out.resolve())
