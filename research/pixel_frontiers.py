#!/usr/bin/env python3
"""Offline paired six-game frontier reporting; never executes a policy/binary."""
import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import re

import numpy as np

import audit_pixel_panel as audit
import pixel_evaluation_plan as bridge
import claim as claim_module
import claim_frontier as frontier_module
from claim import ini, normalized, parse_checkpoints, require, save, sha
from claim_frontier import frontier

SCHEMA = "pixel-frontier-collection-v1"
VIEWER = Path(__file__).with_name("pixel_frontier.html")
VIEWER_JS = Path(__file__).with_name("pixel_frontier.js")
METRICS = {
    "connect4cnn": ("win fraction", "wins", None),
    "pongcnn": ("mean final point fraction (censoring bounds)", "mean_final_point_fraction_lower", "mean_final_point_fraction_upper"),
    "flappycnn": ("mean pipes", "mean_pipes", None),
    "breakoutcnn": ("mean score through terminal/frame cap", "mean_score", None),
    "snakecnn": ("mean ending length", "mean_score", None),
    "mazecnn": ("goal fraction", "success_fraction", None),
}


def read(path):
    return json.loads(Path(path).read_text())


def location(base, name):
    require(isinstance(name, str) and name, "Need a nonempty artifact path")
    return (base / name).resolve()


def checked_plan(path):
    with redirect_stdout(io.StringIO()): bridge.inspect(path)
    return read(path)


def allocation(plan):
    jobs = sorted({binding["job_index"] for binding in plan["bindings"]})
    evaluations = [(index, step) for index, binding in enumerate(plan["bindings"])
                   for step in binding["checkpoint_steps"]]
    return jobs, evaluations


def prepare(plan_path, output):
    plan_path = plan_path.resolve(); plan = checked_plan(plan_path)
    jobs, evaluations = allocation(plan)
    value = dict(schema=SCHEMA, plan=str(plan_path), plan_sha256=sha(plan_path),
                 purpose="development", gpu_execution_authorized=False,
                 training=[dict(job_index=index, receipt=None) for index in jobs],
                 evaluations=[dict(binding_index=index, steps=step, result=None) for index, step in evaluations])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "collection.json", value, exclusive=True)
    return value


def score(task, counts):
    _, low_key, high_key = METRICS[task]
    low = counts[low_key] / counts["episodes"] if task == "connect4cnn" else counts[low_key]
    high = counts[high_key] if high_key else low
    require(type(low) in (int, float) and type(high) in (int, float)
            and math.isfinite(low) and math.isfinite(high) and 0 <= low <= high, "Invalid score/bounds")
    if task in ("connect4cnn", "pongcnn", "mazecnn"): require(high <= 1, "Invalid fraction")
    return float(low), float(high)


def condition(binding):
    # A fixed-training source drawing and a mixed catalog are different treatments.
    training = (f"mixed-s{binding['training_representation_seed']}-c{binding['training_catalog_count']}"
                if binding.get("training_representation_mode") == 1 else
                f"fixed-r{binding.get('training_representation', binding['representation'])}")
    return binding["environment"], training, binding["representation"]


def reduce(plan, points, failures):
    """Scalar statistics only. Seeds are equally weighted; drawings never pooled."""
    expected = {(index, step) for index, binding in enumerate(plan["bindings"])
                for step in binding["checkpoint_steps"]}
    lookup = {(point["binding_index"], point["steps"]): point for point in points}
    require(len(lookup) == len(points) and set(lookup) <= expected, "Duplicate/unexpected observed cell")
    for point in points:
        binding = plan["bindings"][point["binding_index"]]
        task, treatment, drawing = condition(binding)
        require((point["environment"], point["training_appearance"], point["evaluation_representation"], point["model"], point["seed"])
                == (task, treatment, drawing, binding["model"], binding["training_seed"]), "Point identity differs from binding")
        require(type(point["seconds"]) in (int, float) and math.isfinite(point["seconds"]) and point["seconds"] > 0
                and all(type(point[k]) in (int, float) and math.isfinite(point[k]) for k in ("score_lower", "score_upper"))
                and 0 <= point["score_lower"] <= point["score_upper"], "Invalid time/score")
        if task in ("connect4cnn", "pongcnn", "mazecnn"): require(point["score_upper"] <= 1, "Invalid fraction")
    missing = [dict(binding_index=index, steps=step) for index, step in sorted(expected - set(lookup))]
    groups = {}
    for index, binding in enumerate(plan["bindings"]):
        key = condition(binding)
        group = groups.setdefault(key, [])
        group.extend((index, step) for step in binding["checkpoint_steps"])
    means, summaries = [], []
    for key, cells in groups.items():
        relevant = {index for index, _ in cells}
        issues = [error for error in failures if error.get("binding_index") in relevant
                  or any(plan["bindings"][index]["job_index"] == error.get("job_index") for index in relevant)]
        complete = all(cell in lookup for cell in cells) and not issues
        rows = []
        model_steps = sorted({(plan["bindings"][index]["model"], step) for index, step in cells})
        for model, step in model_steps:
            assigned = [index for index, steps in cells if steps == step and plan["bindings"][index]["model"] == model]
            require(len(assigned) == len(plan["training_seeds"])
                    and {plan["bindings"][index]["training_seed"] for index in assigned} == set(plan["training_seeds"]),
                    "Condition lost/duplicated an assigned seed")
            observed = [lookup.get((index, step)) for index in assigned]
            if any(point is None for point in observed): continue
            rows.append(dict(environment=key[0], training_appearance=key[1], evaluation_representation=key[2],
                model=model, steps=step, seeds=len(observed), metric=METRICS[key[0]][0],
                seconds=math.fsum(p["seconds"] for p in observed)/len(observed),
                score_lower=math.fsum(p["score_lower"] for p in observed)/len(observed),
                score_upper=math.fsum(p["score_upper"] for p in observed)/len(observed),
                frontier_seconds_lower=None, frontier_seconds_upper=None))
        if complete:
            for bound in ("lower", "upper"):
                flags = frontier([r["seconds"] for r in rows], [r["score_" + bound] for r in rows])
                for row, flag in zip(rows, flags): row["frontier_seconds_" + bound] = bool(flag)
        means.extend(rows)
        summaries.append(dict(environment=key[0], training_appearance=key[1], evaluation_representation=key[2],
            metric=METRICS[key[0]][0], status="complete_descriptive" if complete else "incomplete",
            expected=len(cells), observed=sum(cell in lookup for cell in cells), failures=len(issues),
            score_bounds_kind="deterministic-censoring-not-confidence" if key[0] == "pongcnn" else "point-estimate"))
    return dict(status="complete_descriptive" if not missing and not failures else "incomplete",
                points=points, means=means, conditions=summaries, missing=missing, failures=failures,
                dominance_certified=False, confidence_bands_enabled=False,
                cross_game_score_average_enabled=False, deployment_checkpoint_selected=False)


class Inputs:
    def __init__(self): self.hashes = {}

    def track(self, path):
        path = Path(path).resolve(); digest = sha(path)
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, "Artifact changed while auditing")
        self.hashes[str(path)] = digest
        return path

    def json(self, path): return read(self.track(path))

    def verify(self):
        require(all(sha(Path(path)) == digest for path, digest in self.hashes.items()), "Input changed during reporting")


def training(inputs, receipt_path, binding, plan, plan_root):
    record = inputs.json(receipt_path); root = receipt_path.parent
    process_path = location(root, record["process"])
    timed = inputs.json(process_path)
    require(timed["status"] in ("ok", "failed", "timeout"), "Training isn't terminal")
    require(type(timed["launch_monotonic_ns"]) is int and type(timed["end_monotonic_ns"]) is int
            and timed["end_monotonic_ns"] > timed["launch_monotonic_ns"], "Need strict process clock receipts")
    require(timed["environment"]["PUFFER_CHECKPOINT_RECEIPTS"] == "1", "Native checkpoint receipts weren't enabled")
    hardware = inputs.json(location(root, record["hardware"]))
    require(set(hardware) == {"gpu_uuid", "gpu_name", "host", "cuda_compiler", "driver"}
            and all(isinstance(v, str) and v.strip() for v in hardware.values()), "Incomplete hardware identity")
    binary = plan["binaries"][binding["environment"]][binding["binary_family"]]
    require(timed["command"] == [binary["path"], "train", "--headless"], "Unexpected training command/override")
    require(sha(inputs.track(binary["path"])) == binary["sha256"], "Training binary changed")
    cwd = Path(timed["cwd"]); require(cwd.is_absolute(), "Need absolute native training cwd")
    config_path = inputs.track(cwd / "config/default.ini")
    require(sha(config_path) == binding["config_sha256"], "Executed training config differs from plan")
    game_path = inputs.track(cwd / f"config/{binding['environment']}.ini")
    expected_game = (plan_root / binding["config"]).parent / f"{binding['environment']}.ini"
    require(sha(game_path) == sha(inputs.track(expected_game)), "Secondary game INI differs")
    config = ini(config_path, game_path)
    checkpoint_root = Path(config["base"]["checkpoint_dir"]) / binding["environment"] / config["base"]["run_id"]
    require(checkpoint_root.is_absolute(), "Need absolute planned checkpoint directory")
    resolved = inputs.track(checkpoint_root / "resolved.ini")
    require(normalized(ini(resolved)) == normalized(config), "Native resolved config differs")
    checkpoints = {int(step): location(root, name) for step, name in record["checkpoints"].items()}
    require(len(checkpoints) == len(record["checkpoints"])
            and list(checkpoints) == binding["checkpoint_steps"][:len(checkpoints)], "Wrong checkpoint prefix/allocation")
    require(checkpoints, "No completed checkpoints")
    parameters = None
    for step, path in checkpoints.items():
        require(path == (checkpoint_root / f"{step:016d}.bin").resolve(), "Unexpected checkpoint path")
        inputs.track(path)
        require(path.stat().st_size > 0 and path.stat().st_size % 4 == 0, "Malformed float32 weights")
        weights = np.fromfile(path, dtype=np.float32)
        require(np.isfinite(weights).all(), "Nonfinite checkpoint")
        require(parameters is None or parameters == weights.size, "Parameter count changed within job")
        parameters = int(weights.size)
    log = inputs.track(location(root, record["log"]))
    times = parse_checkpoints(log.read_text(), timed, binding["checkpoint_steps"], parameters, allow_partial=True)
    require(set(times) == set(checkpoints), "Timing/weight checkpoint allocation differs")
    require(timed["status"] != "ok" or timed["returncode"] == 0, "Successful training had nonzero exit")
    return dict(status=timed["status"], times=times, checkpoints=checkpoints, parameters=parameters,
                process_seconds=(timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9,
                launch_ns=timed["launch_monotonic_ns"], end_ns=timed["end_monotonic_ns"], hardware=hardware)


def effective_config(ev, config, suite, task, checkpoint, directory, graph):
    require(graph in (-1, 1), "Need recorded eager or graph evaluation")
    if task == "connect4cnn":
        config["env"].update(player_pieces="0", env_pieces="0", representation=str(suite["representation"]),
                             representation_mode="0", representation_seed="0")
        config["eval_exact"] = {k: str(suite[k]) for k in ("seed", "episodes", "slots")}
        config["eval_exact"]["episode_offset"] = str(suite["offset"])
    else:
        if task == "mazecnn": config = ev.evaluation_world(config, suite)[0]
        config = ev.suite_config(config, suite)
    config["base"].update(load_model_path=str(checkpoint), eval_agents=str(suite["slots"]),
        run_id="evaluation", checkpoint_dir=str(directory / "unused-checkpoints"), log_dir=str(directory / "metrics"),
        seed="0", cudagraphs=str(graph), **{"async": "0"})
    config["vec"].update(total_agents=str(suite["slots"]), num_buffers="1", num_threads="1",
                         num_policies="1", hist_policy_percent="0")
    config["selfplay"]["enabled"] = "0"
    config["train"].update(gpus="1", horizon="1", minibatch_size=str(suite["slots"]), replay_ratio="1", verb_eps="0")
    config["eval_exact"]["output"] = str(directory / "episodes.csv")
    return config


def completion(task, text, counts, parameters):
    fields = {
        "connect4cnn": ("CONNECT4", ["wins"]),
        "pongcnn": ("PONG", ["completed", "capped", "wins", "right_points", "left_points"]),
        "flappycnn": ("FLAPPY", ["pipes", "capped"]),
        "breakoutcnn": ("BREAKOUT", ["capped", "score"]),
        "snakecnn": ("SNAKE", ["deaths", "horizon_ends", "score", "foods"]),
        "mazecnn": ("MAZE", ["successes", "timeouts"]),
    }
    prefix, keys = fields[task]
    lines = [line for line in text.splitlines() if line.startswith(prefix + "_EXACT_EVAL ")]
    require(len(lines) == 1, "Missing/duplicate native completion summary")
    parts = lines[0].split()[1:]
    require(all(re.fullmatch(r"[a-z_]+=[0-9]+", part) for part in parts), "Malformed native summary")
    pairs = [part.split("=", 1) for part in parts]
    values = dict(pairs); require(len(values) == len(pairs), "Duplicate summary field")
    require(values["version"] == "1" and int(values["requested"]) == counts["episodes"]
            and int(values["params"]) == parameters, "Wrong native completion quota/parameters")
    # Pong distinguishes completed matches from administratively capped matches.
    require(int(values["completed"]) == (counts["completed"] if task in ("pongcnn", "breakoutcnn") else counts["episodes"]),
            "Wrong native completion count")
    if task in ("pongcnn", "breakoutcnn"):
        require(int(values["evaluated"]) == counts["episodes"], "Wrong native assigned quota")
    names = dict(horizon_ends="horizons", right_points="right", left_points="left")
    for key in keys: require(int(values[names.get(key, key)]) == counts[key], "Summary/episode counts differ")


def evaluation(inputs, path, binding, step, trained, plan, root):
    measured = inputs.json(path)
    require(measured["status"] == "ok", "Evaluation failed or wasn't executed")
    if binding["environment"] != "connect4cnn":
        require(measured["prepare_only"] is False, "Prepared inputs aren't completed policy evaluations")
    suite_path = inputs.track(root / binding["suite"])
    binary = plan["binaries"][binding["environment"]][binding["binary_family"]]
    require(measured["suite_sha256"] == sha(suite_path)
            and measured["training_ini_sha256"] == binding["config_sha256"]
            and measured["family"] == binding["policy_family"]
            and measured["binary_sha256"] == binary["sha256"], "Evaluation inputs don't match binding")
    require(step in trained["checkpoints"] and Path(measured["checkpoint"]).resolve() == trained["checkpoints"][step]
            and measured["checkpoint_sha256"] == sha(trained["checkpoints"][step])
            and type(measured["parameters"]) is int and measured["parameters"] == trained["parameters"], "Wrong checkpoint/parameters")
    require(measured["command"] == [binary["path"], "eval_exact", "--headless"], "Wrong evaluation command")
    require(Path(measured["cwd"]).resolve() == path.parent.resolve(), "Evaluation result refers to another cwd")
    directory = path.parent
    require(sha(inputs.track(directory / "training.ini")) == binding["config_sha256"], "Copied training config changed")
    require(sha(inputs.track(directory / "suite/suite.json")) == sha(suite_path), "Copied suite changed")
    for name, digest in bridge.hashes(suite_path.parent).items():
        require(sha(inputs.track(directory / "suite" / name)) == digest, "Copied suite input changed")
    require(sha(inputs.track(directory / "config/default.ini")) == measured["effective_ini_sha256"], "Effective config changed")
    ev = bridge.adapter(binding["environment"])
    loaded = ev.load_suite(suite_path)
    suite = loaded if binding["environment"] == "connect4cnn" else loaded[0]
    actual = ini(directory / "config/default.ini")
    expected = effective_config(ev, ini(directory / "training.ini"), suite, binding["environment"],
                                trained["checkpoints"][step], directory, actual.getint("base", "cudagraphs"))
    require(normalized(actual) == normalized(expected), "Effective evaluation configuration differs from bound recipe")
    env_path = inputs.track(directory / f"config/{binding['environment']}.ini")
    require(env_path.stat().st_size == 0, "Secondary evaluation config overrides recipe")
    episodes = inputs.track(directory / "episodes.csv")
    require(sha(episodes) == measured["episodes_sha256"], "Episode receipts changed")
    counts = (ev.audit_csv(episodes, loaded, binding["environment"]) if binding["environment"] == "connect4cnn"
              else ev.audit_csv(episodes, *loaded))
    require(counts == measured["counts"], "Stored counts differ from raw assigned episodes")
    completion(binding["environment"], inputs.track(directory / "native.log").read_text(), counts, trained["parameters"])
    low, high = score(binding["environment"], counts)
    return dict(seconds=trained["times"][step], score_lower=low, score_upper=high,
                parameters=trained["parameters"], episodes=counts["episodes"], counts=counts,
                checkpoint_sha256=measured["checkpoint_sha256"])


def write_csv(path, rows, fields):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_viewer(path, result):
    """Embed audited values verbatim; no CDN, policy execution or new statistics."""
    template = VIEWER.read_text()
    require(template.count("__DATA__") == template.count("__VIEWER_JS__") == 1,
            "Wrong viewer template markers")
    payload = json.dumps(result, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c")
    payload = payload.replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = template.replace("__VIEWER_JS__", VIEWER_JS.read_text()).replace("__DATA__", payload)
    with path.open("x") as stream: stream.write(page)


def report(collection, output):
    require(not output.exists(), "Use a fresh report directory")
    inputs = Inputs(); value = inputs.json(collection)
    require(value["schema"] == SCHEMA and value["purpose"] == "development"
            and value["gpu_execution_authorized"] is False, "Wrong collection/claim gate")
    plan_path = inputs.track(location(collection.parent, value["plan"]))
    require(sha(plan_path) == value["plan_sha256"], "Binding plan changed")
    plan = checked_plan(plan_path); root = plan_path.parent
    require(root.resolve() not in output.resolve().parents, "Report output overlaps the frozen evaluation packet")
    jobs, expected = allocation(plan)
    readers = [Path(__file__), VIEWER, VIEWER_JS, Path(audit.__file__), Path(bridge.__file__), Path(bridge.panel.__file__),
               Path(claim_module.__file__), Path(frontier_module.__file__)]
    readers.extend(Path(bridge.adapter(task).__file__) for task in plan["tasks"])
    reader_hashes = {str(path.resolve()): sha(inputs.track(path)) for path in readers}
    require(all(type(r["job_index"]) is int for r in value["training"])
            and all(type(r["binding_index"]) is int and type(r["steps"]) is int for r in value["evaluations"]),
            "Collection indices/steps must be integers")
    require([r["job_index"] for r in value["training"]] == jobs, "Training job allocation changed")
    require([(r["binding_index"], r["steps"]) for r in value["evaluations"]] == expected, "Evaluation allocation changed")
    trained, failures, points = {}, [], []
    for record in value["training"]:
        index = record["job_index"]
        if record["receipt"] is None: continue
        try:
            binding = next(b for b in plan["bindings"] if b["job_index"] == index)
            trained[index] = training(inputs, location(collection.parent, record["receipt"]), binding, plan, root)
            if trained[index]["status"] != "ok": failures.append(dict(job_index=index, error="Training did not complete"))
        except (ValueError, OSError, KeyError, TypeError) as error:
            failures.append(dict(job_index=index, error=f"{type(error).__name__}: {error}"))
    if trained:
        hardware = next(iter(trained.values()))["hardware"]
        require(all(record["hardware"] == hardware for record in trained.values()), "Mixed hardware/toolchain campaign")
        intervals = sorted((record["launch_ns"], record["end_ns"]) for record in trained.values())
        require(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])), "Overlapping timed training jobs")
    for record in value["evaluations"]:
        index, step = record["binding_index"], record["steps"]
        if record["result"] is None: continue
        binding = plan["bindings"][index]
        try:
            require(binding["job_index"] in trained, "No audited training time/checkpoint")
            point = evaluation(inputs, location(collection.parent, record["result"]), binding, step,
                               trained[binding["job_index"]], plan, root)
            task, treatment, representation = condition(binding)
            points.append(dict(binding_index=index, steps=step, environment=task, training_appearance=treatment,
                               evaluation_representation=representation, model=binding["model"],
                               seed=binding["training_seed"], **point))
        except (ValueError, OSError, KeyError, TypeError) as error:
            failures.append(dict(binding_index=index, steps=step, error=f"{type(error).__name__}: {error}"))
    result = reduce(plan, points, failures)
    # Explicit identities make missing cells visible without inventing points.
    result["planned_cells"] = []
    for index, step in expected:
        binding = plan["bindings"][index]
        task, treatment, drawing = condition(binding)
        result["planned_cells"].append(dict(binding_index=index, job_index=binding["job_index"], steps=step,
            environment=task, training_appearance=treatment, evaluation_representation=drawing,
            model=binding["model"], seed=binding["training_seed"]))
    job_costs = [dict(job_index=index, status=record["status"], process_seconds=record["process_seconds"],
        checkpoint_decisions=list(record["times"]),
        process_sps=max(record["times"])/record["process_seconds"] if record["status"] == "ok" else None,
        native_sps=None, native_sps_status="not collected by this reader", hardware=record["hardware"])
        for index, record in trained.items()]
    result.update(plan_sha256=sha(plan_path), collection_sha256=sha(collection),
        timing="native post-rename checkpoint CLOCK_MONOTONIC minus process launch; includes startup/writes; excludes evaluation",
        training_processes=len(trained), accounted_training_process_seconds=math.fsum(v["process_seconds"] for v in trained.values()),
        training_cost_accounting_complete=len(trained) == len(jobs), training_job_costs=job_costs,
        qualification="development artifact audit; runtime/math/learner calibration/held-out exclusion and inference remain separate gates",
        input_sha256=inputs.hashes, reader_sources_sha256=reader_hashes,
        numpy_version=np.__version__, reporter_sha256=sha(Path(__file__)))
    inputs.verify(); checked_plan(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "analysis.json", result, exclusive=True)
    write_viewer(output / "curves.html", result)
    fields = ["environment", "training_appearance", "evaluation_representation", "model", "steps", "seed",
              "seconds", "score_lower", "score_upper", "parameters", "episodes", "binding_index"]
    write_csv(output / "observations.csv", points, fields)
    write_csv(output / "means.csv", result["means"], fields[:-1] + ["seeds", "metric", "frontier_seconds_lower", "frontier_seconds_upper"])
    write_csv(output / "missing.csv", result["missing"], ["binding_index", "steps"])
    lines = ["# Pixel learning curves — descriptive audit", "",
        f"Status: **{result['status']}**. Observed {len(points)}/{len(expected)} cells; {len(failures)} failure records.", "",
        "[Interactive full curves](curves.html) · [Every observation](observations.csv) · [Paired-seed means](means.csv) · [Missing cells](missing.csv)", "",
        "All games and drawings stay separate. Means require every assigned training seed; incomplete conditions have no frontier flags.",
        "Pong bounds reflect deterministic censoring, not confidence intervals. Lower/upper envelope flags aren't dominance certificates.",
        "All checkpoint declines and dominated observations remain. No best-seed or best-test-checkpoint selection.",
        "Training time is post-rename monotonic completion minus process launch, including startup/writes. Evaluation is excluded.",
        "One training-process cost is counted per job, even when its weights target many drawings.",
        "No certified superiority, unseen-game transfer or SOTA claim. Inspect analysis.json for all failures and source/input receipts.", "",
        "| Game | Training appearance | Evaluation drawing | Metric | Cells | Status |",
        "|---|---|---:|---|---:|---|"]
    for group in result["conditions"]:
        lines.append(f"| {group['environment']} | {group['training_appearance']} | {group['evaluation_representation']} | "
                     f"{group['metric']} | {group['observed']}/{group['expected']} | {group['status']} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report"); p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.plan, args.out) if args.command == "prepare" else report(args.collection.resolve(), args.out)
    summary = dict(status=value.get("status", "prepared-not-executed"), gpu_executed=False)
    if args.command == "prepare":
        summary.update(planned_training_jobs=len(value["training"]), planned_evaluations=len(value["evaluations"]))
    else:
        summary.update(observations=len(value["points"]), missing=len(value["missing"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
