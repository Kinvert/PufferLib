#!/usr/bin/env python3
"""Independent scalar checks of the transported 45-observation discovery prefix.

No native imports/execution, GPU queries, CNN computation or campaign restart.
Point rows are exported episode aggregates, not independently received episodes.
"""
import argparse
import collections
import configparser
import csv
import hashlib
import json
import math
from pathlib import Path

from learning_exports import DRAWINGS, dominates, number, require, rows, same

FILES = ("analysis.json", "allocation-result.json", "parent-search-result.json",
         "continuation-manifest.json", "completed-history.tsv", "last-native-proposal.json",
         "frozen-search-recipe.ini", "review/combined/curves.json", "review/combined/training.csv",
         "exports/per-environment-checkpoints.csv", "exports/per-environment-final.csv",
         "exports/normalized-final-feedback.csv", "sps-report/training-sps.csv",
         "sps-report/checkpoint-sps.csv", "sps-report/native-bin-mean-sps.csv")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def history(entries):
    return "PUFFER_CROSS_GAME_V1 18\n" + "".join(" ".join(
        ["0", format(e["feedback"]["score"], ".17g"), format(e["feedback"]["cost"], ".17g")]
        + [format(x, ".9g") for x in e["proposal"]["normalized"]]) + "\n" for e in entries)


def graph(policy):
    keys = ["cnn_depth", "cnn_projection", "cnn_global_pool"]
    keys += [f"cnn_{name}_{i}" for i in range(1, policy["cnn_depth"] + 1)
             for name in ("channels", "kernel", "stride", "pool", "residual")]
    return {"encoder": 4, **{k: policy[k] for k in keys}}


def check_history(parent, allocation, replay, ledger):
    require(parent["status"] == "ok" and allocation["status"] == "failed",
            "Allocation failure or successful ancestry was relabeled")
    entries = parent["trials"] + allocation["trials"]
    require(len(parent["trials"]) == 12 and len(allocation["trials"]) == 33
            and [e["index"] for e in entries] == list(range(45)), "Incomplete/reordered prefix")
    require(history(entries[:12]) == parent["history"] and history(entries) == ledger == allocation["history"],
            "Changed history: every completed score/cost/vector must be retained")
    require(replay["trial"] == replay["observations_replayed"] == replay["gp_observations"]
            == replay["success_observations"] == 45
            and replay["failure_observations"] == replay["random"] == 0, "Incomplete final replay")
    for i, e in enumerate(entries):
        require(len(e["proposal"]["normalized"]) == 18 and graph(e["proposal"]["policy"]) == e["architecture"],
                "Effective architecture disagrees with native proposal")
        duplicate = e["duplicate_of"]
        earlier = [j for j in range(i) if entries[j]["architecture"] == e["architecture"]]
        require((duplicate is None and not earlier) or duplicate in earlier, "Incorrect duplicate flag")
    return entries


def audit(root):
    receipt = read(root / "analysis.json")
    allocation, parent = [read(root / name) for name in ("allocation-result.json", "parent-search-result.json")]
    require(sha(root / "allocation-result.json") == receipt["result_sha256"], "Changed failed allocation")
    require(sha(root / "continuation-manifest.json") == receipt["continuation_manifest_sha256"], "Changed manifest")
    require(receipt["allocation_status"] == "failed" and receipt["failed_trial_one_based"] == 46
            and not receipt["partial_feedback_submitted"], "Incorrect prefix qualification")
    available, unavailable = [], []
    for name, digest in receipt["files_sha256"].items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), "Hash path outside packet")
        if path.is_file():
            require(sha(path) == digest, "Changed transported receipt: " + name)
            available.append(name)
        else:
            unavailable.append(name)
    entries = check_history(parent, allocation, read(root / "last-native-proposal.json"),
                            (root / "completed-history.tsv").read_text())
    models = [e["name"] for e in entries] + ["quality-reference", "nature-cnn"]
    require(len(set(models)) == 47, "Duplicate model identities")
    curve = read(root / "review/combined/curves.json")
    require(set(curve["model_catalog"]) == set(models) and not curve["missing"] and not curve["failures"]
            and curve["allocation_status"] == "failed" and curve["excluded_partial_trial_one_based"] == 46,
            "Incomplete or relabeled comparison")
    conditions = {(c["environment"], c["evaluation_representation"]): c for c in curve["conditions"]}
    require(set(conditions) == {(g, d) for g, n in DRAWINGS.items() for d in range(n)}, "Drawing coverage changed")

    training = rows(root / "sps-report/training-sps.csv")
    jobs = {}; comparison = {}
    for row in training:
        key = row["stage"], row["environment"], row["candidate"], int(row["seed"])
        require(key not in jobs and key[1] in DRAWINGS, "Duplicate/unknown training job")
        require(number(row, "process_wall_seconds") > 0 and int(row["decisions"]) > 0
                and same(number(row, "process_sps"), int(row["decisions"]) / number(row, "process_wall_seconds")),
                "Wrong process SPS")
        jobs[key] = row
        if key[0] != "calibration" and key[2] in models:
            short = key[1:]
            require(short not in comparison and key[3] in (64173, 64174), "Unpaired comparison job")
            comparison[short] = row
    partial = [r for k, r in jobs.items() if k[2] not in models]
    require(len(jobs) == 594 and len(comparison) == 564 and len(partial) == 6
            and {r["candidate"] for r in partial} == {"quiet-owl-46"}, "Partial job evidence lost or charged to prefix")
    bins = collections.defaultdict(list)
    native = rows(root / "sps-report/native-bin-mean-sps.csv")
    for row in native:
        key = row["stage"], row["environment"], row["candidate"], int(row["seed"])
        require(key in jobs, "Unknown native bin job")
        for k in ("mean_agent_steps", "native_bin_mean_sps", "native_bin_mean_uptime"):
            number(row, k)
        bins[key].append(int(row["bin_index"]))
    require(len(native) == 10890 and set(bins) == set(jobs)
            and all(sorted(v) == list(range(len(v))) for v in bins.values()), "Native bin coverage lost")
    cp = {}; schedules = collections.defaultdict(list)
    for row in rows(root / "sps-report/checkpoint-sps.csv"):
        key = row["stage"], row["environment"], row["candidate"], int(row["seed"])
        step, seconds = int(row["decisions"]), number(row, "checkpoint_seconds")
        require(key in jobs and (key, step) not in cp and 0 < seconds <= number(jobs[key], "process_wall_seconds"),
                "Checkpoint identity/time invalid")
        require(same(number(row, "cumulative_checkpoint_sps"), step / seconds), "Wrong checkpoint SPS")
        cp[key, step] = seconds; schedules[key].append(row)
    require(len(cp) == 2376 and set(schedules) == set(jobs), "Checkpoint coverage lost")
    for key, values in schedules.items():
        last_step, last_seconds = 0, 0
        values.sort(key=lambda r: int(r["decisions"]))
        require([int(r["decisions"]) for r in values] == [int(jobs[key]["decisions"]) * i // 4 for i in range(1, 5)],
                "Unequal checkpoint schedules")
        for row in values:
            step, seconds = int(row["decisions"]), float(row["checkpoint_seconds"])
            require(seconds > last_seconds and int(row["interval_decisions"]) == step - last_step
                    and same(number(row, "interval_seconds"), seconds - last_seconds)
                    and same(number(row, "interval_checkpoint_sps"), (step - last_step) / (seconds - last_seconds)),
                    "Wrong interval SPS")
            last_step, last_seconds = step, seconds
    combined_jobs = rows(root / "review/combined/training.csv")
    require(len(combined_jobs) == 564, "Comparison training table incomplete")
    covered = set()
    for row in combined_jobs:
        key = row["environment"], row["candidate"], int(row["seed"])
        require(key in comparison and key not in covered, "Duplicate/unassigned comparison training")
        job = comparison[key]; covered.add(key)
        require(int(row["parameters"]) == int(job["parameters"])
                and same(number(row, "train_seconds"), number(job, "process_wall_seconds"))
                and same(number(row, "process_sps"), number(job, "process_sps")), "Training tables disagree")

    groups = collections.defaultdict(list); seeds = collections.defaultdict(list); cells = {}
    for p in curve["points"]:
        g, m, seed, step, d = p["environment"], p["model"], p["seed"], p["steps"], p["evaluation_representation"]
        key = g, m, seed, step, d
        require(key not in cells and (g, m, seed) in comparison and (g, d) in conditions, "Duplicate/unassigned point")
        job = comparison[g, m, seed]
        jobkey = job["stage"], g, m, seed
        require((jobkey, step) in cp and same(p["seconds"], cp[jobkey, step])
                and p["parameters"] == int(job["parameters"]) and p["episodes"] == 33, "Point/job binding mismatch")
        require(math.isfinite(p["score_lower"]) and math.isfinite(p["score_upper"])
                and p["score_lower"] <= p["score_upper"]
                and (g == "pongcnn" or p["score_lower"] == p["score_upper"]), "Invalid score bounds")
        cells[key] = p; groups[g, m, step].append(p); seeds[g, m, seed, step].append(p)
    require(len(cells) == 15416 and len(groups) == 1128 and len(seeds) == 2256, "Point coverage incomplete")
    budgets = {g: {int(r["decisions"]) for (game, _, _), r in comparison.items() if game == g} for g in DRAWINGS}
    require(all(len(v) == 1 for v in budgets.values()), "Per-game budgets unequal")
    budgets = {g: next(iter(v)) for g, v in budgets.items()}
    for g in DRAWINGS:
        for m in models:
            for s in (64173, 64174):
                for i in range(1, 5):
                    values = seeds[g, m, s, budgets[g] * i // 4]
                    require({p["evaluation_representation"] for p in values} == set(range(DRAWINGS[g]))
                            and len(values) == DRAWINGS[g], "Missing assigned drawing/seed/checkpoint")

    means = {(p["environment"], p["model"], p["steps"], p["evaluation_representation"]): p for p in curve["means"]}
    require(len(means) == len(curve["means"]) == 7708, "Paired-mean coverage incomplete")
    for (g, m, step), values in groups.items():
        for d in range(DRAWINGS[g]):
            pair = [p for p in values if p["evaluation_representation"] == d]
            avg = means[g, m, step, d]
            require(avg["seeds"] == 2 and all(same(avg[k], math.fsum(p[k] for p in pair) / 2)
                    for k in ("seconds", "score_lower", "score_upper")), "Paired means changed")

    by_game = collections.defaultdict(list); exported = rows(root / "exports/per-environment-checkpoints.csv")
    observed = {}
    for row in exported:
        key = g, m, step = row["environment"], row["candidate"], int(row["decisions"])
        require(key not in observed and key in groups, "Duplicate/unknown game export")
        values = groups[key]
        require(int(row["drawings"]) == DRAWINGS[g] and int(row["training_seeds"]) == 2
                and same(number(row, "mean_checkpoint_seconds"), math.fsum(p["seconds"] for p in values) / len(values))
                and all(same(number(row, k), math.fsum(p[k] for p in values) / len(values))
                        for k in ("score_lower", "score_upper")), "Game export disagrees with points")
        observed[key] = row
        by_game[g].append(dict(environment=g, candidate=m, decisions=step,
            seconds=number(row, "mean_checkpoint_seconds"), score_lower=number(row, "score_lower"),
            score_upper=number(row, "score_upper"), metric=row["metric"]))
    require(len(observed) == 1128, "Missing game export")
    finals = rows(root / "exports/per-environment-final.csv")
    require(len(finals) == 282 and {(r["environment"], r["candidate"]) for r in finals}
            == {(g, m) for g in DRAWINGS for m in models}, "Final coverage missing")
    for r in finals:
        require(int(r["decisions"]) == budgets[r["environment"]]
                and observed[r["environment"], r["candidate"], int(r["decisions"])] == r, "Nonfinal/changed selection")

    cfg = configparser.ConfigParser(interpolation=None); cfg.read(root / "frozen-search-recipe.ini")
    anchors = {g: (float(cfg["objective." + g]["offset"]), float(cfg["objective." + g]["target"])) for g in DRAWINGS}
    def normalized(p, endpoint):
        offset, target = anchors[p["environment"]]
        require(target > offset, "Invalid anchor")
        return max(0, min(1, (p[endpoint] - offset) / (target - offset)))
    aggregate, seed_scores = [], []
    for m in models:
        for i in range(1, 5):
            score = {k: math.fsum(math.fsum(normalized(p, k) for p in groups[g, m, budgets[g] * i // 4])
                     / (2 * DRAWINGS[g]) for g in DRAWINGS) / 6 for k in ("score_lower", "score_upper")}
            cost = 2 * math.fsum(float(observed[g, m, budgets[g] * i // 4]["mean_checkpoint_seconds"]) for g in DRAWINGS)
            aggregate.append(dict(candidate=m, checkpoint_fraction=i / 4, seconds=cost, **score))
        for s in (64173, 64174):
            score = {k: math.fsum(math.fsum(normalized(p, k) for p in seeds[g, m, s, budgets[g]])
                     / DRAWINGS[g] for g in DRAWINGS) / 6 for k in ("score_lower", "score_upper")}
            seed_scores.append(dict(candidate=m, seed=s, **score))
    final_aggregate = [p for p in aggregate if p["checkpoint_fraction"] == 1]
    supplied = rows(root / "exports/normalized-final-feedback.csv")
    require(len(supplied) == 47 and {r["candidate"] for r in supplied} == set(models), "Missing aggregate export")
    for r in supplied:
        p = next(p for p in final_aggregate if p["candidate"] == r["candidate"])
        require(same(p["score_lower"], number(r, "equal_game_normalized_final_score"))
                and same(p["seconds"], number(r, "total_checkpoint_training_seconds")), "Aggregate clipping/cost mismatch")
    for e in entries:
        p = next(p for p in final_aggregate if p["candidate"] == e["name"])
        require(same(p["score_lower"], e["feedback"]["score"])
                and same(p["score_upper"], e["feedback"]["score_upper"])
                and same(p["seconds"], e["feedback"]["cost"]), "Native feedback disagrees with exported points")
        require(e["feedback"]["training_jobs_charged"] == 12 and len(e["feedback"]["games"]) == 6,
                "Feedback game/cost coverage changed")
        for game in e["feedback"]["games"]:
            g = game["environment"]; values = groups[g, e["name"], budgets[g]]
            require(game["cells"] == len(values) == 2 * DRAWINGS[g]
                    and (game["offset"], game["target"]) == anchors[g]
                    and all(same(game["normalized_" + end], math.fsum(normalized(v, "score_" + end) for v in values)
                                 / len(values)) for end in ("lower", "upper")), "Per-game native feedback changed")
    for values in [aggregate, *by_game.values()]:
        for p in values:
            p["frontier_lower_endpoint"] = not any(dominates(q, p) for q in values)
            p["dominated_using_nonoverlapping_bounds"] = any(dominates(q, p, "score_lower", "score_upper") for q in values)
    for p in aggregate:
        p["final_budget_frontier"] = p["checkpoint_fraction"] == 1 and not any(dominates(q, p) for q in final_aggregate)
    graph_groups = collections.defaultdict(list)
    for e in entries:
        graph_groups[json.dumps(e["architecture"], sort_keys=True)].append(e["name"])
    duplicates = []
    for names in graph_groups.values():
        if len(names) < 2:
            continue
        selected = [p for p in final_aggregate if p["candidate"] in names]
        point_scores = [[(p["score_lower"], p["score_upper"]) for (g, m, s, step, d), p in sorted(cells.items())
                         if m == name] for name in names]
        weight_hashes = [[r["checkpoint_sha256"] for key in sorted(comparison) if key[1] == name
                         for r in schedules[(comparison[key]["stage"], *key)]] for name in names]
        duplicates.append(dict(models=names, score_span=max(p["score_lower"] for p in selected) - min(p["score_lower"] for p in selected),
            all_point_scores_identical=all(v == point_scores[0] for v in point_scores),
            all_checkpoint_hashes_identical=all(v == weight_hashes[0] for v in weight_hashes),
            min_seconds=min(p["seconds"] for p in selected), max_seconds=max(p["seconds"] for p in selected)))
    per_game_seed = []
    for (g, m, s, step), values in sorted(seeds.items()):
        if step == budgets[g]:
            per_game_seed.append(dict(environment=g, candidate=m, seed=s,
                score_lower=math.fsum(p["score_lower"] for p in values) / len(values),
                score_upper=math.fsum(p["score_upper"] for p in values) / len(values),
                worst_drawing_lower=min(p["score_lower"] for p in values),
                best_drawing_lower=max(p["score_lower"] for p in values)))
    return dict(status="transported-artifact-consistency-passed", input_sha256={n: sha(root / n) for n in FILES},
        available_receipt_hashes_verified=available, unavailable_raw_receipt_files=unavailable,
        raw_packet_audited=False, episode_rows_independently_received=False, statistical_dominance_qualified=False,
        allocation_status="failed", completed_observations=45, excluded_partial_trial=46, partial_completed_jobs=6,
        models=models, training_jobs=len(jobs), comparison_jobs=len(comparison), point_rows=len(cells),
        checkpoints=len(cp), effective_graphs=len(graph_groups), aggregate_checkpoints=aggregate,
        final_seed_scores=seed_scores, final_game_seed_scores=per_game_seed, by_game=dict(by_game), duplicates=duplicates,
        architectures=[dict(candidate=e["name"], duplicate_of=e["duplicate_of"], **e["architecture"]) for e in entries])


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("packet", type=Path); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); value = audit(args.packet.resolve()); args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "analysis.json").write_text(json.dumps(value, indent=2) + "\n")
    write_csv(args.out / "aggregate-checkpoint-frontiers.csv", value["aggregate_checkpoints"])
    write_csv(args.out / "final-seed-scores.csv", value["final_seed_scores"])
    write_csv(args.out / "final-game-seed-scores.csv", value["final_game_seed_scores"])
    write_csv(args.out / "per-game-checkpoint-frontiers.csv", [p for v in value["by_game"].values() for p in v])
    print(json.dumps({k: value[k] for k in ("status", "completed_observations", "effective_graphs", "point_rows", "allocation_status")}))


if __name__ == "__main__":
    main()
