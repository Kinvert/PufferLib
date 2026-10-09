#!/usr/bin/env python3
"""Check committed summary exports and describe full checkpoint frontiers.

Scalar/artifact analysis only. No native execution, GPU query, neural math,
continuation, source-packet audit or statistical dominance certification.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

DRAWINGS = dict(connect4cnn=10, pongcnn=7, flappycnn=7, breakoutcnn=5, snakecnn=6, mazecnn=6)
FILES = ("snapshot-summary.json", "training-sps.csv", "checkpoint-sps.csv", "native-bin-mean-sps.csv",
         "per-environment-checkpoints.csv", "per-environment-final.csv", "normalized-final-feedback.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), "Nonfinite " + key)
    return value


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def identity(row):
    return row["stage"], row["environment"], row["candidate"], int(row["seed"])


def point_key(row):
    return row["environment"], row["candidate"], int(row["decisions"])


def dominates(a, b, left="score_lower", right="score_lower"):
    return (a["seconds"] <= b["seconds"] and a[left] >= b[right]
            and (a["seconds"] < b["seconds"] or a[left] > b[right]))


def analyze(root):
    snapshot = json.loads((root / FILES[0]).read_text())
    training, checkpoints, native, points, finals, aggregate = [rows(root / name) for name in FILES[1:]]
    models = {r["candidate"] for r in aggregate}
    require(len(models) == len(aggregate) == 14, "Expected all 12 candidates and two references")
    require(len(training) == snapshot["completed_training_jobs"] == 192, "Incomplete training export")
    require(len(checkpoints) == snapshot["completed_checkpoints"] == 768, "Incomplete checkpoint export")
    require(len(native) == snapshot["native_metric_bins"] == 3520, "Incomplete native metric export")
    require(len(snapshot["panels"]) == 14, "Incomplete stage coverage")
    for stage in snapshot["panels"]:
        require(stage["status"] == "ok"
                and stage["completed_training_jobs"] == stage["planned_training_jobs"]
                and stage["completed_drawing_evaluations"] == stage["planned_drawing_evaluations"]
                and stage["completed_repeat_eager_checks"] == 2 * stage["completed_training_jobs"],
                "Incomplete stage: " + stage["stage"])
    require(sum(s["completed_training_jobs"] for s in snapshot["panels"]) == len(training), "Stage/job count mismatch")
    require(sum(s["completed_drawing_evaluations"] for s in snapshot["panels"]) == 5248, "Drawing count mismatch")

    jobs = {}
    for row in training:
        key = identity(row)
        require(key not in jobs, "Duplicate training identity")
        require(row["environment"] in DRAWINGS and row["candidate"] in models, "Unknown game/model")
        seconds = number(row, "process_wall_seconds"); decisions = int(row["decisions"])
        require(seconds > 0 and decisions > 0 and int(row["parameters"]) > 0, "Invalid job")
        require(same(number(row, "process_sps"), decisions / seconds), "Wrong process SPS")
        expected = {63173,63174} if row["stage"] == "calibration" else {64173,64174}
        require(int(row["seed"]) in expected, "Unexpected seed phase")
        jobs[key] = row

    times = {}; per_job = collections.defaultdict(list)
    for row in checkpoints:
        key = identity(row); step = int(row["decisions"])
        require(key in jobs and (key, step) not in times, "Unknown/duplicate checkpoint")
        seconds = number(row, "checkpoint_seconds")
        require(0 < seconds <= number(jobs[key], "process_wall_seconds"), "Checkpoint outside process")
        require(same(number(row, "cumulative_checkpoint_sps"), step / seconds), "Wrong checkpoint SPS")
        require(len(row["checkpoint_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["checkpoint_sha256"]), "Malformed weight hash")
        times[key, step] = seconds; per_job[key].append(row)
    require(set(per_job) == set(jobs), "Missing checkpoint job")
    for key, values in per_job.items():
        values.sort(key=lambda r:int(r["decisions"])); last_step, last_time = 0, 0.0
        total = int(jobs[key]["decisions"])
        require([int(r["decisions"]) for r in values] == [total*i//4 for i in range(1,5)], "Schedule mismatch")
        for row in values:
            step, seconds = int(row["decisions"]), number(row, "checkpoint_seconds")
            require(seconds > last_time and int(row["interval_decisions"]) == step-last_step
                    and same(number(row,"interval_seconds"),seconds-last_time)
                    and same(number(row,"interval_checkpoint_sps"),(step-last_step)/(seconds-last_time)),
                    "Invalid interval receipt calculation")
            last_step, last_time = step, seconds

    bins = collections.defaultdict(list)
    for row in native:
        key = identity(row); require(key in jobs, "Unknown native bin job")
        for name in ("mean_agent_steps", "native_bin_mean_sps", "native_bin_mean_uptime"):
            number(row, name)
        bins[key].append(int(row["bin_index"]))
    require(set(bins) == set(jobs) and all(sorted(v) == list(range(len(v))) for v in bins.values()), "Native bin coverage mismatch")

    observed = {}; by_game = collections.defaultdict(list)
    require(len(points) == 336 and len(finals) == 84, "Incomplete game/model/checkpoint coverage")
    for row in points:
        key = point_key(row); game, model, step = key
        require(key not in observed and game in DRAWINGS and model in models, "Unknown/duplicate point")
        require(int(row["drawings"]) == DRAWINGS[game] and int(row["training_seeds"]) == 2, "Drawing/seed coverage mismatch")
        paired = [k for k in jobs if k[0] != "calibration" and k[1:3] == (game,model)]
        require(len(paired) == 2 and {k[3] for k in paired} == {64173,64174}, "Missing matched training pair")
        require(same(number(row,"mean_checkpoint_seconds"), math.fsum(times[k,step] for k in paired)/2), "Mean checkpoint cost mismatch")
        lo, hi = number(row,"score_lower"), number(row,"score_upper")
        require(lo <= hi and (game == "pongcnn" or lo == hi), "Invalid score bounds")
        p = dict(environment=game,candidate=model,decisions=step,seconds=number(row,"mean_checkpoint_seconds"),
                 score_lower=lo,score_upper=hi,metric=row["metric"])
        observed[key] = row; by_game[game].append(p)
    require(set(by_game) == set(DRAWINGS), "Missing game")
    for game, values in by_game.items():
        require(len(values) == 56 and {p["candidate"] for p in values} == models, "Missing model/checkpoint")
        require(len({p["metric"] for p in values}) == 1, "Mixed game metrics")
        budgets = {int(j["decisions"]) for j in jobs.values() if j["environment"] == game}
        require(len(budgets) == 1, "Unequal game budgets")
        total = next(iter(budgets))
        require(all({p["decisions"] for p in values if p["candidate"] == m} == {total*i//4 for i in range(1,5)}
                    for m in models), "Unequal checkpoint schedules")
        for p in values:
            p["frontier_lower_endpoint"] = not any(dominates(q,p) for q in values)
            p["frontier_upper_endpoint"] = not any(dominates(q,p,"score_upper","score_upper") for q in values)
            p["dominated_using_nonoverlapping_bounds"] = any(dominates(q,p,"score_lower","score_upper") for q in values)
    final_map = {}
    for row in finals:
        key = point_key(row); require(key not in final_map and observed.get(key) == row, "Changed/duplicate final row")
        require(int(row["decisions"]) == max(k[2] for k in observed if k[:2] == key[:2]), "Nonfinal checkpoint selected")
        final_map[key] = row
    aggregate_points = []
    for row in aggregate:
        model = row["candidate"]; score = number(row,"equal_game_normalized_final_score")
        cost = number(row,"total_checkpoint_training_seconds")
        require(0 <= score <= 1 and same(cost,2*math.fsum(number(r,"mean_checkpoint_seconds")
            for r in finals if r["candidate"] == model)), "Aggregate cost/score invalid")
        aggregate_points.append(dict(candidate=model,seconds=cost,score_lower=score,score_upper=score))
    for p in aggregate_points:
        p["frontier"] = not any(dominates(q,p) for q in aggregate_points)
    comparison = []
    for game in DRAWINGS:
        q,n = [next(r for r in finals if r["environment"] == game and r["candidate"] == model)
               for model in ("quality-reference","nature-cnn")]
        qjobs,njobs = [[r for r in training if r["stage"] == "references" and r["environment"] == game
                       and r["candidate"] == model] for model in ("quality-reference","nature-cnn")]
        pooled = lambda jobs: sum(int(r["decisions"]) for r in jobs)/math.fsum(number(r,"process_wall_seconds") for r in jobs)
        comparison.append(dict(environment=game,quality_seconds=number(q,"mean_checkpoint_seconds"),
            nature_seconds=number(n,"mean_checkpoint_seconds"),quality_lower=number(q,"score_lower"),
            quality_upper=number(q,"score_upper"),nature_lower=number(n,"score_lower"),nature_upper=number(n,"score_upper"),
            checkpoint_time_reduction_percent=100*(1-number(q,"mean_checkpoint_seconds")/number(n,"mean_checkpoint_seconds")),
            quality_process_sps=pooled(qjobs),nature_process_sps=pooled(njobs)))
    return dict(status="export-consistency-passed",input_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES},
        models=sorted(models),training_jobs=len(jobs),checkpoints=len(times),points=len(points),native_bins=len(native),
        raw_packet_audited=False,aggregate_scores_recomputed_from_episode_rows=False,statistical_dominance_qualified=False,
        qualification="Two-seed/all-drawing exported means; endpoint frontiers are descriptive. Pong bounds are censoring, not confidence intervals.",
        aggregate_final=aggregate_points,reference_comparison=comparison,by_game=dict(by_game))


def write_csv(path, values):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)


def report(value):
    text=["# Independent analysis of committed 5090 exports", "",
          "All seven export tables pass cross-file consistency checks. This is not an independent raw-packet audit.", "",
          "Final fixed-anchor aggregate frontier (all 14 models retained in analysis.json):", "",
          "| Model | Score | Total paired training seconds |", "|---|---:|---:|"]
    for p in sorted(value["aggregate_final"],key=lambda r:r["seconds"]):
        if p["frontier"]: text.append(f"| {p['candidate']} | {p['score_lower']:.6f} | {p['seconds']:.3f} |")
    text += ["", "Aggregate scores are supplied audited-export values. Clipping occurs before averaging;", "they cannot generally be reconstructed from game means alone.", "", "Quality versus Nature at final budgets:", "",
             "| Game | Quality score/bounds | Nature score/bounds | Quality time reduction | Quality / Nature process SPS |", "|---|---:|---:|---:|---:|"]
    def score(lo,hi): return f"{lo:.4f}" if lo==hi else f"{lo:.4f}–{hi:.4f}"
    for p in value["reference_comparison"]:
        text.append(f"| {p['environment']} | {score(p['quality_lower'],p['quality_upper'])} | {score(p['nature_lower'],p['nature_upper'])} | {p['checkpoint_time_reduction_percent']:.2f}% | {p['quality_process_sps']:,.0f} / {p['nature_process_sps']:,.0f} |")
    text += ["", "Complete checkpoint frontiers follow: every model's four scheduled checkpoints participates,", "including declines and duplicate trials. Drawings are averaged equally within each game,", "then the two paired seeds are averaged. Time is mean once-per-job checkpoint cost.", "No drawings or seeds are treated as independent training replicas."]
    for game,points in value["by_game"].items():
        text += ["", "## "+game, "", "Lower-endpoint descriptive frontier:", "",
                 "| Model | Decisions | Mean train seconds | Score/bounds |", "|---|---:|---:|---:|"]
        for p in sorted(points,key=lambda r:(r["seconds"],r["candidate"])):
            if p["frontier_lower_endpoint"]: text.append(f"| {p['candidate']} | {p['decisions']:,} | {p['seconds']:.3f} | {score(p['score_lower'],p['score_upper'])} |")
        if game=="pongcnn": text += ["", "The lower-endpoint frontier uses conservative censoring scores. It does not certify", "true win-rate dominance; overlapping lower/upper bounds remain unresolved. The CSV", "also retains the upper-endpoint frontier and dominance using nonoverlapping bounds."]
    text += ["", "Adaptive discovery seeds, two seeds per model, no independent episode/appearance", "rows here and no full-trainer numerical qualification: no publication/SOTA claim.", ""]
    return "\n".join(text)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("exports",type=Path); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); value=analyze(args.exports.resolve()); args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"analysis.json").write_text(json.dumps(value,indent=2)+"\n")
    (args.out/"REPORT.md").write_text(report(value))
    write_csv(args.out/"checkpoint-frontiers.csv",[p for points in value["by_game"].values() for p in points])
    write_csv(args.out/"reference-comparison.csv",value["reference_comparison"])
    write_csv(args.out/"aggregate-final-frontier.csv",value["aggregate_final"])
    from learning_exports_view import write
    write(args.out/"curves.html",value)
    print("Export consistency passed; descriptive frontiers written to",args.out)


if __name__=="__main__": main()
