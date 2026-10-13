"""Reproduce superseded time-slice planning; not the full-frontier analysis."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn")
INPUTS = {
    "5060_historical": ROOT / "research/results/connect4cnn/confirm.ol9tcj5k/analysis/observations.csv",
    "5090_single_seed": ROOT / "research/results/connect4cnn/compare.vtew3n12/analysis/observations.csv",
}
BUDGETS = {"5060_historical": (60, 80, 100, 120, 140), "5090_single_seed": (30, 45, 60, 75)}


def latest_available(points, budget):
    """Select by time/steps, never by evaluated quality; absence stays missing."""
    ordered = sorted(points, key=lambda p: p["steps"])
    for point in ordered:
        if not all(math.isfinite(point[k]) for k in ("steps", "seconds", "wins")):
            raise ValueError("Nonfinite observation")
        if point["steps"] < 0 or point["seconds"] < 0 or not 0 <= point["wins"] <= 1:
            raise ValueError("Invalid observation")
    if any(a["steps"] >= b["steps"] or a["seconds"] >= b["seconds"]
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Duplicate steps or nonmonotonic checkpoint times")
    eligible = [p for p in ordered if p["seconds"] <= budget]
    return eligible[-1] if eligible else None


def normal_planning(effect, sd, comparisons=9, power=.8):
    if effect <= 0 or sd <= 0 or comparisons < 1:
        raise ValueError("Positive planning inputs required")
    normal = NormalDist()
    critical = normal.inv_cdf(1 - .05 / comparisons)
    n = math.ceil(((critical + normal.inv_cdf(power)) * sd / effect) ** 2)
    power_40 = normal.cdf(effect * math.sqrt(40) / sd - critical)
    return n, power_40


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots, summaries, paired = [], [], []
    hashes = {}
    for host, path in INPUTS.items():
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f:
            raw = list(csv.DictReader(f))
        seeds = sorted({int(r["seed"]) for r in raw})
        for budget in BUDGETS[host]:
            values = {}
            for model in MODELS:
                for seed in seeds:
                    points = [dict(steps=int(r["steps"]), seconds=float(r["seconds"]),
                                   wins=float(r["wins"])) for r in raw
                              if r["variant"] == model and int(r["seed"]) == seed]
                    point = latest_available(points, budget)
                    values[model, seed] = point
                    snapshots.append(dict(host=host, budget_seconds=budget, variant=model,
                        seed=seed, available=point is not None,
                        checkpoint_steps=point["steps"] if point else None,
                        checkpoint_seconds=point["seconds"] if point else None,
                        wins=point["wins"] if point else None))
                group = [values[model, seed]["wins"] for seed in seeds if values[model, seed]]
                summaries.append(dict(host=host, budget_seconds=budget, variant=model,
                    seeds=len(seeds), available=len(group),
                    mean_wins=mean(group) if len(group) == len(seeds) else None))
            for reference in MODELS[1:]:
                differences = [values[MODELS[0], seed]["wins"] - values[reference, seed]["wins"]
                    for seed in seeds if values[MODELS[0], seed] and values[reference, seed]]
                complete = len(differences) == len(seeds)
                paired.append(dict(host=host, budget_seconds=budget, reference=reference,
                    paired_seeds=len(differences), total_seeds=len(seeds),
                    mean_delta=mean(differences) if complete else None,
                    sd_delta=stdev(differences) if complete and len(differences) > 1 else None,
                    positive_pairs=sum(d > 0 for d in differences) if complete else None))
    power = []
    for sd in (.05, .10, .15, .20):
        for effect in (.05, .10, .15):
            n, power_40 = normal_planning(effect, sd)
            power.append(dict(paired_sd=sd, true_mean_advantage=effect, null_margin=0,
                              comparisons=9, family_alpha=.05, target_power=.8,
                              normal_approx_seeds=n, normal_approx_power_at_40=power_40))
    for filename, rows in (("snapshots.csv", snapshots), ("summaries.csv", summaries),
                           ("paired_planning.csv", paired), ("power_sensitivity.csv", power)):
        write_csv(args.output / filename, rows)
    (args.output / "inputs.json").write_text(json.dumps({
        "status": "superseded_time_slice_development_planning_only", "sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "latest completed checkpoint <= budget; no interpolation or score maximization",
        "missing": "no eligible checkpoint is missing, never zero",
        "power_limit": "normal approximation with assumed SD/effect; not a coverage or power guarantee",
    }, indent=2) + "\n")
    lines = ["# Connect4 claim design: development evidence only", "",
        "**Superseded design:** these time slices and nine-contrast power calculations do not define or size the current full-frontier study. See research/CONNECT4_CLAIM_PROTOCOL.md. Historical numbers are retained, not new confirmation evidence.", "",
        "Existing results now inform design; they are not new confirmation data. Hardware/source cohorts are kept separate.",
        "Latest eligible checkpoint is selected without consulting its score. No interpolation, best-checkpoint selection, or zero imputation.",
        "", "| Cohort | Seconds | Model | Available seeds | Mean wins |", "|---|---:|---|---:|---:|"]
    for r in summaries:
        score = "missing" if r["mean_wins"] is None else f"{r['mean_wins']:.2%}"
        lines.append(f"| {r['host']} | {r['budget_seconds']} | {r['variant']} | {r['available']}/{r['seeds']} | {score} |")
    lines += ["", "## Power sensitivity — assumed effects, not observed significance", "",
        "One-sided alpha .05/9 for three references at three eligible budgets. Null margin is zero; this does not test that improvement exceeds five percentage points.",
        "| Paired SD | Assumed advantage | Approximate seeds for 80% power | Approximate power at 40 |",
        "|---:|---:|---:|---:|"]
    for r in power:
        lines.append(f"| {r['paired_sd']:.0%} | {r['true_mean_advantage']:.0%} | {r['normal_approx_seeds']} | {r['normal_approx_power_at_40']:.1%} |")
    lines += ["", "The five-seed historical variance is unstable and not a 5090 variance estimate. The single 5090 seed cannot estimate training-seed variance.",
        "Old checkpoints use a full 13.312M-decision annealing schedule and coarse checkpoint intervals. A new wall-budget protocol must validate timing/availability and cannot inherit these scores.",
        "See research/CONNECT4_CLAIM_PROTOCOL.md for the replacement full-frontier/common-learner design and open validation gates.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines))
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
