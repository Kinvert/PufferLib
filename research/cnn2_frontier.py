"""Build observed held-out frontiers from saved CNN2 and reference evaluations."""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
campaign = Path(sys.argv[1]).resolve()
out = campaign / "analysis"
trials = {int(r["index"]): r for r in csv.DictReader((campaign / "results.csv").open())}
evaluations = list(csv.DictReader((out / "evaluations.csv").open()))
points = []
for r in evaluations:
    trial = trials[int(r["index"])]
    checkpoint = campaign / r["checkpoint"]
    final = int(r["steps"]) == int(trial["steps"])
    points.append(dict(family="Flex", name=r["name"], seed=73, steps=int(r["steps"]),
        wins=float(r["win_rate"]), seconds=checkpoint.stat().st_mtime - int(trial["run_id"].split("_")[1]) / 1000,
        native_seconds=float(trial["cost"]) if final else None,
        native_sps=float(trial["native_avg_sps"]) if final else None,
        params=int(r["params"]), kind="final" if final else "winner checkpoint",
        games=int(r["games"]), architecture=json.loads(trial["architecture_json"]),
        training_wins=float(trial["score"]) if final else None))
assert sum(p["kind"] == "final" for p in points) == len(trials) == 128
for run in ("compare.2s__8t8l", "compare.l6d5sbk2"):
    for r in csv.DictReader((ROOT / "research/results/connect4cnn" / run / "results.csv").open()):
        family = {"nature_cnn": "Nature", "impala_cnn": "IMPALA", "impoola_cnn": "Impoola"}[r["variant"]]
        points.append(dict(family=family, name=f"{family} seed {r['seed']}", seed=int(r["seed"]),
            steps=int(r["steps"]), wins=float(r["win_rate"]), seconds=float(r["checkpoint_wall_s"]),
            native_seconds=float(r["native_uptime_s"]) if int(r["steps"]) == 13279232 else None,
            native_sps=float(r["native_avg_sps"]), params=int(r["params"]), kind="reference checkpoint",
            games=int(r["games"]), architecture={}, training_wins=None))
means = []
for family in ("Nature", "IMPALA", "Impoola"):
    for steps in sorted({p["steps"] for p in points if p["family"] == family}):
        group = [p for p in points if p["family"] == family and p["steps"] == steps]
        means.append({**group[0], "name": family + " three-seed mean", "seed": "mean",
            "wins": statistics.mean(p["wins"] for p in group),
            "seconds": statistics.mean(p["seconds"] for p in group),
            "native_sps": statistics.mean(p["native_sps"] for p in group),
            "min_wins": min(p["wins"] for p in group), "max_wins": max(p["wins"] for p in group)})


def frontier(group, axis="seconds"):
    return sorted([p for p in group if not any(q[axis] <= p[axis] and q["wins"] >= p["wins"]
        and (q[axis] < p[axis] or q["wins"] > p["wins"]) for q in group)], key=lambda p: p[axis])


fronts = []
budgets = []
for scope in ("final_only", "with_winner_curve"):
    for view in ("seed73", "mean"):
        group = [p for p in points if p["family"] == "Flex" and
                 (scope == "with_winner_curve" or p["kind"] == "final")] + (
            [p for p in points if p["family"] != "Flex" and p["seed"] == 73] if view == "seed73" else means)
        for axis in ("seconds", "steps"):
            for family in ("Flex", "Nature", "IMPALA", "Impoola", "Combined"):
                front = frontier(group if family == "Combined" else [p for p in group if p["family"] == family], axis)
                fronts += [dict(scope=scope, view=view, axis=axis, frontier_family=family, **p) for p in front]
        for seconds in (30, 60, 90, 120, 150, 300, 650, 1250):
            for family in ("Flex", "Nature", "IMPALA", "Impoola"):
                eligible = [p for p in group if p["family"] == family and p["seconds"] <= seconds]
                budgets.append(dict(scope=scope, view=view, budget_seconds=seconds, family=family,
                    best_observed_wins=max((p["wins"] for p in eligible), default=None)))

data = dict(points=points, means=means, frontiers=fronts, budget_table=budgets,
    campaign=campaign.name, note="Observed checkpoints only; curves do not establish performance between measurements.")
(out / "frontier.json").write_text(json.dumps(data, indent=2) + "\n")
for name, rows in (("observations.csv", points), ("frontiers.csv", fronts), ("budget_table.csv", budgets)):
    with (out / name).open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in r.items()} for r in rows])
template = (ROOT / "research/cnn2_frontier.html").read_text()
(out / "frontier.html").write_text(template.replace("__FRONTIER_DATA__", json.dumps(data).replace("</", "<\\/")))
print("Observed time frontier, seed 73:")
for p in fronts:
    if p["scope"] == "final_only" and p["view"] == "seed73" and p["axis"] == "seconds" and p["frontier_family"] == "Combined":
        print(f"{p['family']:8} {p['name']:20} {p['kind']:19} {p['steps']:9} {p['seconds']:8.2f}s {p['wins']:7.2%}")
print("All 128 final checkpoints included; total evaluated Flex points:", sum(p["family"] == "Flex" for p in points))
