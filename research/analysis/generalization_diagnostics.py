#!/usr/bin/env python3
"""Describe task concentration, matched-time scores and search-space blind spots.

Offline exported-scalar analysis. No CNN/reference math, GPU or native execution.
These are discovery diagnostics, not confirmatory tests or a new search objective.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import continuation_exports as intake


def latest_affordable(points, seconds):
    eligible = [p for p in points if p["seconds"] <= seconds]
    return max(eligible, key=lambda p: p["decisions"]) if eligible else None


def diagnose(root):
    checked = intake.audit(root)
    curve = intake.read(root / "review/combined/curves.json")
    entries = intake.read(root / "parent-search-result.json")["trials"] + intake.read(root / "allocation-result.json")["trials"]
    anchors = {r["environment"]: (r["offset"], r["target"]) for r in entries[0]["feedback"]["games"]}
    groups = collections.defaultdict(list)
    for p in curve["points"]:
        groups[p["environment"], p["model"], p["steps"]].append(p)
    def normalized(game, model, step, endpoint="score_lower"):
        offset, target = anchors[game]; values = groups[game, model, step]
        return math.fsum(max(0, min(1, (p[endpoint] - offset) / (target - offset))) for p in values) / len(values)
    final = {(g, m): max(step for game, model, step in groups if (game, model) == (g, m))
             for g in intake.DRAWINGS for m in checked["models"]}
    game_delta = []
    for g in intake.DRAWINGS:
        new, quality, nature = [normalized(g, m, final[g, m])
                                for m in ("quiet-owl-45", "quality-reference", "nature-cnn")]
        game_delta.append(dict(environment=g, new_lower=new, quality_lower=quality, nature_lower=nature,
                               contribution_to_mean_gain_over_quality=(new - quality) / 6))
    total_delta = math.fsum(p["contribution_to_mean_gain_over_quality"] for p in game_delta)
    for p in game_delta:
        p["fraction_of_net_gain"] = p["contribution_to_mean_gain_over_quality"] / total_delta
    leave_out = []
    for excluded in [None, *intake.DRAWINGS]:
        games = [g for g in intake.DRAWINGS if g != excluded]
        scores = [dict(candidate=m, score=math.fsum(normalized(g, m, final[g, m]) for g in games) / len(games))
                  for m in checked["models"]]
        scores.sort(key=lambda r: -r["score"])
        selected = {m: next(r["score"] for r in scores if r["candidate"] == m)
                    for m in ("quiet-owl-45", "quality-reference", "nature-cnn")}
        # Equal-score duplicate trials receive the same competition rank.
        ranks = {m: 1 + sum(r["score"] > score for r in scores) for m, score in selected.items()}
        leave_out.append(dict(excluded_game=excluded, ranks=ranks, scores=selected,
                              winner=scores[0]["candidate"], winner_score=scores[0]["score"]))
    matched = []
    for g, values in checked["by_game"].items():
        nature = next(p for p in values if p["candidate"] == "nature-cnn" and p["decisions"] == final[g, "nature-cnn"])
        for m in checked["models"]:
            p = latest_affordable([p for p in values if p["candidate"] == m], nature["seconds"])
            if p is None:
                continue
            # Select by decisions/time only: no best-score checkpoint selection.
            signed = p["score_lower"] - nature["score_lower"]
            outcome = ("higher" if signed > 1e-12 else "lower" if signed < -1e-12 else "equal")
            bound_outcome = ("higher" if p["score_lower"] > nature["score_upper"] else
                             "lower" if p["score_upper"] < nature["score_lower"] else "overlap_or_equal")
            matched.append(dict(environment=g, candidate=m, nature_seconds=nature["seconds"],
                checkpoint_decisions=p["decisions"], checkpoint_seconds=p["seconds"],
                score_lower=p["score_lower"], score_upper=p["score_upper"], nature_lower=nature["score_lower"],
                nature_upper=nature["score_upper"], lower_endpoint_comparison=outcome,
                censoring_bound_comparison=bound_outcome))
    coverage = []
    for m in checked["models"]:
        selected = [r for r in matched if r["candidate"] == m]
        coverage.append(dict(candidate=m, games_observed=len(selected),
            games_higher_lower_endpoint=sum(r["lower_endpoint_comparison"] == "higher" for r in selected),
            games_lower_lower_endpoint=sum(r["lower_endpoint_comparison"] == "lower" for r in selected),
            games_higher_nonoverlapping_bounds=sum(r["censoring_bound_comparison"] == "higher" for r in selected)))
    blind = []
    for e in entries:
        p = e["architecture"]; k, s = p["cnn_kernel_1"], p["cnn_stride_1"]
        if k < s:
            blind.append(dict(candidate=e["name"], kernel=k, stride=s, duplicate_of=e["duplicate_of"]))
    source_paths = [root / "review/combined/curves.json", root / "parent-search-result.json",
                    root / "allocation-result.json", root / "frozen-search-recipe.ini"]
    return dict(qualification="Adaptive two-seed diagnostics only; no significance/generalization certification.",
        input_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        normalized_game_contributions=game_delta, leave_one_game_out=leave_out,
        matched_nature_final_time=matched, matched_time_coverage=coverage,
        first_kernel_smaller_than_stride=blind,
        blind_spot_trials=len(blind), blind_spot_unique_graphs=len({json.dumps(e["architecture"], sort_keys=True)
            for e in entries if e["architecture"]["cnn_kernel_1"] < e["architecture"]["cnn_stride_1"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); value = diagnose(args.packet.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "diagnostics.json").write_text(json.dumps(value, indent=2) + "\n")
    intake.write_csv(args.out / "matched-time-comparisons.csv", value["matched_nature_final_time"])
    intake.write_csv(args.out / "matched-time-coverage.csv", value["matched_time_coverage"])
    intake.write_csv(args.out / "game-contributions.csv", value["normalized_game_contributions"])
    print(json.dumps(dict(blind_spot_trials=value["blind_spot_trials"], blind_spot_graphs=value["blind_spot_unique_graphs"],
        matched_time_comparisons=len(value["matched_nature_final_time"]))))


if __name__ == "__main__":
    main()
