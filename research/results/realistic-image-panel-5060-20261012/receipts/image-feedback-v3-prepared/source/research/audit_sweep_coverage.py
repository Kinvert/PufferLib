"""Summarize active architecture and budget coverage in one native sweep CSV.

This is offline reporting: it does not propose candidates or launch training.
Use one campaign at a time because seed and recipe are campaign-level settings.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def summarize(path, long_budget):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "architecture_json" not in rows[0]:
        raise ValueError("Expected a nonempty sweep results.csv with architecture_json")

    by_depth = defaultdict(list)
    exact = Counter()
    shapes = set()
    for row in rows:
        architecture = json.loads(row["architecture_json"])
        depth = int(architecture["policy.cnn_depth"])
        steps = int(row["steps"])
        identity = row["architecture_sha256"]
        shapes.add(identity)
        exact[(identity, steps)] += 1
        by_depth[depth].append((row, architecture, steps))

    def frequencies(group, key):
        return dict(sorted(Counter(str(a[key]) for _, a, _ in group if key in a).items()))

    def stage_frequencies(group, name):
        return dict(sorted(Counter(
            str(a[f"policy.cnn_{name}_{stage}"])
            for _, a, _ in group
            for stage in range(1, int(a["policy.cnn_depth"]) + 1)
            if f"policy.cnn_{name}_{stage}" in a
        ).items()))

    return {
        "source": str(path),
        "long_budget_threshold_decisions": long_budget,
        "trials": len(rows),
        "distinct_active_architectures": len(shapes),
        "duplicate_architecture_and_actual_budget_trials": sum(n - 1 for n in exact.values()),
        "depth": {
            str(depth): {
                "trials": len(group),
                "distinct_active_architectures": len({r["architecture_sha256"] for r, _, _ in group}),
                "actual_budget_min": min(steps for _, _, steps in group),
                "actual_budget_max": max(steps for _, _, steps in group),
                "trials_at_or_above_long_budget": sum(steps >= long_budget for _, _, steps in group),
                "distinct_architectures_at_or_above_long_budget": len({
                    r["architecture_sha256"] for r, _, steps in group if steps >= long_budget
                }),
                "pooling": stage_frequencies(group, "pool"),
                "dilation": stage_frequencies(group, "dilation"),
                "activation": stage_frequencies(group, "activation"),
                "readout": frequencies(group, "policy.cnn_readout"),
                "global_pool": frequencies(group, "policy.cnn_global_pool"),
                "controls": {
                    key: frequencies(group, key)
                    for key in sorted({key for _, architecture, _ in group
                                       for key in architecture if key.startswith("policy.cnn_")})
                },
            }
            for depth, group in sorted(by_depth.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--long-budget", type=int, default=8_000_000)
    args = parser.parse_args()
    if args.long_budget < 1:
        parser.error("--long-budget must be positive")
    print(json.dumps(summarize(args.results_csv, args.long_budget), indent=2))


if __name__ == "__main__":
    main()
