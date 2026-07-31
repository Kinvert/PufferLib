#!/usr/bin/env python3
"""Join native Protein results to fixed Dogfight mastery screens."""

from __future__ import annotations

import argparse
import configparser
import csv
import io
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _number(value: str):
    value = value.strip().replace("_", "")
    if not value:
        return None
    number = float(value)
    if number.is_integer() and all(marker not in value.lower() for marker in (".", "e")):
        return int(number)
    return number


def _series(parser: configparser.ConfigParser, key: str) -> list[float | int]:
    value = parser.get("metrics", key, fallback="")
    return [_number(item) for item in value.split(",") if item.strip()]


def _last(parser: configparser.ConfigParser, key: str):
    values = _series(parser, key)
    return values[-1] if values else None


def _parameter(
    parser: configparser.ConfigParser, dimension: str
) -> float | int | str | None:
    if "." not in dimension:
        return None
    section, key = dimension.split(".", 1)
    value = parser.get(section, key, fallback=None)
    if value is None:
        return None
    try:
        return _number(value)
    except ValueError:
        return value


def _load_native(path: Path, dimensions: Iterable[str]) -> dict:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    run_id = parser.get("base", "run_id")
    return {
        "run_id": run_id,
        "native_log": str(path),
        "seed": parser.getint("base", "seed", fallback=-1),
        "final_agent_steps": _last(parser, "agent_steps"),
        "pool_score": _last(parser, "selfplay/pool_score"),
        "pool_flight_quality": _last(
            parser, "selfplay/pool_flight_quality"
        ),
        "pool_fitness": _last(parser, "selfplay/pool_fitness"),
        "native_perf": _last(parser, "env/perf"),
        "native_score": _last(parser, "env/score"),
        "native_sps": _last(parser, "SPS"),
        "native_uptime_seconds": _last(parser, "uptime"),
        "native_avg_stage": _last(parser, "env/avg_stage"),
        "native_timeout_rate": _last(parser, "env/timeouts"),
        "native_avg_abs_bias": _last(parser, "env/avg_abs_bias"),
        "native_avg_signed_bias": _last(parser, "env/avg_signed_bias"),
        "parameters": {
            dimension: _parameter(parser, dimension) for dimension in dimensions
        },
    }


def _load_fixed_results(experiment: Path) -> dict[str, dict]:
    results = {}
    fixed_dir = experiment / "fixed_eval"
    if not fixed_dir.is_dir():
        return results
    for path in sorted(fixed_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("format") != "dogfight-fixed-stage-screen-v1":
            continue
        result["_result_path"] = str(path)
        results[result["run_id"]] = result
    return results


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    value = sum(
        left * right for left, right in zip(centered_x, centered_y)
    ) / denominator
    return round(value, 12)


def _correlation(rows: list[dict], left: str, right: str) -> dict:
    pairs = [
        (float(row[left]), float(row[right]))
        for row in rows
        if row.get(left) is not None and row.get(right) is not None
    ]
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return {
        "n": len(pairs),
        "pearson": _pearson(xs, ys),
        "spearman": _pearson(_average_ranks(xs), _average_ranks(ys)),
    }


def _rank_key(row: dict):
    return (
        -row["highest_contiguous_stage"],
        -row["fixed_min_perf"],
        -row["fixed_mean_perf"],
        row["fixed_max_abs_signed_bias"],
        -(
            row["pool_fitness"]
            if row["pool_fitness"] is not None
            else -math.inf
        ),
    )


def analyze_experiment(
    experiment: Path,
    *,
    minimum_candidates: int = 16,
    minimum_spearman: float = 0.25,
) -> dict:
    experiment = experiment.resolve()
    manifest_path = experiment / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dimensions = manifest.get("native_sweep_dimensions", [])
    fixed_results = _load_fixed_results(experiment)

    rows = []
    for log_path in sorted((experiment / "logs" / "dogfight").glob("*.ini")):
        row = _load_native(log_path, dimensions)
        fixed = fixed_results.get(row["run_id"])
        if fixed is None:
            row.update(
                {
                    "fixed_status": "unscreened",
                    "highest_contiguous_stage": None,
                    "fixed_min_perf": None,
                    "fixed_mean_perf": None,
                    "fixed_max_abs_signed_bias": None,
                    "fixed_roll_gate_passed": None,
                    "fixed_max_abs_roll_common_mode": None,
                    "evaluated_cells": 0,
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                    "fixed_result": None,
                }
            )
        else:
            row.update(
                {
                    "fixed_status": (
                        "mastered" if fixed["all_mastered"] else "rejected"
                    ),
                    "highest_contiguous_stage": fixed[
                        "highest_contiguous_stage"
                    ],
                    "fixed_min_perf": fixed["min_perf"],
                    "fixed_mean_perf": fixed["mean_perf"],
                    "fixed_max_abs_signed_bias": fixed[
                        "max_abs_signed_bias"
                    ],
                    "fixed_roll_gate_passed": fixed.get(
                        "roll_gate_passed", False
                    ),
                    "fixed_max_abs_roll_common_mode": fixed.get(
                        "max_abs_roll_common_mode"
                    ),
                    "evaluated_cells": fixed["evaluated_cells"],
                    "checkpoint": fixed["checkpoint"],
                    "checkpoint_sha256": fixed["checkpoint_sha256"],
                    "fixed_result": fixed["_result_path"],
                }
            )
        rows.append(row)

    screened = [
        row for row in rows if row["fixed_status"] != "unscreened"
    ]
    ranked = sorted(screened, key=_rank_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    unscreened = sorted(
        (row for row in rows if row["fixed_status"] == "unscreened"),
        key=lambda row: row["run_id"],
    )
    for row in unscreened:
        row["rank"] = None
    leaderboard = ranked + unscreened

    correlations = {
        "pool_fitness_vs_fixed_mean": _correlation(
            screened, "pool_fitness", "fixed_mean_perf"
        ),
        "pool_fitness_vs_fixed_min": _correlation(
            screened, "pool_fitness", "fixed_min_perf"
        ),
        "pool_score_vs_fixed_mean": _correlation(
            screened, "pool_score", "fixed_mean_perf"
        ),
        "pool_score_vs_fixed_min": _correlation(
            screened, "pool_score", "fixed_min_perf"
        ),
        "native_perf_vs_fixed_mean": _correlation(
            screened, "native_perf", "fixed_mean_perf"
        ),
        "native_score_vs_fixed_mean": _correlation(
            screened, "native_score", "fixed_mean_perf"
        ),
    }
    primary = correlations["pool_fitness_vs_fixed_mean"]
    if primary["n"] < minimum_candidates:
        gate_status = "insufficient"
        gate_reason = (
            f"{primary['n']} screened candidates; need {minimum_candidates}"
        )
    elif primary["spearman"] is None:
        gate_status = "fail"
        gate_reason = "pool fitness or fixed performance is constant"
    elif primary["spearman"] < minimum_spearman:
        gate_status = "fail"
        gate_reason = (
            f"Spearman {primary['spearman']:.3f} is below "
            f"{minimum_spearman:.3f}"
        )
    else:
        gate_status = "pass"
        gate_reason = (
            f"Spearman {primary['spearman']:.3f} meets "
            f"{minimum_spearman:.3f}"
        )

    return {
        "format": "dogfight-sweep-calibration-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment),
        "native_sweep_dimensions": dimensions,
        "candidate_count": len(rows),
        "screened_candidate_count": len(screened),
        "mastered_candidate_count": sum(
            row["fixed_status"] == "mastered" for row in screened
        ),
        "correlations": correlations,
        "protein_gate": {
            "status": gate_status,
            "reason": gate_reason,
            "minimum_candidates": minimum_candidates,
            "minimum_spearman": minimum_spearman,
            "note": (
                "This gate only tests whether Protein's native objective is "
                "useful for search. It does not prove curriculum mastery."
            ),
        },
        "leaderboard": leaderboard,
    }


def _atomic_write(path: Path, content: str):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _csv_content(analysis: dict) -> str:
    parameter_names = sorted(
        {
            name
            for row in analysis["leaderboard"]
            for name in row["parameters"]
        }
    )
    fields = [
        "rank",
        "run_id",
        "fixed_status",
        "highest_contiguous_stage",
        "fixed_min_perf",
        "fixed_mean_perf",
        "fixed_max_abs_signed_bias",
        "fixed_roll_gate_passed",
        "fixed_max_abs_roll_common_mode",
        "pool_score",
        "pool_flight_quality",
        "pool_fitness",
        "native_perf",
        "native_score",
        "native_sps",
        "final_agent_steps",
        "checkpoint",
        "checkpoint_sha256",
        *parameter_names,
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in analysis["leaderboard"]:
        flattened = dict(row)
        flattened.update(row["parameters"])
        writer.writerow(flattened)
    return output.getvalue()


def _markdown_content(analysis: dict) -> str:
    primary = analysis["correlations"]["pool_fitness_vs_fixed_mean"]
    lines = [
        "# Dogfight Protein Calibration Report",
        "",
        f"- Experiment: `{analysis['experiment_dir']}`",
        f"- Candidates: `{analysis['candidate_count']}`",
        f"- Screened candidates: `{analysis['screened_candidate_count']}`",
        f"- Mastered candidates: `{analysis['mastered_candidate_count']}`",
        f"- Protein gate: `{analysis['protein_gate']['status']}`",
        f"- Gate reason: {analysis['protein_gate']['reason']}",
        (
            "- Adjusted pool fitness versus fixed mean: "
            f"Spearman `{primary['spearman']}`, "
            f"Pearson `{primary['pearson']}`, n `{primary['n']}`"
        ),
        "",
        "The Protein gate measures search-objective alignment only. Stage 10",
        "still requires every fixed seed/mirror cell to reach at least 90%.",
        "",
        "| Rank | Run | Fixed status | Highest stage | Fixed min | "
        "Fixed mean | Raw pool | Flight quality | Fitness | Roll gate | Bias |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["leaderboard"]:
        lines.append(
            f"| {row['rank'] or ''} | `{row['run_id']}` | "
            f"{row['fixed_status']} | "
            f"{row['highest_contiguous_stage']} | "
            f"{row['fixed_min_perf']} | {row['fixed_mean_perf']} | "
            f"{row['pool_score']} | {row['pool_flight_quality']} | "
            f"{row['pool_fitness']} | {row['fixed_roll_gate_passed']} | "
            f"{row['fixed_max_abs_signed_bias']} |"
        )
    lines.extend(["", "## Swept parameters", ""])
    for row in analysis["leaderboard"]:
        encoded = json.dumps(row["parameters"], sort_keys=True)
        lines.append(f"- `{row['run_id']}`: `{encoded}`")
    return "\n".join(lines) + "\n"


def write_analysis(experiment: Path, analysis: dict) -> Path:
    output_dir = experiment.resolve() / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "calibration_summary.json",
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_dir / "calibration_leaderboard.csv", _csv_content(analysis)
    )
    _atomic_write(
        output_dir / "CALIBRATION_REPORT.md", _markdown_content(analysis)
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--minimum-candidates", type=int, default=10)
    parser.add_argument("--minimum-spearman", type=float, default=0.25)
    args = parser.parse_args()
    if args.minimum_candidates < 2:
        parser.error("--minimum-candidates must be at least 2")
    if not -1.0 <= args.minimum_spearman <= 1.0:
        parser.error("--minimum-spearman must be between -1 and 1")

    analysis = analyze_experiment(
        args.experiment,
        minimum_candidates=args.minimum_candidates,
        minimum_spearman=args.minimum_spearman,
    )
    output_dir = write_analysis(args.experiment, analysis)
    print(
        "dogfight_sweep_analysis "
        f"candidates={analysis['candidate_count']} "
        f"screened={analysis['screened_candidate_count']} "
        f"gate={analysis['protein_gate']['status']} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
