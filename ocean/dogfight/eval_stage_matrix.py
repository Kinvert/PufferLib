#!/usr/bin/env python3
"""Evaluate one Dogfight checkpoint against fixed, paired curriculum stages."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple, Sequence


class EvalSample(NamedTuple):
    stage: int
    seed: int
    mirror: int
    episodes: int
    perf: float
    score: float
    timeouts: float
    player_ground_hits: float
    opponent_ground_hits: float
    avg_abs_bias: float
    avg_signed_bias: float


Runner = Callable[[int, int, int, int], str]


def _last_metrics(output: str, prefix: str) -> dict[str, float]:
    for line in reversed(output.splitlines()):
        offset = line.find(prefix)
        if offset < 0:
            continue
        metrics = {}
        for field in line[offset + len(prefix) :].strip().split():
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            metrics[key] = float(value)
        return metrics
    raise ValueError(f"evaluation output is missing {prefix.strip()!r}")


def parse_eval_output(
    output: str,
    *,
    stage: int,
    seed: int,
    mirror: int,
    requested_episodes: int,
) -> EvalSample:
    controls = _last_metrics(output, "dogfight_eval_controls ")
    outcomes = _last_metrics(output, "dogfight_eval_outcomes ")
    game_counts = re.findall(r"bot_eval=(\d+)/(\d+)", output)
    if not game_counts:
        raise ValueError("evaluation output is missing bot_eval game counts")
    completed_episodes, target_episodes = (
        int(value) for value in game_counts[-1]
    )
    if target_episodes != requested_episodes:
        raise ValueError(
            f"evaluator target {target_episodes} != requested "
            f"{requested_episodes}"
        )
    if completed_episodes < requested_episodes:
        raise ValueError(
            f"evaluator completed only {completed_episodes}/"
            f"{requested_episodes} episodes"
        )
    return EvalSample(
        stage=stage,
        seed=seed,
        mirror=mirror,
        episodes=completed_episodes,
        perf=outcomes["perf"],
        score=outcomes["score"],
        timeouts=outcomes["timeouts"],
        player_ground_hits=outcomes["player_ground_hits"],
        opponent_ground_hits=outcomes["opponent_ground_hits"],
        avg_abs_bias=controls["avg_abs_bias"],
        avg_signed_bias=controls["avg_signed_bias"],
    )


def summarize(
    samples: Sequence[EvalSample],
    *,
    stages: Sequence[int],
    threshold: float,
) -> dict:
    stage_summaries = {}
    for stage in stages:
        stage_samples = [sample for sample in samples if sample.stage == stage]
        perfs = [sample.perf for sample in stage_samples]
        stage_summaries[str(stage)] = {
            "mastered": bool(perfs) and min(perfs) >= threshold,
            "cells": len(stage_samples),
            "episodes": sum(sample.episodes for sample in stage_samples),
            "min_perf": min(perfs) if perfs else None,
            "mean_perf": statistics.fmean(perfs) if perfs else None,
            "max_abs_signed_bias": (
                max(abs(sample.avg_signed_bias) for sample in stage_samples)
                if stage_samples
                else None
            ),
        }

    highest_contiguous_stage = -1
    for stage in stages:
        if not stage_summaries[str(stage)]["mastered"]:
            break
        highest_contiguous_stage = stage

    all_perfs = [sample.perf for sample in samples]
    return {
        "threshold": threshold,
        "highest_contiguous_stage": highest_contiguous_stage,
        "all_mastered": bool(stages)
        and all(stage_summaries[str(stage)]["mastered"] for stage in stages),
        "evaluated_cells": len(samples),
        "min_perf": min(all_perfs) if all_perfs else None,
        "mean_perf": statistics.fmean(all_perfs) if all_perfs else None,
        "max_abs_signed_bias": (
            max(abs(sample.avg_signed_bias) for sample in samples)
            if samples
            else None
        ),
        "stages": stage_summaries,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    stages: Sequence[int],
    seeds: Sequence[int],
    mirrors: Sequence[int],
    episodes: int,
    threshold: float,
    runner: Runner,
) -> tuple[list[EvalSample], dict]:
    del checkpoint
    samples = []
    stopped_after_stage = None
    for stage in stages:
        stage_samples = []
        for seed in seeds:
            for mirror in mirrors:
                output = runner(stage, seed, mirror, episodes)
                try:
                    sample = parse_eval_output(
                        output,
                        stage=stage,
                        seed=seed,
                        mirror=mirror,
                        requested_episodes=episodes,
                    )
                except (KeyError, ValueError) as error:
                    raise RuntimeError(
                        "Dogfight evaluator returned an invalid contract "
                        f"for stage={stage} seed={seed} mirror={mirror}: "
                        f"{error}\nEvaluator output:\n{output}"
                    ) from error
                samples.append(sample)
                stage_samples.append(sample)
        if min(sample.perf for sample in stage_samples) < threshold:
            stopped_after_stage = stage
            break

    result = summarize(samples, stages=stages, threshold=threshold)
    result["stopped_after_stage"] = stopped_after_stage
    return samples, result


def _parse_ints(value: str) -> tuple[int, ...]:
    if ".." in value:
        start, end = (int(part) for part in value.split("..", 1))
        return tuple(range(start, end + 1))
    return tuple(int(part) for part in value.split(","))


def _subprocess_runner(eval_script: Path, checkpoint: Path) -> Runner:
    def run(stage: int, seed: int, mirror: int, episodes: int) -> str:
        command = [
            "bash",
            str(eval_script),
            "headless",
            str(checkpoint),
            str(stage),
            str(episodes),
            str(seed),
            str(mirror),
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Dogfight evaluator failed "
                f"(exit {completed.returncode}):\n{completed.stdout}"
            )
        return completed.stdout

    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stages", default="0..10")
    parser.add_argument("--seeds", default="42,314159,271828")
    parser.add_argument("--mirrors", default="0,1")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).with_name("eval_checkpoint.sh"),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.episodes < 128:
        parser.error("--episodes must be at least 128")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    stages = _parse_ints(args.stages)
    seeds = _parse_ints(args.seeds)
    mirrors = _parse_ints(args.mirrors)
    if not stages or not seeds or set(mirrors) != {0, 1}:
        parser.error("stages/seeds must be non-empty and mirrors must be exactly 0,1")

    samples, result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        stages=stages,
        seeds=seeds,
        mirrors=mirrors,
        episodes=args.episodes,
        threshold=args.threshold,
        runner=_subprocess_runner(args.eval_script, args.checkpoint),
    )
    result["checkpoint"] = str(args.checkpoint)
    result["samples"] = [sample._asdict() for sample in samples]

    for sample in samples:
        print(
            "dogfight_stage_cell "
            f"stage={sample.stage} seed={sample.seed} mirror={sample.mirror} "
            f"perf={sample.perf:.6f} score={sample.score:.6f} "
            f"signed_bias={sample.avg_signed_bias:.6f}"
        )
    print(
        "dogfight_stage_matrix "
        f"highest_contiguous_stage={result['highest_contiguous_stage']} "
        f"all_mastered={int(result['all_mastered'])} "
        f"min_perf={result['min_perf']:.6f}"
    )

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded)
    return 0 if result["all_mastered"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
