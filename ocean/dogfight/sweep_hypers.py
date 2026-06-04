"""Dogfight training-log parsing for bounded hyperparameter probes."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path


SURFACE_SATURATION_KEYS = (
    "final_action_sat_elevator",
    "final_action_sat_aileron",
    "final_action_sat_rudder",
)
SURFACE_SATURATION_REJECT_THRESHOLD = 0.75

TRIALS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    (
        "apricot_control",
        {
            "train.learning-rate": "0.01788370841005079",
            "train.ent-coef": "0.1604920157441345",
            "train.anneal-ent-coef": "0",
            "train.min-ent-coef-ratio": "0.1",
            "train.clip-coef": "0.41000268739635415",
            "train.vf-coef": "5",
            "train.vf-clip-coef": "3.484823007218511",
            "train.max-grad-norm": "3.8432803314849577",
            "train.replay-ratio": "2.8448025380102187",
            "train.momentum": "0.95",
            "train.prio-alpha": "0.5506490443590013",
            "train.prio-beta0": "1",
            "train.vtrace-rho-clip": "3.243009815021134",
            "train.vtrace-c-clip": "2.15402866995608",
        },
    ),
]

RANDOM_SPACE: dict[str, tuple[str, float | int, float | int]] = {
    "train.learning-rate": ("log", 0.00008, 0.0008),
    "train.ent-coef": ("log", 0.0002, 0.04),
    "train.min-ent-coef-ratio": ("log", 0.01, 0.5),
    "train.clip-coef": ("linear", 0.06, 0.30),
    "train.vf-coef": ("linear", 0.5, 4.0),
    "train.vf-clip-coef": ("linear", 0.2, 3.0),
    "train.max-grad-norm": ("linear", 0.5, 3.0),
    "train.momentum": ("linear", 0.90, 0.995),
    "train.vtrace-rho-clip": ("linear", 0.8, 4.0),
    "train.vtrace-c-clip": ("linear", 0.8, 4.0),
    "train.prio-alpha": ("linear", 0.5, 1.0),
    "train.prio-beta0": ("linear", 0.2, 1.0),
}

DISCRETE_RANDOM_SPACE: dict[str, tuple[str, ...]] = {
    "train.anneal-ent-coef": ("1",),
    "train.replay-ratio": ("0.5", "1.0", "1.5", "2.0"),
    "policy.hidden-size": ("128", "256"),
    "policy.num-layers": ("2", "3"),
}

METRICS = (
    "agent_steps",
    "epoch",
    "SPS",
    "entropy",
    "old_kl",
    "kl",
    "clipfrac",
    "curriculum_target",
    "mastery_stage",
    "stage9_bank_deg",
    "base_stage_kill_rate",
    "base_stage_kills",
    "base_stage_eps",
    "base_stage_ground_rate",
    "base_stage_timeout_rate",
    "base_stage_episode_length",
    "base_stage_action_saturation",
    "base_stage_signed_bias",
    "action_sat_elevator",
    "action_sat_aileron",
    "action_sat_rudder",
    "action_sat_trigger",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROMOTION_RE = re.compile(
    r"\[CURRICULUM\].*?target\s+([0-9.]+)\s+->\s+([0-9.]+)"
    r"\s+kill_rate=([0-9.]+)\s+episodes=([0-9.]+)"
)
TUI_PREFIXES = ("╭", "╰", "├", "┤", "│")


def parse_number(raw: str) -> float:
    value = raw.strip()
    multiplier = 1.0
    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000.0
        value = value[:-1]
    return float(value) * multiplier


def metric_regex(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_/-]){re.escape(name)}\s+(-?[0-9.]+[KMB]?)")


METRIC_RES = {name: metric_regex(name) for name in METRICS}


def maybe_float(value: object) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def error_scan_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(TUI_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).lower()


def add_derived_kill_rate(summary: dict[str, object], values: dict[str, list[float]]) -> None:
    if summary.get("final_base_stage_kill_rate") != "":
        return

    kills = values.get("base_stage_kills", [])
    episodes = values.get("base_stage_eps", [])
    rates = [
        kill / eps
        for kill, eps in zip(kills, episodes)
        if eps > 0.0
    ]
    if not rates:
        return

    summary["final_base_stage_kill_rate"] = rates[-1]
    summary["best_base_stage_kill_rate"] = max(rates)


def add_selection_fields(summary: dict[str, object]) -> None:
    surface_values = [maybe_float(summary.get(key)) for key in SURFACE_SATURATION_KEYS]
    if any(value is None for value in surface_values):
        summary["surface_saturation"] = ""
        summary["selection_score"] = ""
        summary["rejected"] = True
        return

    surface_saturation = sum(float(value) for value in surface_values) / len(surface_values)
    best_kill_rate = maybe_float(summary.get("best_base_stage_kill_rate")) or 0.0
    max_target = maybe_float(summary.get("max_target")) or 0.9

    summary["surface_saturation"] = surface_saturation
    summary["selection_score"] = max_target + 0.25 * best_kill_rate - 2.0 * surface_saturation
    summary["rejected"] = surface_saturation > SURFACE_SATURATION_REJECT_THRESHOLD


def parse_log(path: Path) -> dict[str, object]:
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))
    summary: dict[str, object] = {
        "log": str(path),
        "promotions": 0,
        "max_target": 0.9,
        "last_promotion_kill_rate": "",
        "last_promotion_episodes": "",
        "error": "",
    }

    for match in PROMOTION_RE.finditer(text):
        summary["promotions"] = int(summary["promotions"]) + 1
        summary["max_target"] = max(float(summary["max_target"]), float(match.group(2)))
        summary["last_promotion_kill_rate"] = match.group(3)
        summary["last_promotion_episodes"] = match.group(4)

    lowered = error_scan_text(text)
    for marker in ("traceback", "exception", "assert", "nan", "crash"):
        if marker in lowered:
            summary["error"] = marker
            break

    values_by_metric: dict[str, list[float]] = {}
    for name, regex in METRIC_RES.items():
        values = [parse_number(match.group(1)) for match in regex.finditer(text)]
        values_by_metric[name] = values
        summary[f"final_{name}"] = values[-1] if values else ""
        summary[f"best_{name}"] = max(values) if values else ""

    add_derived_kill_rate(summary, values_by_metric)
    best_curriculum_target = maybe_float(summary.get("best_curriculum_target"))
    if best_curriculum_target is not None:
        summary["max_target"] = max(float(summary["max_target"]), best_curriculum_target)
    add_selection_fields(summary)
    return summary


def sample_value(kind: str, low: float | int, high: float | int, rng: random.Random) -> str:
    if kind == "log":
        value = 10 ** rng.uniform(math.log10(float(low)), math.log10(float(high)))
    elif kind == "linear":
        value = rng.uniform(float(low), float(high))
    else:
        raise ValueError(f"unknown sample kind {kind}")
    return f"{value:.6g}"


def random_trials(max_runs: int, seed: int) -> list[tuple[str, dict[str, str]]]:
    rng = random.Random(seed)
    trials: list[tuple[str, dict[str, str]]] = [("baseline", {})]
    for idx in range(1, max_runs):
        overrides = {
            key: sample_value(kind, low, high, rng)
            for key, (kind, low, high) in RANDOM_SPACE.items()
        }
        for key, choices in DISCRETE_RANDOM_SPACE.items():
            overrides[key] = rng.choice(choices)
        trials.append((f"random_{idx:04d}", overrides))
    return trials


def trials_from_summary(path: Path, top_k: int) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[float, str, dict[str, str]]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            if not (status.startswith("ok") or status == "parsed"):
                continue
            if row.get("rejected") != "False":
                continue
            score = maybe_float(row.get("selection_score"))
            if score is None:
                continue
            overrides = json.loads(row.get("overrides") or "{}")
            rows.append((score, row["trial"], overrides))

    rows.sort(reverse=True, key=lambda item: item[0])
    return [(name, overrides) for _, name, overrides in rows[:top_k]]
