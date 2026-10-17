"""Audit received Connect4 hardware receipts without extracting or executing archive files."""
import argparse
import configparser
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"flex_quality": "Ours quality", "nature_cnn": "Nature", "impala_cnn": "IMPALA", "impoola_cnn": "Impoola"}
PARAMS = {"flex_quality": 160736, "nature_cnn": 138528, "impala_cnn": 270496, "impoola_cnn": 151712}
EVAL = re.compile(r"^CUDA_EVAL env=connect4cnn score=([-+\d.eE]+) perf=([-+\d.eE]+) games=(\d+) params=(\d+)$", re.M)


def frontier(points, axis):
    return sorted([p for p in points if not any(q[axis] <= p[axis] and q["wins"] >= p["wins"] and
        (q[axis] < p[axis] or q["wins"] > p["wins"]) for q in points)], key=lambda p: p[axis])


def csv_out(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(archive, output):
    with tarfile.open(archive) as tar:
        members = {}
        for member in tar.getmembers():
            name = str(PurePosixPath(member.name))
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            assert member.isfile(), "Only regular evidence files accepted"
            assert name not in members, "Duplicate archive path"
            members[name] = member

        def data(name):
            return tar.extractfile(members[name]).read()

        def text(name):
            return data(name).decode()

        def ini(name):
            c = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
            c.read_string(text(name))
            return c

        protocol, jobs, finished = [json.loads(text(n)) for n in ("protocol.json", "jobs.json", "finished.json")]
        assert finished == {"status": "ok", "jobs": 4, "evaluations": 52, "logging_failures": 0}
        assert protocol["precision"] == "float32" and protocol["seeds"] == [173] and protocol["eval_seed"] == 20173
        assert protocol["completed_steps"] == protocol["steps"] == 13312000
        assert set(protocol["variants"]) == set(LABELS)
        assert protocol["checkpoint_steps"] == list(range(1024000, 13312001, 1024000))
        assert hashlib.sha256(data("recipe.ini")).hexdigest() == protocol["recipe_sha256"]
        mismatches = []
        for name, expected in protocol["source_sha256"].items():
            assert hashlib.sha256(data("source/" + name)).hexdigest() == expected, name
            tracked = subprocess.run(["git", "show", protocol["revision"] + ":" + name], cwd=ROOT, capture_output=True)
            if tracked.returncode or hashlib.sha256(tracked.stdout).hexdigest() != expected:
                mismatches.append(name)
        assert len(jobs) == 4 and {j["variant"] for j in jobs} == set(LABELS)
        common = None
        lookup = {}
        selected = json.loads(text("source/ocean/connect4cnn/confirmation.json"))
        for job in jobs:
            variant, seed = job["variant"], job["seed"]
            assert seed == 173 and job["status"] == "ok"
            trial = f"{variant}-s{seed}"
            c = ini(trial + "/metrics/connect4cnn/trial.ini")
            assert c.getint("base", "seed") == seed and c.getint("train", "total_timesteps") == 13312000
            assert c.getint("env", "representation") == 0
            assert c.getint("policy", "hidden_size") == 128 and c.getint("policy", "num_layers") == 1
            for key, value in selected["policies"].get(variant, {"encoder": 0}).items():
                assert c.getfloat("policy", key) == value
            if "config_sha256" in job:
                assert hashlib.sha256(data(trial + "/config/default.ini")).hexdigest() == job["config_sha256"]
            effective = {s: dict(c[s]) for s in c if s != "metrics"}
            for key in ("env_name", "seed", "run_id", "checkpoint_dir", "log_dir"):
                effective["base"].pop(key, None)
            effective["policy"] = {k: v for k, v in effective["policy"].items() if k != "encoder" and not k.startswith("cnn_")}
            if common is None:
                common = effective
            assert effective == common, "Non-encoder effective settings differ"
            assert set(job["checkpoint_sha256"]) == {f"{s:016d}.bin" for s in protocol["checkpoint_steps"]}
            assert all(re.fullmatch("[0-9a-f]{64}", h) for h in job["checkpoint_sha256"].values())
            lookup[variant] = (job, c)
        raw = list(csv.DictReader(io.StringIO(text("results.csv"))))
        assert len(raw) == 52
        points, seen = [], set()
        for r in raw:
            variant, step = r["variant"], int(r["steps"])
            assert (variant, step) not in seen
            seen.add((variant, step))
            assert int(r["seed"]) == 173 and int(r["eval_seed"]) == 20173
            assert int(r["params"]) == PARAMS[variant] and int(r["games"]) >= 1024
            job, c = lookup[variant]
            assert r["checkpoint"] == f"{variant}-s173/checkpoints/connect4cnn/trial/{step:016d}.bin"
            match, = EVAL.findall(text(f"{variant}-s173/eval-{step:016d}.log"))
            assert tuple(map(float, match)) == (float(r["score"]), float(r["win_rate"]), int(r["games"]), int(r["params"]))
            wall, native = float(r["train_process_wall_s"]), float(r["native_uptime_s"])
            assert math.isclose(wall, job["train_process_wall_s"], rel_tol=1e-10)
            assert math.isclose(float(c["metrics"]["uptime"].split(",")[-1]), native, rel_tol=1e-10)
            assert math.isclose(float(r["process_sps"]), 13312000 / wall, rel_tol=1e-10)
            assert math.isclose(float(r["native_avg_sps"]), 13312000 / native, rel_tol=1e-10)
            point = dict(variant=variant, seed=173, steps=step, wins=float(r["win_rate"]),
                         seconds=float(r["checkpoint_wall_s"]), train_seconds=wall,
                         process_sps=float(r["process_sps"]), native_sps=float(r["native_avg_sps"]), params=int(r["params"]))
            assert all(math.isfinite(v) for v in point.values() if isinstance(v, (int, float)))
            assert 0 <= point["wins"] <= 1 and 0 < point["seconds"] <= wall
            points.append(point)
        assert seen == set(itertools.product(LABELS, protocol["checkpoint_steps"]))
        for variant in LABELS:
            curve = sorted((p for p in points if p["variant"] == variant), key=lambda p: p["steps"])
            assert all(a["seconds"] < b["seconds"] for a, b in zip(curve, curve[1:]))
        edges = [{"axis": axis, **p} for axis in ("seconds", "steps") for p in frontier(points, axis)]
        finals = [p for p in points if p["steps"] == 13312000]
        result = dict(status="receipt_audit_passed", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      revision=protocol["revision"], source_hashes_verified=len(protocol["source_sha256"]),
                      source_files_differing_from_git_revision=mismatches, evaluation_logs_verified=52,
                      configurations_matched=4, total_training_seconds=sum(p["train_seconds"] for p in finals),
                      binary_and_checkpoint_arrays="Not present; remote audit claims retained, not independently repeated",
                      remote_audit=json.loads(text("audit-5090.json")))
        output.mkdir(parents=True, exist_ok=True)
        (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        csv_out(output / "observations.csv", points)
        csv_out(output / "finals.csv", finals)
        csv_out(output / "frontiers.csv", edges)
        lines = ["# RTX 5090: received evidence audit and complete observed frontier", "",
                 f"Revision `{protocol['revision']}`; source hashes verified: {len(protocol['source_sha256'])}; 52 CSV observations match their raw evaluation logs; four non-encoder configs match.",
                 f"Captured files differing from the recorded Git revision: {mismatches or 'none'}.",
                 "One training seed (173). No seed confidence intervals or population-dominance claim. All 52 chronological observations remain in observations.csv, including declines.",
                 "Executable binaries and checkpoint arrays were omitted from the transfer. Their recorded hashes and the remote audit are retained, but this audit cannot independently repeat byte-hash/finiteness checks on absent files.", ""]
        for axis in ("seconds", "steps"):
            lines += [f"## Complete observed {axis} frontier", "", "| Model | Decisions | Checkpoint wall seconds | Held-out wins |", "|---|---:|---:|---:|"]
            for p in frontier(points, axis):
                lines.append(f"| {LABELS[p['variant']]} | {p['steps']:,} | {p['seconds']:.3f} | {p['wins']:.2%} |")
            lines.append("")
        lines += ["Checkpoint timing includes process startup and checkpoint writes; it is distinct from native uptime. Early checkpoints share the full-run learning-rate schedule and are not independent short-budget training runs.",
                  "Archive provenance and hardware metadata remain in the received archive. Historical 5060 data used different source/compiler/host settings; matching-source cross-host confirmation remains pending.", ""]
        (output / "REPORT.md").write_text("\n".join(lines))
        print(json.dumps(result, indent=2))
        print("Time frontier:", [(p["variant"], p["steps"], round(p["seconds"], 2), round(p["wins"] * 100, 2)) for p in frontier(points, "seconds")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.output)
