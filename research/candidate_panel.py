#!/usr/bin/env python3
"""Expand native CNN candidates across games/drawings; external launch/audit glue.

No Python optimizer, learner or CPU neural execution. Native PROTEIN still
proposes architectures. This separate development protocol never upgrades the
older fixed-baseline packets or certifies a publication claim.
"""
import argparse
from contextlib import redirect_stdout
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
import prepare_pixel_robustness as panel
import pixel_evaluation_plan as bridge
from claim import normalized, parse_checkpoints, process, require, save, sha
from sweep import LIMITS
from wandb_sidecar import display_name, read_ini, trials
import checkpoint_preflight as policy_layout

PROTOCOL = "native-cnn-candidate-panel-v1"
BASELINE_PROTOCOL = "native-cnn-candidate-panel-v2"
TASKS = tuple(panel.ENVIRONMENTS)
BASELINES = {
    "nature_cnn": dict(name="nature-cnn", architecture={"encoder": 2}, binary_family="default", policy_family="nature"),
    "impala_cnn": dict(name="impala-cnn", architecture={"encoder": 0}, binary_family="impala", policy_family="impala"),
    "impoola_cnn": dict(name="impoola-cnn", architecture={"encoder": 0}, binary_family="impoola", policy_family="impoola"),
}


def registry_builds(registry):
    if registry["protocol"] == "native-cnn-candidate-build-v1":
        return {task: {"default": record} for task, record in registry["binaries"].items()}
    require(registry["protocol"] == "native-cnn-candidate-build-v2", "Unknown build registry")
    require(all(set(builds) == {"default", "impala", "impoola"} for builds in registry["binaries"].values()),
            "Need all baseline build families")
    return registry["binaries"]


def candidate_families(candidate):
    return candidate.get("binary_family", "default"), candidate.get("policy_family", "flex")


def metadata_source_binding(registry, native_registry, sources):
    # Owned build_arch/weights_create files, excluding unrelated test/docs.
    for name, digest in registry["sources"].items():
        production = name.startswith("src/") or (name.startswith("ocean/connect4cnn/")
            and "/tests/" not in name and Path(name).suffix in (".h", ".cu", ".c"))
        if production:
            require(native_registry["source_sha256"].get(name) == sources.get(name) == digest,
                    "Metadata and training production source differ: " + name)
    for record in registry["actions"].values():
        require(native_registry["source_sha256"].get(record["source"]) == sources.get(record["source"]) == record["sha256"],
                "Metadata and training action source differ")


def inspect_job_layout(value, root, index, directory=None, execution=False):
    job = value["jobs"][index]; family = job.get("policy_family", "flex")
    declaration = value["policy_layout"]; recorded = Path(declaration["preparation_root"])
    destination = directory or root / "policy-layouts" / f"job-{index:04d}"
    record = policy_layout.inspect_description(root / "policy-metadata", root / job["config"],
        job["environment"], family, destination, recorded / "policy-metadata", recorded / job["config"],
        Path(declaration["repository_root"]),
        timeout=json.loads((destination / "result.json").read_text())["timing"]["timeout"] if execution else 60)
    require(0 < record["timing"]["timeout"] <= 60, "Unbounded native layout process")
    expected = job["policy_layout"]
    if execution:
        require(record["descriptor_sha256"] == expected["descriptor_sha256"]
                and record["parameters"] == expected["parameters"] and record["bytes"] == expected["bytes"],
                "Repeated registration differs from preparation")
    else: require(record == expected, "Job layout differs from retained native proof")
    return record


def checkpoint_size(job, checkpoint):
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0 and checkpoint.stat().st_size % 4 == 0,
            "Missing/malformed float32 checkpoint")
    if "policy_layout" in job:
        require(checkpoint.stat().st_size == job["policy_layout"]["bytes"],
                "Checkpoint size differs from native policy registration")
    return checkpoint.stat().st_size // 4


def capture(out):
    sources = {ROOT / "build.sh", Path(__file__).resolve()}
    for directory in (ROOT / "src", ROOT / "config", ROOT / "research"):
        sources.update(p for p in directory.glob("*") if p.suffix in (".h", ".c", ".cu", ".py", ".ini", ".sh"))
    for task in (*TASKS, "connect4", "pong", "flappy", "breakout", "snakebench", "maze"):
        sources.update(p for p in (ROOT / "ocean" / task).rglob("*")
                       if p.is_file() and p.suffix in (".h", ".c", ".cu", ".py", ".ini", ".sh", ".json"))
    entries = {}
    for path in sorted(sources):
        name = str(path.relative_to(ROOT)); digest = sha(path)
        target = out / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): require(sha(target) == digest, "Source changed after scalar preparation")
        else: shutil.copyfile(path, target)
        require(sha(path) == sha(target) == digest, "Source changed during capture")
        entries[name] = digest
    return entries


def verify_sources(out, entries, current=False):
    for name, digest in entries.items():
        require(sha(out / "source" / name) == digest, "Changed frozen source: " + name)
        if current: require(sha(ROOT / name) == digest, "Changed executing source: " + name)


def build(out, baselines=False):
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    entries = capture(out); (out / "bin").mkdir()
    environment = dict(os.environ)
    require(all(not environment.get(k) for k in ("NVCC_EXTRA", "NVCC_APPEND_FLAGS", "NVCC_FLAGS")), "Unset extra compiler flags")
    require(environment.get("NVCC_PREPEND_FLAGS", "") in ("", "--threads 1"), "Only compiler thread limit permitted")
    environment.update(NVCC_ARCH="sm_120", NVCC_PREPEND_FLAGS="--threads 1")
    compiler = subprocess.check_output([str(Path(environment.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"), "--version"], text=True, timeout=20)
    records = {}
    try:
        for task in TASKS:
            records[task] = {}
            variants = (("default", ""), ("impala", "-DC4_IMPALA_CNN"), ("impoola", "-DC4_IMPOOLA_CNN")) if baselines else (("default", ""),)
            for family, flag in variants:
                verify_sources(out, entries, current=True)
                basename = f"{task}-{family}" if baselines else task
                binary = out / "bin" / basename
                compiler_env = dict(environment, NVCC_PREPEND_FLAGS="--threads 1" + (" " + flag if flag else ""))
                command = ["bash", "build.sh", task, str(binary), "--float"]
                process(command, ROOT, out / f"{basename}-build.txt", 1200, compiler_env)
                identity = json.loads(subprocess.check_output([str(binary), "eval_exact_info"], text=True, timeout=30))
                bridge.validate_identity(task, family, identity, panel.ENVIRONMENTS[task])
                records[task][family] = dict(path=str(binary), sha256=sha(binary), info=identity, command=command,
                    compiler_prepend_flags=compiler_env["NVCC_PREPEND_FLAGS"])
            if not baselines: records[task] = records[task]["default"]
        verify_sources(out, entries, current=True)
        save(out / "registry.json", dict(protocol="native-cnn-candidate-build-v2" if baselines else "native-cnn-candidate-build-v1", binaries=records,
            source_sha256=entries, compiler=compiler, compiler_environment={k: environment.get(k) for k in
            ("CUDA_HOME", "NCCL_ROOT", "NVCC_ARCH", "NVCC_PREPEND_FLAGS", "CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH")},
            gpu_queried=False, policy_executed=False, vendor_system_link_closure_complete=False))
    except BaseException as error:
        save(out / "failure.json", dict(error=str(error), status="failed-build", policy_executed=False))
        raise
    return out / "registry.json"


def architecture(path):
    config = read_ini(path)
    require(config.has_section("policy") and config.getint("policy", "encoder") == 4,
            "This first panel accepts legacy encoder 4 only; encoder 5 retains separate gates")
    depth = config.getint("policy", "cnn_depth")
    require(depth in (1, 2, 3), "Invalid depth")
    keys = ("cnn_depth", "cnn_projection", "cnn_global_pool") + tuple(
        f"cnn_{key}_{stage}" for stage in range(1, depth+1) for key in ("channels", "kernel", "stride", "pool", "residual"))
    spec = {"encoder": 4}
    for key in keys:
        value = config.getfloat("policy", key)
        require(value in LIMITS[4]["policy."+key][1], "Unsupported architecture coordinate: " + key)
        spec[key] = int(value)
    # Core is fixed across candidates/games; imported discovery cores cannot silently change.
    for key, expected in (("hidden_size", 128), ("num_layers", 1)):
        require(config.getint("policy", key, fallback=expected) == expected, "Candidate core must be H128/L1")
    return spec


def candidates(named, campaigns):
    records = []
    for value in named:
        name, separator, filename = value.partition("=")
        require(separator and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name), "Use name=INI with a simple friendly name")
        path = Path(filename).resolve()
        records.append(dict(name=name, architecture=architecture(path), original=str(path), sha256=sha(path), origin="explicit"))
    for campaign in campaigns:
        campaign = campaign.resolve()
        # Keep every completed trial, including duplicates/dominated ones. No score filter.
        values = trials(campaign)
        require(values, "No completed native PROTEIN trials: " + str(campaign))
        log_hash = sha(campaign / "sweep.log")
        task = (campaign / "environment.txt").read_text().strip() if (campaign / "environment.txt").exists() else "connect4cnn"
        for trial in values:
            path = campaign / "metrics" / task / (trial["run_id"] + ".ini")
            records.append(dict(name=f"native-{len(records)+1}-{display_name(campaign, trial['index'])}",
                architecture=architecture(path), original=str(path), sha256=sha(path), origin="native-protein",
                proposal_task=task, trial_index=trial["index"], discovery_score=trial["score"],
                discovery_seconds=trial["cost"], discovery_log_sha256=log_hash,
                discovery_checkpoint_sha256=trial["checkpoint_sha256"]))
    require(1 <= len(records) <= 512 and len({r["name"] for r in records}) == len(records), "Need unique candidate names (one to 512)")
    return records


def job_order(records, seeds):
    for si, seed in enumerate(seeds):
        for ti, task in enumerate(TASKS):
            offset = (si + ti) % len(records)
            for candidate in records[offset:] + records[:offset]:
                yield task, seed, candidate


def base_config(source, task, budget, appearance_seed, recipe=None):
    base = panel.read(source / "config/default.ini", source / f"config/{task}.ini", source / f"ocean/{task}/compare.ini")
    if recipe: panel.apply_learner(base, panel.learner_recipe(recipe, panel.read(source / "config/default.ini")))
    for section in list(base.sections()):
        # The native train parser still requires [sweep] defaults.
        if section.startswith("sweep.") or section in ("eval_exact", "metrics"): base.remove_section(section)
    for key in list(base["policy"]):
        if key.startswith("cnn_") or key == "encoder": base.remove_option("policy", key)
    base["policy"].update(hidden_size="128", num_layers="1")
    require(base.getint("vec", "total_agents") == 64 and base.getint("train", "horizon") == 32
            and base.getint("train", "minibatch_size") == 2048 and base.getint("train", "gpus") == 1,
            "First panel requires shared 64-slot/H32/M2048 learner")
    base["base"].update(env_name=task, load_model_path="None", eval_episodes="0", eval_agents="-1",
                         checkpoint_interval=str(budget["checkpoint_steps"]//2048))
    base["train"]["total_timesteps"] = str(budget["steps"])
    base["env"].update(representation="0", representation_mode="1", representation_seed=str(appearance_seed))
    if task in ("pongcnn", "flappycnn"): base["env"]["representation_mix_catalog"] = "1"
    require(base.getint("selfplay", "enabled") == 0 and base.getint("vec", "num_policies") == 1, "No selfplay/multiple policies")
    return base


def apply_job(config, candidate, seed, directory):
    config["policy"].update({k: str(v) for k, v in candidate["architecture"].items()})
    config["base"].update(seed=str(seed), run_id=candidate["name"], checkpoint_dir=str(directory / "checkpoints"), log_dir=str(directory / "metrics"))


def check_starts(task, binary, config, suite_path, directory):
    if task == "connect4cnn": return dict(proof="declared-empty-boards-and-identity-seeds", native_manifest_executed=False)
    ev = bridge.adapter(task); suite, manifest = ev.load_suite(suite_path)
    directory.mkdir(parents=True)
    with (directory / "manifest.ini").open("x") as stream: ev.suite_config(config, suite).write(stream)
    with (directory / "episodes.csv").open("x") as stream, (directory / "stderr.txt").open("x") as errors:
        subprocess.run([str(binary), "eval_exact_manifest", str(directory / "manifest.ini")],
                       stdout=stream, stderr=errors, text=True, timeout=120, check=True)
    require(ev.validate_manifest(directory / "episodes.csv", suite) == manifest, "Native family/config worlds or raster differ")
    return dict(proof="native-family-world-and-raster", native_manifest_executed=True,
                manifest=str(directory / "episodes.csv"), manifest_sha256=sha(directory / "episodes.csv"))


def prepare(args):
    records = candidates(args.candidate, args.native_campaign)
    baselines = getattr(args, "baselines", [])
    require(len(set(baselines)) == len(baselines) and set(baselines) <= set(BASELINES), "Unknown/duplicate baseline")
    for name in baselines: records.append(dict(BASELINES[name], kind="baseline", baseline=name, origin="fixed-native-reference"))
    require(len({r["name"] for r in records}) == len(records), "Candidate name collides with fixed baseline")
    expanded = bool(baselines)
    require(args.seeds and len(set(args.seeds)) == len(args.seeds) and all(type(s) is int and 0 <= s < 2**32 for s in args.seeds), "Invalid paired training seeds")
    require(0 <= args.appearance_seed < 2**32 and 0 <= args.eval_seed <= 2**32-len(args.seeds), "Invalid appearance/evaluation seeds")
    require(1 <= args.episodes <= 1000000 and 1 <= args.slots <= 1024, "Invalid exact episode quota/slots")
    require(1 <= args.pong_max_decisions <= 16777216 and 1 <= args.breakout_max_frames <= 16777216, "Invalid administrative caps")
    overrides = {}
    for raw in getattr(args, "task_budget", []):
        task, separator, budget = raw.partition("=")
        require(separator and task in TASKS and task not in overrides, "Invalid/duplicate per-game budget")
        steps, cadence = map(int, budget.split(":"))
        overrides[task] = dict(steps=steps, checkpoint_steps=cadence)
    budgets = panel.resolve_budgets(TASKS, args.steps, args.checkpoint_steps, overrides)
    require(all(b["steps"] % 2048 == b["checkpoint_steps"] % 2048 == 0 for b in budgets.values()),
            "Budget/cadence must be positive 2048-decision multiples")
    recipes = {}
    for raw in getattr(args, "learner_recipe", []):
        task, separator, filename = raw.partition("=")
        require(separator and task in TASKS and task not in recipes, "Invalid/duplicate per-game learner")
        recipes[task] = Path(filename).resolve()
    registry_path = args.registry.resolve(); registry = json.loads(registry_path.read_text())
    require(registry["protocol"] == ("native-cnn-candidate-build-v2" if expanded else "native-cnn-candidate-build-v1")
            and set(registry["binaries"]) == set(TASKS), "Need complete matching fresh build registry; build --baselines for references")
    builds = registry_builds(registry)
    verify_sources(registry_path.parent, registry["source_sha256"], current=True)
    for task, variants in builds.items():
        for family, record in variants.items():
            require(sha(Path(record["path"])) == record["sha256"], "Changed native build")
            bridge.validate_identity(task, family, record["info"], panel.ENVIRONMENTS[task])
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        bases = {}
        for task in TASKS:
            bases[task] = base_config(ROOT, task, budgets[task], args.appearance_seed, recipes.get(task))
        geometry = panel.native_geometry(out, bases)
        assignments = panel.mixed_assignments(out, bases, args.appearance_seed)
        sources = capture(out)
        metadata = getattr(args, "policy_metadata", None)
        if metadata is not None:
            metadata_registry = policy_layout.freeze(metadata, out / "policy-metadata")
            metadata_source_binding(metadata_registry, registry, sources)
        shutil.copyfile(registry_path, out / "registry.json")
        learner_receipts = {}
        for task, path in recipes.items():
            target = out / "learners" / f"{task}.ini"; target.parent.mkdir(exist_ok=True)
            digest = sha(path); shutil.copyfile(path, target)
            require(sha(target) == sha(path) == digest, "Learner recipe changed during capture")
            learner_receipts[task] = dict(snapshot=str(target.relative_to(out)), sha256=digest, original=str(path))
        for index, record in enumerate(records):
            target = out / "candidates" / f"{index:04d}.ini"; target.parent.mkdir(exist_ok=True)
            if record.get("kind") == "baseline":
                config = panel.read(); config["policy"] = {"hidden_size": "128", "num_layers": "1", "encoder": str(record["architecture"]["encoder"])}
                with target.open("x") as stream: config.write(stream)
                record["sha256"] = sha(target)
            else:
                shutil.copyfile(record["original"], target)
                require(sha(target) == sha(Path(record["original"])) == record["sha256"], "Candidate changed during capture")
            record["snapshot"] = str(target.relative_to(out))
        jobs, suites, worlds = [], {}, {}
        for task, seed, candidate in job_order(records, args.seeds):
            budget = budgets[task]
            steps = list(range(budget["checkpoint_steps"], budget["steps"], budget["checkpoint_steps"])) + [budget["steps"]]
            config = panel.read(); config.read_dict({s: dict(bases[task][s]) for s in bases[task].sections()})
            directory = out / "jobs" / task / f"{candidate['name']}-s{seed}"; (directory / "config").mkdir(parents=True)
            apply_job(config, candidate, seed, directory)
            path = directory / "config/default.ini"
            with path.open("x") as stream: config.write(stream)
            (directory / "config" / f"{task}.ini").write_text("# Complete configuration is in default.ini.\n")
            binary_family, policy_family = candidate_families(candidate)
            selected_build = builds[task][binary_family]
            binary = Path(selected_build["path"]); ev = bridge.adapter(task)
            ev.load_config(path, selected_build["info"], policy_family)
            targets = []
            for representation in range(panel.ENVIRONMENTS[task]):
                key = f"{task}/s{seed}/r{representation}"; suite_path = out / "suites" / key / "suite.json"
                if key not in suites:
                    kwargs = dict(binary=binary, config=path, out=suite_path.parent, seed=args.eval_seed+args.seeds.index(seed),
                        offset=0, episodes=args.episodes, slots=args.slots, representation=representation, purpose="development",
                        max_decisions=args.pong_max_decisions, max_frames=args.breakout_max_frames, level_offset=0,
                        training_level_count=0, level_count=config.getint("env", "num_maps", fallback=1))
                    with redirect_stdout(io.StringIO()): ev.create_suite(SimpleNamespace(**kwargs))
                    manifest = bridge.rows(suite_path.parent / "episodes.csv")
                    world = bridge.worlds(manifest); world_key = (task, seed)
                    require(world_key not in worlds or world == worlds[world_key], "Different worlds/RNG across drawings")
                    worlds[world_key] = world
                    suites[key] = str(suite_path.relative_to(out))
                target = dict(representation=representation, suite=suites[key])
                if expanded:
                    target["start_check"] = check_starts(task, binary, config, suite_path, out / "host-checks" / f"job-{len(jobs):04d}" / f"r{representation}")
                targets.append(target)
            jobs.append(dict(id=str(directory.relative_to(out)), environment=task, candidate=candidate["name"], seed=seed,
                config=str(path.relative_to(out)), cwd=str(directory), checkpoint_steps=steps, targets=targets,
                command=[str(binary), "train", "--headless"], binary_sha256=selected_build["sha256"]))
            if expanded: jobs[-1].update(binary_family=binary_family, policy_family=policy_family, kind=candidate.get("kind", "candidate"))
            if metadata is not None:
                jobs[-1]["policy_layout"] = policy_layout.describe(out / "policy-metadata", path, task, policy_family,
                    out / "policy-layouts" / f"job-{len(jobs)-1:04d}", 60)
        verify_sources(out, sources, current=True)
        frozen = {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}
        value = dict(protocol=BASELINE_PROTOCOL if expanded else PROTOCOL, purpose="development-plumbing", candidates=records, jobs=jobs, suites=suites,
            seeds=args.seeds, steps=args.steps, task_budgets=budgets, learner_recipes=learner_receipts,
            episodes=args.episodes, slots=args.slots,
            appearance_seed=args.appearance_seed, assignments=assignments, learner_geometry=geometry,
            eval_seed=args.eval_seed, caps=dict(pong_max_decisions=args.pong_max_decisions, breakout_max_frames=args.breakout_max_frames),
            files_sha256=frozen, source_sha256=sources, native_registry=registry,
            planned_training_jobs=len(jobs), planned_evaluations=sum(len(j["targets"])*len(j["checkpoint_steps"]) for j in jobs),
            initialization="random-per-game-run", proposal_optimizer="external-existing-native-PROTEIN-or-explicit-candidates",
            cross_game_adaptive_protein_implemented=False, quality_selection_allowed=False, publication_claim_qualified=False,
            numerical_runtime_acceptance_complete=False, caps_calibrated=False, native_vector_initialization_validated=False)
        if expanded: value.update(baselines=baselines, native_working_root=str(out), gpu_executed=False,
                                  declared_decisions=sum(budgets[j["environment"]]["steps"] for j in jobs))
        if metadata is not None:
            value["policy_layout"] = dict(protocol="native-candidate-policy-layout-v1", preparation_root=str(out),
                repository_root=str(ROOT), checkpoint_bytes_checked_at_preparation=False,
                numerical_acceptance=False, full_learner_memory_accepted=False)
        save(out / "plan.json", value, exclusive=True)
        inspect(out / "plan.json")
        return value
    except BaseException as error:
        save(out / "failure.json", dict(status="failed-preparation", error=str(error), policy_executed=False))
        raise


def inspect(path, current=False):
    value = json.loads(path.read_text()); root = path.parent
    require(value["protocol"] in (PROTOCOL, BASELINE_PROTOCOL) and not (root / "failure.json").exists(), "Wrong/failed candidate packet")
    expanded = value["protocol"] == BASELINE_PROTOCOL
    for name, digest in value["files_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe frozen path")
        require(sha(root / name) == digest, "Changed frozen input: " + name)
    verify_sources(root, value["source_sha256"], current)
    builds = registry_builds(value["native_registry"])
    if "policy_layout" in value:
        require(value["policy_layout"]["protocol"] == "native-candidate-policy-layout-v1"
                and value["policy_layout"]["checkpoint_bytes_checked_at_preparation"] is False
                and value["policy_layout"]["numerical_acceptance"] is False
                and value["policy_layout"]["full_learner_memory_accepted"] is False, "Wrong candidate policy-layout scope")
        require(not current or root.resolve() == Path(value["policy_layout"]["preparation_root"]), "Cannot launch relocated layout packet")
        metadata_source_binding(policy_layout.inspect_registry(root / "policy-metadata", current), value["native_registry"], value["source_sha256"])
    else: require(all("policy_layout" not in j for j in value["jobs"]), "Layout jobs lack declaration")
    if expanded:
        require(value["native_registry"]["protocol"] == "native-cnn-candidate-build-v2"
                and value["baselines"] and len(set(value["baselines"])) == len(value["baselines"])
                and set(value["baselines"]) <= set(BASELINES), "Wrong fixed-baseline declaration")
        require({c.get("baseline") for c in value["candidates"] if c.get("kind") == "baseline"} == set(value["baselines"]),
                "Fixed baseline lost or relabeled")
        require(not current or root.resolve() == Path(value["native_working_root"]), "Cannot launch relocated packet paths")
        for candidate in value["candidates"]:
            snapshot = root / candidate["snapshot"]
            require(sha(snapshot) == candidate["sha256"], "Candidate/reference snapshot differs")
            if candidate.get("kind") == "baseline":
                require(all(candidate[k] == v for k, v in BASELINES[candidate["baseline"]].items()), "Baseline architecture/build changed")
                require(dict(panel.read(snapshot)["policy"]) == dict(encoder=str(candidate["architecture"]["encoder"]), hidden_size="128", num_layers="1"), "Reference snapshot has extra settings")
            else: require(architecture(snapshot) == candidate["architecture"], "Candidate differs from actual discovery snapshot")
    require(len(value["jobs"]) == len(value["candidates"])*len(TASKS)*len(value["seeds"]), "Incomplete candidate/game cross product")
    expected = len(value["candidates"])*len(value["seeds"])*sum(panel.ENVIRONMENTS[t]*
        ((b["steps"]-1)//b["checkpoint_steps"]+1) for t, b in value["task_budgets"].items())
    require(value["planned_evaluations"] == expected, "Incomplete drawing/checkpoint cross product")
    common, bases = {}, {}
    order = list(job_order(value["candidates"], value["seeds"]))
    for index, job in enumerate(value["jobs"]):
        if "policy_layout" in value: inspect_job_layout(value, root, index)
        config = panel.read(root / job["config"])
        candidate = next(c for c in value["candidates"] if c["name"] == job["candidate"])
        if candidate.get("kind") != "baseline":
            require(architecture(root / job["config"]) == candidate["architecture"], "Architecture changed across games")
        require(config.getint("base", "seed") == job["seed"] and config.getint("env", "representation_mode") == 1
                and config.getint("env", "representation_seed") == value["appearance_seed"], "Wrong training RNG/appearance")
        task = job["environment"]; budget = value["task_budgets"][task]
        require(config.getint("train", "total_timesteps") == budget["steps"]
                and config.getint("base", "checkpoint_interval") == budget["checkpoint_steps"]//2048, "Per-game budget/cadence changed")
        shared = normalized(config)
        shared.pop("policy")
        for key in ("seed", "run_id", "checkpoint_dir", "log_dir"): shared["base"].pop(key, None)
        require(task not in common or shared == common[task], "Learner/world changed across CNN candidates")
        common[task] = shared
        if expanded:
            require((task, job["seed"], job["candidate"]) == (order[index][0], order[index][1], order[index][2]["name"]), "Outcome-independent job order changed")
            binary_family, policy_family = candidate_families(candidate)
            selected = builds[task][binary_family]
            require(job["binary_family"] == binary_family and job["policy_family"] == policy_family
                    and job["binary_sha256"] == selected["sha256"]
                    and job["command"] == [selected["path"], "train", "--headless"], "Wrong policy/build family or command")
            directory = Path(value["native_working_root"]) / "jobs" / task / f"{candidate['name']}-s{job['seed']}"
            require(job["cwd"] == str(directory) and job["id"] == str(directory.relative_to(Path(value["native_working_root"])))
                    and job["config"] == job["id"]+"/config/default.ini", "Wrong declared native output paths")
            if task not in bases:
                recipe = value["learner_recipes"].get(task)
                bases[task] = base_config(root / "source", task, budget, value["appearance_seed"], root / recipe["snapshot"] if recipe else None)
            expected = panel.read(); expected.read_dict({s: dict(bases[task][s]) for s in bases[task].sections()})
            apply_job(expected, candidate, job["seed"], directory)
            require(normalized(expected) == normalized(config), "Resolved job differs from frozen per-game recipe, including all-family edits")
            secondary = root / job["id"] / "config" / f"{task}.ini"
            require(secondary.read_text() == "# Complete configuration is in default.ini.\n", "Secondary game INI can override matched settings")
        require([t["representation"] for t in job["targets"]] == list(range(panel.ENVIRONMENTS[job["environment"]])), "Missing drawing target")
        for target in job["targets"]:
            ev = bridge.adapter(task); loaded = ev.load_suite(root / target["suite"])
            if expanded:
                proof = target["start_check"]
                if task == "connect4cnn": require(proof == dict(proof="declared-empty-boards-and-identity-seeds", native_manifest_executed=False), "Wrong Connect4 start proof")
                else:
                    manifest = root / "host-checks" / f"job-{index:04d}" / f"r{target['representation']}" / "episodes.csv"
                    require(proof["native_manifest_executed"] is True and proof["proof"] == "native-family-world-and-raster"
                            and sha(manifest) == proof["manifest_sha256"]
                            and ev.validate_manifest(manifest, loaded[0]) == loaded[1], "Baseline/config start manifest differs")
        if current: require(sha(Path(job["command"][0])) == job["binary_sha256"], "Changed executing binary")
    return value


def hardware(expected):
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    require(not busy.strip(), "GPU busy; stop without interrupting anyone")
    text = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20)
    rows = list(csv.reader(text.splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 3, "Require one visible GPU")
    name, uuid, driver = [v.strip() for v in rows[0]]
    require(expected in name, "Wrong GPU for this scheduled mode")
    compiler = subprocess.check_output([str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"), "--version"], text=True, timeout=20)
    return dict(gpu_name=name, gpu_uuid=uuid, driver=driver, host=socket.gethostname(), cuda_compiler=compiler)


def candidate_frontiers(value, observations, mode):
    """Full per-game/drawing scalar envelopes; no raw cross-game aggregation."""
    import pixel_frontiers as reporting
    bindings, identities = [], {}
    for index, job in enumerate(value["jobs"]):
        for target in job["targets"]:
            task = job["environment"]; representation = target["representation"]
            key = task, job["candidate"], job["seed"], representation
            require(key not in identities, "Duplicate candidate/drawing/training seed")
            identities[key] = len(bindings)
            bindings.append(dict(environment=task, representation=representation, training_seed=job["seed"],
                model=job["candidate"], job_index=index, checkpoint_steps=job["checkpoint_steps"],
                training_representation=None, training_representation_mode=1,
                training_representation_seed=value["appearance_seed"],
                training_catalog_count=value["assignments"][task]["catalog_count"]))
    points = []
    for row in observations:
        key = row["environment"], row["candidate"], row["seed"], row["representation"]
        require(key in identities, "Unknown observed candidate/game/drawing")
        index = identities[key]; binding = bindings[index]
        task, treatment, drawing = reporting.condition(binding)
        points.append(dict(binding_index=index, steps=row["decisions"], environment=task,
            training_appearance=treatment, evaluation_representation=drawing, model=row["candidate"], seed=row["seed"],
            seconds=row["train_seconds"], score_lower=row["score_lower"], score_upper=row["score_upper"]))
    reduced = reporting.reduce(dict(bindings=bindings, training_seeds=value["seeds"]), points, [])
    reduced["cost_scope"] = "native-training-through-checkpoint-once-per-job"
    reduced["cross_game_adaptive_protein_implemented"] = False
    reduced["publication_claim_qualified"] = False
    reduced["quality_selection_allowed"] = False
    reduced["mode"] = mode
    # Complete tiny smokes still cannot supply a learned-quality frontier.
    if mode != "development":
        for row in reduced["means"]:
            row["frontier_seconds_lower"] = row["frontier_seconds_upper"] = None
    return reduced


def write_frontiers(out, result):
    import pixel_frontiers as reporting
    save(out / "curves.json", result)
    reporting.write_csv(out / "paired-means.csv", result["means"], ["environment", "training_appearance", "evaluation_representation", "model",
        "steps", "seeds", "metric", "seconds", "score_lower", "score_upper", "frontier_seconds_lower", "frontier_seconds_upper"])
    reporting.write_csv(out / "conditions.csv", result["conditions"], ["environment", "training_appearance", "evaluation_representation", "metric", "status", "expected", "observed", "failures", "score_bounds_kind"])
    reporting.write_csv(out / "missing.csv", result["missing"], ["binding_index", "steps"])


def coverage(path, out):
    """GPU-free explicit missing ledger, rather than plotting fabricated scores."""
    value = inspect(path.resolve())
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = candidate_frontiers(value, [], "preparation")
    write_frontiers(out, result)
    save(out / "allocation.json", dict(plan_sha256=sha(path), jobs=value["planned_training_jobs"],
        evaluations=value["planned_evaluations"], planned_episode_executions=value["planned_evaluations"]*value["episodes"],
        declared_decisions=sum(value["task_budgets"][j["environment"]]["steps"] for j in value["jobs"]),
        conditions=len(result["conditions"]), observations=0, policy_executed=False, gpu_queried=False))
    return result


def run_hardware(value, mode):
    if mode == "canary5090":
        require(len(value["candidates"]) == 1 and len(value["seeds"]) == 1
                and max(b["steps"] for b in value["task_budgets"].values()) <= 131072
                and len(value["jobs"]) == 6
                and all(len(j["checkpoint_steps"]) == 1 for j in value["jobs"])
                and value["planned_evaluations"] <= 41
                and value["episodes"] <= 17 and value["slots"] <= 16
                and value["candidates"][0].get("kind") != "baseline",
                "5090 canary exceeds bounded single-candidate allocation")
        return "5090"
    if mode == "research":
        require(len(value["candidates"]) == 1 and 1 <= len(value["seeds"]) <= 2
                and max(b["steps"] for b in value["task_budgets"].values()) <= 1048576
                and all(len(j["checkpoint_steps"]) <= 4 for j in value["jobs"])
                and value["planned_evaluations"] <= 328
                and value["episodes"] <= 33 and value["slots"] <= 32
                and value["candidates"][0].get("kind") != "baseline",
                "Research exceeds bounded single-candidate local allocation")
        return "5060"
    if mode == "mini":
        require(len(value["candidates"]) == 2 and 1 <= len(value["seeds"]) <= 2
                and max(b["steps"] for b in value["task_budgets"].values()) <= 1048576
                and all(len(j["checkpoint_steps"]) <= 4 for j in value["jobs"])
                and len(value["jobs"]) <= 24 and value["planned_evaluations"] <= 656
                and value["episodes"] <= 33 and value["slots"] <= 32,
                "Mini exceeds bounded short local allocation")
        require({c.get("baseline") for c in value["candidates"] if c.get("kind") == "baseline"} == {"nature_cnn"}
                and sum(c.get("kind") != "baseline" for c in value["candidates"]) == 1,
                "Mini requires one frozen quality candidate and Nature only")
        candidate = next(c for c in value["candidates"] if c.get("kind") != "baseline")
        require(candidate["architecture"] == architecture(ROOT / "research/recipes/panel_smoke_quality.ini"),
                "Mini accepts the frozen quality architecture only")
        return "5060"
    if mode == "smoke":
        require(len(value["candidates"]) <= 2 and len(value["seeds"]) == 1 and max(b["steps"] for b in value["task_budgets"].values()) <= 131072
                and value["episodes"] <= 33 and value["slots"] <= 32 and value["planned_evaluations"] <= 164,
                "Smoke exceeds bounded local allocation")
        return "5060"
    require(mode == "development", "Unknown execution mode")
    return "5090"


def run(args):
    require(args.allow_gpu, "Scheduled GPU execution requires --allow-gpu")
    root = args.plan.resolve().parent; value = inspect(args.plan.resolve(), current=True)
    require(args.timeout > 0 and args.train_timeout > 0 and args.eval_timeout > 0, "Positive process/campaign deadlines required")
    expected = run_hardware(value, args.mode)
    if args.mode in ("mini", "research", "canary5090"):
        require(args.timeout <= (600 if args.mode == "canary5090" else 900)
                and args.train_timeout <= 30 and args.eval_timeout <= 30,
                "Mini requires bounded campaign/process deadlines")
    out = root / "execution"; out.mkdir(exist_ok=False)
    deadline = time.monotonic() + args.timeout
    record = dict(status="running", mode=args.mode, plan_sha256=sha(args.plan), jobs=[], evaluations=[], repeat_checks=[],
                  learning_quality_claim=False, started_monotonic_ns=time.monotonic_ns())
    if "policy_layout" in value: record["layout_checks"] = []
    def remaining(limit):
        seconds = deadline-time.monotonic()
        require(seconds > 0, "Campaign deadline reached; preserve partial evidence")
        return min(limit, seconds)
    try:
        for index, job in enumerate(value["jobs"]):
            inspect(args.plan.resolve(), current=True)
            directory = root / job["id"]; task = job["environment"]
            if "policy_layout" in value:
                destination = out / "policy-layouts" / f"job-{index:04d}"
                policy_layout.describe(root / "policy-metadata", root / job["config"], task,
                    job.get("policy_family", "flex"), destination, remaining(60))
                inspect_job_layout(value, root, index, destination, execution=True)
                record["layout_checks"].append(dict(job_index=index, result=str(destination / "result.json"),
                    result_sha256=sha(destination / "result.json")))
                save(out / "progress.json", record)
            lock_path = ROOT / "build/connect4cnn/hardware-benchmark.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                info = hardware(expected)
                if "hardware" in record: require(info == record["hardware"], "Hardware changed within campaign")
                else: record["hardware"] = info
                environment = dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS="1")
                timed = process(job["command"], directory, directory / "train.txt", remaining(args.train_timeout), environment)
            checkpoint_root = directory / "checkpoints" / task / job["candidate"]
            paths = {step: checkpoint_root / f"{step:016d}.bin" for step in job["checkpoint_steps"]}
            sizes = {checkpoint_size(job, path) for path in paths.values()}
            require(len(sizes) == 1, "Invalid checkpoint byte sizes")
            parameters = next(iter(sizes))
            for path in paths.values(): require(np.isfinite(np.fromfile(path, dtype=np.float32)).all(), "Nonfinite checkpoint")
            config = panel.read(root / job["config"]); resolved = panel.read(checkpoint_root / "resolved.ini")
            require(normalized(config) == normalized(resolved), "Native resolved training INI differs")
            times = parse_checkpoints((directory / "train.txt").read_text(), timed, job["checkpoint_steps"], parameters)
            job_record = dict(job_index=index, id=job["id"], process=timed, parameters=parameters,
                process_seconds=(timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9,
                checkpoint_seconds=times, checkpoints={str(s): dict(path=str(p), sha256=sha(p)) for s, p in paths.items()})
            job_record["process_sps"] = value["task_budgets"][task]["steps"]/job_record["process_seconds"]
            record["jobs"].append(job_record)
            save(out / "progress.json", record)
            for step, checkpoint in paths.items():
                for target in job["targets"]:
                    # Adapter owns its reservation. Don't hold the training lock here.
                    destination = out / "evaluations" / f"job-{index:04d}" / f"s{step}-r{target['representation']}"
                    kwargs = dict(binary=Path(job["command"][0]), config=root / job["config"], checkpoint=checkpoint,
                        suite=root / target["suite"], out=destination, family=job.get("policy_family", "flex"), timeout=remaining(args.eval_timeout), eager=False, prepare_only=False)
                    with redirect_stdout(io.StringIO()): bridge.adapter(task).run(SimpleNamespace(**kwargs))
                    result = json.loads((destination / "result.json").read_text())
                    require(result["status"] == "ok", "Evaluator failed")
                    record["evaluations"].append(dict(job_index=index, step=step, representation=target["representation"],
                        result=str(destination / "result.json"), result_sha256=sha(destination / "result.json"),
                        counts=result["counts"], training_seconds=times[step]))
                    save(out / "progress.json", record)
            # A same-batch repeat/eager smoke for every candidate/game, using
            # drawing 0 and its final checkpoint. This is not independent math.
            if args.mode in ("smoke", "mini", "canary5090"):
                target = job["targets"][0]; step = job["checkpoint_steps"][-1]
                reference = out / "evaluations" / f"job-{index:04d}" / f"s{step}-r0" / "episodes.csv"
                for mode in ("repeat", "eager"):
                    destination = out / "repeat-checks" / f"job-{index:04d}" / mode
                    kwargs = dict(binary=Path(job["command"][0]), config=root / job["config"], checkpoint=paths[step],
                        suite=root / target["suite"], out=destination, family=job.get("policy_family", "flex"), timeout=remaining(args.eval_timeout), eager=mode=="eager", prepare_only=False)
                    with redirect_stdout(io.StringIO()): bridge.adapter(task).run(SimpleNamespace(**kwargs))
                    require((destination / "episodes.csv").read_bytes() == reference.read_bytes(), "Repeat/graph-eager episode bytes differ")
                    record["repeat_checks"].append(dict(job_index=index, mode=mode, result=str(destination / "result.json"),
                        result_sha256=sha(destination / "result.json"), episodes_sha256=sha(reference)))
                    save(out / "progress.json", record)
        require(len(record["evaluations"]) == value["planned_evaluations"], "Incomplete deterministic drawing panel")
        inspect(args.plan.resolve(), current=True)
        record["status"] = "ok"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_monotonic_ns"] = time.monotonic_ns()
        save(out / "result.json", record)
        save(out / "progress.json", record)
    return record


def audit(path, out):
    """Offline audit/recompute of retained native receipts, never a GPU query."""
    import pixel_frontiers as reporting
    path = path.resolve(); root = path.parent; value = inspect(path)
    result_path = root / "execution/result.json"; result = json.loads(result_path.read_text())
    inputs = reporting.Inputs(); inputs.track(path); inputs.track(result_path)
    for name in value["files_sha256"]: inputs.track(root / name)
    reporting_plan = dict(binaries=registry_builds(value["native_registry"]))
    def evaluate(entry, destination):
        job = value["jobs"][entry["job_index"]]
        target = next(t for t in job["targets"] if t["representation"] == entry["representation"])
        trained_entry = next(e for e in result["jobs"] if e["job_index"] == entry["job_index"])
        binding = dict(environment=job["environment"], suite=target["suite"], config_sha256=sha(root / job["config"]),
                       binary_family=job.get("binary_family", "default"), policy_family=job.get("policy_family", "flex"))
        trained = dict(parameters=trained_entry["parameters"], times={int(s): t for s, t in trained_entry["checkpoint_seconds"].items()},
                       checkpoints={int(s): Path(record["path"]) for s, record in trained_entry["checkpoints"].items()})
        return reporting.evaluation(inputs, destination, binding, entry["step"], trained, reporting_plan, root)
    require(result["plan_sha256"] == sha(path), "Result belongs to a different plan")
    if "policy_layout" in value:
        checks = result["layout_checks"]
        require([r["job_index"] for r in result["jobs"]] == list(range(len(result["jobs"]))),
                "Completed training is not the declared prefix of layout-checked jobs")
        require([r["job_index"] for r in checks] == list(range(len(checks))) and len(checks) <= len(value["jobs"]),
                "Duplicate/missing/unassigned layout checks")
        require(len(checks) >= len(result["jobs"]) and (result["status"] != "ok" or len(checks) == len(value["jobs"])),
                "Training lacks repeated pre-GPU registration")
        for entry in checks:
            destination = root / "execution/policy-layouts" / f"job-{entry['job_index']:04d}"
            require(Path(entry["result"]) == destination / "result.json" and sha(destination / "result.json") == entry["result_sha256"],
                    "Changed repeated registration identity")
            inspect_job_layout(value, root, entry["job_index"], destination, execution=True)
            for name in ("result.json", "descriptor.json", "native.txt", "native.txt.json"):
                inputs.track(destination / name)
    observations, training, previous_end = [], [], None
    for entry in result["jobs"]:
        job = value["jobs"][entry["job_index"]]; directory = root / job["id"]
        if "policy_layout" in value:
            require(entry["parameters"] == job["policy_layout"]["parameters"], "Training parameter count differs from registration")
        timed = json.loads((directory / "train.txt.json").read_text())
        inputs.track(directory / "train.txt.json"); inputs.track(directory / "train.txt")
        require(timed == entry["process"] and timed["status"] == "ok" and timed["returncode"] == 0
                and timed["command"] == job["command"] and timed["cwd"] == job["cwd"]
                and timed["environment"]["PUFFER_CHECKPOINT_RECEIPTS"] == "1", "Wrong native training process receipt")
        require(previous_end is None or timed["launch_monotonic_ns"] >= previous_end, "Training processes overlap")
        previous_end = timed["end_monotonic_ns"]
        seconds = (timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9
        require(seconds > 0 and seconds == entry["process_seconds"], "Invalid training clock")
        times = parse_checkpoints((directory / "train.txt").read_text(), timed, job["checkpoint_steps"], entry["parameters"])
        require({str(s): t for s, t in times.items()} == entry["checkpoint_seconds"], "Altered native checkpoint times")
        checkpoint_root = directory / "checkpoints" / job["environment"] / job["candidate"]
        inputs.track(checkpoint_root / "resolved.ini")
        require(normalized(panel.read(checkpoint_root / "resolved.ini")) == normalized(panel.read(root / job["config"])), "Changed native resolved learner")
        for step, checkpoint in entry["checkpoints"].items():
            checkpoint_path = Path(checkpoint["path"])
            inputs.track(checkpoint_path)
            require(checkpoint_path == checkpoint_root / f"{int(step):016d}.bin" and sha(checkpoint_path) == checkpoint["sha256"], "Changed checkpoint identity")
            require(checkpoint_size(job, checkpoint_path) == entry["parameters"], "Audited checkpoint layout differs")
            weights = np.fromfile(checkpoint_path, dtype=np.float32)
            require(weights.size == entry["parameters"] and np.isfinite(weights).all(), "Changed/malformed weights")
        metric_path = directory / "metrics" / job["environment"] / (job["candidate"]+".ini")
        require(metric_path.is_file(), "Missing native training metrics")
        inputs.track(metric_path)
        training.append(dict(candidate=job["candidate"], environment=job["environment"], seed=job["seed"], parameters=entry["parameters"],
            train_seconds=seconds, process_sps=value["task_budgets"][job["environment"]]["steps"]/seconds,
            native_avg_sps=None, native_metrics_ini=str(metric_path)))
    seen = set()
    for entry in result["evaluations"]:
        job = value["jobs"][entry["job_index"]]; key = (entry["job_index"], entry["step"], entry["representation"])
        require(key not in seen and entry["step"] in job["checkpoint_steps"], "Duplicate/unassigned evaluation")
        seen.add(key)
        target = next(t for t in job["targets"] if t["representation"] == entry["representation"])
        destination = root / "execution/evaluations" / f"job-{entry['job_index']:04d}" / f"s{entry['step']}-r{entry['representation']}"
        require(Path(entry["result"]) == destination / "result.json" and sha(Path(entry["result"])) == entry["result_sha256"], "Changed evaluation result")
        receipt = json.loads(Path(entry["result"]).read_text()); ev = bridge.adapter(job["environment"])
        checkpoint = root / job["id"] / "checkpoints" / job["environment"] / job["candidate"] / f"{entry['step']:016d}.bin"
        suite_path = root / target["suite"]; loaded = ev.load_suite(suite_path)
        suite = loaded if job["environment"] == "connect4cnn" else loaded[0]
        for name, artifact in (("checkpoint_sha256", checkpoint), ("binary_sha256", Path(job["command"][0])),
                               ("training_ini_sha256", root / job["config"]), ("suite_sha256", suite_path),
                               ("episodes_sha256", destination / "episodes.csv"), ("effective_ini_sha256", destination / "config/default.ini")):
            require(receipt[name] == sha(artifact), "Changed evaluation identity/input: " + name)
        require(receipt["status"] == "ok" and receipt["family"] == job.get("policy_family", "flex"), "Unsuccessful/wrong-family evaluation")
        counts = ev.audit_csv(destination / "episodes.csv", suite, job["environment"]) if job["environment"] == "connect4cnn" else ev.audit_csv(destination / "episodes.csv", *loaded)
        require(counts == receipt["counts"] == entry["counts"], "Stored score differs from assigned episode audit")
        require(counts["episodes"] == value["episodes"], "Wrong deterministic quota")
        train_entry = next(e for e in result["jobs"] if e["job_index"] == entry["job_index"])
        require(entry["training_seconds"] == train_entry["checkpoint_seconds"][str(entry["step"])], "Drawing was charged different training time")
        measured = evaluate(entry, destination / "result.json")
        observations.append(dict(candidate=job["candidate"], environment=job["environment"], seed=job["seed"], representation=entry["representation"],
            decisions=entry["step"], train_seconds=entry["training_seconds"], score_lower=measured["score_lower"], score_upper=measured["score_upper"], counts=counts))
    expected = {(i, s, t["representation"]) for i, j in enumerate(value["jobs"]) for s in j["checkpoint_steps"] for t in j["targets"]}
    if result["status"] == "ok": require(seen == expected and len(training) == len(value["jobs"]), "Incomplete successful panel")
    repeat_keys = set()
    for entry in result["repeat_checks"]:
        key = (entry["job_index"], entry["mode"])
        require(key not in repeat_keys and entry["mode"] in ("repeat", "eager"), "Duplicate/unknown repeat check")
        repeat_keys.add(key)
        require(sha(Path(entry["result"])) == entry["result_sha256"], "Changed repeat result")
        repeat = json.loads(Path(entry["result"]).read_text())
        require(repeat["status"] == "ok" and repeat["episodes_sha256"] == entry["episodes_sha256"]
                == sha(Path(entry["result"]).parent / "episodes.csv"), "Repeat bytes changed")
        job = value["jobs"][entry["job_index"]]
        reference = root / "execution/evaluations" / f"job-{entry['job_index']:04d}" / f"s{job['checkpoint_steps'][-1]}-r0/episodes.csv"
        require(sha(reference) == entry["episodes_sha256"], "Repeat differs from original evaluation")
        evaluate(dict(job_index=entry["job_index"], step=job["checkpoint_steps"][-1], representation=0), Path(entry["result"]))
    if result["status"] == "ok" and result["mode"] in ("smoke", "mini", "canary5090"):
        require(repeat_keys == {(i, mode) for i in range(len(value["jobs"])) for mode in ("repeat", "eager")}, "Missing smoke repeat/eager check")
    inputs.verify()
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    analysis = dict(status=result["status"], mode=result["mode"], plan_sha256=sha(path), result_sha256=sha(result_path),
        planned_jobs=len(value["jobs"]), audited_jobs=len(training), planned_evaluations=len(expected), audited_evaluations=len(seen),
        missing_evaluations=[list(k) for k in sorted(expected-seen)], training=training, observations=observations,
        repeat_checks=len(result["repeat_checks"]), hardware=result.get("hardware"), inputs_sha256=inputs.hashes, frontier_claim_qualified=False,
        quality_selection_allowed=False, cross_game_aggregate_score=None, native_vector_initialization_validated=False,
        warning="Short plumbing results are not learning evidence, independent numerical acceptance, calibrated budgets/caps or a publication frontier.")
    if value["protocol"] == BASELINE_PROTOCOL:
        curves = candidate_frontiers(value, observations, result["mode"])
        analysis["curve_status"] = curves["status"]
        write_frontiers(out, curves)
    save(out / "analysis.json", analysis)
    with (out / "training.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training[0]) if training else ["candidate", "environment"]); writer.writeheader(); writer.writerows(training)
    with (out / "observations.csv").open("x", newline="") as stream:
        fields = ["candidate", "environment", "seed", "representation", "decisions", "train_seconds", "score_lower", "score_upper", "counts"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({**row, "counts": json.dumps(row["counts"], sort_keys=True)} for row in observations)
    lines = ["# Native CNN multi-game development panel", "", f"Status: {result['status']}. {len(training)}/{len(value['jobs'])} training jobs; {len(seen)}/{len(expected)} drawing evaluations; {len(result['repeat_checks'])} exact-byte repeat/eager checks.", "",
        "One CNN architecture per candidate across all six games; learners are fixed per game. Each checkpoint is evaluated on every fixed drawing with paired episode identities/RNG. Training is charged once per job. All scores, including zero scores and censoring, are retained. No cross-game raw-score average or best-game winner.", "",
        analysis["warning"], "", "| Game | Candidate | Process seconds | Process SPS | Parameters |", "|---|---|---:|---:|---:|"]
    lines.extend(f"| {r['environment']} | {r['candidate']} | {r['train_seconds']:.3f} | {r['process_sps']:,.0f} | {r['parameters']:,} |" for r in training)
    lines.extend(["", "Native metrics retain SPS/uptime/performance arrays in their original INIs. Process SPS above includes startup/checkpoints and is not native average SPS. See observations.csv for every drawing/score and analysis.json for complete counts and missing cells.", ""])
    (out / "REPORT.md").write_text("\n".join(lines))
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    compiler = sub.add_parser("build"); compiler.add_argument("--out", type=Path, required=True)
    compiler.add_argument("--baselines", action="store_true", help="Compile all three native families for every game; no GPU query")
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True); prep.add_argument("--registry", type=Path, required=True)
    prep.add_argument("--candidate", action="append", default=[]); prep.add_argument("--native-campaign", action="append", default=[], type=Path)
    prep.add_argument("--baselines", nargs="+", choices=tuple(BASELINES), default=[], help="Fixed references, not discovery dimensions")
    prep.add_argument("--policy-metadata", type=Path, help="Existing three-family host registration build; check layouts before GPU use")
    prep.add_argument("--seeds", nargs="+", type=int, default=[57173]); prep.add_argument("--appearance-seed", type=int, default=35173)
    prep.add_argument("--steps", type=int, default=65536); prep.add_argument("--checkpoint-steps", type=int, default=65536)
    prep.add_argument("--task-budget", action="append", default=[], help="ENV=STEPS:CHECKPOINT_DECISIONS, matched across CNNs")
    prep.add_argument("--learner-recipe", action="append", default=[], help="ENV=INI numeric learner overlay, matched across CNNs")
    prep.add_argument("--eval-seed", type=int, default=67173); prep.add_argument("--episodes", type=int, default=17)
    prep.add_argument("--slots", type=int, default=16); prep.add_argument("--pong-max-decisions", type=int, default=512)
    prep.add_argument("--breakout-max-frames", type=int, default=2048)
    check = sub.add_parser("inspect"); check.add_argument("--plan", type=Path, required=True)
    review = sub.add_parser("audit"); review.add_argument("--plan", type=Path, required=True); review.add_argument("--out", type=Path, required=True)
    ledger = sub.add_parser("coverage"); ledger.add_argument("--plan", type=Path, required=True); ledger.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("run"); execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--mode", choices=("smoke", "mini", "research", "canary5090", "development"), required=True); execute.add_argument("--allow-gpu", action="store_true")
    execute.add_argument("--timeout", type=int, default=900); execute.add_argument("--train-timeout", type=int, default=60)
    execute.add_argument("--eval-timeout", type=int, default=30)
    args = parser.parse_args()
    if args.command == "build": print(build(args.out, args.baselines))
    elif args.command == "prepare":
        value = prepare(args); print(json.dumps({k: value[k] for k in ("planned_training_jobs", "planned_evaluations", "episodes", "steps")}, indent=2))
    elif args.command == "inspect":
        value = inspect(args.plan); print(json.dumps(dict(status="valid-prepared-inputs", jobs=len(value["jobs"]), evaluations=value["planned_evaluations"], policy_executed=False), indent=2))
    elif args.command == "audit":
        value = audit(args.plan, args.out); print(json.dumps({k: value[k] for k in ("status", "audited_jobs", "audited_evaluations", "repeat_checks")}, indent=2))
    elif args.command == "coverage":
        value = coverage(args.plan, args.out); print(json.dumps(dict(status=value["status"], missing=len(value["missing"]), observed=len(value["points"])), indent=2))
    else:
        value = run(args); print(json.dumps(dict(status=value["status"], jobs=len(value["jobs"]), evaluations=len(value["evaluations"])), indent=2))


if __name__ == "__main__": main()
