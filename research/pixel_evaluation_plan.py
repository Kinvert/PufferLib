#!/usr/bin/env python3
"""Bind prepared native pixel jobs to paired development suites; no policies/GPU.

Only eval_exact_info and eval_exact_manifest native commands execute. This is
external configuration/provenance glue, not a training or search framework.
"""
import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import audit_pixel_panel as audit
import eval_acceptance as acceptance
import prepare_pixel_robustness as panel
from claim import require, save, sha
import deterministic_eval as connect4

PROTOCOL = "pixel-development-evaluation-binding-v1"
TRANSFER_PROTOCOL = "pixel-development-evaluation-binding-v2"
MIXED_PROTOCOL = "pixel-development-evaluation-binding-v3"
BUILD_FAMILIES = ("default", "impala", "impoola")
POLICY_FAMILIES = dict(flex_quality="flex", nature_cnn="nature", impala_cnn="impala", impoola_cnn="impoola")


def binding_specs(value, appearances, catalogs=None):
    require(appearances in ("matched", "all"), "Wrong evaluation appearance mode")
    catalogs = panel.ENVIRONMENTS if catalogs is None else catalogs
    for index, job in enumerate(value["jobs"]):
        targets = range(catalogs[job["environment"]]) if appearances == "all" else [job["representation"]]
        for representation in targets:
            yield index, job, representation


def adapter(task):
    return connect4 if task == "connect4cnn" else acceptance.adapter(task)


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def verify(root, entries):
    require(hashes(root) == entries, "Changed/missing/extra frozen packet files")


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def worlds(manifest):
    # Keep all game/RNG/level columns. Only the raster may change with drawing.
    return [{k: v for k, v in row.items() if k not in ("representation", "start_observation_hash")}
            for row in manifest]


def validate_identity(task, family, identity, count):
    require(identity["environment"] == task and identity["rules"] == panel.ENVIRONMENT_RULES[task]
            and identity["float32"] is True and identity["receipt_version"] == (2 if task == "connect4cnn" else 1),
            "Wrong binary environment/rules/precision")
    require(identity["default_encoder"] == ("tiny" if family == "default" else family), "Wrong compiled encoder family")
    if task in ("pongcnn", "flappycnn") or "representation_count" in identity:
        available = identity.get("representation_count", 4 if task == "flappycnn" else 0)
        require(type(available) is int and available >= count, "Stale native drawing catalog")


def binary_registry(path, tasks, catalogs):
    value = json.loads(path.read_text())
    require(isinstance(value, dict) and set(value) == set(tasks), "Registry must cover exactly the panel tasks")
    registry = {}
    for task in tasks:
        require(isinstance(value[task], dict) and set(value[task]) == set(BUILD_FAMILIES), "Need all three native build families")
        registry[task] = {}
        for family, name in value[task].items():
            require(isinstance(name, str) and name, "Invalid binary path")
            binary = (path.parent / name).resolve()
            digest = sha(binary)
            ev = adapter(task)
            identity = (json.loads(subprocess.check_output([str(binary), "eval_exact_info"], text=True, timeout=30))
                        if task == "connect4cnn" else ev.info(binary))
            validate_identity(task, family, identity, catalogs[task])
            require(sha(binary) == digest, "Binary changed during host metadata query")
            registry[task][family] = dict(path=str(binary), sha256=digest, info=identity)
    return registry


def check_binaries(registry):
    for builds in registry.values():
        for build in builds.values():
            require(sha(Path(build["path"])) == build["sha256"], "Registered binary changed")


def prepare(args):
    source = args.panel.resolve(); value = audit.load(source)
    catalogs = audit.catalog_counts(source.parent, value)
    appearances = getattr(args, "evaluation_appearances", "matched")
    require(appearances in ("matched", "all"), "Wrong evaluation appearance mode")
    mixed = value["version"] == "pixel-robustness-preparation-v4"
    require(not mixed or appearances == "all", "Mixed training requires explicit all-drawing fixed evaluation")
    require(type(args.seed) is int and 0 <= args.seed <= 2**32-len(value["seeds"]), "Evaluation seed allocation overflows")
    require(type(args.episodes) is int and 1 <= args.episodes <= 1000000
            and type(args.slots) is int and 1 <= args.slots <= 1024, "Wrong episode quota/batch")
    require(type(args.pong_max_decisions) is int and 1 <= args.pong_max_decisions <= 16777216
            and type(args.breakout_max_frames) is int and 1 <= args.breakout_max_frames <= 16777216, "Wrong development caps")
    out = args.out.resolve()
    require(not out.is_relative_to(source.parent) and not source.parent.is_relative_to(out), "Output overlaps input panel")
    out.mkdir(parents=True, exist_ok=False)
    try:
        original = hashes(source.parent); registry_digest = sha(args.registry)
        registry = binary_registry(args.registry.resolve(), value["environments"], catalogs)
        tool_paths = {Path(__file__), Path(audit.__file__), Path(panel.__file__), Path(acceptance.__file__),
                      Path(connect4.__file__), panel.ROOT / "ocean/connect4cnn/claim.py"}
        tool_paths.update(Path(adapter(task).__file__) for task in value["environments"])
        tools = {str(p.relative_to(panel.ROOT)): sha(p) for p in tool_paths}
        shutil.copytree(source.parent, out / "panel")
        verify(out / "panel", original); audit.load(out / "panel/protocol.json")
        shutil.copyfile(args.registry, out / "registry.json")
        for name in tools:
            target = out / "tools" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(panel.ROOT / name, target)
        require(hashes(out / "tools") == tools, "Tool changed during capture")
        conditions, bindings, reference_worlds = {}, [], {}
        for binding_index, (index, job, representation) in enumerate(binding_specs(value, appearances, catalogs)):
            task, seed = (job[k] for k in ("environment", "seed"))
            ev = adapter(task); family = dict(impala_cnn="impala", impoola_cnn="impoola").get(job["model"], "default")
            binary = Path(registry[task][family]["path"])
            config = out / "panel" / job["config"]
            ev.load_config(config, registry[task][family]["info"], POLICY_FAMILIES[job["model"]])
            key = (task, representation, seed)
            suite_name = f"suites/{task}/r{representation}/s{seed}/suite.json"
            suite_path = out / suite_name
            if key not in conditions:
                # Allocation never depends on model, drawing, checkpoint or observed score.
                evaluation_seed = args.seed + value["seeds"].index(seed)
                kwargs = dict(binary=binary, config=config, out=suite_path.parent, seed=evaluation_seed,
                              offset=0, episodes=args.episodes, slots=args.slots, representation=representation,
                              purpose="development", max_decisions=args.pong_max_decisions,
                              max_frames=args.breakout_max_frames, level_offset=0, training_level_count=0)
                if task == "mazecnn": kwargs["level_count"] = audit.config(config).getint("env", "num_maps")
                with redirect_stdout(io.StringIO()): ev.create_suite(SimpleNamespace(**kwargs))
                loaded = ev.load_suite(suite_path)
                suite = loaded if task == "connect4cnn" else loaded[0]
                manifest = rows(suite_path.parent / "episodes.csv")
                world_key = (task, seed)
                if world_key in reference_worlds:
                    require(worlds(manifest) == reference_worlds[world_key], "Starting worlds changed across drawings")
                else: reference_worlds[world_key] = worlds(manifest)
                conditions[key] = dict(suite=suite, manifest=manifest)
            condition = conditions[key]
            proof = "declared-empty-boards-and-identity-seeds" if task == "connect4cnn" else "native-host-world-and-raster"
            check_name = None
            if task != "connect4cnn":
                check_id = f"binding-{binding_index:04d}" if appearances == "all" else f"job-{index:04d}"
                directory = out / "host-checks" / check_id; directory.mkdir(parents=True)
                with (directory / "manifest.ini").open("x") as stream:
                    ev.suite_config(audit.config(config), condition["suite"]).write(stream)
                with (directory / "episodes.csv").open("x") as stream, (directory / "stderr.txt").open("x") as errors:
                    subprocess.run([str(binary), "eval_exact_manifest", str(directory / "manifest.ini")],
                                   stdout=stream, stderr=errors, text=True, timeout=120, check=True)
                manifest = ev.validate_manifest(directory / "episodes.csv", condition["suite"])
                require(manifest == condition["manifest"], "Native family/config starting worlds or pixels differ")
                check_name = str(directory.relative_to(out))
            binding = dict(job_index=index, environment=task, representation=representation, training_seed=seed,
                model=job["model"], policy_family=POLICY_FAMILIES[job["model"]], binary_family=family,
                config="panel/"+job["config"], config_sha256=job["config_sha256"], suite=suite_name,
                evaluation_seed=condition["suite"]["seed"], checkpoint_steps=job["checkpoint_steps"],
                evaluation_gate=job["evaluation_gate"], start_proof=proof, host_check=check_name)
            if appearances == "all":
                binding.update(training_representation=None if mixed else job["representation"], evaluation_representation=representation)
                if mixed:
                    record = value["appearance_assignments"][task]
                    binding.update(training_representation_mode=1, training_representation_seed=value["appearance_seed"],
                                   training_catalog_count=record["catalog_count"], training_mix_catalog=record["mix_catalog"])
            bindings.append(binding)
        verify(source.parent, original); check_binaries(registry)
        require(sha(args.registry) == registry_digest and all(sha(panel.ROOT / name) == digest for name, digest in tools.items()),
                "Preparation input/tool changed")
        result = dict(protocol=MIXED_PROTOCOL if mixed else TRANSFER_PROTOCOL if appearances == "all" else PROTOCOL,
            status="prepared-not-executed", purpose="development",
            publication_confirmation_launchable=False, gpu_execution_authorized=False, policy_executed=False,
            caps_calibrated=False, maze_unseen_level_exclusion_certified=False,
            compilation_source_closure_certified=False, paired_worlds_checked=True,
            tasks=value["environments"], training_seeds=value["seeds"], evaluation_seed=args.seed,
            episodes=args.episodes, slots=args.slots, conditions=len(conditions), bindings=bindings, binaries=registry,
            tools_sha256=tools, caps=dict(pong_max_decisions=args.pong_max_decisions, breakout_max_frames=args.breakout_max_frames),
            files_sha256=hashes(out), warning="Binary SHA/metadata are not a current-source build certificate. Host manifests are not policy episodes. Runtime/math/learning/inference gates remain open.")
        if appearances == "all":
            result.update(evaluation_appearances="all", training_jobs=len(value["jobs"]),
                transfer_scope="same-policy-checkpoints-across-fixed-drawings",
                unseen_appearance_selection_exclusion_certified=False)
        if mixed: result["training_appearances"] = "mixed-fixed-per-slot"
        save(out / "plan.json", result)
        inspect(out / "plan.json", check_native=True)
        return result
    except Exception as error:
        save(out / "failure.json", dict(status="failed-preparation", error=str(error), policy_executed=False))
        raise


def inspect(path, check_native=False):
    root = path.parent; plan = json.loads(path.read_text())
    require(not (root / "failure.json").exists(), "Preparation failed; preserve it and regenerate a fresh packet")
    require(plan["protocol"] in (PROTOCOL, TRANSFER_PROTOCOL, MIXED_PROTOCOL) and plan["status"] == "prepared-not-executed"
            and plan["purpose"] == "development",
            "Wrong evaluation binding protocol/status")
    mixed = plan["protocol"] == MIXED_PROTOCOL
    transfer = plan["protocol"] in (TRANSFER_PROTOCOL, MIXED_PROTOCOL)
    appearances = "all" if transfer else "matched"
    if transfer:
        require(plan["evaluation_appearances"] == "all"
                and plan["transfer_scope"] == "same-policy-checkpoints-across-fixed-drawings"
                and plan["unseen_appearance_selection_exclusion_certified"] is False,
                "Wrong transfer mode or improperly certified unseen appearances")
    else:
        require(not any(k in plan for k in ("evaluation_appearances", "training_jobs", "transfer_scope",
                "unseen_appearance_selection_exclusion_certified")), "Legacy binding cannot declare transfer")
    for key in ("publication_confirmation_launchable", "gpu_execution_authorized", "policy_executed", "caps_calibrated",
                "maze_unseen_level_exclusion_certified", "compilation_source_closure_certified"):
        require(plan[key] is False, "Preparation improperly upgraded an execution/claim gate")
    entries = hashes(root); entries.pop("plan.json", None)
    require(entries == plan["files_sha256"], "Changed/missing/extra evaluation packet files")
    require(hashes(root / "tools") == plan["tools_sha256"], "Frozen tool receipts differ")
    value = audit.load(root / "panel/protocol.json")
    catalogs = audit.catalog_counts(root / "panel", value)
    require(mixed == (value["version"] == "pixel-robustness-preparation-v4"), "Mixed training requires the mixed binding protocol")
    if mixed:
        require(plan["training_appearances"] == "mixed-fixed-per-slot", "Wrong mixed training declaration")
    else:
        require("training_appearances" not in plan, "Fixed training cannot declare mixed appearances")
    specifications = list(binding_specs(value, appearances, catalogs))
    require(plan["tasks"] == value["environments"] and plan["training_seeds"] == value["seeds"]
            and len(plan["bindings"]) == len(specifications), "Wrong task/seed/job allocation")
    if transfer:
        require(type(plan["training_jobs"]) is int and plan["training_jobs"] == len(value["jobs"]),
                "Transfer bindings must not invent additional training jobs")
    require(set(plan["binaries"]) == set(plan["tasks"]), "Wrong binary task allocation")
    for task, builds in plan["binaries"].items():
        require(set(builds) == set(BUILD_FAMILIES), "Wrong binary family allocation")
        for family, build in builds.items():
            require(Path(build["path"]).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", build["sha256"]) is not None,
                    "Wrong binary path/hash receipt")
            validate_identity(task, family, build["info"], catalogs[task])
    condition_names, reference_worlds = set(), {}
    for binding_index, (binding, (index, job, representation)) in enumerate(zip(plan["bindings"], specifications)):
        task = job["environment"]; ev = adapter(task)
        require(binding["job_index"] == index and binding["environment"] == task and binding["representation"] == representation
                and binding["training_seed"] == job["seed"] and binding["model"] == job["model"]
                and binding["config"] == "panel/"+job["config"] and binding["config_sha256"] == job["config_sha256"]
                and binding["checkpoint_steps"] == job["checkpoint_steps"] and binding["evaluation_gate"] == job["evaluation_gate"],
                "Job binding differs from captured training panel")
        if transfer:
            require((binding["training_representation"] is None if mixed else
                    type(binding["training_representation"]) is int and binding["training_representation"] == job["representation"])
                    and type(binding["evaluation_representation"]) is int
                    and binding["evaluation_representation"] == representation, "Wrong training/evaluation drawing allocation")
        else:
            require(not any(k in binding for k in ("training_representation", "evaluation_representation")),
                    "Legacy binding cannot declare transfer drawings")
        mixed_keys = ("training_representation_mode", "training_representation_seed", "training_catalog_count", "training_mix_catalog")
        if mixed:
            record = value["appearance_assignments"][task]
            require(all(type(binding[k]) is int for k in mixed_keys[:-1])
                    and binding["training_representation_mode"] == 1 and binding["training_representation_seed"] == value["appearance_seed"]
                    and binding["training_catalog_count"] == record["catalog_count"]
                    and binding["training_mix_catalog"] == record["mix_catalog"], "Wrong mixed training binding seed/catalog")
        else:
            require(not any(k in binding for k in mixed_keys), "Fixed training cannot declare mixed binding keys")
        family = dict(impala_cnn="impala", impoola_cnn="impoola").get(job["model"], "default")
        require(binding["binary_family"] == family and binding["policy_family"] == POLICY_FAMILIES[job["model"]], "Wrong policy/build family")
        config = audit.relative(root, binding["config"])
        ev.load_config(config, plan["binaries"][task][family]["info"], binding["policy_family"])
        expected_name = f"suites/{task}/r{representation}/s{job['seed']}/suite.json"
        require(binding["suite"] == expected_name, "Wrong suite allocation")
        suite_path = root / expected_name; loaded = ev.load_suite(suite_path)
        suite = loaded if task == "connect4cnn" else loaded[0]
        expected_seed = plan["evaluation_seed"] + value["seeds"].index(job["seed"])
        require(suite["seed"] == binding["evaluation_seed"] == expected_seed and suite["offset"] == 0
                and suite["episodes"] == plan["episodes"] and suite["slots"] == plan["slots"]
                and suite["representation"] == representation and suite["purpose"] == "development", "Wrong paired episode block")
        if task == "pongcnn": require(suite["max_decisions"] == plan["caps"]["pong_max_decisions"], "Wrong Pong cap")
        if task == "breakoutcnn": require(suite["max_frames"] == plan["caps"]["breakout_max_frames"], "Wrong Breakout cap")
        if task == "mazecnn": require(suite["level_offset"] == suite["training_level_count"] == 0
            and suite["level_count"] == audit.config(config).getint("env", "num_maps"), "Wrong development level panel")
        manifest = rows(suite_path.parent / "episodes.csv"); key = (task, job["seed"])
        if key in reference_worlds: require(worlds(manifest) == reference_worlds[key], "Drawing worlds differ")
        else: reference_worlds[key] = worlds(manifest)
        if task == "connect4cnn":
            require(binding["host_check"] is None and binding["start_proof"] == "declared-empty-boards-and-identity-seeds", "Invented Connect4 host proof")
        else:
            check_id = f"binding-{binding_index:04d}" if transfer else f"job-{index:04d}"
            require(binding["host_check"] == "host-checks/"+check_id
                    and binding["start_proof"] == "native-host-world-and-raster", "Missing native family start check")
            directory = root / binding["host_check"]
            require(ev.validate_manifest(directory / "episodes.csv", suite) == manifest, "Family native starts differ")
            expected = ev.suite_config(audit.config(config), suite); actual = audit.config(directory / "manifest.ini")
            require({s: dict(expected[s]) for s in expected.sections()} == {s: dict(actual[s]) for s in actual.sections()},
                    "Family host config differs")
        condition_names.add(expected_name)
    require(plan["conditions"] == len(condition_names) and plan["paired_worlds_checked"] is True, "Wrong checked condition count")
    if check_native: check_binaries(plan["binaries"])
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    for name in ("panel", "registry", "out"): create.add_argument("--"+name, type=Path, required=True)
    create.add_argument("--seed", type=int, default=56123); create.add_argument("--episodes", type=int, default=1000)
    create.add_argument("--slots", type=int, default=64)
    create.add_argument("--evaluation-appearances", choices=("matched", "all"), default="matched",
                        help="Bind the same trained checkpoints to their training drawing or every fixed drawing; no retraining")
    create.add_argument("--pong-max-decisions", type=int, default=16384)
    create.add_argument("--breakout-max-frames", type=int, default=8192)
    check = sub.add_parser("inspect"); check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--check-binaries", action="store_true", help="Byte hashes only; no native commands or GPU query")
    args = parser.parse_args()
    value = prepare(args) if args.command == "prepare" else inspect(args.plan.resolve(), args.check_binaries)
    print(json.dumps(dict(status=value["status"], tasks=len(value["tasks"]), conditions=value["conditions"],
        jobs=value.get("training_jobs", len(value["bindings"])), evaluation_bindings=len(value["bindings"]),
        episodes_per_condition=value["episodes"], policy_executed=False,
        publication_confirmation_launchable=False), indent=2))


if __name__ == "__main__": main()
