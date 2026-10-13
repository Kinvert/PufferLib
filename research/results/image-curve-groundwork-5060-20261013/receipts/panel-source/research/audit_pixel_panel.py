#!/usr/bin/env python3
"""GPU-free fixed pixel-panel configuration/source/fairness audit; no model."""
import argparse
import configparser
import csv
import json
import re
from pathlib import Path
import sys

import prepare_pixel_robustness as panel
sys.path.insert(0, str(panel.ROOT / "ocean/connect4cnn"))
from claim import require, sha


def relative(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe panel path")
    return root / name


def config(path):
    value = configparser.ConfigParser(interpolation=None)
    require(value.read(path), "Missing full panel INI")
    return value


def catalog_counts(root, value):
    result = {}
    for environment in value["environments"]:
        text = (root / "source" / panel.APPEARANCE_SOURCES[environment]).read_text()
        match = re.search(r"^#define " + environment.upper() + r"_NUM_REPRESENTATIONS (\d+)$", text, re.M)
        require(match is not None and 1 <= int(match[1]) <= 64, "Missing/invalid captured drawing catalog")
        result[environment] = int(match[1])
    return result


def mixed_catalog(root, environment):
    text = (root / "source" / panel.APPEARANCE_SOURCES[environment]).read_text()
    return 1 if re.search(r"^#define " + environment.upper() + r"_LEGACY_REPRESENTATIONS \d+$", text, re.M) else None


def common(value):
    result = {section: dict(value[section]) for section in value.sections()}
    for key in ("checkpoint_dir", "log_dir"): result["base"].pop(key)
    result["policy"].pop("encoder")
    return result


def expected_config(root, value, environment, representation, seed, model, recorded_root, directory):
    steps, cadence = task_budget(value, environment)
    source = root / "source"
    c = panel.read(source / "config/default.ini", source / "config" / f"{environment}.ini",
                   source / "ocean" / environment / "compare.ini")
    if environment in value.get("learner_recipes", {}):
        recipe = value["learner_recipes"][environment]
        panel.apply_learner(c, panel.learner_recipe(relative(source, recipe["snapshot"]), panel.read(source / "config/default.ini")))
    for section in list(c):
        if section.startswith("sweep."): c.remove_section(section)
    for key in list(c["policy"]):
        if key.startswith("cnn_"): c.remove_option("policy", key)
    c["base"].update(env_name=environment, load_model_path="None", eval_episodes="0", seed=str(seed),
        checkpoint_interval=str(cadence//batch_steps(value, environment)), run_id="trial",
        checkpoint_dir=str(recorded_root / directory / "checkpoints"), log_dir=str(recorded_root / directory / "metrics"))
    mixed = value["version"] == "pixel-robustness-preparation-v4"
    c["env"].update(representation=str(representation), representation_mode="1" if mixed else "0",
                    representation_seed=str(value["appearance_seed"]) if mixed else "0")
    if mixed and mixed_catalog(root, environment) is not None: c["env"]["representation_mix_catalog"] = "1"
    c["policy"].update({key: str(number) for key, number in value["architecture"].items()})
    c["policy"].update(hidden_size="128", num_layers="1",
        encoder=str(dict(flex_quality=4, nature_cnn=2, impala_cnn=0, impoola_cnn=0)[model]))
    c["train"]["total_timesteps"] = str(steps)
    return {section: dict(c[section]) for section in c.sections()}


def batch_steps(value, environment):
    return 2048 if value["version"] == "pixel-robustness-preparation-v1" else value["rollout_geometry"][environment]["batch_steps"]


def task_budget(value, environment):
    budget = value["task_budgets"][environment] if value["version"] in ("pixel-robustness-preparation-v3", "pixel-robustness-preparation-v4") else value
    return budget["steps"], budget["checkpoint_steps"]


def learner_validation(root, value):
    if value["version"] == "pixel-robustness-preparation-v1":
        require("learner_recipes" not in value and "rollout_geometry" not in value, "Legacy panel upgraded with uncaptured learners")
        return
    require(value["learning_calibrated"] is False and value["gpu_execution_authorized"] is False,
            "Learner preparation upgraded into accepted learning/GPU execution")
    v3 = value["version"] == "pixel-robustness-preparation-v3"
    mixed = value["version"] == "pixel-robustness-preparation-v4"
    require(value.get("learner_contract") == ("shared-native-geometry-v3" if mixed else "shared-native-geometry-v2" if v3 else "shared-native-geometry-v1"),
            "Incomplete/prototype shared-learner contract; regenerate rather than modify old receipts")
    require(sha(root / "build-only.sh") == value["build_only_sha256"], "Changed shared-learner build-only script")
    recipes = value["learner_recipes"]
    require(isinstance(recipes, dict) and (recipes or v3 or mixed) and set(recipes) <= set(value["environments"]), "Wrong learner allocation")
    for environment, recipe in recipes.items():
        require(recipe["snapshot"] == f"learner-recipes/{environment}.ini"
                and value["source_sha256"][recipe["snapshot"]] == recipe["sha256"], "Wrong learner snapshot/hash")
        panel.learner_recipe(relative(root / "source", recipe["snapshot"]), panel.read(root / "source/config/default.ini"))
    require(set(value["rollout_geometry"]) == set(value["environments"]), "Missing/extra learner geometry tasks")
    path_build = root / "validation/build.json"
    require(sha(path_build) == value["learner_preflight_build_sha256"], "Changed learner preflight build receipt")
    build = json.loads(path_build.read_text())
    require(build["policy_executed"] is False and build["gpu_runtime_validated"] is False
            and set(build["source_sha256"]) == {"research/learner_geometry.c", "src/ini.h", "src/pufferl.cu"}, "Wrong scalar preflight source/status")
    require(all(value["source_sha256"][name] == digest for name, digest in build["source_sha256"].items()),
            "Scalar preflight differs from captured learner/parser/helper")
    if (root / "validation/learner_geometry").exists():
        require(sha(root / "validation/learner_geometry") == build["binary_sha256"], "Changed scalar preflight executable")
    for environment, record in value["rollout_geometry"].items():
        steps, cadence = task_budget(value, environment)
        require(record["config"] == f"validation/{environment}.ini" and record["receipt"] == f"validation/{environment}.json",
                "Wrong scalar preflight allocation")
        path = relative(root, record["config"]); path_result = relative(root, record["receipt"])
        require(sha(path) == record["config_sha256"] and sha(path_result) == record["receipt_sha256"], "Changed learner geometry input/result")
        c = config(path); result = json.loads(path_result.read_text())
        p, h, mb = c.getint("vec", "total_agents"), c.getint("train", "horizon"), c.getint("train", "minibatch_size")
        require(result["protocol"] == "native-learner-geometry-v1" and result["environment"] == environment
                and result["float32_geometry"] is True and result["policy_executed"] is False
                and result["gpu_runtime_validated"] is False and result["world_size"] == 1
                and result["agents_per_rank"] == p and result["horizon"] == h
                and result["minibatch_decisions"] == mb and record["batch_steps"] == p*h == result["rollout_decisions_per_rank"],
                "Wrong native learner geometry identity/shape/status")
        require(type(record["batch_steps"]) is int and record["batch_steps"] > 0
                and steps % record["batch_steps"] == cadence % record["batch_steps"] == 0
                and steps >= record["batch_steps"] and cadence >= record["batch_steps"], "Unaligned learner budget/cadence")
        require(result["actual_global_decisions"] == result["requested_global_decisions"] == steps
                and result["dropped_requested_decisions"] == 0 and result["nonzero_training_updates"] is True
                and result["optimizer_minibatches_per_epoch"] > 0
                and result["epochs"] == steps//record["batch_steps"]
                and result["optimizer_updates_per_rank"] == result["epochs"]*result["optimizer_minibatches_per_epoch"], "Learner has zero/rounded/inconsistent updates")
        # Match the retained preflight full INI to a job's complete captured
        # recipe. Only launch-specific seed/outputs/model selection differ.
        job = next(j for j in value["jobs"] if j["environment"] == environment)
        expected = expected_config(root, value, environment, job["representation"], job["seed"], job["model"],
            Path(job["cwd"]).parents[2], str(Path(job["config"]).parent.parent))
        require(all(dict(c[s]) == expected[s] for s in ("train", "vec", "selfplay", "env")), "Preflight learner/world differs from captured resolved recipe")
        require(c.getint("policy", "hidden_size") == 128 and c.getint("policy", "num_layers") == 1, "Preflight core differs")
        require(all(c["base"][key] == expected["base"][key] for key in ("async", "cudagraphs", "reset_every_horizon", "checkpoint_interval")),
                "Preflight actor/cadence differs")


def appearance_validation(root, value):
    keys = ("appearance_seed", "appearance_contract", "appearance_assignments", "appearance_preflight_build_sha256")
    if value["version"] != "pixel-robustness-preparation-v4":
        require(not any(k in value for k in keys), "Legacy panel cannot declare mixed training")
        return
    seed = value["appearance_seed"]
    require(type(seed) is int and 0 <= seed <= 2**32-1
            and value["appearance_contract"] == "native-slot-fixed-full-catalog-v1", "Wrong mixed appearance contract/seed")
    records = value["appearance_assignments"]
    require(isinstance(records, dict) and set(records) == set(value["environments"]), "Missing/extra appearance tasks")
    path = root / "validation/appearance-build.json"
    require(sha(path) == value["appearance_preflight_build_sha256"], "Appearance build receipt changed")
    build = json.loads(path.read_text())
    expected_sources = {"research/appearance_assignments.c", "src/ini.h", "ocean/connect4cnn/appearance.h"}
    expected_sources.update(panel.APPEARANCE_SOURCES[e] for e in value["environments"])
    require(set(build["source_sha256"]) == expected_sources
            and all(value["source_sha256"][name] == digest for name, digest in build["source_sha256"].items())
            and build["policy_executed"] is False and build["gpu_runtime_validated"] is False
            and build["game_executed"] is False, "Wrong appearance source/status receipt")
    if (root / "validation/appearance_assignments").exists():
        require(sha(root / "validation/appearance_assignments") == build["binary_sha256"], "Appearance scalar binary changed")
    catalogs = catalog_counts(root, value)
    for environment, record in records.items():
        source = root / "source" / panel.APPEARANCE_SOURCES[environment]
        match = re.search(r"^#define " + environment.upper() + r"_NUM_REPRESENTATIONS (\d+)$", source.read_text(), re.M)
        count = catalogs[environment]
        require(match is not None and int(match[1]) == count, "Captured native catalog differs")
        c = config(root / f"validation/{environment}.ini")
        slots = c.getint("vec", "total_agents")
        require(all(type(record[k]) is int for k in ("mode", "seed", "slots", "catalog_count"))
                and (type(record["mix_catalog"]) is int if mixed_catalog(root, environment) is not None else record["mix_catalog"] is None)
                and record["mode"] == 1 and record["seed"] == seed and record["slots"] == slots
                and record["catalog_count"] == count and record["mix_catalog"] == mixed_catalog(root, environment)
                and record["native_vector_initialization_validated"] is False, "Wrong mixed assignment metadata")
        name = f"validation/{environment}-appearances.csv"
        require(record["csv"] == name and sha(root / name) == record["csv_sha256"], "Appearance CSV changed or rebound")
        with (root / name).open() as stream: actual = list(csv.DictReader(stream))
        expected = [dict(slot=str(i), representation=str(panel.appearance_assignment(seed, i, count))) for i in range(slots)]
        require(actual == expected, "Mixed slots differ from declared seed/catalog")
        counts = [sum(int(row["representation"]) == i for row in actual) for i in range(count)]
        require(isinstance(record["counts"], list) and all(type(n) is int for n in record["counts"])
                and record["counts"] == counts and record["all_drawings_assigned"] is bool(all(counts)), "Wrong mixed coverage counts")


def load(path):
    root = path.parent; value = json.loads(path.read_text())
    require(not (root / "failure.json").exists(), "Failed panel preparation; retain it and use a fresh ID")
    require(value["version"] in ("pixel-robustness-preparation-v1", "pixel-robustness-preparation-v2", "pixel-robustness-preparation-v3", "pixel-robustness-preparation-v4") and value["status"] == "prepared-not-executed"
            and value["publication_confirmation_launchable"] is False and value["precision"] == "float32"
            and value["initialization"] == "random", "Wrong panel status/precision/initialization")
    environments, seeds = value["environments"], value["seeds"]
    require(isinstance(environments, list) and environments and len(set(environments)) == len(environments)
            and all(e in panel.ENVIRONMENTS for e in environments), "Wrong environment allocation")
    require(isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
            and all(type(seed) is int and 0 <= seed <= 2**32-1 for seed in seeds), "Wrong training seeds")
    mixed = value["version"] == "pixel-robustness-preparation-v4"
    require(value["appearances"] in (("mixed",) if mixed else ("default", "all"))
            and value["core"] == dict(hidden_size=128, num_layers=1), "Wrong drawings/core")
    steps, cadence = value["steps"], value["checkpoint_steps"]
    require(type(steps) is int and type(cadence) is int and steps > 0 and cadence > 0, "Wrong budget/cadence")
    if value["version"] == "pixel-robustness-preparation-v1":
        require(steps >= 2048 and cadence >= 2048 and steps % 2048 == cadence % 2048 == 0, "Wrong legacy budget/cadence")
    if value["version"] in ("pixel-robustness-preparation-v3", "pixel-robustness-preparation-v4"):
        overrides = value["task_budget_overrides"]
        require(isinstance(overrides, dict) and (overrides or mixed), "Missing explicit task budget allocation")
        resolved = panel.resolve_budgets(environments, steps, cadence, overrides)
        require(isinstance(value["task_budgets"], dict) and set(value["task_budgets"]) == set(environments), "Wrong resolved task allocation")
        checked = panel.resolve_budgets(environments, steps, cadence, value["task_budgets"])
        require(checked == value["task_budgets"] == resolved, "Resolved task budgets differ from declarations/defaults")
        max_steps = max(b["steps"] for b in resolved.values())
    else:
        require("task_budgets" not in value and "task_budget_overrides" not in value, "Legacy panel upgraded with uncaptured task budgets")
        max_steps = steps
    require(value["purpose"] == ("plumbing" if max_steps <= 65536 else "development-design-not-calibrated"), "Wrong purpose")
    require(isinstance(value["source_sha256"], dict) and value["source_sha256"], "Missing source receipts")
    for name, digest in value["source_sha256"].items():
        require(sha(relative(root / "source", name)) == digest, "Changed source snapshot: " + name)
    selection = json.loads((root / "source/ocean/connect4cnn/confirmation.json").read_text())
    require(value["architecture"] == selection["policies"]["flex_quality"], "Frozen architecture changed")
    catalogs = catalog_counts(root, value)
    expected = []
    for ei, environment in enumerate(environments):
        for representation in range(catalogs[environment]) if value["appearances"] == "all" else [0]:
            for si, seed in enumerate(seeds):
                offset = (ei+representation+si) % 4
                for model in panel.MODELS[offset:]+panel.MODELS[:offset]:
                    expected.append((environment, representation, seed, model))
    require(len(value["jobs"]) == len(expected), "Missing/extra panel jobs")
    learner_validation(root, value)
    appearance_validation(root, value)
    recorded_root = Path(value["jobs"][0]["cwd"]).parents[2]
    require(recorded_root.is_absolute(), "Panel cwd must be absolute")
    groups = {}
    for job, allocation in zip(value["jobs"], expected):
        environment, representation, seed, model = allocation
        steps, cadence = task_budget(value, environment)
        require(tuple(job[key] for key in ("environment", "representation", "seed", "model")) == allocation,
                "Wrong/duplicate/order-changed allocation")
        directory = f"{environment}/r{representation}/{model}-s{seed}"
        require(job["config"] == directory+"/config/default.ini" and job["cwd"] == str(recorded_root / directory), "Wrong job paths")
        path_config = relative(root, job["config"])
        require(sha(path_config) == job["config_sha256"], "Panel configuration bytes changed")
        # The native loader reads this game INI after default.ini. It must not
        # silently override an otherwise audited full configuration.
        require((path_config.parent / f"{environment}.ini").read_text() ==
                "# Full resolved settings in default.ini.\n", "Native game INI overrides the resolved panel")
        c = config(path_config)
        require({section: dict(c[section]) for section in c.sections()} ==
                expected_config(root, value, environment, representation, seed, model, recorded_root, directory),
                "Resolved job differs from captured recipe/declared overrides")
        require(all(section in c for section in ("base", "env", "vec", "train", "policy", "selfplay"))
                and not any(section.startswith("sweep.") for section in c), "Incomplete or sweep-contaminated INI")
        require(c.get("base", "env_name") == environment and c.get("base", "load_model_path") == "None"
                and c.getint("base", "seed") == seed and c.getint("base", "eval_episodes") == 0
                and c.getint("base", "checkpoint_interval") == cadence//batch_steps(value, environment), "Wrong training initialization/seed/cadence")
        require(c.get("base", "checkpoint_dir") == str(recorded_root / directory / "checkpoints")
                and c.get("base", "log_dir") == str(recorded_root / directory / "metrics"), "Wrong training output paths")
        require(c.getint("env", "representation") == representation and c.getint("env", "representation_mode") == (1 if mixed else 0)
                and c.getint("env", "representation_seed") == (value["appearance_seed"] if mixed else 0), "Wrong training drawing mode/seed")
        require(c.getint("vec", "total_agents")*c.getint("train", "horizon") == batch_steps(value, environment)
                and c.getint("train", "gpus") == 1 and c.getint("train", "total_timesteps") == steps,
                "Wrong learner batch/GPU/budget")
        if value["version"] == "pixel-robustness-preparation-v1":
            require(c.getint("vec", "total_agents") == 64 and c.getint("train", "horizon") == 32, "Changed legacy rollout geometry")
        require(c.getint("policy", "hidden_size") == 128 and c.getint("policy", "num_layers") == 1, "Changed recurrent core")
        encoder = dict(flex_quality=4, nature_cnn=2, impala_cnn=0, impoola_cnn=0)[model]
        require(c.getint("policy", "encoder") == encoder, "Model label/selector differs")
        for key, number in value["architecture"].items():
            if key != "encoder": require(c.getint("policy", key) == number, "Architecture knob changed: " + key)
        require(job["rules"] == panel.ENVIRONMENT_RULES[environment]
                and job["evaluation_gate"] == panel.EVALUATION_GATES[environment], "Wrong rules/evaluation gate; regenerate stale panel")
        require(job["checkpoint_steps"] == list(range(cadence, steps, cadence))+[steps], "Missing/changed checkpoints")
        family = dict(impala_cnn="impala", impoola_cnn="impoola").get(model, "default")
        require(job["train_command"] == ["env", "PUFFER_CHECKPOINT_RECEIPTS=1",
            str(recorded_root / "bin" / f"{environment}-{family}"), "train", "--headless"], "Wrong native train command")
        groups.setdefault(allocation[:3], []).append(common(c))
    require(all(len(group) == 4 and all(c == group[0] for c in group) for group in groups.values()),
            "Learner/world/policy controls differ within a paired condition")
    expected_builds = []
    for environment in environments:
        for family, flags in (("default", ""), ("impala", "-DC4_IMPALA_CNN"), ("impoola", "-DC4_IMPOOLA_CNN")):
            binary = str(recorded_root / "bin" / f"{environment}-{family}")
            expected_builds.append(dict(environment=environment, family=family, binary=binary,
                command=["env", "NVCC_PREPEND_FLAGS="+flags, "bash", "build.sh", environment, binary, "--float"]))
    require(value["builds"] == expected_builds, "Wrong native builds/precision/family flags")
    lines = (root / "source.sha256").read_text().splitlines()
    entries = {}
    for line in lines:
        digest, name = line.split(None, 1); name = name.strip()
        require(name not in entries, "Duplicate source/build receipt"); entries[name] = digest
    require(entries == value["source_sha256"], "Source/build manifest differs")
    if value["version"] != "pixel-robustness-preparation-v1":
        repo_entries = {}
        for line in (root / "source.repo.sha256").read_text().splitlines():
            digest, name = line.split(None, 1); name = name.strip()
            require(name not in repo_entries, "Duplicate repository source receipt"); repo_entries[name] = digest
        require(repo_entries == {name: digest for name, digest in entries.items() if not name.startswith("learner-recipes/")},
                "Repository build manifest mixes external snapshots or omits sources")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--panel", type=Path, required=True)
    value = load(parser.parse_args().panel.resolve())
    print(json.dumps(dict(status="prepared-panel-configuration-consistent", tasks=len(value["environments"]),
        jobs=len(value["jobs"]), builds=len(value["builds"]), learning_evidence_audited=False,
        publication_confirmation_launchable=False), indent=2))


if __name__ == "__main__": main()
