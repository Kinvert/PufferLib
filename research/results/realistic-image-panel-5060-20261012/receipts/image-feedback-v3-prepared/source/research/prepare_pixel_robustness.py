#!/usr/bin/env python3
"""GPU-free fixed-architecture panel preparation; training/inference stay native."""
import argparse
import configparser
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {"connect4cnn": 10, "pongcnn": 7, "flappycnn": 7, "breakoutcnn": 5, "snakecnn": 6, "mazecnn": 6}
ENVIRONMENT_RULES = {"connect4cnn": "connect4-full-board-draw-v2", "pongcnn": "pong-native-v1",
                     "flappycnn": "flappy-native-v1", "breakoutcnn": "breakout-native-v1",
                     "snakecnn": "snake-local-episodic-v1", "mazecnn": "maze-native-v1"}
MODELS = ("flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn")
EVALUATION_GATES = {"connect4cnn": "connect4-exact-available",
                    "flappycnn": "flappy-exact-compiled-gpu-validation-pending",
                    "pongcnn": "pong-exact-compiled-gpu-validation-pending", "breakoutcnn": "breakout-exact-compiled-gpu-validation-pending",
                    "snakecnn": "snake-exact-compiled-gpu-validation-pending",
                    "mazecnn": "maze-exact-compiled-gpu-validation-pending"}
APPEARANCE_SOURCES = {e: f"ocean/{e}/{e}.h" for e in ENVIRONMENTS}
APPEARANCE_SOURCES["snakecnn"] = "ocean/snakecnn/game.h"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(*paths):
    config = configparser.ConfigParser(interpolation=None)
    if config.read(paths) != [str(path) for path in paths]:
        raise ValueError("Missing configuration input")
    return config


def learner_recipe(path, default):
    value = read(path)
    allowed = dict(base={"async", "cudagraphs", "reset_every_horizon"},
        vec={"total_agents", "num_buffers", "num_threads"},
        train=set(default["train"])-{"gpus", "total_timesteps"})
    if value.defaults() or not value.sections() or not any(dict(value[s]) for s in value.sections()):
        raise ValueError("Need explicit nonempty learner settings without DEFAULT inheritance")
    for section in value.sections():
        if section not in allowed or not set(value[section]) <= allowed[section]:
            raise ValueError("Learner recipes cannot alter architecture, world, seeds, budgets, GPU count or output controls")
        for raw in value[section].values():
            try: number = float(raw.replace("_", ""))
            except ValueError: raise ValueError("Learner settings must be numeric scalars") from None
            if not math.isfinite(number): raise ValueError("Learner settings must be finite")
    return value


def apply_learner(config, recipe):
    for section in recipe.sections():
        config[section].update(dict(recipe[section]))


def native_geometry(out, bases):
    # External preparation only. The compiled helper performs scalar INI
    # arithmetic, never neural inference/training or CUDA discovery.
    from audit_learning_recipes import source_guard, geometry
    source_guard(ROOT / "src/pufferl.cu")
    directory = out / "validation"; directory.mkdir()
    binary = directory / "learner_geometry"
    source_hashes = {}
    for name in ("research/learner_geometry.c", "src/ini.h", "src/pufferl.cu"):
        original = ROOT / name; digest = sha(original)
        target = out / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha(target) != digest: raise ValueError("Native geometry source changed during capture")
        source_hashes[name] = digest
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
               str(out / "source/research/learner_geometry.c"), "-lm", "-o", str(binary)]
    try:
        with (directory / "build.txt").open("x") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=60)
        records = {}
        for environment, base in bases.items():
            path = directory / f"{environment}.ini"
            with path.open("x") as stream: base.write(stream)
            value = geometry(binary, [path])
            receipt = directory / f"{environment}.json"
            receipt.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
            if not value["nonzero_training_updates"] or value["dropped_requested_decisions"] or value["world_size"] != 1:
                raise ValueError("Learner panel needs positive updates, exact decisions and one GPU")
            records[environment] = dict(batch_steps=value["rollout_decisions_per_rank"],
                config=str(path.relative_to(out)), config_sha256=sha(path),
                receipt=str(receipt.relative_to(out)), receipt_sha256=sha(receipt))
        build = dict(command=command, compiler=subprocess.check_output(["cc", "--version"], text=True, timeout=20),
                     binary_sha256=sha(binary), source_sha256=source_hashes,
                     policy_executed=False, gpu_runtime_validated=False)
        (directory / "build.json").write_text(json.dumps(build, indent=2, sort_keys=True)+"\n")
        return records
    except Exception as error:
        (out / "failure.json").write_text(json.dumps(dict(status="failed-learner-preflight", error=str(error),
                                                         policy_executed=False), indent=2)+"\n")
        raise


def resolve_budgets(environments, steps, checkpoint_steps, overrides):
    if type(steps) is not int or type(checkpoint_steps) is not int or steps <= 0 or checkpoint_steps <= 0:
        raise ValueError("Decision budget/cadence must be positive integers")
    if not isinstance(overrides, dict) or not set(overrides) <= set(environments):
        raise ValueError("Task budgets must name requested environments")
    for value in overrides.values():
        if not isinstance(value, dict) or set(value) != {"steps", "checkpoint_steps"} or any(
                type(v) is not int or v <= 0 for v in value.values()):
            raise ValueError("Task budgets need exactly positive integer steps/checkpoint_steps")
    return {e: dict(overrides.get(e, dict(steps=steps, checkpoint_steps=checkpoint_steps))) for e in environments}


def appearance_assignment(seed, slot, count):
    # Independent scalar translation for native CSV checking, not game RNG.
    def mix(x):
        x ^= x >> 16; x = (x * 0x7feb352d) & 0xffffffff
        x ^= x >> 15; x = (x * 0x846ca68b) & 0xffffffff
        return x ^ (x >> 16)
    draw = mix(slot ^ mix(seed ^ 0x9e3779b9))
    threshold = (2**32 - count) % count
    while draw < threshold: draw = mix((draw + 0x9e3779b9) & 0xffffffff)
    return draw % count


def mixed_assignments(out, bases, seed):
    directory = out / "validation"
    binary = directory / "appearance_assignments"
    names = ("research/appearance_assignments.c", "src/ini.h", "ocean/connect4cnn/appearance.h")
    source_hashes = {}
    for name in names:
        source = ROOT / name; digest = sha(source)
        target = out / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != digest: raise ValueError("Appearance source changed after scalar preflight")
        shutil.copyfile(source, target)
        if sha(target) != digest: raise ValueError("Appearance source changed during capture")
        source_hashes[name] = digest
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
               str(out / "source/research/appearance_assignments.c"), "-lm", "-o", str(binary)]
    try:
        with (directory / "appearance-build.txt").open("x") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=60)
        records = {}
        for environment, base in bases.items():
            catalog = ROOT / APPEARANCE_SOURCES[environment]; digest = sha(catalog)
            match = re.search(r"^#define " + environment.upper() + r"_NUM_REPRESENTATIONS (\d+)$", catalog.read_text(), re.M)
            if not match or int(match[1]) != ENVIRONMENTS[environment]: raise ValueError("Native drawing catalog differs")
            source_hashes[APPEARANCE_SOURCES[environment]] = digest
            path = directory / f"{environment}-appearances.csv"
            with path.open("x") as stream, (directory / f"{environment}-appearances-stderr.txt").open("x") as errors:
                subprocess.run([str(binary), str(directory / f"{environment}.ini"), str(ENVIRONMENTS[environment])],
                               stdout=stream, stderr=errors, check=True, timeout=60)
            with path.open() as stream: actual = list(csv.DictReader(stream))
            slots = base.getint("vec", "total_agents"); count = ENVIRONMENTS[environment]
            expected = [dict(slot=str(i), representation=str(appearance_assignment(seed, i, count))) for i in range(slots)]
            if actual != expected: raise ValueError("Native slot assignment differs from independent arithmetic")
            counts = [sum(int(row["representation"]) == i for row in actual) for i in range(count)]
            records[environment] = dict(mode=1, seed=seed, slots=slots, catalog_count=count,
                mix_catalog=1 if environment in ("pongcnn", "flappycnn") else None,
                csv=str(path.relative_to(out)), csv_sha256=sha(path), counts=counts,
                all_drawings_assigned=all(counts), native_vector_initialization_validated=False)
        receipt = dict(command=command, binary_sha256=sha(binary), source_sha256=source_hashes,
                       policy_executed=False, gpu_runtime_validated=False, game_executed=False,
                       compiler=subprocess.check_output(["cc", "--version"], text=True, timeout=20))
        (directory / "appearance-build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
        return records
    except Exception as error:
        (out / "failure.json").write_text(json.dumps(dict(status="failed-appearance-preflight", error=str(error),
                                                         policy_executed=False), indent=2)+"\n")
        raise


def prepare(out, environments, appearances, seeds, steps, checkpoint_steps, learner_recipes=None, task_budgets=None, appearance_seed=None):
    if not environments or len(set(environments)) != len(environments) or any(e not in ENVIRONMENTS for e in environments):
        raise ValueError("Need distinct supported environments")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s <= 2**32-1 for s in seeds):
        raise ValueError("Need distinct unsigned 32-bit training seeds")
    if appearances not in ("default", "all", "mixed"):
        raise ValueError("Use default, all fixed, or mixed appearances")
    mixed = appearances == "mixed"
    if mixed:
        appearance_seed = 0 if appearance_seed is None else appearance_seed
        if type(appearance_seed) is not int or not 0 <= appearance_seed <= 2**32-1:
            raise ValueError("Appearance seed must be an unsigned 32-bit integer")
    elif appearance_seed is not None:
        raise ValueError("Appearance seed requires mixed training")
    task_budgets = {} if task_budgets is None else task_budgets
    budgets = resolve_budgets(environments, steps, checkpoint_steps, task_budgets)
    learner_recipes = {} if learner_recipes is None else learner_recipes
    if not isinstance(learner_recipes, dict) or not set(learner_recipes) <= set(environments):
        raise ValueError("Learner recipes must name distinct requested environments")
    recipes, bases = {}, {}
    extended = bool(learner_recipes or task_budgets or mixed)
    default = read(ROOT / "config/default.ini")
    for environment in environments:
        task_steps, task_cadence = budgets[environment]["steps"], budgets[environment]["checkpoint_steps"]
        base = read(ROOT / "config/default.ini", ROOT / "config" / f"{environment}.ini",
                    ROOT / "ocean" / environment / "compare.ini")
        if environment in learner_recipes:
            path = Path(learner_recipes[environment]).resolve()
            digest = sha(path); recipe = learner_recipe(path, default)
            if sha(path) != digest: raise ValueError("Learner recipe changed while reading")
            apply_learner(base, recipe)
            recipes[environment] = dict(original_path=str(path), name=path.stem,
                snapshot=f"learner-recipes/{environment}.ini", sha256=digest)
        batch = base.getint("vec", "total_agents") * base.getint("train", "horizon")
        if batch <= 0 or task_steps < batch or task_cadence < batch or task_steps % batch or task_cadence % batch:
            raise ValueError(f"Decision budget/cadence must be positive rollout multiples ({batch}) for {environment}")
        for section in list(base):
            if section.startswith("sweep."):
                base.remove_section(section)
        base.set("base", "env_name", environment)
        base.set("base", "load_model_path", "None")
        base.set("base", "eval_episodes", "0")
        base.set("base", "checkpoint_interval", str(task_cadence // batch))
        base.set("train", "total_timesteps", str(task_steps))
        base.set("policy", "hidden_size", "128")
        base.set("policy", "num_layers", "1")
        base.set("env", "representation", "0")
        base.set("env", "representation_mode", "1" if mixed else "0")
        base.set("env", "representation_seed", str(appearance_seed) if mixed else "0")
        if mixed and environment in ("pongcnn", "flappycnn"): base.set("env", "representation_mix_catalog", "1")
        # One frozen encoder across games. Remove inherited inactive architecture knobs.
        for key in list(base["policy"]):
            if key.startswith("cnn_"):
                base.remove_option("policy", key)
        if base.getint("train", "gpus") != 1 or (not extended and
                (base.getint("vec", "total_agents") != 64 or base.getint("train", "horizon") != 32)):
            raise ValueError("Panel requires common H128/L1, 64 slots, horizon 32 and one GPU")
        bases[environment] = base
    out = out.resolve(); out.mkdir(parents=True, exist_ok=False)
    geometry = native_geometry(out, bases) if extended else {}
    assignments = mixed_assignments(out, bases, appearance_seed) if mixed else {}
    selection = json.loads((ROOT / "ocean/connect4cnn/confirmation.json").read_text())
    quality = selection["policies"]["flex_quality"]
    jobs, builds = [], []
    for ei, environment in enumerate(environments):
        base = bases[environment]
        task_steps, task_cadence = budgets[environment]["steps"], budgets[environment]["checkpoint_steps"]
        for family, flags in (("default", []), ("impala", ["-DC4_IMPALA_CNN"]),
                              ("impoola", ["-DC4_IMPOOLA_CNN"])):
            binary = out / "bin" / f"{environment}-{family}"
            build = ["env", "NVCC_PREPEND_FLAGS=" + " ".join(flags), "bash", "build.sh",
                     environment, str(binary), "--float"]
            builds.append(dict(environment=environment, family=family, binary=str(binary), command=build))
        representations = range(ENVIRONMENTS[environment]) if appearances == "all" else [0]
        for representation in representations:
            for si, seed in enumerate(seeds):
                # Rotate which family runs first. Never choose order based on outcomes.
                offset = (ei + representation + si) % len(MODELS)
                order = MODELS[offset:] + MODELS[:offset]
                for model in order:
                    config = read()
                    config.read_dict({section: dict(base[section]) for section in base.sections()})
                    config.set("base", "seed", str(seed))
                    config.set("env", "representation", str(representation))
                    for key, value in quality.items():
                        config.set("policy", key, str(value))
                    encoder = {"flex_quality": 4, "nature_cnn": 2, "impala_cnn": 0, "impoola_cnn": 0}[model]
                    config.set("policy", "encoder", str(encoder))
                    family = {"impala_cnn": "impala", "impoola_cnn": "impoola"}.get(model, "default")
                    job = out / environment / f"r{representation}" / f"{model}-s{seed}"
                    (job / "config").mkdir(parents=True)
                    config.set("base", "run_id", "trial")
                    config.set("base", "checkpoint_dir", str(job / "checkpoints"))
                    config.set("base", "log_dir", str(job / "metrics"))
                    with (job / "config/default.ini").open("x") as stream:
                        config.write(stream)
                    (job / "config" / f"{environment}.ini").write_text("# Full resolved settings in default.ini.\n")
                    command = ["env", "PUFFER_CHECKPOINT_RECEIPTS=1",
                               str(out / "bin" / f"{environment}-{family}"), "train", "--headless"]
                    jobs.append(dict(environment=environment, rules=ENVIRONMENT_RULES[environment],
                                     representation=representation, model=model, seed=seed,
                                     config=str((job / "config/default.ini").relative_to(out)),
                                     config_sha256=sha(job / "config/default.ini"), cwd=str(job), train_command=command,
                                     checkpoint_steps=list(range(task_cadence, task_steps, task_cadence)) + [task_steps],
                                     evaluation_gate=EVALUATION_GATES[environment]))
    sources = {ROOT / "build.sh", ROOT / "config/default.ini", Path(__file__).resolve(),
               ROOT / "ocean/connect4cnn/confirmation.json"}
    if extended:
        sources.update(ROOT / "research" / name for name in ("learner_geometry.c", "audit_learning_recipes.py", "audit_pixel_panel.py"))
    if mixed: sources.add(ROOT / "research/appearance_assignments.c")
    for directory in (ROOT / "src", ROOT / "ocean/connect4cnn"):
        sources.update(path for path in directory.rglob("*") if path.is_file() and path.suffix in (".h", ".c", ".cu", ".py", ".ini", ".sh"))
    for environment in environments:
        sources.add(ROOT / "config" / f"{environment}.ini")
        names = (environment, "snakebench") if environment == "snakecnn" else (environment, environment.removesuffix("cnn"))
        for name in names:
            sources.update(path for path in (ROOT / "ocean" / name).rglob("*")
                           if path.is_file() and path.suffix in (".h", ".c", ".cu", ".py", ".ini", ".sh"))
    hashes = {}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha(source)
    for environment, recipe in recipes.items():
        target = out / "source" / recipe["snapshot"]; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(recipe["original_path"], target)
        if sha(target) != recipe["sha256"]: raise ValueError("Learner recipe changed during capture")
        hashes[recipe["snapshot"]] = recipe["sha256"]
    if extended:
        build = json.loads((out / "validation/build.json").read_text())
        if any(hashes[name] != digest for name, digest in build["source_sha256"].items()):
            raise ValueError("Learner/parser source changed after native geometry validation")
    if mixed:
        receipt = json.loads((out / "validation/appearance-build.json").read_text())
        if any(hashes[name] != digest for name, digest in receipt["source_sha256"].items()):
            raise ValueError("Appearance/catalog source changed after native validation")
    protocol = dict(version="pixel-robustness-preparation-v4" if mixed else "pixel-robustness-preparation-v3" if task_budgets else "pixel-robustness-preparation-v2" if recipes else "pixel-robustness-preparation-v1", status="prepared-not-executed",
                    publication_confirmation_launchable=False,
                    purpose="plumbing" if max(b["steps"] for b in budgets.values()) <= 65536 else "development-design-not-calibrated",
                    environments=environments, appearances=appearances, seeds=seeds, steps=steps,
                    checkpoint_steps=checkpoint_steps, precision="float32", initialization="random",
                    architecture=quality, core=dict(hidden_size=128, num_layers=1), jobs=jobs, builds=builds,
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=hashes,
                    evaluation_warning="No pooled-v1 scores qualify this panel for paper claims. Native exact adapters compile for all six tasks; non-Connect4 GPU acceptance, Maze supervision/level exclusion and inference/replication gates remain open.")
    if extended:
        protocol.update(learner_recipes=recipes, rollout_geometry=geometry,
            learner_preflight_build_sha256=sha(out / "validation/build.json"),
            learner_contract="shared-native-geometry-v3" if mixed else "shared-native-geometry-v2" if task_budgets else "shared-native-geometry-v1",
            learning_calibrated=False, gpu_execution_authorized=False)
    if task_budgets or mixed:
        protocol.update(task_budget_overrides=task_budgets, task_budgets=budgets)
    if mixed:
        protocol.update(appearance_seed=appearance_seed, appearance_contract="native-slot-fixed-full-catalog-v1",
            appearance_assignments=assignments, appearance_preflight_build_sha256=sha(out / "validation/appearance-build.json"))
    (out / "bin").mkdir()
    # Builds only. Deliberately provide no batch launch script while evaluation gates are open.
    script = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(ROOT)),
              "source ocean/connect4cnn/runtime_env.sh", "export NVCC_ARCH=${NVCC_ARCH:-sm_120}",
              "test -z \"${NVCC_PREPEND_FLAGS:-}\"", "test -z \"${NVCC_EXTRA:-}\"",
              "test -z \"${NVCC_APPEND_FLAGS:-}\"", "test -z \"${NVCC_FLAGS:-}\""]
    (out / "source.sha256").write_text("".join(f"{value}  {path}\n" for path, value in hashes.items()))
    if extended:
        # External overlays have virtual snapshot paths, not repository paths.
        # Check their immutable copies separately from source compiled at ROOT.
        repo_hashes = {name: digest for name, digest in hashes.items() if not name.startswith("learner-recipes/")}
        (out / "source.repo.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in repo_hashes.items()))
        checks = ["(", "  cd " + shlex.quote(str(out / "source")),
                  "  sha256sum --check " + shlex.quote(str(out / "source.sha256")), ")",
                  "sha256sum --check " + shlex.quote(str(out / "source.repo.sha256"))]
    else:
        checks = ["sha256sum --check " + shlex.quote(str(out / "source.sha256"))]
    script += checks
    script += [shlex.join(build["command"]) + " > " + shlex.quote(str(out / "bin" / f"{build['environment']}-{build['family']}.build.log")) + " 2>&1" for build in builds]
    script += checks
    script += ["sha256sum " + shlex.quote(str(out / "bin")) + "/*-default " + shlex.quote(str(out / "bin")) + "/*-impala " + shlex.quote(str(out / "bin")) + "/*-impoola > " + shlex.quote(str(out / "binaries.sha256"))]
    (out / "build-only.sh").write_text("\n".join(script) + "\n")
    if extended: protocol["build_only_sha256"] = sha(out / "build-only.sh")
    # Publish only after the complete preparation/build-only packet exists.
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
    parser.add_argument("--appearances", choices=("default", "all", "mixed"), default="default")
    parser.add_argument("--appearance-seed", type=int, help="Unsigned 32-bit fixed-slot mixed drawing seed; mixed default is 0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[53111])
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--checkpoint-steps", type=int, default=32768)
    parser.add_argument("--learner-recipe", action="append", default=[], metavar="ENV=INI",
                        help="Shared numeric learner/vector/actor overlay; no world/architecture/seed changes")
    parser.add_argument("--task-budget", action="append", default=[], metavar="ENV=STEPS:CHECKPOINT_STEPS",
                        help="Per-game decisions/cadence, shared by all models/drawings/seeds; defaults apply elsewhere")
    args = parser.parse_args()
    recipes = {}
    for item in args.learner_recipe:
        if "=" not in item: parser.error("Use --learner-recipe ENV=INI")
        environment, path = item.split("=", 1)
        if not path or environment in recipes: parser.error("Need distinct environments and nonempty learner paths")
        recipes[environment] = Path(path)
    budgets = {}
    for item in args.task_budget:
        try:
            environment, pair = item.split("=", 1)
            task_steps, cadence = (int(number) for number in pair.split(":"))
        except ValueError: parser.error("Use --task-budget ENV=STEPS:CHECKPOINT_STEPS")
        if environment in budgets: parser.error("Need distinct task budget environments")
        budgets[environment] = dict(steps=task_steps, checkpoint_steps=cadence)
    protocol = prepare(args.out, args.environments, args.appearances, args.seeds, args.steps, args.checkpoint_steps, recipes, budgets, args.appearance_seed)
    print(f"Prepared {len(protocol['jobs'])} fixed-model jobs and {len(protocol['builds'])} builds; no GPU accessed. {args.out}/protocol.json")


if __name__ == "__main__":
    main()
