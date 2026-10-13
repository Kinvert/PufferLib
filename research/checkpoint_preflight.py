"""Native registration/stat receipts for evaluator acceptance; no neural math."""
import configparser
import json
import math
from pathlib import Path
import re
import shutil
import shlex

from claim import process, require, save, sha

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"flex": ("default", "flex"), "nature": ("default", "nature_cnn"),
            "impala": ("impala", "impala_cnn"), "impoola": ("impoola", "impoola_cnn")}
HEADERS = {task: f"ocean/{task}/{task}.h" for task in
           ("connect4cnn", "pongcnn", "flappycnn", "breakoutcnn", "mazecnn")}
HEADERS["snakecnn"] = "ocean/snakecnn/game.h"
FALSE_FLAGS = ("cuda_executed", "gpu_queried", "weights_initialized", "policy_executed",
               "checkpoint_contents_validated", "numerical_acceptance")
IMAGE_SCHEMA = "native-image-metadata-registry-v1"


def validate_image_contract(value):
    shape = value.get("shape_hwc")
    require(value.get("version") == "human-geometry-v1" and value.get("layout") == "NHWC"
            and value.get("dtype") == "float32" and value.get("preprocessing") == "none"
            and isinstance(shape, list) and len(shape) == 3 and all(type(v) is int for v in shape)
            and all(36 <= v <= 1024 for v in shape[:2]) and shape[2] in (1, 3),
            "Invalid native image contract")


def binary_key(registry, task, family):
    build_family = FAMILIES[family][0]
    return f"{task}/{build_family}" if registry.get("schema") == IMAGE_SCHEMA else build_family


def freeze_images(builds, output):
    """Freeze explicit per-game host builds; no GPU query or neural execution.

    builds maps each task to its default and optional impala/impoola build dirs.
    Each supplied family must use exactly that game's declared image contract.
    """
    require(set(builds) == set(HEADERS), "Need all six image metadata games")
    output.mkdir(parents=True, exist_ok=False)
    records = dict(schema=IMAGE_SCHEMA, sources={}, binaries={}, actions={}, builds={}, image_contracts={},
                   vendor_system_link_closure_complete=False)
    for task, families in builds.items():
        require("default" in families and set(families) <= {"default", "impala", "impoola"},
                "Missing default or unsupported image metadata family")
        for family, original in families.items():
            original = Path(original).resolve(); key = f"{task}/{family}"
            info = json.loads((original / "metadata-info.json").read_text())
            require(info["schema"] == "native-image-build-v1" and info["environment"] == task
                    and info["compiled_family"] == family and info["gpu_queried"] is False
                    and info["cuda_executed"] is False, "Wrong image metadata build identity")
            contract = info["image_contract"]; validate_image_contract(contract)
            if task in records["image_contracts"]:
                require(records["image_contracts"][task] == contract, "Image families use different inputs")
            records["image_contracts"][task] = contract
            sources = source_hashes(original / "source.sha256")
            for name, digest in sources.items():
                require(sha(original / "source" / name) == sha(ROOT / name) == digest,
                        "Image metadata source differs; compile fresh targets")
                require(name not in records["sources"] or records["sources"][name] == digest,
                        "Image builds use inconsistent sources")
                records["sources"][name] = digest
                target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists(): shutil.copyfile(original / "source" / name, target)
            hashes = {}
            for line in (original / "binaries.sha256").read_text().splitlines():
                digest, name = line.split(maxsplit=1); path = Path(name.lstrip("*"))
                require(path.parent == original and path.name in ("puffer", "metadata")
                        and path.name not in hashes and sha(path) == digest, "Image build binary differs")
                hashes[path.name] = digest
            require("metadata" in hashes, "Missing image host metadata binary")
            if "puffer" in hashes:
                trainer = json.loads((original / "trainer-info.json").read_text())
                require(trainer["environment"] == task and trainer["image_contract"] == contract,
                        "Trainer and metadata image contracts differ")
            binary = output / key; binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / "metadata", binary); binary.chmod(0o700)
            records["binaries"][key] = hashes["metadata"]
            receipt_dir = output / "builds" / key; receipt_dir.mkdir(parents=True)
            receipts = {}
            for name in ("source.sha256", "source-files.txt", "metadata-info.json", "image-flags.txt",
                         "metadata-command.txt", "metadata-build.txt", "compiler.txt", "environment.txt",
                         "revision.txt", "worktree.txt", "source-check.txt", "copied-source-check.txt", "binaries.sha256"):
                shutil.copyfile(original / name, receipt_dir / name); receipts[name] = sha(receipt_dir / name)
            records["builds"][key] = dict(original=str(original), receipts=receipts)
        name = HEADERS[task]; header = ROOT / name
        match = re.search(r"^#define ACT_SIZES \{([0-9]+)\}$", header.read_text(), re.M)
        require(match is not None, "Unsupported image action declaration")
        records["actions"][task] = dict(count=int(match[1]), source=name, sha256=sha(header))
        target = output / "action-headers" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, target)
    (output / "source.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(records["sources"].items())))
    save(output / "registry.json", records, exclusive=True)
    inspect_registry(output, current=True)
    return records


def inspect_image_registry(directory, value):
    require(set(value["image_contracts"]) == set(HEADERS) and set(value["builds"]) == set(value["binaries"]),
            "Incomplete image metadata registry")
    require({f"{task}/default" for task in HEADERS} <= set(value["binaries"]), "Missing per-game image tools")
    for task, contract in value["image_contracts"].items(): validate_image_contract(contract)
    for key, build in value["builds"].items():
        parts = key.split("/")
        require(len(parts) == 2 and parts[0] in HEADERS and parts[1] in ("default", "impala", "impoola"),
                "Unsupported image tool key")
        task, family = parts; receipt_dir = directory / "builds" / key
        expected = {"source.sha256", "source-files.txt", "metadata-info.json", "image-flags.txt",
                    "metadata-command.txt", "metadata-build.txt", "compiler.txt", "environment.txt",
                    "revision.txt", "worktree.txt", "source-check.txt", "copied-source-check.txt", "binaries.sha256"}
        require(set(build["receipts"]) == expected, "Incomplete image build receipts")
        for name, digest in build["receipts"].items():
            require(sha(receipt_dir / name) == digest, "Changed image build receipt")
        sources = source_hashes(receipt_dir / "source.sha256")
        require(all(value["sources"].get(n) == d for n, d in sources.items()), "Image source binding differs")
        info = json.loads((receipt_dir / "metadata-info.json").read_text())
        require(info["environment"] == task and info["compiled_family"] == family
                and info["schema"] == "native-image-build-v1" and info["gpu_queried"] is False
                and info["cuda_executed"] is False and info["image_contract"] == value["image_contracts"][task],
                "Changed image build identity")
        flags = (receipt_dir / "image-flags.txt").read_text().split()
        h, w, c = value["image_contracts"][task]["shape_hwc"]
        expected_flags = ["-DCNN_IMAGE_V1", f"-DCNN_IMAGE_HEIGHT={h}", f"-DCNN_IMAGE_WIDTH={w}", f"-DCNN_IMAGE_CHANNELS={c}"]
        if family != "default": expected_flags.append("-DC4_IMPALA_CNN" if family == "impala" else "-DC4_IMPOOLA_CNN")
        require(flags == expected_flags, "Image compiler flags and contract differ")
        command = shlex.split((receipt_dir / "metadata-command.txt").read_text())
        require(all(command.count(flag) == 1 for flag in expected_flags)
                and f"-DPUFFER_{task.upper()}" in command
                and f'-DENV_HEADER="ocean/{task}/{task}.h"' in command
                and f'-DPUFFER_ENV_NAME="{task}"' in command
                and "research/policy_metadata.cu" in command, "Image metadata compile command differs")
        require((receipt_dir / "source-files.txt").read_text().splitlines() == sorted(sources),
                "Image source file manifest differs")
        binary_hashes = {}
        for line in (receipt_dir / "binaries.sha256").read_text().splitlines():
            digest, name = line.split(maxsplit=1); path = Path(name.lstrip("*"))
            require(path.parent == Path(build["original"]) and path.name in ("metadata", "puffer")
                    and path.name not in binary_hashes, "Wrong image build binary receipt")
            binary_hashes[path.name] = digest
        require(binary_hashes.get("metadata") == value["binaries"][key], "Image metadata binary provenance differs")


def source_hashes(path):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in result
                and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Malformed metadata source manifest")
        result[name] = digest
    require("research/policy_metadata.cu" in result and "src/pufferl.cu" in result,
            "Missing metadata implementation provenance")
    return result


def freeze(directory, output):
    """Copy existing compiled host tools/owned source; never build or query GPU."""
    directory = directory.resolve()
    if (directory / "registry.json").is_file():
        registry = inspect_registry(directory, current=True)
        require(registry.get("schema") == IMAGE_SCHEMA, "Only explicit image registries may be copied")
        shutil.copytree(directory, output)
        return inspect_registry(output, current=True)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes(directory / "source.sha256")
    records = {"sources": sources, "binaries": {}, "actions": {}, "original": str(directory),
               "vendor_system_link_closure_complete": False}
    for name, digest in sources.items():
        require(sha(directory / "source" / name) == digest == sha(ROOT / name),
                "Metadata owned source differs; compile fresh local targets")
        target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "source" / name, target)
    binary_rows = {}
    for line in (directory / "binaries.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1); path = Path(name.lstrip("*"))
        require(path.parent == directory and path.name in ("default", "impala", "impoola")
                and path.name not in binary_rows, "Wrong metadata binary manifest")
        binary_rows[path.name] = digest
    require(set(binary_rows) == {"default", "impala", "impoola"}, "Incomplete metadata families")
    for family, digest in binary_rows.items():
        require(sha(directory / family) == digest, "Metadata binary changed")
        shutil.copyfile(directory / family, output / family)
        (output / family).chmod(0o700)
        records["binaries"][family] = sha(output / family)
    for name in ("source.sha256", "source-files.txt", "compiler.txt", "environment.txt", "revision.txt",
                 "worktree.txt", "binaries.sha256", "source-check.txt", "copied-source-check.txt",
                 *(f"{f}.{s}.txt" for f in binary_rows for s in ("command", "compiler"))):
        shutil.copyfile(directory / name, output / name)
    for task, name in HEADERS.items():
        header = ROOT / name; match = re.search(r"^#define ACT_SIZES \{([0-9]+)\}$", header.read_text(), re.M)
        require(match is not None, "Unsupported discrete action declaration")
        target = output / "action-headers" / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, target)
        records["actions"][task] = dict(count=int(match[1]), source=name, sha256=sha(header))
    save(output / "registry.json", records, exclusive=True)
    inspect_registry(output, current=True)
    return records


def inspect_registry(directory, current=False):
    """Artifact-only; do not run any native executable when inspecting."""
    value = json.loads((directory / "registry.json").read_text())
    if value.get("schema") == IMAGE_SCHEMA: inspect_image_registry(directory, value)
    else: require(set(value["binaries"]) == {"default", "impala", "impoola"}, "Incomplete metadata tools")
    for family, digest in value["binaries"].items():
        require(sha(directory / family) == digest, "Frozen metadata binary changed")
    require(source_hashes(directory / "source.sha256") == value["sources"], "Metadata source declaration changed")
    for name, digest in value["sources"].items():
        require(sha(directory / "source" / name) == digest, "Frozen metadata source changed")
        if current: require(sha(ROOT / name) == digest, "Current metadata source changed")
    require(set(value["actions"]) == set(HEADERS), "Incomplete action declarations")
    for task, record in value["actions"].items():
        require(record["source"] == HEADERS[task], "Wrong action header")
        path = directory / "action-headers" / record["source"]
        match = re.search(r"^#define ACT_SIZES \{([0-9]+)\}$", path.read_text(), re.M)
        require(sha(path) == record["sha256"] and match is not None and int(match[1]) == record["count"],
                "Changed action declaration")
        if current: require(sha(ROOT / record["source"]) == record["sha256"], "Current action header changed")
    return value


def validate_descriptor(value, config, family, actions, checkpoint_bytes, image_contract=None, task=None):
    schema = "native-policy-metadata-v2" if image_contract is not None else "native-policy-metadata-v1"
    require(family in FAMILIES and value["schema"] == schema
            and value["model"] == FAMILIES[family][1], "Wrong native policy descriptor/family")
    require(all(value[key] is False for key in FALSE_FLAGS), "Metadata initialized/executed a model")
    shape = [1, 36, 44]
    if image_contract is not None:
        validate_image_contract(image_contract)
        require(task in HEADERS and value["environment"] == task and value["image_contract"] == image_contract,
                "Wrong image environment or preprocessing")
        h, w, c = image_contract["shape_hwc"]; shape = [c, h, w]
    require(value["observation_shape"] == shape and value["discrete_actions"] == actions,
            "Wrong observation or action dimensions")
    require(value["hidden"] == config.getint("policy", "hidden_size")
            and value["layers"] == config.getint("policy", "num_layers")
            and value["horizon"] == config.getint("train", "horizon"), "Wrong policy/core metadata")
    require(value["raw_checkpoint_contiguous"] is True and checkpoint_bytes > 0
            and value["checkpoint_bytes"] == value["parameter_payload_bytes"] == value["parameter_allocator_bytes"]
            == checkpoint_bytes == value["total_parameters"] * 4, "Wrong native checkpoint byte count")
    offset = 0; counts = dict(encoder=0, decoder=0, core=0); groups = []
    for index, tensor in enumerate(value["parameter_tensors"]):
        group = tensor["group"]; shape = tensor["shape"]
        require(tensor["index"] == index and group in counts and shape
                and all(type(n) is int and n > 0 for n in shape), "Invalid ordered parameter tensor")
        require(tensor["elements"] == math.prod(shape) and tensor["offset_bytes"] == offset,
                "Parameter shape/offset mismatch")
        offset += tensor["elements"] * 4; counts[group] += tensor["elements"]; groups.append(group)
    require(offset == checkpoint_bytes and groups == sorted(groups, key=("encoder", "decoder", "core").index),
            "Parameter layout differs from contiguous payload")
    require(all(counts[g] == value[g+"_parameters"] for g in counts), "Component counts disagree")


def validate_registration(value, config, family, actions, image_contract=None, task=None):
    require(value["checkpoint_bytes"] is None, "Registration unexpectedly checked a checkpoint")
    payload = value["parameter_payload_bytes"]
    require(type(payload) is int and payload > 0, "Invalid registered parameter payload")
    validate_descriptor({**value, "checkpoint_bytes": payload}, config, family, actions, payload, image_contract, task)


def register(metadata, config_path, task, family, output, timeout, checkpoint=None):
    """Native host registration, optionally stat an existing checkpoint; no math."""
    require(family in FAMILIES and task in HEADERS, "Metadata gate excludes state/other graphs/encoder 5")
    registry = inspect_registry(metadata, current=True); binary = metadata / binary_key(registry, task, family)
    image_contract = registry.get("image_contracts", {}).get(task)
    actions = registry["actions"][task]["count"]
    command = [str(binary), "--check-flex-bytes" if checkpoint is not None else "--describe-flex", str(config_path), str(actions)]
    if checkpoint is not None: command.append(str(checkpoint))
    inputs = {str(p): sha(p) for p in ((binary, config_path, checkpoint) if checkpoint is not None else (binary, config_path))}
    output.mkdir(parents=True, exist_ok=False)
    timed = process(command, ROOT, output / "native.txt", timeout)
    value = json.loads((output / "native.txt").read_text())
    # Use the same full INI that native registration parsed; no hand-written
    # architecture formula and no model weights are loaded by the host tool.
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(config_path), "Missing full metadata INI")
    if checkpoint is None: validate_registration(value, config, family, actions, image_contract, task)
    else: validate_descriptor(value, config, family, actions, checkpoint.stat().st_size, image_contract, task)
    require(all(sha(Path(p)) == digest for p, digest in inputs.items()), "Metadata inputs changed during registration")
    inspect_registry(metadata, current=True)
    save(output / "descriptor.json", value, exclusive=True)
    record = dict(family=family, actions=actions, descriptor_sha256=sha(output / "descriptor.json"),
                parameters=value["total_parameters"], bytes=value["parameter_payload_bytes"],
                command=command, timing=timed, inputs_sha256=inputs,
                numerical_acceptance=False, runtime_binary_source_certified=False)
    save(output / "result.json", record, exclusive=True)
    return record


def describe(metadata, config_path, task, family, output, timeout):
    return register(metadata, config_path, task, family, output, timeout)


def check(metadata, config_path, checkpoint, task, family, output, timeout):
    return register(metadata, config_path, task, family, output, timeout, checkpoint)


def inspect_description(metadata, config_path, task, family, directory, recorded_metadata,
                        recorded_config, repository_root, timeout=60):
    """Reparse retained native stdout/process/shape proof, without execution."""
    registry = inspect_registry(metadata)
    record = json.loads((directory / "result.json").read_text())
    descriptor = json.loads((directory / "descriptor.json").read_text())
    require(descriptor == json.loads((directory / "native.txt").read_text())
            and sha(directory / "descriptor.json") == record["descriptor_sha256"], "Changed registration descriptor")
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(config_path), "Missing registration INI")
    actions = registry["actions"][task]["count"]
    validate_registration(descriptor, config, family, actions, registry.get("image_contracts", {}).get(task), task)
    binary_family = binary_key(registry, task, family)
    command = [str(recorded_metadata / binary_family), "--describe-flex", str(recorded_config), str(actions)]
    timing = json.loads((directory / "native.txt.json").read_text())
    require(record["command"] == timing["command"] == command and timing == record["timing"]
            and timing["cwd"] == str(repository_root) and timing["status"] == "ok" and timing["returncode"] == 0
            and timing["timeout"] == timeout and type(timing["launch_monotonic_ns"]) is int
            and type(timing["end_monotonic_ns"]) is int and timing["launch_monotonic_ns"] < timing["end_monotonic_ns"],
            "Wrong registration process receipt")
    require(record["family"] == family and record["actions"] == actions
            and record["parameters"] == descriptor["total_parameters"] and record["bytes"] == descriptor["parameter_payload_bytes"]
            and record["inputs_sha256"] == {str(recorded_metadata / binary_family): registry["binaries"][binary_family],
                str(recorded_config): sha(config_path)} and record["numerical_acceptance"] is False
            and record["runtime_binary_source_certified"] is False, "Wrong registration input/layout binding")
    return record
