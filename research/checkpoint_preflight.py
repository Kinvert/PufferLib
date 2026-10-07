"""Native registration/stat receipts for evaluator acceptance; no neural math."""
import configparser
import json
import math
from pathlib import Path
import re
import shutil

from claim import process, require, save, sha

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"flex": ("default", "flex"), "nature": ("default", "nature_cnn"),
            "impala": ("impala", "impala_cnn"), "impoola": ("impoola", "impoola_cnn")}
HEADERS = {task: f"ocean/{task}/{task}.h" for task in
           ("connect4cnn", "pongcnn", "flappycnn", "breakoutcnn", "mazecnn")}
HEADERS["snakecnn"] = "ocean/snakecnn/game.h"
FALSE_FLAGS = ("cuda_executed", "gpu_queried", "weights_initialized", "policy_executed",
               "checkpoint_contents_validated", "numerical_acceptance")


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
    directory = directory.resolve(); output.mkdir(parents=True, exist_ok=False)
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
    require(set(value["binaries"]) == {"default", "impala", "impoola"}, "Incomplete metadata tools")
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


def validate_descriptor(value, config, family, actions, checkpoint_bytes):
    require(family in FAMILIES and value["schema"] == "native-policy-metadata-v1"
            and value["model"] == FAMILIES[family][1], "Wrong native policy descriptor/family")
    require(all(value[key] is False for key in FALSE_FLAGS), "Metadata initialized/executed a model")
    require(value["observation_shape"] == [1, 36, 44] and value["discrete_actions"] == actions,
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


def validate_registration(value, config, family, actions):
    require(value["checkpoint_bytes"] is None, "Registration unexpectedly checked a checkpoint")
    payload = value["parameter_payload_bytes"]
    require(type(payload) is int and payload > 0, "Invalid registered parameter payload")
    validate_descriptor({**value, "checkpoint_bytes": payload}, config, family, actions, payload)


def register(metadata, config_path, task, family, output, timeout, checkpoint=None):
    """Native host registration, optionally stat an existing checkpoint; no math."""
    require(family in FAMILIES and task in HEADERS, "Metadata gate excludes state/other graphs/encoder 5")
    registry = inspect_registry(metadata, current=True); binary = metadata / FAMILIES[family][0]
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
    if checkpoint is None: validate_registration(value, config, family, actions)
    else: validate_descriptor(value, config, family, actions, checkpoint.stat().st_size)
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
    validate_registration(descriptor, config, family, actions)
    binary_family = FAMILIES[family][0]
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
