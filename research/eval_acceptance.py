#!/usr/bin/env python3
"""Prepare/audit repeat, graph/eager and tail checks for native exact evaluators.

prepare executes only native host manifests and byte/configuration inspection.
run requires a separately scheduled GPU window and uses each existing native
checkpoint supervisor. No training, architecture search or CPU model executes.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from claim import require, save, sha

TASKS = ("flappycnn", "breakoutcnn", "pongcnn", "snakecnn", "mazecnn")
FAMILIES = ("state", "tiny", "experimental", "nature", "compact", "flex", "impala", "impoola")
MODES = ("graph", "repeat", "eager", "tail")
SUITE_FILES = ("suite.json", "episodes.csv", "source.ini", "manifest.ini")
PROTOCOL = "native-exact-evaluator-acceptance-v1"
LAYOUT_PROTOCOL = "native-exact-evaluator-acceptance-v2"
_adapters = {}


def adapter(task):
    require(task in TASKS, "Unsupported pending evaluator task")
    if task not in _adapters:
        path = ROOT / "ocean" / task / "exact_eval.py"
        source_hash = sha(path)
        spec = importlib.util.spec_from_file_location("acceptance_" + task, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        require(sha(path) == source_hash, "Adapter changed during import")
        module.acceptance_source_sha256 = source_hash
        _adapters[task] = module
    require(sha(Path(_adapters[task].__file__)) == _adapters[task].acceptance_source_sha256,
            "Imported adapter source changed; start a fresh process")
    return _adapters[task]


def hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_files(directory, entries):
    for name, digest in entries.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe receipt path")
        require(sha(directory / name) == digest, f"Immutable file changed: {name}")


def read_cases(path):
    cases = json.loads(path.read_text())
    require(isinstance(cases, list) and 1 <= len(cases) <= 64, "Need one to 64 cases")
    names, signatures, resolved = set(), set(), []
    for case in cases:
        require(isinstance(case, dict) and set(case) == {"name", "task", "family", "binary", "config", "checkpoint", "suite"}, "Wrong case fields")
        name = case["name"]
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
                and name not in names, "Invalid/duplicate case name")
        names.add(name)
        require(case["task"] in TASKS and case["family"] in FAMILIES, "Unsupported task/family; Flex2 has separate gates")
        item = dict(case)
        for key in ("binary", "config", "checkpoint", "suite"):
            require(isinstance(case[key], str), f"Invalid {key} path")
            target = (path.parent / case[key]).resolve()
            require(target.is_file(), f"Missing {key}: {target}")
            item[key] = str(target)
        signature = tuple(item[k] for k in ("task", "family", "binary", "config", "checkpoint", "suite"))
        require(signature not in signatures, "Duplicate checkpoint/suite case")
        signatures.add(signature); resolved.append(item)
    return resolved


def validate_panel_suite(suite):
    require(suite["purpose"] == "development", "Acceptance must not expose a heldout suite")
    slots, episodes = suite["slots"], suite["episodes"]
    require(slots < episodes < 2 * slots, "Need two waves with a partial last wave, e.g. 65 IDs/64 slots")


def prepare(args):
    source, out = args.cases.resolve(), args.out.resolve()
    metadata = getattr(args, "policy_metadata", None)
    protocol = LAYOUT_PROTOCOL if metadata is not None else PROTOCOL
    require(type(args.timeout) is int and 1 <= args.timeout <= 3600, "Invalid per-native-process timeout")
    cases, source_hash = read_cases(source), sha(source)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "request.json", dict(protocol=protocol, cases_path=str(source), cases_sha256=source_hash,
                                    status="preparing-host-inputs", gpu_execution_authorized=False), exclusive=True)
    try:
        shutil.copyfile(source, out / "cases.json")
        tools = {"research/eval_acceptance.py": Path(__file__),
                 "ocean/connect4cnn/claim.py": ROOT / "ocean/connect4cnn/claim.py",
                 "ocean/connect4cnn/deterministic_eval.py": ROOT / "ocean/connect4cnn/deterministic_eval.py"}
        if metadata is not None:
            import checkpoint_preflight as preflight
            tools["research/checkpoint_preflight.py"] = Path(preflight.__file__)
            preflight.freeze(metadata, out / "policy-metadata")
        for task in {case["task"] for case in cases}:
            tools[f"ocean/{task}/exact_eval.py"] = Path(adapter(task).__file__)
        tool_hashes = {}
        for name, path in tools.items():
            target = out / "tooling" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target); tool_hashes[name] = sha(path)
            require(sha(target) == tool_hashes[name], "Tool changed during snapshot")
        entries = []
        for case in cases:
            ev = adapter(case["task"]); name = case["name"]
            suite_path = Path(case["suite"]); suite, _ = ev.load_suite(suite_path)
            validate_panel_suite(suite)
            directory = out / "inputs" / name; (directory / "suite").mkdir(parents=True)
            shutil.copyfile(case["config"], directory / "training.ini")
            for filename in SUITE_FILES: shutil.copyfile(suite_path.parent / filename, directory / "suite" / filename)
            layout = None
            if metadata is not None:
                require(suite["environment"] == case["task"], "Native layout gate requires a pixel checkpoint")
                layout = preflight.check(out / "policy-metadata", directory / "training.ini",
                    Path(case["checkpoint"]), case["task"], case["family"], directory / "layout", args.timeout)
            # Keep batch size fixed. A fresh first wave must reproduce the last
            # wave of the original run without inheriting recurrent state.
            tail_offset = suite["offset"] + suite["slots"]
            tail_args = argparse.Namespace(**dict(suite, binary=Path(case["binary"]),
                config=directory / "suite/source.ini", offset=tail_offset,
                episodes=suite["episodes"]-suite["slots"], out=directory / "tail-suite"))
            ev.create_suite(tail_args)
            tail, _ = ev.load_suite(directory / "tail-suite/suite.json")
            require(tail["slots"] == suite["slots"] and tail["offset"] == tail_offset, "Tail changed batch/identity")
            item = dict(case, original_sha256={k: sha(Path(case[k])) for k in ("binary", "config", "checkpoint", "suite")},
                        checkpoint_bytes=Path(case["checkpoint"]).stat().st_size, timeout=args.timeout,
                        suite_environment=suite["environment"], suite_seed=suite["seed"],
                        suite_offset=suite["offset"], suite_episodes=suite["episodes"], slots=suite["slots"], tail_offset=tail_offset,
                        tail_episodes=tail["episodes"], modes={})
            if layout is not None: item["checkpoint_layout"] = layout
            for mode in MODES:
                selected = directory / ("tail-suite" if mode == "tail" else "suite") / "suite.json"
                target = out / "prepared" / name / mode
                ev.run(argparse.Namespace(binary=Path(case["binary"]), config=directory / "training.ini",
                    checkpoint=Path(case["checkpoint"]), suite=selected, out=target, family=case["family"],
                    timeout=args.timeout, eager=mode == "eager", prepare_only=True))
                result = json.loads((target / "result.json").read_text())
                require(result["status"] == "prepared-not-executed" and not result["gpu_runtime_qualified"], "Unexpected execution")
                require(result["checkpoint_sha256"] == item["original_sha256"]["checkpoint"]
                        and result["binary_sha256"] == item["original_sha256"]["binary"], "Inputs changed during preparation")
                item["modes"][mode] = dict(suite=str(selected.relative_to(out)),
                    prepared=str(target.relative_to(out)), effective_ini_sha256=result["effective_ini_sha256"])
            entries.append(item)
        for name, path in tools.items():
            require(sha(path) == tool_hashes[name] and sha(out / "tooling" / name) == tool_hashes[name],
                    "Tool changed during preparation")
        require(sha(source) == source_hash, "Cases changed during preparation")
        for case in entries:
            for key, digest in case["original_sha256"].items(): require(sha(Path(case[key])) == digest, "Original input changed")
        packet = dict(protocol=protocol, status="prepared-not-executed", gpu_execution_authorized=False,
                      policy_math_certified=False, publication_confirmation_launchable=False,
                      cases=entries, tooling_sha256=tool_hashes, files_sha256=hashes(out),
                      note="Host starts/byte/configuration preparation only. Compiled build provenance and scheduled GPU "
                           "acceptance remain required. Python orchestrates tests; policy execution stays native.")
        if metadata is not None:
            packet["preparation_root"] = str(out)
        save(out / "packet.json", packet, exclusive=True)
    except BaseException as error:
        save(out / "preparation_failure.json", dict(status="failed", error=f"{type(error).__name__}: {error}"), exclusive=True)
        raise
    print(out / "packet.json")


def load_packet(path, current_inputs=False):
    packet = json.loads(path.read_text())
    require(packet["protocol"] in (PROTOCOL, LAYOUT_PROTOCOL) and packet["status"] == "prepared-not-executed"
            and packet["gpu_execution_authorized"] is False
            and not packet["publication_confirmation_launchable"] and not packet["policy_math_certified"], "Wrong packet/gates")
    verify_files(path.parent, packet["files_sha256"])
    registry = None
    if packet["protocol"] == LAYOUT_PROTOCOL:
        import checkpoint_preflight as preflight
        registry = preflight.inspect_registry(path.parent / "policy-metadata", current=current_inputs)
        require(Path(packet["preparation_root"]).is_absolute(), "Missing original preparation root")
    for name, digest in packet["tooling_sha256"].items():
        require(sha(path.parent / "tooling" / name) == digest, "Frozen tool changed")
        if current_inputs: require(sha(ROOT / name) == digest, "Current tool differs; regenerate preparation")
    require(isinstance(packet["cases"], list) and 1 <= len(packet["cases"]) <= 64, "Invalid case count")
    names = set()
    for case in packet["cases"]:
        require(isinstance(case["name"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", case["name"])
                and case["name"] not in names and set(case["modes"]) == set(MODES), "Invalid/duplicate/incomplete case")
        require(case["family"] in FAMILIES and type(case["timeout"]) is int and 1 <= case["timeout"] <= 3600,
                "Wrong family/timeout")
        require(case["checkpoint_bytes"] > 0 and case["checkpoint_bytes"] % 4 == 0, "Malformed weight byte count")
        names.add(case["name"]); ev = adapter(case["task"]); starts = {}
        if registry is not None:
            layout = case["checkpoint_layout"]; family = case["family"]
            require(family in preflight.FAMILIES and layout["family"] == family
                    and layout["numerical_acceptance"] is False and layout["runtime_binary_source_certified"] is False,
                    "Wrong checkpoint-layout scope")
            folder = path.parent / "inputs" / case["name"] / "layout"
            require(json.loads((folder / "result.json").read_text()) == layout, "Native layout proof differs from case")
            descriptor = json.loads((folder / "descriptor.json").read_text())
            require(sha(folder / "descriptor.json") == layout["descriptor_sha256"]
                    and json.loads((folder / "native.txt").read_text()) == descriptor, "Native descriptor changed")
            preflight.validate_descriptor(descriptor, ev.read(path.parent / "inputs" / case["name"] / "training.ini"),
                family, registry["actions"][case["task"]]["count"], case["checkpoint_bytes"])
            recorded = Path(packet["preparation_root"])
            binary = recorded / "policy-metadata" / preflight.FAMILIES[family][0]
            config = recorded / "inputs" / case["name"] / "training.ini"
            command = [str(binary), "--check-flex-bytes", str(config), str(layout["actions"]), case["checkpoint"]]
            timing = json.loads((folder / "native.txt.json").read_text())
            require(layout["command"] == timing["command"] == command and timing == layout["timing"]
                    and timing["status"] == "ok" and timing["returncode"] == 0
                    and type(timing["launch_monotonic_ns"]) is int and type(timing["end_monotonic_ns"]) is int
                    and timing["launch_monotonic_ns"] < timing["end_monotonic_ns"]
                    and timing["timeout"] == case["timeout"], "Wrong native layout process receipt")
            require(layout["inputs_sha256"] == {str(binary): registry["binaries"][preflight.FAMILIES[family][0]],
                    str(config): case["original_sha256"]["config"], case["checkpoint"]: case["original_sha256"]["checkpoint"]}
                    and layout["parameters"] == descriptor["total_parameters"] and layout["bytes"] == case["checkpoint_bytes"]
                    and layout["actions"] == registry["actions"][case["task"]]["count"], "Wrong layout input/shape binding")
        for mode in MODES:
            expected_suite = f"inputs/{case['name']}/{'tail-suite' if mode == 'tail' else 'suite'}/suite.json"
            require(case["modes"][mode]["suite"] == expected_suite, "Wrong mode suite path")
            require(case["modes"][mode]["prepared"] == f"prepared/{case['name']}/{mode}", "Wrong prepared mode path")
            suite, starts[mode] = ev.load_suite(path.parent / expected_suite)
            require(suite["slots"] == case["slots"] and suite["seed"] == case["suite_seed"], "Case/suite differs")
            require(suite["environment"] == case["suite_environment"] and suite["purpose"] == "development", "Wrong mode task/purpose")
            require(suite["episodes"] == (case["tail_episodes"] if mode == "tail" else case["suite_episodes"]), "Wrong mode quota")
            require(suite["offset"] == (case["tail_offset"] if mode == "tail" else case["suite_offset"]), "Wrong mode offset")
            if mode != "tail": validate_panel_suite(suite)
        require(case["tail_offset"] == case["suite_offset"] + case["slots"]
                and case["tail_episodes"] == case["suite_episodes"]-case["slots"], "Tail isn't the last original wave")
        require(starts["graph"] == starts["repeat"] == starts["eager"]
                and starts["tail"] == starts["graph"][case["slots"]:], "Native tail starts differ")
        if current_inputs:
            for key, digest in case["original_sha256"].items(): require(sha(Path(case[key])) == digest, "Original input changed")
    return packet


def receipt_rows(path):
    with path.open() as stream: rows = list(csv.DictReader(stream))
    indexed = {int(row["episode_id"]): row for row in rows}
    require(len(indexed) == len(rows), "Duplicate episode receipt")
    return indexed


def configuration_receipt(ev, directory):
    """Compare the full INI, allowing only output-directory relocation."""
    config = ev.read(directory / "config/default.ini")
    # An evidence archive may have moved; preserve the recorded execution cwd
    # rather than rewriting historical INIs to their present storage location.
    recorded = Path(json.loads((directory / "result.json").read_text())["cwd"])
    require(recorded.is_absolute() and recorded.name == directory.name
            and recorded.parent.name == directory.parent.name, "Wrong recorded case/mode directory")
    expected_paths = (("base", "checkpoint_dir", recorded / "unused-checkpoints"),
                      ("base", "log_dir", recorded / "metrics"),
                      ("eval_exact", "output", recorded / "episodes.csv"))
    for section, key, expected in expected_paths:
        require(config.get(section, key) == str(expected), "Unexpected execution output path")
        config[section][key] = "<output>/" + expected.name
    return {section: dict(config[section]) for section in config.sections()}


def compare_case(packet_root, case, output):
    ev = adapter(case["task"]); rows_by_mode, records = {}, {}
    if "checkpoint_layout" in case: inspect_execution_layout(case, output / case["name"] / "layout")
    for mode in MODES:
        directory = output / case["name"] / mode
        result = json.loads((directory / "result.json").read_text())
        suite_path = packet_root / case["modes"][mode]["suite"]
        suite, manifest = ev.load_suite(suite_path)
        counts = ev.audit_csv(directory / "episodes.csv", suite, manifest)
        require(result["status"] == "ok" and result["family"] == case["family"]
                and result["counts"] == counts and result["prepare_only"] is False, "Failed/mismatched result")
        require(result["checkpoint_sha256"] == case["original_sha256"]["checkpoint"]
                and result["binary_sha256"] == case["original_sha256"]["binary"]
                and result["training_ini_sha256"] == case["original_sha256"]["config"]
                and result["suite_sha256"] == sha(suite_path), "Result/input identity differs")
        require(result["effective_ini_sha256"] == sha(directory / "config/default.ini")
                and result["episodes_sha256"] == sha(directory / "episodes.csv"), "Result file changed")
        effective = ev.read(directory / "config/default.ini")
        prepared = json.loads((packet_root / case["modes"][mode]["prepared"] / "result.json").read_text())
        require(configuration_receipt(ev, directory) ==
                configuration_receipt(ev, packet_root / case["modes"][mode]["prepared"]),
                "Prepared/executed configuration differs")
        require(dict(effective["policy"]) == result["policy"] == prepared["policy"], "Policy/core changed during execution")
        require(effective.getint("base", "cudagraphs") == (-1 if mode == "eager" else 1), "Wrong eager/graph mode")
        require(type(result["parameters"]) is int and result["parameters"] > 0
                and result["parameters"] * 4 == case["checkpoint_bytes"], "Wrong parameter count")
        timing = json.loads((directory / "native.log.json").read_text())
        start, end = timing["launch_monotonic_ns"], timing["end_monotonic_ns"]
        require(type(start) is int and type(end) is int and end >= start and timing["returncode"] == 0, "Bad process receipt")
        seconds = result["evaluation_process_seconds"]
        require(math.isfinite(seconds) and seconds == (end-start)/1e9, "Wrong process timing")
        rows_by_mode[mode] = receipt_rows(directory / "episodes.csv"); records[mode] = result
    first = rows_by_mode["graph"]
    require(first == rows_by_mode["repeat"], "Repeat differs; retain all receipts")
    require(first == rows_by_mode["eager"], "Eager/graph differs; retain all receipts")
    tail = rows_by_mode["tail"]
    require(tail == {key: first[key] for key in tail}, "Prior waves affected tail; keep batch size fixed")
    return dict(name=case["name"], task=case["task"], family=case["family"],
                parameters=records["graph"]["parameters"], episodes=case["suite_episodes"], slots=case["slots"],
                tail_offset=case["tail_offset"], tail_episodes=case["tail_episodes"],
                repeat_equal=True, eager_equal=True, independent_tail_equal=True,
                policy_math_certified=False, frontier_superiority_certified=False)


def inspect_execution_layout(case, directory):
    """Offline reparse of the repeated host check; not merely outer file hashes."""
    expected = case["checkpoint_layout"]
    result = json.loads((directory / "result.json").read_text())
    timing = json.loads((directory / "native.txt.json").read_text())
    require({k: v for k, v in result.items() if k != "timing"}
            == {k: v for k, v in expected.items() if k != "timing"}, "Repeated layout binding differs")
    require(timing == result["timing"] and timing["command"] == expected["command"]
            and timing["status"] == "ok" and timing["returncode"] == 0
            and timing["timeout"] == case["timeout"]
            and type(timing["launch_monotonic_ns"]) is int and type(timing["end_monotonic_ns"]) is int
            and timing["launch_monotonic_ns"] < timing["end_monotonic_ns"], "Repeated layout process differs")
    require(sha(directory / "descriptor.json") == expected["descriptor_sha256"]
            and json.loads((directory / "native.txt").read_text())
            == json.loads((directory / "descriptor.json").read_text()), "Repeated descriptor differs")


def run(args):
    path, out = args.packet.resolve(), args.out.resolve()
    packet = load_packet(path, current_inputs=True)
    if packet["protocol"] == LAYOUT_PROTOCOL:
        require(str(path.parent) == packet["preparation_root"], "Regenerate relocated packets before execution")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="running-scoped-native-checks", packet_sha256=sha(path), cases=[],
                  policy_math_certified=False, frontier_superiority_certified=False)
    save(out / "request.json", record, exclusive=True)
    try:
        for case in packet["cases"]:
            ev = adapter(case["task"])
            if packet["protocol"] == LAYOUT_PROTOCOL:
                import checkpoint_preflight as preflight
                repeated = preflight.check(path.parent / "policy-metadata",
                    path.parent / "inputs" / case["name"] / "training.ini", Path(case["checkpoint"]),
                    case["task"], case["family"], out / case["name"] / "layout", case["timeout"])
                require(repeated["descriptor_sha256"] == case["checkpoint_layout"]["descriptor_sha256"],
                        "Native parameter layout changed before GPU acceptance")
                inspect_execution_layout(case, out / case["name"] / "layout")
            for mode in MODES:
                ev.run(argparse.Namespace(binary=Path(case["binary"]),
                    config=path.parent / "inputs" / case["name"] / "training.ini",
                    checkpoint=Path(case["checkpoint"]), suite=path.parent / case["modes"][mode]["suite"],
                    out=out / case["name"] / mode, family=case["family"], timeout=case["timeout"],
                    eager=mode == "eager", prepare_only=False))
                load_packet(path, current_inputs=True)
                require(sha(path) == record["packet_sha256"], "Packet metadata changed during execution")
            record["cases"].append(compare_case(path.parent, case, out))
        record.update(status="scoped-receipt-comparisons-passed", files_sha256=hashes(out))
        save(out / "acceptance.json", record, exclusive=True)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(out / "failure.json", record, exclusive=True)
        raise
    print(out / "acceptance.json")


def audit(args):
    path, output = args.packet.resolve(), args.execution.resolve()
    packet = load_packet(path)
    record = json.loads((output / "acceptance.json").read_text())
    require(record["status"] == "scoped-receipt-comparisons-passed" and record["packet_sha256"] == sha(path), "Wrong execution")
    verify_files(output, record["files_sha256"])
    comparisons = [compare_case(path.parent, case, output) for case in packet["cases"]]
    require(comparisons == record["cases"], "Acceptance comparisons changed")
    print(json.dumps(dict(status="artifact-comparisons-passed", cases=comparisons,
                         policy_math_certified=False, frontier_superiority_certified=False), indent=2))


def inspect(args):
    packet = load_packet(args.packet.resolve(), current_inputs=True)
    cases = []
    for case in packet["cases"]:
        for mode in MODES:
            directory = args.packet.resolve().parent / case["modes"][mode]["prepared"]
            result = json.loads((directory / "result.json").read_text())
            require(result["status"] == "prepared-not-executed" and result["prepare_only"] is True
                    and result["gpu_runtime_qualified"] is False and not (directory / "episodes.csv").exists(),
                    "Prepared mode was executed or changed")
            require(sha(directory / "config/default.ini") == case["modes"][mode]["effective_ini_sha256"],
                    "Prepared configuration hash differs")
            configuration_receipt(adapter(case["task"]), directory)
        cases.append({key: case[key] for key in ("name", "task", "family", "checkpoint_bytes", "suite_episodes",
                                                "slots", "tail_offset", "tail_episodes")})
    print(json.dumps(dict(status="host-preparation-validated", cases=cases,
                         gpu_execution_authorized=False, policy_math_certified=False,
                         frontier_superiority_certified=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare", help="GPU-free existing-checkpoint gate preparation")
    create.add_argument("--cases", type=Path, required=True); create.add_argument("--out", type=Path, required=True)
    create.add_argument("--timeout", type=int, default=600)
    create.add_argument("--policy-metadata", type=Path,
        help="Opt-in v2 native pre-GPU checkpoint-size gate; fresh three-family metadata build directory")
    execute = sub.add_parser("run", help="Scheduled native GPU acceptance only; never trains")
    execute.add_argument("--packet", type=Path, required=True); execute.add_argument("--out", type=Path, required=True)
    review = sub.add_parser("audit", help="GPU-free completed receipt comparisons; not neural math certification")
    review.add_argument("--packet", type=Path, required=True); review.add_argument("--execution", type=Path, required=True)
    check = sub.add_parser("inspect", help="GPU-free prepared inputs/starts/configuration validation")
    check.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "audit": audit, "inspect": inspect}[args.command](args)


if __name__ == "__main__": main()
