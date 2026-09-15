"""Read completed native CNN trials; optionally mirror them to W&B.

Uses the saved-INI pattern from F-Zero/Admiral. Final PROTEIN observations
come from sweep stdout, since native metric arrays contain binned means.
"""
import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


RESULT = re.compile(r"sweep run=(\d+) score=([-+\d.eE]+) cost=([-+\d.eE]+) steps=([-+\d.eE]+) random=(\d+) gp_obs=(\d+) pareto=(\d+)")
SCHEMA = 2


def display_name(root, index):
    adjectives = ("happy", "mystic", "brave", "gentle", "bright", "quiet", "swift", "lucky",
                  "cosmic", "golden", "clever", "merry", "calm", "wild", "silver", "sunny")
    nouns = ("cat", "tree", "fox", "otter", "owl", "river", "panda", "wolf",
             "falcon", "cedar", "badger", "comet", "tiger", "maple", "robin", "bear")
    digest = hashlib.sha256(f"{Path(root).name}/{index}".encode()).digest()
    return f"{adjectives[digest[0] % len(adjectives)]}-{nouns[digest[1] % len(nouns)]}-{index + 1}"


def read_ini(path):
    ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    ini.optionxform = str
    with Path(path).open() as f:
        ini.read_file(f)
    return ini


def scalar(value):
    try:
        number = float(value.replace("_", ""))
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def architecture(config):
    encoder = config["policy.encoder"]
    common = ("policy.hidden_size", "policy.num_layers")
    if encoder == 1:
        version = "connect4-cnn-v1"
        keys = ("policy.cnn_channels", "policy.cnn_blocks", "policy.cnn_global_pool") + common
    elif encoder == 2:
        version, keys = "connect4-nature-v1", common
    elif encoder == 3:
        version = "connect4-compact-v1"
        keys = ("policy.cnn_channels", "policy.cnn_depth", "policy.cnn_stride", "policy.cnn_projection") + common
    elif encoder == 4:
        version = "connect4-flex-v1"
        keys = ("policy.cnn_depth", "policy.cnn_projection", "policy.cnn_global_pool") + common
        keys += tuple(f"policy.cnn_{key}_{stage}" for stage in range(1, int(config["policy.cnn_depth"]) + 1)
                      for key in ("channels", "kernel", "stride", "pool", "residual"))
    else:
        raise ValueError("Unsupported sweep encoder")
    return {"version": version, "observation": [1, 36, 44], **{k: config[k] for k in keys}}


def trials(root):
    root = Path(root)
    log = root / "sweep.log"
    if not log.exists():
        return []
    observed = {}
    for line in log.read_text().splitlines():
        match = RESULT.fullmatch(line)
        if match:
            i, score, cost, steps, random, gp, pareto = match.groups()
            observed[int(i)] = dict(score=float(score), cost=float(cost), steps=int(float(steps)),
                                    random=int(random), gp_obs=int(gp), pareto=int(pareto))
    result = []
    for path in sorted((root / "metrics/connect4cnn").glob("*.ini")):
        index = int(path.stem.rsplit("_", 1)[1])
        if index not in observed:
            continue
        final = observed[index]
        if not all(math.isfinite(final[k]) for k in ("score", "cost", "steps")) or final["cost"] <= 0:
            raise ValueError(f"Invalid final observation: {path}")
        ini = read_ini(path)
        if not ini.has_section("metrics") or ini.get("base", "env_name") != "connect4cnn":
            raise ValueError(f"Invalid completed native log: {path}")
        config = {f"{s}.{k}": scalar(v) for s in ini.sections() if s != "metrics" for k, v in ini[s].items()}
        spec = architecture(config)
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        checkpoint = root / "checkpoints/connect4cnn" / path.stem / f"{final['steps']:016d}.bin"
        if not checkpoint.is_file() or checkpoint.stat().st_size % 4:
            raise ValueError(f"Missing or malformed final checkpoint: {checkpoint}")
        columns = {k: [float(x) for x in v.split(",")] for k, v in ini["metrics"].items()}
        steps = columns.pop("agent_steps")
        history = []
        for i, step in enumerate(steps):
            row = {"agent_steps": int(round(step))}
            for key, values in columns.items():
                j = i - (len(steps) - len(values))
                if 0 <= j < len(values) and math.isfinite(values[j]):
                    row[key] = values[j]
            history.append(row)
        result.append(dict(index=index, run_id=path.stem, config=config, architecture=spec,
                           architecture_sha256=identity, checkpoint=str(checkpoint.relative_to(root)),
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           params=checkpoint.stat().st_size // 4, history=history,
                           native_avg_sps=final["steps"] / final["cost"], **final))
    return sorted(result, key=lambda r: r["index"])


def sync(root, mode, project, entity=None, client=None):
    root = Path(root).resolve()
    payloads = root / "sidecar"
    payloads.mkdir(exist_ok=True)
    destination = hashlib.sha256(f"{mode}:{entity}:{project}".encode()).hexdigest()[:12]
    state_path = payloads / f"state-{destination}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if mode != "disabled" and client is None:
        for key, name in (("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "data"), ("WANDB_CONFIG_DIR", "config")):
            os.environ.setdefault(key, str(payloads / name))
        import wandb as client
    count = 0
    for trial in trials(root):
        run_id = trial["run_id"]
        content = json.dumps(trial, indent=2, sort_keys=True) + "\n"
        (payloads / f"{run_id}.json").write_text(content)
        if isinstance(state.get(run_id), dict) and state[run_id].get("schema") == SCHEMA:
            continue
        wandb_id = hashlib.sha256(f"{root.name}/{run_id}".encode()).hexdigest()[:24]
        if mode != "disabled":
            run = client.init(project=project, entity=entity, id=wandb_id, name=display_name(root, trial["index"]),
                              group=root.name, mode=mode, dir=str(payloads), config=trial["config"],
                              resume="allow" if mode == "online" else None,
                              settings=client.Settings(console="off", disable_git=True, x_disable_stats=True))
            try:
                run.define_metric("agent_steps")
                run.define_metric("*", step_metric="agent_steps")
                run.define_metric("binned/*", hidden=True)
                for row in trial["history"]:
                    run.log(row, step=row["agent_steps"])
                for key in list(run.summary.keys()):
                    if key.startswith("binned/"):
                        del run.summary[key]
                run.summary.update(trial["history"][-1])
                run.summary.update({"protein/score": trial["score"], "protein/cost_seconds": trial["cost"],
                                    "protein/random": trial["random"], "protein/gp_observations": trial["gp_obs"],
                                    "native/agent_steps": trial["steps"], "native/average_sps": trial["native_avg_sps"],
                                    "native/parameters": trial["params"], "native/checkpoint": trial["checkpoint"],
                                    "native/run_id": run_id,
                                    "architecture": trial["architecture"]["version"],
                                    "architecture_sha256": trial["architecture_sha256"],
                                    "metric_scope": "native training history (downsampled); final PROTEIN stdout rounded to 4 score / 2 cost decimals"})
                run.finish()
            except BaseException:
                run.finish(exit_code=1)
                raise
        state[run_id] = {"id": wandb_id, "schema": SCHEMA}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", choices=("disabled", "offline", "online"), default="offline")
    parser.add_argument("--project", default="puffer-cnn")
    parser.add_argument("--entity")
    parser.add_argument("--follow", action="store_true", help="Check completed trials every 5 seconds until finished.json exists")
    args = parser.parse_args()
    while True:
        count = sync(args.run, args.mode, args.project, args.entity)
        if count:
            print(f"Sidecar: recorded {count} completed trials ({args.mode})", flush=True)
        if not args.follow or (args.run / "finished.json").exists():
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
