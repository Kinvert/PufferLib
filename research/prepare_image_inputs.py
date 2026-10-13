#!/usr/bin/env python3
"""Freeze realistic image inputs and bounded validation configs, without GPU work.

No learner or neural reference runs here. These short validation recipes are
not calibrated overnight budgets and do not replace historical recipes.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "connect4cnn": (576,672), "pongcnn": (640,620), "flappycnn": (640,420),
    "breakoutcnn": (330,576), "snakecnn": (616,616), "mazecnn": (616,616),
}
CONTROLS = {
    "nature": {"encoder": 2},
    "quality": {"encoder": 4, "cnn_depth": 1, "cnn_projection": 64, "cnn_global_pool": 0,
                "cnn_channels_1": 16, "cnn_kernel_1": 7, "cnn_stride_1": 4, "cnn_pool_1": 0, "cnn_residual_1": 0},
}


def architecture(channels=8, residual=0, stride=4, gap=0):
    if channels not in (8,16) or residual not in (0,1) or stride not in (1,2,4,8) or gap not in (0,1):
        raise ValueError("Invalid targeted architecture")
    return dict(encoder=4, cnn_depth=1, cnn_projection=64, cnn_global_pool=gap,
                cnn_channels_1=channels, cnn_kernel_1=8, cnn_stride_1=stride,
                cnn_pool_1=0, cnn_residual_1=residual)


def config(game, arch):
    if game not in SHAPES: raise ValueError("Unknown image game")
    value = configparser.ConfigParser(interpolation=None)
    paths = [ROOT/"config/default.ini", ROOT/f"config/{game}.ini", ROOT/f"ocean/{game}/compare.ini"]
    if value.read(paths) != [str(p) for p in paths]: raise ValueError("Missing frozen learner input")
    # Bounded plumbing geometry, identical across architectures within a game.
    # Main-run batch/horizon selection needs new-image memory and learning calibration.
    overlays = {
        "base": dict(async_=0, seed=7, eval_episodes=0, load_model_path="None", checkpoint_interval=2),
        "vec": dict(total_agents=4, num_buffers=1, num_threads=1, num_policies=1, hist_policy_percent=0),
        "train": dict(gpus=1, horizon=16, minibatch_size=16, total_timesteps=256, replay_ratio=1),
        "selfplay": dict(enabled=0),
        "policy": dict(hidden_size=128, num_layers=1, **arch),
        "env": dict(representation=0, representation_mode=1, representation_seed=17,
                    representation_mix_catalog=1 if game in ("pongcnn","flappycnn") else 0),
    }
    for section, values in overlays.items():
        for key, data in values.items(): value.set(section, "async" if key == "async_" else key, str(data))
    for section in list(value.sections()):
        if section.startswith("sweep."): value.remove_section(section)
    return value, paths


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    candidates = {**CONTROLS, **{f"c{c}-res{r}": architecture(c,r) for c in (8,16) for r in (0,1)},
                  **{f"stride{s}-gap": architecture(stride=s,gap=1) for s in (1,2,4,8)}}
    entries=[]; sources={}
    for game,(height,width) in SHAPES.items():
        for name,arch in candidates.items():
            value,paths=config(game,arch)
            path=out/game/(name+".ini"); path.parent.mkdir(exist_ok=True)
            with path.open("w") as stream: value.write(stream)
            entries.append(dict(game=game,candidate=name,config=str(path.relative_to(out)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape_hwc=[height,width,3]))
            sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    result=dict(schema="realistic-image-validation-inputs-v1",entries=entries,source_sha256=sources,
        native_frame_and_policy_shape_equal=True,preprocessing="none",layout="NHWC",dtype="float32",
        historical_input_default_unchanged=True,seed_sweep_allowed=False,
        training_seeds=[7],appearance_seed=17,calibrated_for_learning=False,gpu_executed=False)
    (out/"inputs.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); prepare(args.out.resolve()); print(args.out.resolve()/"inputs.json")
