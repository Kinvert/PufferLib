import configparser
import importlib.util
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOGFIGHT_DIR = REPO_ROOT / "ocean" / "dogfight"
SCRIPT = DOGFIGHT_DIR / "native_sweep.py"
WRAPPER = DOGFIGHT_DIR / "sweep_vanilla_selfplay.sh"


def run_wrapper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("dogfight_native_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_translates_friendly_max_runs_to_native_5c_syntax():
    result = run_wrapper(
        "--dry-run",
        "--max-runs",
        "1000",
        "--timesteps",
        "8388608",
        "--session",
        "df42-sweep-test",
    )

    assert result.returncode == 0, result.stdout
    assert "sweep.max_runs=1000" in result.stdout
    assert "train.total_timesteps=8388608" in result.stdout
    assert "--max-runs 1000" not in result.stdout
    assert "--wandb" in result.stdout
    assert "--wandb-project=df42" in result.stdout
    assert "/home/claude/PufferLib/.venv/bin/python" in result.stdout
    assert "tmux new-session" in result.stdout


def test_launcher_prefers_clone_local_virtualenv():
    wrapper = WRAPPER.read_text()
    launcher = SCRIPT.read_text()
    local_choice = wrapper.index('repo_root/.venv')
    compatibility_fallback = wrapper.index('/home/claude/PufferLib/.venv')

    assert local_choice < compatibility_fallback
    assert "/home/claude/PufferLib" not in launcher
    assert "python3.12" not in launcher


def test_prepare_stages_only_the_intended_native_sweep_dimensions(tmp_path):
    experiment = tmp_path / "df42-sweep-canary"
    result = subprocess.run(
        [
            "/home/claude/PufferLib/.venv/bin/python",
            str(SCRIPT),
            "prepare",
            "--experiment-dir",
            str(experiment),
            "--max-runs",
            "1",
            "--timesteps",
            "8388608",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    staged_default = configparser.ConfigParser(interpolation=None)
    staged_default.read(experiment / "config" / "default.ini")
    assert not [
        section
        for section in staged_default.sections()
        if section.startswith("sweep.")
    ]

    staged_dogfight = configparser.ConfigParser(interpolation=None)
    staged_dogfight.read(experiment / "config" / "dogfight.ini")
    dimensions = {
        section.removeprefix("sweep.")
        for section in staged_dogfight.sections()
        if section.startswith("sweep.")
    }
    assert dimensions == {
        "env.native_spawn_total_steps",
        "env.native_frontier_fraction",
        "vec.frozen_bank_pct",
    }
    assert staged_dogfight["sweep"]["max_runs"] == "1"
    assert staged_dogfight["train"]["total_timesteps"] == "8388608"
    checkpoint_interval = staged_dogfight.getint(
        "base", "checkpoint_interval"
    )
    assert checkpoint_interval > 0
    assert staged_dogfight["base"]["wandb_project"] == "df43"
    assert Path(staged_dogfight["base"]["checkpoint_dir"]).is_absolute()
    assert Path(staged_dogfight["base"]["log_dir"]).is_absolute()

    manifest = json.loads((experiment / "manifest.json").read_text())
    assert manifest["max_runs"] == 1
    assert manifest["wandb_project"] == "df43"
    assert manifest["native_sweep_dimensions"] == sorted(dimensions)
    assert manifest["native_command"][-1] == "train.total_timesteps=8388608"


def test_default_profile_is_a_bounded_screening_run(tmp_path):
    experiment = tmp_path / "df42-sweep-screen"
    result = subprocess.run(
        [
            "/home/claude/PufferLib/.venv/bin/python",
            str(SCRIPT),
            "prepare",
            "--experiment-dir",
            str(experiment),
            "--max-runs",
            "1000",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    staged = configparser.ConfigParser(interpolation=None)
    staged.read(experiment / "config" / "dogfight.ini")
    total_timesteps = staged.getint("train", "total_timesteps")
    spawn_timesteps = staged.getint("env", "native_spawn_total_steps")
    assert total_timesteps > 0
    assert spawn_timesteps == total_timesteps
    spawn_sweep = staged["sweep.env.native_spawn_total_steps"]
    spawn_min = int(spawn_sweep["min"].replace("_", ""))
    spawn_max = int(spawn_sweep["max"].replace("_", ""))
    assert 0 < spawn_min < spawn_max
    assert spawn_min <= spawn_timesteps <= spawn_max
    assert staged["env"]["native_acquisition_steps"] == "0"
    assert staged.getint("base", "checkpoint_interval") > 0


def test_wandb_payload_is_bounded_and_uses_native_agent_steps(tmp_path):
    module = load_sweep_module()
    log_path = tmp_path / "sweep_1_0000.ini"
    extra_metrics = "\n".join(
        f"env/extra_{index} = {index}.0,{index + 1}.0"
        for index in range(40)
    )
    log_path.write_text(
        f"""
[base]
run_id = sweep_1_0000

[train]
total_timesteps = 8388608

[metrics]
agent_steps = 4194304,8388608
SPS = 1000000,1100000
uptime = 4.0,8.0
env/perf = 0.25,0.5
{extra_metrics}
""".strip()
        + "\n"
    )

    payload = module.build_wandb_payload(log_path)

    assert payload["run_id"] == "sweep_1_0000"
    assert [row["step"] for row in payload["history"]] == [4194304, 8388608]
    assert all(len(row["metrics"]) <= 31 for row in payload["history"])
    assert payload["summary"]["final_agent_steps"] == 8388608
    assert payload["summary"]["final_env/perf"] == 0.5


def test_wandb_uses_native_id_without_overriding_random_display_name(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    log_path = tmp_path / "sweep_1_0000.ini"
    log_path.write_text(
        """
[base]
run_id = sweep_1_0000

[metrics]
agent_steps = 8388608
env/perf = 0.5
""".strip()
        + "\n"
    )

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, *args, **kwargs):
            pass

        def finish(self):
            pass

    class FakeWandb:
        def __init__(self):
            self.init_calls = []

        def Settings(self, **kwargs):
            return kwargs

        def init(self, **kwargs):
            self.init_calls.append(kwargs)
            return FakeRun()

    fake_wandb = FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    module.upload_wandb_run(tmp_path, log_path, "df42")
    module.upload_fixed_wandb_run(
        {
            "run_id": "sweep_1_0000",
            "highest_contiguous_stage": -1,
            "all_mastered": False,
            "min_perf": 0.5,
            "mean_perf": 0.5,
            "stopped_after_stage": 0,
            "checkpoint_sha256": "abc123",
            "samples": [],
        },
        "df42",
    )

    assert len(fake_wandb.init_calls) == 2
    assert all(call["id"] == "sweep_1_0000" for call in fake_wandb.init_calls)
    assert all("name" not in call for call in fake_wandb.init_calls)


def test_dogfight_exports_at_most_31_values_before_core_appends_n():
    source = (DOGFIGHT_DIR / "dogfight_puffer.h").read_text()
    match = re.search(
        r"void puf_log\(Log\* log, Dict\* out\) \{(?P<body>.*?)\n\}",
        source,
        re.DOTALL,
    )
    assert match is not None
    keys = re.findall(
        r'dict_set\(\s*out\s*,\s*"([^"]+)"',
        match.group("body"),
        re.DOTALL,
    )

    assert len(keys) <= 31
    assert len(keys) + 1 <= 32
    assert len(keys) == len(set(keys))
