import configparser
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "ocean" / "dogfight" / "wandb_sidecar.py"
CONFIG = REPO_ROOT / "config" / "dogfight.ini"


def load_module():
    spec = importlib.util.spec_from_file_location("dogfight_wandb_sidecar", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_log(path: Path, include_pool_score: bool = True) -> None:
    # Native INI serialization backfills this final-only metric across all
    # downsample points; the sidecar must still publish it only at the end.
    pool = (
        "selfplay/pool_score = 0.75,0.75\n"
        if include_pool_score
        else ""
    )
    path.write_text(
        f"""
[base]
env_name = dogfight
run_id = {path.stem}
checkpoint_dir = checkpoints

[selfplay]
enabled = 1
eval_games = 4096
eval_pool_size = 8

[train]
total_timesteps = 200

[metrics]
agent_steps = 100,200
SPS = 50,80
uptime = 2,4
env/perf = 0.25,0.5
{pool}""".strip()
        + "\n",
        encoding="ascii",
    )


class FakeRun:
    def __init__(self, run_id: str, name: str):
        self.id = run_id
        self.name = name
        self.summary = {}
        self.history = []
        self.finished = False

    def log(self, values, step):
        self.history.append((values, step))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.runs = []

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run_id = kwargs.get("id", "generated-id-1")
        run = FakeRun(run_id, "random-name-7")
        self.runs.append(run)
        return run


def test_native_pool_score_is_the_fitness_and_aligns_to_final_step(tmp_path):
    module = load_module()
    log_path = tmp_path / "sweep_1_0000.ini"
    write_log(log_path)

    payload = module.load_native_log(log_path)

    assert payload["valid_pool_eval"] is True
    assert payload["summary"]["protein/fitness"] == 0.75
    assert payload["summary"]["selfplay/pool_winrate"] == 0.75
    assert module.POOL_SCORE_KEY not in payload["history"][0]["metrics"]
    assert payload["history"][1]["metrics"][module.POOL_SCORE_KEY] == 0.75
    assert payload["summary"]["native/mean_sps"] == 50


def test_first_upload_lets_wandb_generate_name_and_resume_uses_saved_id(
    tmp_path,
):
    module = load_module()
    log_path = tmp_path / "sweep_1_0000.ini"
    state_path = tmp_path / "state.json"
    write_log(log_path)
    state = module.load_or_create_state(
        state_path, [], include_existing=True
    )
    fake_wandb = FakeWandb()

    module.upload_native_log(
        log_path,
        "df43",
        state,
        state_path,
        tmp_path / "checkpoints",
        "dogfight",
        wandb_module=fake_wandb,
    )

    first = fake_wandb.init_calls[0]
    assert "id" not in first
    assert "name" not in first
    assert "group" not in first
    assert state["runs"][log_path.name]["wandb_id"] == "generated-id-1"
    assert state["runs"][log_path.name]["wandb_name"] == "random-name-7"
    assert state["runs"][log_path.name]["status"] == "uploaded"

    state["runs"][log_path.name]["status"] = "failed"
    module.upload_native_log(
        log_path,
        "df43",
        state,
        state_path,
        tmp_path / "checkpoints",
        "dogfight",
        wandb_module=fake_wandb,
    )

    resumed = fake_wandb.init_calls[1]
    assert resumed["id"] == "generated-id-1"
    assert resumed["resume"] == "allow"
    assert "name" not in resumed


def test_new_state_ignores_old_logs_and_discovers_new_logs(tmp_path):
    module = load_module()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old_log = log_dir / "old.ini"
    new_log = log_dir / "new.ini"
    write_log(old_log)
    state = module.load_or_create_state(
        tmp_path / "state.json", [old_log], include_existing=False
    )

    assert module.discover_logs(log_dir, state, 0) == []
    write_log(new_log)
    assert module.discover_logs(log_dir, state, 0) == [new_log]


def test_missing_pool_score_is_explicitly_invalid(tmp_path):
    module = load_module()
    log_path = tmp_path / "sweep_1_0000.ini"
    write_log(log_path, include_pool_score=False)

    payload = module.load_native_log(log_path)

    assert payload["valid_pool_eval"] is False
    assert payload["summary"]["selfplay/valid_pool_eval"] is False
    assert payload["summary"]["protein/fitness"] is None


def test_dogfight_enables_robocode_style_pool_evaluation():
    config = configparser.ConfigParser(interpolation=None)
    config.read(CONFIG)

    assert config["selfplay"]["eval_games"] == "4096"
    assert config["selfplay"]["eval_pool_size"] == "8"
    assert config["sweep"]["metric"] == "score"


def test_native_match_horizon_satisfies_advantage_vector_width():
    source = (REPO_ROOT / "src" / "pufferl.cu").read_text()

    assert 'puf_ini_put(ini, "train.horizon", "1");' in source
    assert 'puf_ini_put(ini, "train.horizon", "8");' not in source
