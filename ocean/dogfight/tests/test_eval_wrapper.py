import subprocess
from pathlib import Path


def prepare_eval_source(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "ocean" / "dogfight" / "build_eval.sh"
    core_source = repo_root / "src" / "pufferl.cu"
    generated_source = tmp_path / "pufferl_dogfight_eval.cu"
    original = core_source.read_bytes()

    result = subprocess.run(
        ["bash", str(script), "--prepare-only", str(generated_source)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert core_source.read_bytes() == original
    return repo_root, generated_source.read_text()


def test_eval_wrapper_prepares_copy_without_touching_core(tmp_path):
    _, generated = prepare_eval_source(tmp_path)

    assert 'puf_ini_put(ini, "train.horizon", "8");' in generated
    assert 'strcmp(mode, "render") == 0' in generated
    assert "run_eval(&ini, &ctx, EVAL_RENDER, 1);" in generated


def test_eval_wrapper_removes_small_batch_survivorship_bias(tmp_path):
    _, generated = prepare_eval_source(tmp_path)

    assert "if (eval_agents > num_games && num_games >= 1024)" not in generated
    assert "if (eval_agents > num_games)" in generated


def test_headless_and_visible_eval_use_the_same_dogfight_contract(tmp_path):
    _, generated = prepare_eval_source(tmp_path)

    assert generated.count('puf_ini_put(&ini, "vec.num_buffers", "2");') >= 2
    assert 'puf_ini_put(&ini, "base.eval_agents", "2");' in generated
    assert generated.count(
        'puf_ini_put(&ini, "env.domain_randomization", "0");'
    ) >= 2
    assert generated.count(
        'puf_ini_put(&ini, "env.vertical_spawn_prob", "0");'
    ) >= 2
    assert generated.count(
        'puf_ini_put(&ini, "env.role_randomization", "0");'
    ) >= 2
    assert 'puf_ini_put(&ini, "env.dr", "0");' not in generated
    assert "if (num_games <= 0) {" in generated
    assert "dogfight_eval_controls avg_abs_bias=" in generated
    assert "dogfight_eval_outcomes perf=" in generated
    assert 'dict_get(&log, "env/timeouts")' in generated
    assert 'dict_get(&log, "env/player_ground")' in generated
    assert 'dict_get(&log, "env/opponent_ground")' in generated
    assert "env/target_az_neg_aileron_sum" in generated
    assert "env/target_az_pos_aileron_sum" in generated


def test_checkpoint_eval_script_builds_matching_commands(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "ocean" / "dogfight" / "eval_checkpoint.sh"
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"test")

    common = [
        "bash",
        str(script),
        "--dry-run",
        str(checkpoint),
        "2",
        "64",
        "123",
    ]
    headless = subprocess.run(
        [*common, "headless"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    visible = subprocess.run(
        [*common, "visible"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert headless.returncode == 0, headless.stdout
    assert visible.returncode == 0, visible.stdout
    for output in (headless.stdout, visible.stdout):
        assert "base.load_model_path=" in output
        assert "base.seed=123" in output
        assert "base.num_games=64" in output
        assert "base.eval_agents=2" in output
        assert "env.fixed_stage=2" in output
        assert "env.num_agents=1" in output
        assert "env.domain_randomization=0" in output
        assert "env.vertical_spawn_prob=0" in output
        assert "env.role_randomization=0" in output
        assert "env.eval_lateral_mirror=0" in output
        assert "selfplay.enabled=0" in output
        assert "vec.num_frozen_banks=0" in output
    assert " eval_bot dogfight " in f" {headless.stdout} "
    assert " render dogfight " in f" {visible.stdout} "

    mirrored = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "headless",
            str(checkpoint),
            "2",
            "64",
            "123",
            "1",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert mirrored.returncode == 0, mirrored.stdout
    assert "env.eval_lateral_mirror=1" in mirrored.stdout


def test_checkpoint_eval_prefers_clone_local_venv():
    repo_root = Path(__file__).resolve().parents[3]
    script = (
        repo_root / "ocean" / "dogfight" / "eval_checkpoint.sh"
    ).read_text()

    assert '"$repo_root/.venv"' in script
    assert "/lib/python3.12/" not in script
