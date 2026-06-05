from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_native_rollout_passes_trainer_global_step_to_dogfight_envs():
    root = repo_root()
    df36_train = Path(
        "/home/claude/dogfight3/pufferlib/ocean/dogfight/train_dual_selfplay.py"
    ).read_text()
    dogfight_binding = (root / "ocean" / "dogfight" / "binding.c").read_text()
    vecenv_h = (root / "src" / "vecenv.h").read_text()
    bindings_cu = (root / "src" / "bindings.cu").read_text()

    assert "vec_set_global_step" in df36_train
    assert "MY_GLOBAL_STEP" in dogfight_binding
    assert "my_set_global_step(Env* env, long global_step)" in dogfight_binding
    assert "static_vec_set_global_step(StaticVec* vec, long global_step)" in vecenv_h
    assert "my_set_global_step(&envs[i], step_global)" in vecenv_h

    set_pos = bindings_cu.index("static_vec_set_global_step(pufferl.vec, pufferl.global_step)")
    rollout_pos = bindings_cu.index("static_vec_omp_step(pufferl.vec)")
    increment_pos = bindings_cu.index(
        "pufferl.global_step += pufferl.hypers.horizon * pufferl.hypers.total_agents"
    )

    assert set_pos < rollout_pos < increment_pos
