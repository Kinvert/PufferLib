from pathlib import Path


def test_native_selfplay_accepts_external_seed_and_excludes_loaded_opponent():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "pufferl.cu"
    ).read_text(encoding="utf-8")

    assert (
        "const char* selfplay_sample(Selfplay* sp, const char* exclude)"
        in source
    )
    assert '"load_enemy_model_path", initial_opponent_buf' in source
    assert "selfplay_add_checkpoint(&selfplay, configured_opponent);" in source
    assert (
        "configured_opponent ? configured_opponent"
        in source
    )
    assert "selfplay_sample(&selfplay, bank->current_path)" in source
