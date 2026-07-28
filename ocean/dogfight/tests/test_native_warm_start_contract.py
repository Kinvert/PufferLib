from pathlib import Path


def _source() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "src" / "pufferl.cu").read_text()


def _between(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_checkpoint_loader_requires_exact_byte_count():
    loader = _between(
        _source(),
        "void puf_load_weights_into(",
        "void pufferl_load_primary_weights(",
    )

    assert "fstat(" in loader
    assert "checkpoint size mismatch" in loader
    assert "expected=%" in loader
    assert "actual=%" in loader


def test_primary_load_synchronizes_async_actor_snapshot():
    helper = _between(
        _source(),
        "void pufferl_load_primary_weights(",
        "void pufferl_load_frozen_bank(",
    )

    assert "puf_load_weights_into(" in helper
    assert "actor_param_puf" in helper
    assert "cudaStreamSynchronize" in helper


def test_train_load_precedes_selfplay_initial_checkpoint():
    train = _between(_source(), "TrainResult run_train(", "TrainResult launch_train(")

    create_index = train.index("PuffeRL* pufferl = create_pufferl(")
    load_index = train.index("pufferl_load_primary_weights(")
    selfplay_index = train.index("Selfplay selfplay")
    initial_save_index = train.index("puf_save_weights(pufferl, initial_checkpoint)")

    assert create_index < load_index < selfplay_index < initial_save_index
    warm_start = train[create_index:selfplay_index]
    assert "puf_checkpoint_path_key(ini," in warm_start
    assert '"load_model_path"' in warm_start
