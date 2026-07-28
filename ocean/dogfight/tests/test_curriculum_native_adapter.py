from pathlib import Path


def test_curriculum_poll_precedes_dashboard_throttle():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "src" / "pufferl.cu").read_text()

    poll = source.index("Dict curriculum_log = {0};")
    throttle = source.index("if (!is_eval && last_log.size")

    assert poll < throttle
    assert "vec_log(pufferl->vec, &curriculum_log, 0);" in source[poll:throttle]
    assert "puf_curriculum_update(" in source[poll:throttle]
    assert "puf_curriculum_counters_cleared(&pufferl->curriculum);" in source
