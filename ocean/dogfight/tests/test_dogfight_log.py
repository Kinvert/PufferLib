import importlib
from pathlib import Path
import re
import sys


def load_dogfight_log():
    dogfight_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(dogfight_dir))
    try:
        return importlib.import_module("dogfight_log")
    finally:
        sys.path.pop(0)


def reset_logger(module):
    for handler in list(module.logger.handlers):
        module.logger.removeHandler(handler)
        handler.close()


def test_dogfight_log_creates_timestamped_named_file(tmp_path):
    module = load_dogfight_log()
    reset_logger(module)

    path = Path(module.init_log(tmp_path, "unit_named"))
    try:
        assert path.exists()
        assert path.name.startswith("unit_named_")
        assert re.search(r"\d{4}-\d{2}-\d{2}_\d{6}\.log$", path.name)
    finally:
        reset_logger(module)


def test_dogfight_log_creates_nested_default_file(tmp_path):
    module = load_dogfight_log()
    reset_logger(module)

    nested = tmp_path / "a" / "b" / "c"
    path = Path(module.init_log(nested))
    try:
        assert nested.is_dir()
        assert path.name.startswith("dogfight_")
    finally:
        reset_logger(module)


def test_dogfight_log_writes_structured_timestamped_lines(tmp_path):
    module = load_dogfight_log()
    reset_logger(module)

    path = Path(module.init_log(tmp_path, "wrote"))
    try:
        module.log("[ROUND] num=42 event=start")
        module.log("[RATING] player=foo rating=1234")
        module.log("[ERROR] something broke")
        for handler in list(module.logger.handlers):
            handler.flush()

        content = path.read_text()
        for tag in ("[ROUND]", "[RATING]", "[ERROR]", "log_started"):
            assert tag in content
        assert re.search(r"^\d{2}:\d{2}:\d{2} ", content, re.MULTILINE)
    finally:
        reset_logger(module)
