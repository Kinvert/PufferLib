import re
import subprocess
from pathlib import Path


DF36_COMMIT = "171482a9b9889bebaa427ab658c95c0fc391c031"


def repo_root():
    return Path(__file__).resolve().parents[3]


def df36_file(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            "/home/claude/dogfight3",
            "show",
            f"{DF36_COMMIT}:{path}",
        ],
        text=True,
    )


def extract_balanced_block(source, start_pattern):
    match = re.search(start_pattern, source)
    assert match is not None, f"missing block matching {start_pattern}"

    start = match.start()
    open_brace = source.index("{", match.end() - 1)
    depth = 0
    for idx in range(open_brace, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"unterminated block matching {start_pattern}")


def remove_function(source, name):
    pattern = rf"static inline [^;]+ {name}\s*\([^;]*?\)\s*\{{"
    block = extract_balanced_block(source, pattern)
    return source.replace(block, "")


def normalize_flightlib(source):
    source = source.replace("INDUCED_DRAG_K", "K")
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"\s+", " ", source)
    source = re.sub(r"\s*([{}(),;=+*/<>?:&|\\-])\s*", r"\1", source)
    return source.strip()


def test_flightlib_matches_df36_except_macro_name_and_state_rng_helper():
    current = (repo_root() / "ocean" / "dogfight" / "flightlib.h").read_text()
    reference = df36_file("pufferlib/ocean/dogfight/flightlib.h")

    current_without_state_rng = remove_function(current, "rndf_state")
    current_without_state_rng = remove_function(
        current_without_state_rng,
        "randomize_flight_params_rng",
    )

    assert normalize_flightlib(current_without_state_rng) == normalize_flightlib(
        reference
    )
