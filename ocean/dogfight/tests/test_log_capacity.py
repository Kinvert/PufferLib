import re
from pathlib import Path


def _body_after(text, anchor):
    start = text.index(anchor)
    return text[start:]


def _first_create_dict_capacity(text, anchor):
    body = _body_after(text, anchor)
    match = re.search(r"create_dict\((\d+)\)", body)
    assert match is not None, f"missing create_dict after {anchor}"
    return int(match.group(1))


def test_runtime_log_dict_capacity_covers_dogfight_metrics():
    repo_root = Path(__file__).resolve().parents[3]
    binding_c = (repo_root / "ocean" / "dogfight" / "binding.c").read_text()
    pufferlib_cu = (repo_root / "src" / "pufferlib.cu").read_text()
    bindings_cu = (repo_root / "src" / "bindings.cu").read_text()

    my_log_body = _body_after(binding_c, "void my_log").split("\n}\n", 1)[0]
    dogfight_metrics = my_log_body.count("dict_set(out,")
    required_capacity = dogfight_metrics + 1  # static_vec_log/static_vec_eval_log add "n"

    capacities = {
        "pufferl_log": _first_create_dict_capacity(
            pufferlib_cu, "Dict* log_environments_impl"
        ),
        "pufferl_eval_log": _first_create_dict_capacity(
            bindings_cu, "pybind11::dict puf_eval_log"
        ),
        "vec_log": _first_create_dict_capacity(bindings_cu, "py::dict vec_log"),
    }

    too_small = {
        name: capacity
        for name, capacity in capacities.items()
        if capacity < required_capacity
    }

    assert too_small == {}
