from pathlib import Path


CORE_DOGFIGHT_FILES = (
    "binding.c",
    "dogfight.c",
    "dogfight.h",
    "flightlib.h",
    "autopilot.h",
    "autoace.h",
    "dogfight_observations.h",
    "dogfight_spawn.h",
    "dogfight_render.h",
    "p40.glb",
)


def test_core_dogfight_port_files_are_present():
    dogfight_dir = Path(__file__).resolve().parents[1]

    missing = [
        filename
        for filename in CORE_DOGFIGHT_FILES
        if not (dogfight_dir / filename).is_file()
    ]

    assert missing == []
