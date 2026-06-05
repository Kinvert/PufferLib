import re
import subprocess
from pathlib import Path


DF36_COMMIT = "171482a9b9889bebaa427ab658c95c0fc391c031"
EARLY_STAGE_SPAWNERS = [
    "spawn_tail_chase",
    "spawn_head_on",
    "spawn_crossing",
    "spawn_vertical",
    "spawn_gentle_turns",
    "spawn_offset",
    "spawn_angled",
]
ADVANCED_STAGE_SPAWNERS = [
    "spawn_dive_attack",
    "spawn_zoom_attack",
    "spawn_rear",
    "spawn_full_predictable",
    "spawn_full_random",
    "spawn_medium_turns",
    "spawn_hard_maneuvering",
]


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


def normalize_c(source):
    source = source.replace("dogfight_rndf(env,", "rndf(")
    source = source.replace("dogfight_rand_int(env,", "rand_int(")
    source = source.replace("env->low_altitude_variant = 1;", "")
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"\s+", " ", source)
    source = re.sub(r"\s*([{}(),;=+*/<>?:&|-])\s*", r"\1", source)
    return source.strip()


def normalize_side_spawner(source):
    source = source.replace(
        "if (dogfight_rndf(env, 0, 1) < "
        "clampf(env->side_energy_spawn_prob, 0.0f, 1.0f)) {",
        "if (rndf(0, 1) < 0.2f) {",
    )
    source = source.replace("        env->side_spawn_variant = SIDE_SPAWN_ENERGY;\n", "")
    source = source.replace("    env->side_spawn_variant = SIDE_SPAWN_STANDARD;\n", "")
    source = re.sub(
        r"\n\s*float bank_deg = resolve_side_bank_deg\(env, cfg\);"
        r"\n\s*bank_deg = clampf\(bank_deg, 0.0f, 90.0f\);",
        "",
        source,
    )
    source = source.replace("if (bank_deg > 0.0f)", "if (cfg->bank > 0)")
    source = source.replace(
        "env->opponent_ap.target_bank = bank_deg * DEG_TO_RAD;",
        "env->opponent_ap.target_bank = (float)cfg->bank * DEG_TO_RAD;",
    )
    source = source.replace("\n        env->opponent_ap.target_bank = 0.0f;", "")
    return normalize_c(source)


def test_advanced_curriculum_stage_table_matches_df36():
    current = (repo_root() / "ocean" / "dogfight" / "dogfight.h").read_text()
    reference = df36_file("pufferlib/ocean/dogfight/dogfight.h")

    current_table = extract_balanced_block(
        current,
        r"static const StageConfig STAGES\[CURRICULUM_COUNT\]\s*=\s*\{",
    )
    reference_table = extract_balanced_block(
        reference,
        r"static const StageConfig STAGES\[CURRICULUM_COUNT\]\s*=\s*\{",
    )

    assert normalize_c(current_table) == normalize_c(reference_table)


def test_advanced_stage_spawners_match_df36_except_env_rng_helper():
    current = (repo_root() / "ocean" / "dogfight" / "dogfight_spawn.h").read_text()
    reference = df36_file("pufferlib/ocean/dogfight/dogfight_spawn.h")

    for spawner in ADVANCED_STAGE_SPAWNERS:
        pattern = rf"static void {spawner}\s*\([^;]*?\)\s*\{{"
        current_block = extract_balanced_block(current, pattern)
        reference_block = extract_balanced_block(reference, pattern)

        assert normalize_c(current_block) == normalize_c(reference_block), spawner


def test_early_stage_spawners_match_df36_except_env_rng_and_telemetry():
    current = (repo_root() / "ocean" / "dogfight" / "dogfight_spawn.h").read_text()
    reference = df36_file("pufferlib/ocean/dogfight/dogfight_spawn.h")

    for spawner in EARLY_STAGE_SPAWNERS:
        pattern = rf"static void {spawner}\s*\([^;]*?\)\s*\{{"
        current_block = extract_balanced_block(current, pattern)
        reference_block = extract_balanced_block(reference, pattern)

        assert normalize_c(current_block) == normalize_c(reference_block), spawner


def test_side_stage_spawner_matches_df36_except_env_rng_config_and_telemetry():
    current = (repo_root() / "ocean" / "dogfight" / "dogfight_spawn.h").read_text()
    reference = df36_file("pufferlib/ocean/dogfight/dogfight_spawn.h")

    pattern = r"static void spawn_side\s*\([^;]*?\)\s*\{"
    current_block = extract_balanced_block(current, pattern)
    reference_block = extract_balanced_block(reference, pattern)

    assert normalize_side_spawner(current_block) == normalize_side_spawner(
        reference_block
    )
