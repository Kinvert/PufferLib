import math


class FakeBackend:
    def __init__(self):
        self.targets = []

    def set_curriculum_target(self, pufferl, target):
        self.targets.append((pufferl, target))


class FakePufferl:
    def __init__(self, global_step=0):
        self.global_step = global_step


def dogfight_args():
    return {
        "env_name": "dogfight",
        "env": {
            "curriculum_enabled": 1,
            "stage9_bank_deg": -1.0,
        },
        "curriculum": {
            "enabled": 1,
            "initial_target": 0.9,
            "max_target": 18.0,
            "step": 1.0,
            "promote_threshold": 0.90,
            "min_episodes": 10,
            "warmup_steps": 0,
            "eval_interval": 0,
            "stage9_bank_curriculum": 1,
            "stage9_bank_start_target": 8.5,
            "stage9_bank_full_target": 8.9,
            "stage9_bank_step": 0.1,
        },
    }


def minimal_dogfight_args():
    return {
        "env_name": "dogfight",
        "env": {
            "curriculum_enabled": 1,
        },
        "curriculum": {
            "enabled": 1,
            "initial_target": 0.9,
            "max_target": 18.0,
            "step": 1.0,
            "promote_threshold": 0.90,
        },
    }


def test_setup_curriculum_initializes_enabled_dogfight_curriculum():
    import pufferlib.pufferl as pufferl

    backend = FakeBackend()
    pufferl_obj = object()

    state = pufferl.setup_curriculum(dogfight_args(), backend, pufferl_obj)

    assert state["target"] == 0.9
    assert state["max_target"] == 18.0
    assert backend.targets == [(pufferl_obj, 0.9)]


def test_stage9_defaults_to_dogfight3_full_bank_without_hidden_ramp():
    import pufferlib.pufferl as pufferl

    state = pufferl.setup_curriculum(minimal_dogfight_args(), FakeBackend(), object())
    state["target"] = 8.4

    assert state["stage9_bank_curriculum"] == 0
    assert math.isclose(state["stage9_bank_override"], 30.0)
    assert math.isclose(pufferl.stage9_bank_for_target({"env": {}}, 8.5), 30.0)
    assert math.isclose(pufferl.stage9_bank_for_state(state, 8.5), 30.0)
    assert math.isclose(pufferl.next_curriculum_target(state), 9.4)


def test_step_curriculum_promotes_after_mastery_stage_success():
    import pufferlib.pufferl as pufferl

    backend = FakeBackend()
    pufferl_obj = object()
    state = pufferl.setup_curriculum(dogfight_args(), backend, pufferl_obj)
    logs = {
        "env/n": 10.0,
        "env/base_stage_eps": 1.0,
        "env/base_stage_kills": 0.9,
    }

    pufferl.step_curriculum(state, backend, pufferl_obj, logs, epoch=3)

    assert math.isclose(state["target"], 1.9)
    assert backend.targets[-1] == (pufferl_obj, 1.9)
    assert math.isclose(logs["env/base_stage_kill_rate"], 0.9)
    assert math.isclose(logs["env/curriculum_target"], 1.9)
    assert logs["env/mastery_stage"] == 2


def test_step_curriculum_respects_warmup_and_eval_interval():
    import pufferlib.pufferl as pufferl

    args = dogfight_args()
    args["curriculum"]["warmup_steps"] = 1_000
    args["curriculum"]["eval_interval"] = 500
    backend = FakeBackend()
    pufferl_obj = FakePufferl(global_step=900)
    state = pufferl.setup_curriculum(args, backend, pufferl_obj)
    logs = {
        "env/n": 20.0,
        "env/base_stage_eps": 1.0,
        "env/base_stage_kills": 0.9,
    }

    pufferl.step_curriculum(state, backend, pufferl_obj, logs, epoch=1)

    assert math.isclose(state["target"], 0.9)
    assert state["base_stage_eps"] == 0.0
    assert backend.targets == [(pufferl_obj, 0.9)]
    assert state["last_eval_step"] == 1_000

    pufferl_obj.global_step = 1_200
    pufferl.step_curriculum(state, backend, pufferl_obj, logs, epoch=2)

    assert math.isclose(state["target"], 0.9)
    assert state["base_stage_eps"] == 20.0
    assert state["last_eval_step"] == 1_000
    assert backend.targets == [(pufferl_obj, 0.9)]

    pufferl_obj.global_step = 1_500
    pufferl.step_curriculum(state, backend, pufferl_obj, logs, epoch=3)

    assert math.isclose(state["target"], 1.9)
    assert state["mastered_stage"] == 1
    assert state["last_eval_step"] == 1_500
    assert backend.targets[-1] == (pufferl_obj, 1.9)


def test_mastery_metrics_derive_side_spawn_variant_rates():
    import pufferlib.pufferl as pufferl

    logs = {
        "env/n": 200.0,
        "env/base_stage_eps": 0.5,
        "env/base_stage_kills": 0.4,
        "env/base_stage_side_standard_eps": 0.25,
        "env/base_stage_side_standard_kills": 0.125,
        "env/base_stage_side_standard_ground": 0.025,
        "env/base_stage_side_standard_timeouts": 0.05,
        "env/base_stage_side_standard_episode_length_sum": 200.0,
        "env/base_stage_side_standard_action_saturation_sum": 0.05,
        "env/base_stage_side_standard_signed_bias_sum": -0.10,
        "env/base_stage_side_energy_eps": 0.25,
        "env/base_stage_side_energy_kills": 0.075,
        "env/base_stage_side_energy_ground": 0.075,
        "env/base_stage_side_energy_timeouts": 0.025,
        "env/base_stage_side_energy_episode_length_sum": 250.0,
        "env/base_stage_side_energy_action_saturation_sum": 0.15,
        "env/base_stage_side_energy_signed_bias_sum": 0.20,
    }

    pufferl.add_mastery_stage_metrics(logs)

    assert math.isclose(logs["env/base_stage_side_standard_kill_rate"], 0.5)
    assert math.isclose(logs["env/base_stage_side_standard_ground_rate"], 0.1)
    assert math.isclose(logs["env/base_stage_side_standard_timeout_rate"], 0.2)
    assert math.isclose(logs["env/base_stage_side_standard_episode_length"], 800.0)
    assert math.isclose(logs["env/base_stage_side_standard_action_saturation"], 0.2)
    assert math.isclose(logs["env/base_stage_side_standard_signed_bias"], -0.4)
    assert math.isclose(logs["env/base_stage_side_energy_kill_rate"], 0.3)
    assert math.isclose(logs["env/base_stage_side_energy_ground_rate"], 0.3)
    assert math.isclose(logs["env/base_stage_side_energy_timeout_rate"], 0.1)
    assert math.isclose(logs["env/base_stage_side_energy_episode_length"], 1000.0)
    assert math.isclose(logs["env/base_stage_side_energy_action_saturation"], 0.6)
    assert math.isclose(logs["env/base_stage_side_energy_signed_bias"], 0.8)


def test_stage9_curriculum_uses_substeps_before_full_bank():
    import pufferlib.pufferl as pufferl

    state = pufferl.setup_curriculum(dogfight_args(), FakeBackend(), object())
    state["target"] = 8.4

    assert math.isclose(pufferl.next_curriculum_target(state), 8.5)
    state["target"] = 8.5
    assert math.isclose(pufferl.next_curriculum_target(state), 8.6)
