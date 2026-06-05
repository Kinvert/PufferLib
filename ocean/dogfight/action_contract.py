"""Dogfight continuous-action compatibility contract.

The df36 trainer sampled/logprobed raw Normal actions, then clipped the NumPy
actions to the Box(-1, 1) env boundary before stepping C. PufferLib 5 native
rollout also samples/logprobs raw Normal actions, but Dogfight clamps inside
``c_step`` because native actions are written directly into env buffers.
"""

CONTINUOUS_ACTION_DISTRIBUTION = "unsquashed_normal"

DF36_ACTION_CLIP_LOCATION = "python_before_env_step"
DOGFIGHT5_ACTION_CLIP_LOCATION = "dogfight_c_step_before_physics"

DOGFIGHT_ENV_CLAMPS_ACTIONS_BEFORE_STEP = True
PPO_LOGPROBS_USE_RAW_UNSQUASHED_ACTIONS = True
