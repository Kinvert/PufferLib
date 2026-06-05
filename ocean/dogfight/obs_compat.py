"""Dogfight observation-scheme compatibility notes.

The verified df36 run ``kinvert-k/df36/8wv5m6ru`` used the old Dogfight3
``obs_scheme=2``. In that codebase, scheme 2 was the 26-wide
``OBS_OPPONENT_AWARE`` contract. Dogfight5 removed the old 17-wide scheme 0 and
renumbered the active schemes, so current scheme 1 is the same 26-wide contract.

Keep this module as data only. Runtime code should not silently remap user
configs unless a caller explicitly asks for legacy interpretation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsScheme:
    name: str
    size: int


DF36_OBS_SCHEMES = {
    0: ObsScheme("OBS_MOMENTUM_GFORCE", 17),
    1: ObsScheme("OBS_PILOT", 22),
    2: ObsScheme("OBS_OPPONENT_AWARE", 26),
}

DOGFIGHT5_OBS_SCHEMES = {
    0: ObsScheme("OBS_PILOT", 22),
    1: ObsScheme("OBS_OPPONENT_AWARE", 26),
}

DF36_KNOWN_GOOD_RUN = "kinvert-k/df36/8wv5m6ru"
DF36_KNOWN_GOOD_COMMIT = "171482a9b9889bebaa427ab658c95c0fc391c031"
DF36_KNOWN_GOOD_OBS_SCHEME = 2
DOGFIGHT5_DEFAULT_OBS_SCHEME = 1


def dogfight5_scheme_for_df36_scheme(df36_scheme: int) -> int | None:
    """Return the Dogfight5 scheme id with the same observation contract."""
    df36_contract = DF36_OBS_SCHEMES.get(df36_scheme)
    if df36_contract is None:
        return None

    for scheme, contract in DOGFIGHT5_OBS_SCHEMES.items():
        if contract == df36_contract:
            return scheme
    return None


def df36_scheme_for_dogfight5_scheme(dogfight5_scheme: int) -> int | None:
    """Return the df36 scheme id with the same observation contract."""
    current_contract = DOGFIGHT5_OBS_SCHEMES.get(dogfight5_scheme)
    if current_contract is None:
        return None

    for scheme, contract in DF36_OBS_SCHEMES.items():
        if contract == current_contract:
            return scheme
    return None
