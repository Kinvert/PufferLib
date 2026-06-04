def test_downsample_logs_keeps_late_derived_metrics():
    import pufferlib.pufferl as pufferl

    logs = [
        {
            "agent_steps": 1,
            "env/score": 0.1,
        },
        {
            "agent_steps": 2,
            "env/score": 0.2,
            "env/base_stage_side_standard_ground_rate": 0.3,
        },
    ]

    metrics = pufferl.downsample_logs(logs, 2)

    assert metrics["env/score"] == [0.15000000000000002, 0.2]
    assert metrics["env/base_stage_side_standard_ground_rate"] == [0.3, 0.3]
