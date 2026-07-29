import subprocess
from pathlib import Path


def test_reproduction_profile_emits_proven_stock_core_command():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "ocean" / "dogfight" / "train_reproduction.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "tdd_canary",
            "1048576",
            "0",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    expected = [
        "train dogfight",
        "base.run_id=tdd_canary",
        "base.seed=42",
        "vec.total_agents=4096",
        "vec.num_buffers=4",
        "vec.num_frozen_banks=0",
        "vec.frozen_bank_pct=0",
        "selfplay.enabled=0",
        "policy.hidden_size=64",
        "policy.num_layers=3",
        "train.total_timesteps=1048576",
        "train.learning_rate=0.0025",
        "train.minibatch_size=4096",
        "train.gamma=0.996",
        "train.gae_lambda=0.999",
        "train.ent_coef=0.02",
        "train.clip_coef=0.06",
        "train.vf_coef=4.6",
        "train.vf_clip_coef=1.5",
        "train.max_grad_norm=3.4",
        "train.momentum=0.9896",
        "train.prio_alpha=0.4",
        "train.prio_beta0=0.82",
        "train.replay_ratio=0.95",
        "train.vtrace_rho_clip=0.1",
        "train.vtrace_c_clip=2.5",
        "env.num_agents=1",
        "env.fixed_stage=0",
        "env.role_randomization=0",
        "env.domain_randomization=0.05",
        "env.vertical_spawn_prob=0.02",
        "env.warmup_steps=1000000",
        "env.eval_interval=60000",
        "env.min_eval_episodes=50",
    ]
    for argument in expected:
        assert argument in result.stdout
