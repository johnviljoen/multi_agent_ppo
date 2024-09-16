
# sane defaults from cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
ppo_params = {
    "total_timesteps": 10000000,
    "learning_rate": 2.5e-4,
    "num_envs": 8,
    "num_steps": 128,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.1,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
}

