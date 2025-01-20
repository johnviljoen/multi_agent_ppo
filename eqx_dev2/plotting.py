import jax
import equinox as eqx
from tqdm import tqdm

# BRAX rendering
from IPython.display import HTML, clear_output
from brax.io import model
from brax.io import html, image
from brax import envs
import numpy as np

def rollout_and_render(env, actor, obs_rms, save_path="tmp/test.html"):
    print("rolling out policy to render")
    rollout = []
    rng = jax.random.PRNGKey(seed=0)
    env_state = env.reset(rng=rng)
    env_step = jax.jit(env.step)
    actor_call = eqx.filter_jit(actor)
    for _ in tqdm(range(1000)):
        rollout.append(env_state.pipeline_state)
        _rng, rng = jax.random.split(rng)
        obs = env_state.obs
        obs_normalized = obs_rms.normalize(obs)
        act, _ = actor_call(_rng, obs_normalized)
        env_state = env_step(env_state, act)

    print("rendering")
    with open(save_path, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))

def eval_rollout(env, actor, obs_rms, num_eval_envs):
    env = envs.training.EvalWrapper(envs.training.wrap(env))
    rollout = []
    key = jax.random.PRNGKey(seed=0)
    reset_key, rng = jax.random.split(key)
    reset_key = jax.random.split(reset_key, num_eval_envs)
    env_state = env.reset(rng=reset_key)
    env_step = jax.jit(env.step)
    actor_call = eqx.filter_jit(actor)
    cum_rew = 0
    for _ in tqdm(range(1000)):
        rollout.append(env_state.pipeline_state)
        obs = env_state.obs
        obs_normalized = obs_rms.normalize(obs)
        act, _ = jax.vmap(actor_call)(jax.random.split(rng, num_eval_envs), obs_normalized)
        env_state = env_step(env_state, act)
    

    # with open(save_path, 'w') as f:
    #     f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
    eval_metrics = env_state.info['eval_metrics']
    metrics = {}
    for fn in [np.mean, np.std]:
        suffix = '_std' if fn == np.std else ''
        metrics.update(
            {
                f'eval/episode_{name}{suffix}': (
                    fn(value)
                )
                for name, value in eval_metrics.episode_metrics.items()
            }
        )
    return metrics
