import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import gymnasium as gym

import buffers
# Environment setup
def make_env(env_id, num_envs):
    def thunk():
        env = gym.vector.make(env_id, num_envs=num_envs)
        return env
    return thunk

# Assuming args are provided as a namespace or dictionary with appropriate keys
args = {
    'env_id': 'CartPole-v1',  # Example environment
    'seed': 42,
    'num_envs': 4
}

# Initialize environment
envs = make_env(args['env_id'], args['num_envs'])()
obs, info = envs.reset(seed=args['seed'])

# Initialize the episode statistics buffer
init_stats_buffer, update_stats_buffer = buffers.episode_statistics(args['num_envs'])
episode_stats = init_stats_buffer()

def step_env_wrapped(episode_stats, envs, action):
    # Convert action from JAX array to NumPy array for compatibility with Gymnasium
    action_np = np.array(action)
    
    # Step through the environment
    next_obs, reward, terminated, truncated, info = envs.step(action_np)
    
    # Convert next_obs and reward back to JAX arrays for further processing
    reward = jnp.array(reward)
    terminated = jnp.array(terminated)
    truncated = jnp.array(truncated)
    
    # Update the episode statistics buffer
    episode_stats = update_stats_buffer(episode_stats, reward, terminated, truncated)
    
    return episode_stats, (next_obs, reward, terminated, truncated, info)

assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


# Example of using the step_env_wrapped function
actions = jnp.array([0, 1, 0, 1])  # Example actions for 4 environments
episode_stats, (next_obs, reward, terminated, truncated, info) = step_env_wrapped(episode_stats, envs, actions)

print('fin')