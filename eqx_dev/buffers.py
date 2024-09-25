import jax
import jax.numpy as jnp
import jax.random as jr

def replay(num_envs, horizon, obs_size, act_size):
    # Initialize the buffer
    def init():
        buffer = {
            "obs": jnp.zeros((horizon, num_envs, obs_size)),
            "actions": jnp.zeros((horizon, num_envs, act_size)),
            "logprobs": jnp.zeros((horizon, num_envs)),
            "dones": jnp.zeros((horizon, num_envs), dtype=bool),
            "values": jnp.zeros((horizon, num_envs)),
            "advantages": jnp.zeros((horizon, num_envs)),
            "returns": jnp.zeros((horizon, num_envs)),
            "rewards": jnp.zeros((horizon, num_envs)),
            "size": 0,
            "ptr": 0
        }
        return buffer

    # Store an experience in the buffer
    def store(buffer, obs, actions, logprobs, dones, values, advantages, returns, rewards):
        idx = buffer['ptr']
        buffer = {
            "obs": buffer['obs'].at[idx].set(obs),
            "actions": buffer['actions'].at[idx].set(actions),
            "logprobs": buffer['logprobs'].at[idx].set(logprobs),
            "dones": buffer['dones'].at[idx].set(dones),
            "values": buffer['values'].at[idx].set(values),
            "advantages": buffer['advantages'].at[idx].set(advantages),
            "returns": buffer['returns'].at[idx].set(returns),
            "rewards": buffer['rewards'].at[idx].set(rewards),
            "size": jnp.minimum(buffer['size'] + 1, horizon),
            "ptr": (idx + 1) % horizon
        }
        return buffer

    # Sample a batch from the buffer
    def sample_batch(buffer, batch_size, rng):
        max_size = buffer['size']
        indices = jr.randint(rng, (batch_size,), 0, max_size)
        return (
            buffer['obs'][indices],
            buffer['actions'][indices],
            buffer['logprobs'][indices],
            buffer['dones'][indices],
            buffer['values'][indices],
            buffer['advantages'][indices],
            buffer['returns'][indices],
            buffer['rewards'][indices]
        )

    return init, store, sample_batch

def episode_statistics(num_envs):
    # Initialize the buffer
    def init():
        buffer = {
            "episode_returns": jnp.zeros(num_envs, dtype=jnp.float32),
            "episode_lengths": jnp.zeros(num_envs, dtype=jnp.int32),
            "returned_episode_returns": jnp.zeros(num_envs, dtype=jnp.float32),
            "returned_episode_lengths": jnp.zeros(num_envs, dtype=jnp.int32),
        }
        return buffer

    # Update the episode statistics
    def update(buffer, rewards, dones, truncated):
        new_episode_returns = buffer["episode_returns"] + rewards
        new_episode_lengths = buffer["episode_lengths"] + 1

        buffer = {
            "episode_returns": new_episode_returns * (1 - dones) * (1 - truncated),
            "episode_lengths": new_episode_lengths * (1 - dones) * (1 - truncated),
            "returned_episode_returns": jnp.where(
                dones + truncated, new_episode_returns, buffer["returned_episode_returns"]
            ),
            "returned_episode_lengths": jnp.where(
                dones + truncated, new_episode_lengths, buffer["returned_episode_lengths"]
            ),
        }
        return buffer

    return init, update

def running_mean_std():

    # Initialize the running mean and std
    def init(mean=None, var=None, epsilon=1e-4, shape=()):
        if mean is None:
            mean = jnp.zeros(shape)
        if var is None:
            var = jnp.ones(shape)
        count = epsilon
        rms = {
            'mean': mean,
            'var': var,
            'count': count
        }
        return rms

    # Update the running mean and std with a new batch of data
    def update(rms, arr):
        arr = jax.lax.stop_gradient(arr)
        batch_mean = jnp.mean(arr, axis=0)
        batch_var = jnp.var(arr, axis=0)
        batch_count = arr.shape[0]

        # Update from batch moments
        def update_from_moments(rms, batch_mean, batch_var, batch_count):
            delta = batch_mean - rms['mean']
            tot_count = rms['count'] + batch_count
            new_mean = rms['mean'] + delta * batch_count / tot_count
            m_a = rms['var'] * rms['count']
            m_b = batch_var * batch_count
            m_2 = (
                m_a
                + m_b
                + delta ** 2 * rms['count'] * batch_count / tot_count
            )
            new_var = m_2 / tot_count
            new_count = tot_count
            new_rms = {
                'mean': new_mean,
                'var': new_var,
                'count': new_count
            }
            return new_rms

        return update_from_moments(rms, batch_mean, batch_var, batch_count)

    # Normalize an array using the running mean and std
    def normalize(rms, arr):
        return (arr - rms['mean']) / jnp.sqrt(rms['var'] + 1e-5)

    # Denormalize an array using the running mean and std
    def denormalize(rms, arr):
        return arr * jnp.sqrt(rms['var'] + 1e-5) + rms['mean']

    return init, update, normalize, denormalize

if __name__ == "__main__":

    # Usage
    num_envs, horizon, obs_size, act_size = 4, 10, 3, 2
    buffer_init, buffer_store, buffer_sample = replay(num_envs, horizon, obs_size, act_size)

    # Initialize buffer
    buffer_state = buffer_init()

    # Example data for storing
    obs = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    actions = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    logprobs = jnp.array([0.1, 0.2, 0.3, 0.4])
    dones = jnp.array([False, True, False, True])
    values = jnp.array([1.0, 2.0, 3.0, 4.0])
    advantages = jnp.array([0.5, 0.6, 0.7, 0.8])
    returns = jnp.array([1.5, 2.5, 3.5, 4.5])
    rewards = jnp.array([1.1, 2.1, 3.1, 4.1])

    # Store an experience
    buffer_state = buffer_store(buffer_state, obs, actions, logprobs, dones, values, advantages, returns, rewards)

    # Sample a batch
    rng = jr.PRNGKey(0)
    sampled_obs, sampled_actions, sampled_logprobs, sampled_dones, sampled_values, sampled_advantages, sampled_returns, sampled_rewards = buffer_sample(buffer_state, 2, rng)

    print('fin')