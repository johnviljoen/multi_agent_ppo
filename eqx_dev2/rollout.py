import jax
import jax.random as jr

from eqx_ops import filter_scan

def generate_rollout_v(key, env_state, training_state, env_v, unroll_length):
    
    def step_fn(carry, _):
        state, key = carry
        obs = state.obs # this is one obs across the batch
        batch_size = obs.shape[0]
        _key, key = jr.split(key)
        obs_normalized = training_state.obs_rms.normalize(obs)
        action, raw_action = jax.vmap(training_state.model.actor_network)(jr.split(_key, batch_size), obs_normalized)
        log_prob = jax.vmap(training_state.model.actor_network.log_prob)(obs_normalized, action)
        value = jax.vmap(training_state.model.value_network)(obs_normalized)
        data = {
            'obs': obs,
            'action': action,
            'reward': next_state.reward,
            'discount': 1 - next_state.done, # how we decide if truncation is terminal or not
            'truncation': next_state.info['truncation'],
            'log_prob': log_prob,
            'value': value,
            'next_obs': next_state.obs,
            'raw_action': raw_action, # action without gaussian applied to it
        }
        return (next_state, key), data

    (final_state, _), data_seq = filter_scan(
        step_fn, (env_state, key), None, length=unroll_length
    )

    return data_seq, final_state