import functools
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import dataclasses

from brax import envs

# My code
from models import PPOStochasticActor, PPOValueNetwork
from running_mean_std import RunningMeanStd
from losses import compute_ppo_loss
from plotting import rollout_and_render

# Define your AgentModel to hold both the actor and critic networks
class AgentModel(eqx.Module):
    actor_network: PPOStochasticActor
    value_network: PPOValueNetwork

# Define the TrainingState as an Equinox module
class TrainingState(eqx.Module):
    optimizer_state: optax.OptState
    model: AgentModel
    observation_rms: RunningMeanStd
    env_steps: int

def train(
        env,  # The environment, assumed to be JAX-compatible
        num_timesteps: int,
        episode_length: int,
        num_envs: int = 1,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        seed: int = 0,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
    ):

    key = jr.PRNGKey(seed)

    obs_size = env.observation_size
    act_size = env.action_size

    # Wrap the environment for training
    env_v = envs.training.wrap(
        env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=None
    )

    # Initialize models
    key1, key2, key = jr.split(key, 3)
    actor_network = PPOStochasticActor(key1, layer_sizes=[obs_size, 64, 64, act_size])
    value_network = PPOValueNetwork(key2, layer_sizes=[obs_size, 64, 64, 1])
    model = AgentModel(actor_network=actor_network, value_network=value_network)

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Initialize observation RMS
    observation_rms = RunningMeanStd(
        mean=jnp.zeros(obs_size), var=jnp.ones(obs_size), count=1e-4
    )

    # Initialize TrainingState
    training_state = TrainingState(
        optimizer_state=optimizer_state,
        model=model,
        observation_rms=observation_rms,
        env_steps=0
    )

    # Functions for training
    def minibatch_step(carry, minibatch_data):
        optimizer_state, params, key = carry
        key, subkey = jr.split(key)

        def loss_fn(model):
            loss, metrics = compute_ppo_loss(
                actor_network=model.actor_network,
                value_network=model.value_network,
                observation_rms=training_state.observation_rms,
                data=minibatch_data,
                rng=subkey,
                entropy_cost=entropy_cost,
                discounting=discounting,
                reward_scaling=reward_scaling,
                gae_lambda=gae_lambda,
                clipping_epsilon=clipping_epsilon,
                normalize_advantage=normalize_advantage
            )
            return loss, metrics

        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_optimizer_state = optimizer.update(
            grads, optimizer_state, params
        )
        new_params = eqx.apply_updates(params, updates)

        return (new_optimizer_state, new_params, key), metrics

    def sgd_step(carry, _, data, observation_rms):
        optimizer_state, params, key = carry
        key, subkey = jr.split(key)
        permutation = jr.permutation(subkey, batch_size)
        shuffled_data = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:])[permutation], data)

        minibatch_size = batch_size // num_minibatches
        def get_minibatch(i):
            return jax.tree_util.tree_map(lambda x: x[i * minibatch_size:(i + 1) * minibatch_size], shuffled_data)

        init_carry = (optimizer_state, params, key)
        (optimizer_state, params, key), metrics = jax.lax.scan(
            minibatch_step,
            init_carry,
            xs=[get_minibatch(i) for i in range(num_minibatches)]
        )

        return (optimizer_state, params, key), metrics

    def training_step(carry, _):
        training_state, env_state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        # Generate data
        data, env_state = generate_unroll(key_generate_unroll, env_state, training_state)
        # Update observation RMS
        new_observation_rms = training_state.observation_rms.update(data['obs'])
        training_state = dataclasses.replace(
            training_state,
            observation_rms=new_observation_rms
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                    sgd_step, data=data, observation_rms=observation_rms),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=num_updates_per_batch)

        return (new_training_state, env_state, new_key), metrics

    def training_epoch(training_state, env_state, key):
        (training_state, env_state, key), metrics = jax.lax.scan(
            training_step,
            (training_state, env_state, key),
            xs=None,
            length=1  # Adjust as needed
        )
        return training_state, env_state, metrics

    # Generate unroll function
    def generate_unroll(key, env_state, training_state):
        def step_fn(carry, _):
            state, key = carry
            obs = state.obs  # Assuming env_state has 'obs' attribute
            key, subkey = jr.split(key)
            # Normalize observation
            obs_normalized = training_state.observation_rms.normalize(obs)
            # Get action from policy
            actions, raw_actions = jax.vmap(training_state.model.actor_network)(
                jr.split(subkey, batch_size), obs_normalized)
            # Step environment
            next_state = env_v.step(state, actions)
            # Compute log_prob and value
            log_probs = jax.vmap(training_state.model.actor_network.log_prob)(obs_normalized, actions)
            values = jax.vmap(training_state.model.value_network)(obs_normalized)
            # Collect data
            data = {
                'obs': obs,
                'actions': actions,
                'rewards': next_state.reward,
                'discount': 1 - next_state.done,  # For termination
                'truncation': next_state.info['truncation'],
                'termination': next_state.info['termination'],
                'log_probs': log_probs,
                'values': values,
                'next_obs': next_state.obs,
                'raw_actions': raw_actions,
            }
            return (next_state, key), data

        (final_state, _), data_seq = jax.lax.scan(
            step_fn, (env_state, key), xs=None, length=unroll_length
        )

        # Stack data across the unroll length
        data = jax.tree_util.tree_map(lambda x: jnp.stack(x), data_seq)

        # Compute bootstrap value (value at the last next_obs)
        obs_normalized = training_state.observation_rms.normalize(data['next_obs'][-1])
        bootstrap_value = training_state.model.value_network(obs_normalized)
        data['bootstrap_value'] = bootstrap_value

        return data, final_state

    # Initialize environment state
    key, env_key = jr.split(key)
    env_state = env_v.reset(jr.split(env_key, batch_size))
    key, training_key = jr.split(key)

    total_steps = 0
    while total_steps < num_timesteps:
        print(f"Total Steps: {total_steps}")

        training_state, env_state, metrics = training_epoch(training_state, env_state, training_key)
        key, training_key = jr.split(key)

        # Print metrics if needed
        # print(metrics)

        total_steps += unroll_length * batch_size

    return training_state

