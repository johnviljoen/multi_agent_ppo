import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
import dataclasses

from brax import envs

from models import PPOStochasticActor, PPOValueNetwork
from running_mean_std import RunningMeanStd
from losses import compute_ppo_loss

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

    # this provides truncation information, vmaps the env, and auto resets finished envs
    env = envs.training.wrap(
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
    
    @eqx.filter_jit
    def update_step(key, training_state, data, optimizer,
            entropy_cost=entropy_cost, 
            discounting=discounting, 
            reward_scaling=reward_scaling, 
            gae_lambda=gae_lambda, 
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage, 
            unroll_length=unroll_length, 
            num_envs=num_envs
        ):
        def loss_fn(model):
            total_loss, _ = compute_ppo_loss(
                actor_network=model.actor_network,
                value_network=model.value_network,
                observation_rms=training_state.observation_rms,
                data=data,
                rng=key,
                entropy_cost=entropy_cost,
                discounting=discounting,
                reward_scaling=reward_scaling,
                gae_lambda=gae_lambda,
                clipping_epsilon=clipping_epsilon,
                normalize_advantage=normalize_advantage
            )
            return total_loss

        grads = eqx.filter_grad(loss_fn)(training_state.model)
        updates, new_optimizer_state = optimizer.update(
            grads, training_state.optimizer_state, training_state.model
        )
        new_model = eqx.apply_updates(training_state.model, updates)
        new_training_state = dataclasses.replace(
            training_state,
            optimizer_state=new_optimizer_state,
            model=new_model,
            env_steps=training_state.env_steps + unroll_length * num_envs
        )

        return new_training_state

    def generate_unroll(key, env_state, training_state, env=env, unroll_length=unroll_length, batch_size=batch_size):

        def step_fn(carry, _):
            state, key = carry
            obs = state.obs  # Assuming env_state has 'observation' attribute
            _key, key = jr.split(key)
            # Normalize observation
            obs_normalized = training_state.observation_rms.normalize(obs)
            # Get action from policy
            action, raw_action = jax.vmap(training_state.model.actor_network)(jr.split(_key, batch_size), obs_normalized)
            # Step environment
            next_state = env.step(state, action)
            # Compute log_prob and value
            log_prob = jax.vmap(training_state.model.actor_network.log_prob)(obs_normalized, action)
            value = jax.vmap(training_state.model.value_network)(obs_normalized)
            # Collect data
            data = {
                'obs': obs,
                'action': action,
                'reward': next_state.reward,
                'discount': 1 - next_state.done, # how we decide if truncation is terminal or not
                'truncation': next_state.info['truncation'],
                'log_prob': log_prob,
                'value': value,
                'next_obs': next_state.obs,  # Assuming next_state has 'observation'
                'raw_action': raw_action # action without gaussian applied to it
            }
            return (next_state, key), data

        (final_state, _), data_seq = jax.lax.scan(
            step_fn, (env_state, key), None, length=unroll_length
        )

        # # Stack data across the unroll length
        # data = {k: jnp.stack([d[k] for d in data_seq]) for k in data_seq[0]}

        return data_seq, final_state
    
    # Initialize environment state
    _key, key = jr.split(key)
    # env_state = jax.vmap(env.reset)(_key); _key, key = jr.split(key)

    env_reset_jv = jax.jit(env.reset)
    generate_unroll_jv = eqx.filter_jit(generate_unroll)# , in_axes=[0,0,None]))

    total_steps = 0

    while total_steps < num_timesteps:

        # Generate data
        env_state = env_reset_jv(jr.split(_key, num=batch_size)); _key, key = jr.split(key)
        data, env_state = generate_unroll_jv(_key, env_state, training_state); _key, key = jr.split(key)

        # Update observation_rms
        new_observation_rms = training_state.observation_rms.update(data['obs'])
        training_state = dataclasses.replace(
            training_state,
            observation_rms=new_observation_rms
        )

        # Update model parameters
        training_state = update_step(key, training_state, data, optimizer)

        total_steps += unroll_length * num_envs

    return training_state


if __name__ == "__main__":


    env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'positional'  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name,
                            backend=backend)

    # known to work parameters for ant
    training_state = train(
        env=env,
        num_timesteps=50_000_000,
        episode_length=1000,
        num_envs=4096,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        seed=1,
        unroll_length=5,
        batch_size=2048,
        num_minibatches=32,
        num_updates_per_batch=4,
        reward_scaling=10.,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        normalize_advantage=True,
    )

    print('fin')

