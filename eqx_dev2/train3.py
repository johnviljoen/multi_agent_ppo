from functools import partial
import numpy as np
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
from rollout import generate_rollout_v

class AgentModel(eqx.Module):
    actor_network: PPOStochasticActor
    value_network: PPOValueNetwork

class TrainingState(eqx.Module):
    opt_state: optax.OptState
    model: AgentModel
    obs_rms: RunningMeanStd
    env_steps: int

def train(
        env,  # The environment, assumed to be JAX-compatible
        num_timesteps: int,
        episode_length: int,
        num_envs: int = 128,
        action_repeat: int = 1,
        num_evals: int = 1,
        num_resets_per_eval: int = 0,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        seed: int = 0,
        unroll_length: int = 10,
        minibatch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
    ):

    key = jr.PRNGKey(seed)
    _key, key = jr.split(key)

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (minibatch_size * unroll_length * num_minibatches * action_repeat)
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
            num_timesteps
            / (
                    num_evals_after_init
                    * env_step_per_training_step
                    * max(num_resets_per_eval, 1)
            )
    ).astype(int)

    env_v = envs.training.wrap(env, episode_length=episode_length)
    actor_network = PPOStochasticActor(_key, layer_sizes=[env.observation_size, 64, 64, env.action_size]); _key, key = jr.split(key)
    value_network = PPOValueNetwork(_key, layer_sizes=[env.observation_size, 64, 64, 1]); _key, key = jr.split(key)
    model = AgentModel(actor_network=actor_network, value_network=value_network)
    opt = optax.adam(learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    obs_rms = RunningMeanStd(mean=jnp.zeros(env.observation_size), var=jnp.ones(env.observation_size), count=1e-4)
    training_state = TrainingState(opt_state, model, obs_rms, env_steps=0)
    generate_unroll_jv = eqx.filter_jit(partial(generate_rollout_v, env_v=env_v, unroll_length=unroll_length))

    def minibatch_step(carry, minibatch_data):
        training_state, key = carry
        key_loss, new_key = jr.split(key)
        def loss_fn(model):
            loss, metrics = compute_ppo_loss(
                actor_network=model.actor_network,
                value_network=model.value_network,
                observation_rms=training_state.observation_rms,
                data=minibatch_data,
                rng=key_loss,
                entropy_cost=entropy_cost,
                discounting=discounting,
                reward_scaling=reward_scaling,
                gae_lambda=gae_lambda,
                clipping_epsilon=clipping_epsilon,
                normalize_advantage=normalize_advantage
            )
            return loss, metrics
        
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.model)
        updates, new_optimizer_state = opt.update(
            grads, training_state.opt_state, training_state.model
        )
        new_model = eqx.apply_updates(training_state.model, updates)
        new_training_state = dataclasses.replace(
            training_state,
            optimizer_state=new_optimizer_state,
            model=new_model,
            env_steps=training_state.env_steps + unroll_length * num_envs
        )

        return (new_training_state, new_key), metrics
    
    def sgd_step(carry, _, data):
        training_state, key = carry
        key_perm, key_grad, new_key = jr.split(key)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (new_training_state, _), metrics = jax.lax.scan(
                minibatch_step,
                (training_state, key_grad),
                shuffled_data,
                length=num_minibatches)
        return (new_training_state, new_key), metrics
    
    def training_step(carry, _):
        training_state, env_state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        data, env_state = generate_unroll_jv(key_generate_unroll, env_state, training_state)
        new_observation_rms = training_state.observation_rms.update(data['obs'])
        training_state = dataclasses.replace(
            training_state,
            observation_rms=new_observation_rms
        )

        (new_training_state, _), metrics = jax.lax.scan(
            partial(
                    sgd_step, data=data),
            (training_state, key_sgd), (),
            length=num_updates_per_batch)
        
        return (new_training_state, new_key), metrics
    
    def training_epoch(key, training_state, env_state):
        (new_training_state, new_env_state, _), loss_metrics = jax.lax.scan(
            training_step, (training_state, env_state, key), (),
            length=num_training_steps_per_epoch)
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return new_training_state, new_env_state, loss_metrics

    env_reset_jv = jax.jit(env_v.reset)
    total_steps = 0
    env_state = env_reset_jv(jr.split(_key, num=minibatch_size)); _key, key = jr.split(key)

    for it in range(num_evals_after_init):
        
        training_state, env_state, training_metrics = training_epoch(_key, training_state, env_state); _key, key = jr.split(key)
        
        current_step = training_state.env_steps

        print(current_step)

if __name__ == "__main__":


    env_name = 'ant'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    backend = 'positional'  # @param ['generalized', 'positional', 'spring']

    env = envs.get_environment(env_name=env_name, backend=backend)

    # known to work parameters for ant
    training_state = train(
        env=env,
        num_timesteps=50_000_000,
        episode_length=1000,
        num_envs=int(4096*1),
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        seed=1,
        unroll_length=5,
        minibatch_size=2048,
        num_minibatches=32,
        num_updates_per_batch=4,
        reward_scaling=10.,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        normalize_advantage=True,
    )

    print('fin')
