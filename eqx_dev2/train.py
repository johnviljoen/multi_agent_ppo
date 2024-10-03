import functools
from typing import Callable, Optional, Tuple, Union

from data_structures import ReplayBuffer, RunningMeanStd, Transition

import losses as ppo_losses
import models as ppo_networks

import numpy as np
import jax
import jax.numpy as jnp
from brax.v1 import envs as envs_v1
import jax.random as jr

import optax

from brax import envs
from time import time
from dataclasses import dataclass
from absl import logging

from models import PPOStochasticActor, ValueNetwork

@dataclass
class Config:
    env: envs.Env
    num_timesteps: int
    episode_length: int
    action_repeat: int = 1
    num_envs: int = 1
    max_devices_per_host: Optional[int] = None
    num_eval_envs: int = 128
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    discounting: float = 0.9
    seed: int = 0
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_evals: int = 1
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    deterministic_eval: bool = False
    progress_fn: Callable = lambda *args: None
    normalize_advantage: bool = True
    eval_env: Optional[envs.Env] = None
    policy_params_fn: Callable[..., None] = lambda *args: None
    randomization_fn = None
    restore_checkpoint_path: Optional[str] = None


def train(c): # c is the config

    assert c.batch_size * c.num_minibatches % c.num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if c.max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, c.max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d', jax.device_count(), process_count,
        process_id, local_device_count, local_devices_to_use)
    device_count = local_devices_to_use * process_count


    # The number of environment steps executed for every training step.
    env_step_per_training_step = (c.batch_size * c.unroll_length * c.num_minibatches * c.action_repeat)
    num_evals_after_init = max(c.num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        c.num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(c.num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(c.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    assert c.num_envs % device_count == 0

    v_randomization_fn = None
    if c.randomization_fn is not None:
        randomization_batch_size = c.num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            c.randomization_fn, rng=randomization_rng
        )

    if isinstance(c.environment, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = wrap_for_training(
        c.environment,
        episode_length=c.episode_length,
        action_repeat=c.action_repeat,
        randomization_fn=v_randomization_fn,
    )

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, c.num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    normalize = lambda x, y: x

    actor_network = PPOStochasticActor([])
    value_network = ValueNetwork([])

    optimizer = optax.adam(learning_rate=c.learning_rate)

    loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            actor_network=actor_network,
            value_network=value_network,
            rms = RMS,
            entropy_cost=c.entropy_cost,
            discounting=c.discounting,
            reward_scaling=c.reward_scaling,
            gae_lambda=c.gae_lambda,
            clipping_epsilon=c.clipping_epsilon,
            normalize_advantage=c.normalize_advantage)

    gradient_update_fn = gradients.gradient_update_fn(
            loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    def minibatch_step(
            carry, data: Transition,
            normalizer_params: RunningMeanStd):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics

    def sgd_step(carry, unused_t, data: Transition,
                             normalizer_params: RunningMeanStd):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (c.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params),
                (optimizer_state, params, key_grad),
                shuffled_data,
                length=c.num_minibatches)
        return (optimizer_state, params, key), metrics

    def training_step(
            carry: Tuple[TrainingState, envs.State, jr.PRNGKey],
            unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
                (training_state.normalizer_params, training_state.params.policy))

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    unroll_length,
                    extra_fields=('truncation',))
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
                f, (state, key_generate_unroll), (),
                length=batch_size * num_minibatches // num_envs)
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                                                    data)
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME)

        (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(
                        sgd_step, data=data, normalizer_params=normalizer_params),
                (training_state.optimizer_state, training_state.params, key_sgd), (),
                length=num_updates_per_batch)

        new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_step_per_training_step)
        return (new_training_state, state, new_key), metrics

    def training_epoch(training_state: TrainingState, state: envs.State,
                                         key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step, (training_state, state, key), (),
                length=num_training_steps_per_epoch)
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
            training_state: TrainingState, env_state: envs.State,
            key: jr.PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (num_training_steps_per_epoch *
                     env_step_per_training_step *
                     max(num_resets_per_eval, 1)) / epoch_training_time
        metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                **{f'training/{name}': value for name, value in metrics.items()}
        }
        return training_state, env_state, metrics    # pytype: disable=bad-return-type    # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
    )

    training_state = TrainingState(    # pytype: disable=wrong-arg-types    # jax-ndarray
            optimizer_state=optimizer.init(init_params),    # pytype: disable=wrong-arg-types    # numpy-scalars
            params=init_params,
            normalizer_params=running_statistics.init_state(
                    specs.Array(env_state.obs.shape[-1:], jnp.dtype('float32'))),
            env_steps=0)

    if num_timesteps == 0:
        return (
                make_policy,
                (training_state.normalizer_params, training_state.params),
                {},
        )

    if (
            restore_checkpoint_path is not None
            and epath.Path(restore_checkpoint_path).exists()
    ):
        logging.info('restoring from checkpoint %s', restore_checkpoint_path)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        target = training_state.normalizer_params, init_params
        (normalizer_params, init_params) = orbax_checkpointer.restore(
                restore_checkpoint_path, item=target
        )
        training_state = training_state.replace(
                normalizer_params=normalizer_params, params=init_params
        )

    training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use])

    if not eval_env:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
        )
    eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
    )

    evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key)

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
                _unpmap(
                        (training_state.normalizer_params, training_state.params.policy)),
                training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                    training_epoch_with_timing(training_state, env_state, epoch_keys)
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                    lambda x, s: jax.random.split(x[0], s),
                    in_axes=(0, None))(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                    _unpmap(
                            (training_state.normalizer_params, training_state.params.policy)),
                    training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics)
            params = _unpmap(
                    (training_state.normalizer_params, training_state.params)
            )
            policy_params_fn(current_step, make_policy, params)

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
            (training_state.normalizer_params, training_state.params.policy))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)

if __name__ == "__main__":

    # provided good hyperparameters for the ant environment
    config = Config(
        env="ant", 
        num_timesteps=50_000_000,
        num_evals=10, 
        reward_scaling=10, 
        episode_length=1000, 
        normalize_observations=True, 
        action_repeat=1, 
        unroll_length=5, 
        num_minibatches=32, 
        num_updates_per_batch=4, 
        discounting=0.97, 
        learning_rate=3e-4, 
        entropy_cost=1e-2, 
        num_envs=4096, 
        batch_size=2048, 
        seed=1
    )

    train(config)