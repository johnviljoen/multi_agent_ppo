# source: https://github.com/google/brax/blob/main/brax/training/agents/ppo/losses.py

import jax
import jax.numpy as jnp

def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
        truncation: A float32 tensor of shape [T, B] with truncation signal.
        termination: A float32 tensor of shape [T, B] with termination signal.
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
        lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount *
                    (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)

def compute_ppo_loss(
        policy,  # Policy network (Equinox module)
        value,   # Value network (Equinox module)
        rms_params,  # Running mean std parameters
        transition,  # Transition data
        rng: jnp.ndarray,
        entropy_cost: float = 1e-4,
        discounting: float = 0.99,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.2,
        normalize_advantage: bool = True
    ):
    """
    Computes PPO loss for continuous action spaces.

    Args:
        policy: Policy network (Equinox module).
        value: Value network (Equinox module).
        rms_params: Running mean std parameters (for observation normalization).
        transition: Transition data with leading dimensions [Time, Batch]. Required fields are:
            - 'observation': Observations.
            - 'next_observation': Next observations.
            - 'reward': Rewards.
            - 'discount': Discounts.
            - 'action': Actions taken.
            - 'log_prob': Log probabilities under the behavior policy.
            - 'extras': Dictionary containing 'state_extras' with 'truncation' flag.
        rng: Random key.
        entropy_cost: Entropy cost coefficient.
        discounting: Discount factor.
        reward_scaling: Reward scaling factor.
        gae_lambda: Generalized Advantage Estimation lambda.
        clipping_epsilon: PPO clipping epsilon.
        normalize_advantage: Whether to normalize advantage estimates.

    Returns:
        A tuple of (total_loss, metrics).
    """
    # Swap axes to bring time dimension first
    transition = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transition)

    # Normalize observations using running mean and std
    def normalize_observation(rms_params, observation):
        return (observation - rms_params['mean']) / (jnp.sqrt(rms_params['var'] + 1e-8))
    
    observations = normalize_observation(rms_params, transition['observation'])
    next_observations = normalize_observation(rms_params, transition['next_observation'])

    # Compute policy mean and baseline values
    policy_mean = jax.vmap(jax.vmap(policy))(observations)
    baseline = jax.vmap(jax.vmap(value))(observations)

    # Compute bootstrap value for the last next_observation
    bootstrap_value = jax.vmap(value)(next_observations[-1])

    # Compute scaled rewards and flags
    rewards = transition['reward'] * reward_scaling
    truncation = transition['extras']['state_extras']['truncation']
    termination = (1 - transition['discount']) * (1 - truncation)

    # Create the policy distribution
    std = policy.std  # Assuming fixed std
    dist = distrax.MultivariateNormalDiag(loc=policy_mean, scale_diag=std)

    # Compute log probabilities under the current policy
    target_action_log_probs = dist.log_prob(transition['action'])

    # Behaviour action log probabilities (from data)
    behaviour_action_log_probs = transition['log_prob']

    # Compute Generalized Advantage Estimation (GAE)
    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting
    )

    # Normalize advantages if required
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Importance sampling ratio
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    # Surrogate losses
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages

    # Policy loss
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = 0.5 * jnp.mean(v_error ** 2)

    # Entropy loss
    entropy = jnp.mean(dist.entropy())
    entropy_loss = -entropy_cost * entropy

    # Total loss
    total_loss = policy_loss + v_loss + entropy_loss

    # Metrics
    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'entropy_loss': entropy_loss
    }

    return total_loss, metrics


# def compute_ppo_loss(
#         policy, # policy function - equinox datastructure
#         value,  # value function - equinox datastructure
#         rms_params, # running mean std parameters
#         transition,
#         rng: jnp.ndarray,
#         entropy_cost: float = 1e-4,
#         discounting: float = 0.9,
#         reward_scaling: float = 1.0,
#         gae_lambda: float = 0.95,
#         clipping_epsilon: float = 0.3,
#         normalize_advantage: bool = True
#     ):
#     """Computes PPO loss.

#     Args:
#         params: Network parameters,
#         normalizer_params: Parameters of the normalizer.
#         data: Transition that with leading dimension [B, T]. extra fields required
#         are ['state_extras']['truncation'] ['policy_extras']['raw_action']
#             ['policy_extras']['log_prob']
#         rng: Random key
#         ppo_network: PPO networks.
#         entropy_cost: entropy cost.
#         discounting: discounting,
#         reward_scaling: reward multiplier.
#         gae_lambda: General advantage estimation lambda.
#         clipping_epsilon: Policy loss clipping epsilon
#         normalize_advantage: whether to normalize advantage estimate

#     Returns:
#         A tuple (loss, metrics)
#     """
#     parametric_action_distribution = ppo_network.parametric_action_distribution
#     policy_apply = ppo_network.policy_network.apply
#     value_apply = ppo_network.value_network.apply

#     # Put the time dimension first.
#     data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
#     policy_logits = policy_apply(normalizer_params, params.policy,
#                                 data.observation)

#     baseline = value_apply(normalizer_params, params.value, data.observation)

#     bootstrap_value = value_apply(normalizer_params, params.value,
#                                     data.next_observation[-1])

#     rewards = data.reward * reward_scaling
#     truncation = data.extras['state_extras']['truncation']
#     termination = (1 - data.discount) * (1 - truncation)

#     target_action_log_probs = parametric_action_distribution.log_prob(
#         policy_logits, data.extras['policy_extras']['raw_action'])
#     behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

#     vs, advantages = compute_gae(
#         truncation=truncation,
#         termination=termination,
#         rewards=rewards,
#         values=baseline,
#         bootstrap_value=bootstrap_value,
#         lambda_=gae_lambda,
#         discount=discounting)
#     if normalize_advantage:
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#     rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

#     surrogate_loss1 = rho_s * advantages
#     surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon,
#                                 1 + clipping_epsilon) * advantages

#     policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

#     # Value function loss
#     v_error = vs - baseline
#     v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

#     # Entropy reward
#     entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
#     entropy_loss = entropy_cost * -entropy

#     total_loss = policy_loss + v_loss + entropy_loss
#     return total_loss, {
#         'total_loss': total_loss,
#         'policy_loss': policy_loss,
#         'v_loss': v_loss,
#         'entropy_loss': entropy_loss
#     }

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    import equinox as eqx
    from typing import List, Callable
    import distrax

    from models import StochasticActor, ValueNetwork

    # Define observation and action dimensions
    obs_dim = 4
    act_dim = 2
    hidden_sizes = [64, 64]
    
    # Random keys
    key = jr.PRNGKey(0)
    key_policy, key_value, key_sample = jr.split(key, 3)

    # Define policy and value networks
    policy_layer_sizes = [obs_dim] + hidden_sizes + [act_dim]
    policy = StochasticActor(policy_layer_sizes, key_policy)

    value_layer_sizes = [obs_dim] + hidden_sizes + [1]
    value = ValueNetwork(value_layer_sizes, key_value)

    # Initialize running mean std parameters
    rms_params = {
        'mean': jnp.zeros(obs_dim),
        'var': jnp.ones(obs_dim)
    }

    # Create fake transition data
    batch_size = 32
    time_steps = 5
    rng = key_sample
    obs = jr.normal(rng, (time_steps, batch_size, obs_dim))
    next_obs = jr.normal(rng, (time_steps, batch_size, obs_dim))
    rewards = jr.normal(rng, (time_steps, batch_size))
    discounts = jnp.ones((time_steps, batch_size)) * 0.99
    actions = jr.normal(rng, (time_steps, batch_size, act_dim))
    truncation = jnp.zeros((time_steps, batch_size))
    termination = jnp.zeros((time_steps, batch_size))
    log_probs = jr.normal(rng, (time_steps, batch_size))

    # Compute log_probs under policy
    def compute_log_probs(policy, obs, actions):
        def single_step(o, a):
            mean = policy(o)
            dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=policy.std)
            log_prob = dist.log_prob(a)
            return log_prob
        log_probs = jax.vmap(jax.vmap(single_step))(obs, actions)
        return log_probs

    # Compute log_probs
    log_probs = compute_log_probs(policy, obs, actions)

    # Build transition data
    transition = {
        'observation': obs,
        'next_observation': next_obs,
        'reward': rewards,
        'discount': discounts,
        'action': actions,
        'log_prob': log_probs,
        'extras': {
            'state_extras': {
                'truncation': truncation
            }
        }
    }

    # Compute loss
    total_loss, metrics = compute_ppo_loss(
        policy=policy,
        value=value,
        rms_params=rms_params,
        transition=transition,
        rng=rng,
        entropy_cost=1e-4,
        discounting=0.99,
        reward_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        normalize_advantage=True
    )

    print("Total loss:", total_loss)
    print("Metrics:", metrics)

    # Test gradient computation
    def loss_fn(policy_params, value_params):
        # Reconstruct policy and value with new params
        policy_updated = eqx.tree_at(lambda tree: tree, policy, policy_params)
        value_updated = eqx.tree_at(lambda tree: tree, value, value_params)
        loss, _ = compute_ppo_loss(
            policy=policy_updated,
            value=value_updated,
            rms_params=rms_params,
            transition=transition,
            rng=rng,
            entropy_cost=1e-4,
            discounting=0.99,
            reward_scaling=1.0,
            gae_lambda=0.95,
            clipping_epsilon=0.2,
            normalize_advantage=True
        )
        return loss

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    policy_params = eqx.filter(policy, eqx.is_inexact_array)
    value_params = eqx.filter(value, eqx.is_inexact_array)
    (loss_value), (grads_policy, grads_value) = grad_fn(policy_params, value_params)

    print("Loss value:", loss_value)
    print("Gradients computed successfully.")
