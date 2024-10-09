import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import distributions

from typing import Callable, List
from dataclasses import dataclass

class PPOStochasticActor(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable
    std: jnp.ndarray  # Standard deviation
    layer_sizes: List[int]
    action_distribution: dataclass = eqx.field(static=True)

    def __init__(self, key, layer_sizes, activation=jax.nn.relu, action_distribution=distributions.NormalTanhDistribution()):
        keys = jr.split(key, num=len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=keys[i])
            for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]
        self.activation = activation
        self.std = jnp.ones(layer_sizes[-1])  # Fixed std; can be made trainable
        self.layer_sizes = layer_sizes
        self.action_distribution = action_distribution

    def __call__(self, key, x, normalize_fn):
        mean = self.forward_deterministic(x, normalize_fn)
        action = self.action_distribution.sample(key, mean, self.std)
        return action
    
    def forward_deterministic(self, x, normalize_fn):
        x = normalize_fn(x)
        for linear in self.layers[:-1]:
            x = self.activation(linear(x))
        mean = self.layers[-1](x)  # Output mean
        return mean

    def log_prob(self, x, action, normalize_fn):
        return self.action_distribution.log_prob(loc=self.forward_deterministic(x, normalize_fn), scale=self.std, actions=action)

    def entropy(self, key, x, normalize_fn):
        return self.action_distribution.entropy(key=key, loc=self.forward_deterministic(x, normalize_fn), scale=self.std)
    

class ValueNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, key, layer_sizes, activation=jax.nn.relu):
        keys = jr.split(key, num=len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(in_f, out_f, key=k)
            for in_f, out_f, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.squeeze(-1)  # Output scalar value


class SACDoubleCritic(eqx.Module):
    critic_1_layers: list[eqx.nn.Linear]
    critic_2_layers: list[eqx.nn.Linear]
    critic_1_bias: list[jax.Array]
    critic_2_bias: list[jax.Array]
    activation: callable

    def __init__(self, key, layer_sizes, activation=jax.nn.relu):

        keys = jr.split(key, num=len(layer_sizes)*2) # fully consume the mlp_key

        self.critic_1_layers = [eqx.nn.Linear(in_features, out_features, key=keys[i]) 
                       for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
        self.critic_2_layers = [eqx.nn.Linear(in_features, out_features, key=keys[i+len(layer_sizes)]) 
                       for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
        
        self.critic_1_bias = [jnp.zeros(out_features) for out_features in layer_sizes[1:]]
        self.critic_2_bias = [jnp.zeros(out_features) for out_features in layer_sizes[1:]]

        self.activation = activation

    def __call__(self, x):

        def mlp_forward(v, layers, bias):
            # pass through critic_1
            for linear, b in zip(layers[:-1], bias[:-1]):
                v = linear(v)
                v += b
                v = self.activation(v)
            v = layers[-1](v) + bias[-1] # dont apply act to final output
            return v
        
        v1 = mlp_forward(x, self.critic_1_layers, self.critic_1_bias)
        v2 = mlp_forward(x, self.critic_2_layers, self.critic_2_bias)

        v = jnp.min(jnp.hstack([v1,v2]))

        return v


if __name__ == "__main__":

    from functools import partial
    from jax import jit, vmap

    global_key = jr.PRNGKey(0)
    global_key, actor_key, critic_key, dataset_key, sample_key = jr.split(global_key, num=5)

    batch_size = 30
    num_actions = 2
    num_observations = 2

    actor = PPOStochasticActor(actor_key, [num_observations,10,10,num_actions])
    value = ValueNetwork(critic_key, [num_actions,10,10,1])

    x = jr.normal(dataset_key, shape=[batch_size,2])
    normalize_fn = lambda x: x

    actor_call = partial(actor, normalize_fn=normalize_fn)
    actor_log_prob = partial(actor.log_prob, normalize_fn=normalize_fn)
    actor_entropy = partial(actor.entropy, normalize_fn=normalize_fn)

    actor_call_jv = jit(vmap(actor_call))
    actor_log_prob_jv = jit(vmap(actor_log_prob))
    actor_entropy_jv = jit(vmap(actor_entropy))

    u = actor_call_jv(jr.split(sample_key, num=batch_size), x)
    lp = actor_log_prob_jv(x, u)
    e = actor_entropy_jv(jr.split(sample_key, num=batch_size), x)

    print('fin')

