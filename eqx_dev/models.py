import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import distrax
from typing import Callable, List

class StochasticActor(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable
    std: jnp.ndarray  # Standard deviation, possibly trainable
    layer_sizes: List[int]

    def __init__(self, layer_sizes, mlp_key, activation=jax.nn.relu):
        mlp_keys = jr.split(mlp_key, num=len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=mlp_keys[i])
            for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]
        self.activation = activation
        self.std = jnp.ones(layer_sizes[-1])  # Fixed std; can be made trainable
        self.layer_sizes = layer_sizes

    def __call__(self, x):
        for linear in self.layers[:-1]:
            x = self.activation(linear(x))
        mean = self.layers[-1](x)  # Output mean
        return mean

    def sample_action(self, x, sample_key):
        mean = self(x)
        action = mean + self.std * jr.normal(sample_key, shape=mean.shape)
        return action

    def log_prob(self, x, action):
        mean = self(x)
        std = self.std
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        log_prob = dist.log_prob(action)
        return log_prob

    def entropy(self, x):
        mean = self(x)
        std = self.std
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        entropy = dist.entropy()
        return entropy
    
class ValueNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, layer_sizes, key, activation=jax.nn.relu):
        keys = jr.split(key, num=len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(in_f, out_f, key=k)
            for in_f, out_f, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.squeeze(-1)  # Output scalar value

# class StochasticActor(eqx.Module):
#     layers: list[eqx.nn.Linear]
#     bias: list[jax.Array]
#     activation: callable
#     std: jax.Array
#     layer_sizes: list

#     def __init__(self, layer_sizes, mlp_key, activation=jax.nn.relu):

#         mlp_keys = jr.split(mlp_key, num=len(layer_sizes)) # fully consume the mlp_key

#         self.layers = [eqx.nn.Linear(in_features, out_features, key=mlp_keys[i]) 
#                        for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
#         self.bias = [jnp.zeros(out_features) for out_features in layer_sizes[1:]]
#         self.activation = activation
#         self.std = jnp.ones(layer_sizes[-1]) # vector of standard deviations
#         self.layer_sizes = layer_sizes

#     def __call__(self, x, sample_key):
#         for linear, b in zip(self.layers[:-1], self.bias[:-1]):
#             x = linear(x)
#             x += b
#             x = self.activation(x)
#         x = self.layers[-1](x) + self.bias[-1] # dont apply act to final output
#         return jr.normal(sample_key, shape=(self.layer_sizes[-1],)) * self.std + x
        
#     def forward_deterministic(self, x):
#         for linear, b in zip(self.layers[:-1], self.bias[:-1]):
#             x = linear(x)
#             x += b
#             x = self.activation(x)
#         x = self.layers[-1](x) + self.bias[-1] # dont apply act to final output
#         return x


class DoubleCritic(eqx.Module):
    critic_1_layers: list[eqx.nn.Linear]
    critic_2_layers: list[eqx.nn.Linear]
    critic_1_bias: list[jax.Array]
    critic_2_bias: list[jax.Array]
    activation: callable

    def __init__(self, layer_sizes, mlp_key, activation=jax.nn.relu):

        mlp_keys = jr.split(mlp_key, num=len(layer_sizes)*2) # fully consume the mlp_key

        self.critic_1_layers = [eqx.nn.Linear(in_features, out_features, key=mlp_keys[i]) 
                       for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))]
        self.critic_2_layers = [eqx.nn.Linear(in_features, out_features, key=mlp_keys[i+len(layer_sizes)]) 
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

    global_key = jr.PRNGKey(0)
    global_key, actor_key, critic_key, dataset_key, sample_key = jr.split(global_key, num=5)

    batch_size = 30
    num_actions = 2
    num_observations = 2

    actor = StochasticActor([num_observations,10,10,num_actions], actor_key)
    critic = DoubleCritic([num_actions,10,10,1], critic_key)

    x = jr.normal(dataset_key, shape=[batch_size,2])

    u = jax.vmap(actor)(x, jr.split(sample_key, num=batch_size))
    v = jax.vmap(critic)(u)

    print('fin')

