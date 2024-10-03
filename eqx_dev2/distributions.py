# Source: https://github.com/google/brax/blob/main/brax/training/distribution.py

import jax
import jax.numpy as jnp
import dataclasses


@dataclasses.dataclass
class NormalDistribution:
    loc: jnp.array
    scale: jnp.array

    def sample(self, key):
        return jax.random.normal(key, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization
        return entropy * jnp.ones_like(self.loc)


@dataclasses.dataclass
class NormalTanhDistribution:
    _min_std: float = 0.001
    _var_scale: float = 1.0

    def create_dist(self, loc, scale):
        scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
        return NormalDistribution(loc=loc, scale=scale)
    
    def sample_no_postprocess(self, loc, scale, key):
        return self.create_dist(loc, scale).sample(key=key)
    
    def sample(self, loc, scale, key):
        return jnp.tanh(self.sample_no_postprocess(loc, scale, key))

    def mode(self, loc, scale):
        return jnp.tanh(self.create_dist(loc, scale).mode())

    # the forward log det of the jacobian of the tanh bijector
    def tanh_log_det_jac(self, x):
        return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))

    def log_prob(self, loc, scale, actions):
        dist = self.create_dist(loc, scale)
        log_probs = dist.log_prob(actions)
        log_probs -= self.tanh_log_det_jac(actions)
        # if self._event_ndims == 1:
        log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs
    
    def entropy(self, loc, scale, key):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(loc, scale)
        entropy = dist.entropy()
        entropy += self.tanh_log_det_jac(dist.sample(key=key))
        # if self._event_ndims == 1:
        entropy = jnp.sum(entropy, axis=-1)
        return entropy


if __name__ == "__main__":

    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np

    # Set random seed for reproducibility
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Parameters for the Normal Distribution
    loc = jnp.array(0.0)
    scale = jnp.array(1.0)

    normal_dist = NormalDistribution(loc=loc, scale=scale)

    # Generate samples from the Normal Distribution
    num_samples = 100000
    sample_shape = (num_samples,)
    key, subkey = jax.random.split(key)
    subkey = jax.random.split(subkey, sample_shape)
    samples_normal = jax.vmap(normal_dist.sample)(subkey)

    # Convert samples to NumPy array for plotting
    samples_normal_np = np.array(samples_normal)

    # Plot histogram of the samples and overlay the analytical PDF
    x = np.linspace(-5, 5, 1000)
    pdf = norm.pdf(x, loc=loc, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.hist(samples_normal_np, bins=100, density=True, alpha=0.6, color='skyblue', label='Sampled Histogram')
    plt.plot(x, pdf, 'r', lw=2, label='Analytical PDF')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("test1.png", dpi=500)
    plt.close()

    # Now test the NormalTanhDistribution
    # Parameters for the NormalTanhDistribution
    # event_size = 1  # For simplicity, use 1-dimensional data
    loc_tanh = jnp.array([0.0]*num_samples)  # Mean of the underlying normal distribution
    scale_tanh = jnp.array([1.0]*num_samples)  # Scale of the underlying normal distribution

    normal_tanh_dist = NormalTanhDistribution()

    # Generate samples from the NormalTanhDistribution
    key, subkey = jax.random.split(key)
    subkey = jax.random.split(subkey, sample_shape)
    samples_normal_tanh = jax.vmap(normal_tanh_dist.sample)(loc_tanh, scale_tanh, subkey)

    # Convert samples to NumPy array for plotting
    samples_normal_tanh_np = np.array(samples_normal_tanh).flatten()

    # Plot histogram of the samples
    plt.figure(figsize=(10, 6))
    plt.hist(samples_normal_tanh_np, bins=100, density=True, alpha=0.6, color='lightgreen', label='Sampled Histogram')
    plt.title('Normal Distribution Transformed by Tanh')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("test2.png", dpi=500)
    plt.close()

    # Plot the tanh of the underlying normal distribution
    # Generate samples from the underlying normal distribution
    key, subkey = jax.random.split(key)
    subkey = jax.random.split(subkey, sample_shape)
    samples_underlying = jax.vmap(normal_tanh_dist.sample_no_postprocess)(loc_tanh, scale_tanh, subkey)
    samples_underlying_np = np.array(samples_underlying).flatten()

    # Apply tanh manually to the samples
    samples_tanh_np = np.tanh(samples_underlying_np)

    # Plot histogram of the tanh-transformed samples (should match the previous histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(samples_tanh_np, bins=100, density=True, alpha=0.6, color='orange', label='Tanh of Underlying Samples')
    plt.title('Tanh of Underlying Normal Distribution Samples')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("test3.png", dpi=500)
    plt.close()

    # Plot the effect of tanh on the PDF 
    # NOTE the analytical line here is approximate - the real sampled dist is correct.
    x_underlying = np.linspace(-5, 5, num_samples)
    pdf_underlying = norm.pdf(x_underlying, loc=loc_tanh, scale=scale_tanh)
    x_tanh = np.tanh(x_underlying)
    # Compute the transformed PDF using change of variables
    pdf_tanh = pdf_underlying / (1 - x_tanh**2)
    # Since x_tanh is not monotonically increasing over the full range, we need to sort it for plotting
    sorted_indices = np.argsort(x_tanh)
    x_tanh_sorted = x_tanh[sorted_indices]
    pdf_tanh_sorted = pdf_tanh[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(x_tanh_sorted, pdf_tanh_sorted, 'b', lw=2, label='Transformed PDF')
    plt.hist(samples_normal_tanh_np, bins=100, density=True, alpha=0.6, color='lightgreen', label='Sampled Histogram')
    plt.title('Transformed PDF vs Sampled Histogram')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig("test4.png", dpi=500)
    plt.close()
