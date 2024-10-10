import jax
import jax.numpy as jnp
import equinox as eqx
import dataclasses

class RunningMeanStd(eqx.Module):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    epsilon: float = 1e-4

    def update(self, arr) -> None:
        arr = jax.lax.stop_gradient(arr)
        batch_mean = jnp.mean(arr, axis=tuple(range(arr.ndim - 1)))
        batch_var = jnp.var(arr, axis=tuple(range(arr.ndim - 1)))
        batch_count = int(jnp.prod(jnp.array(arr.shape[:-1]))) # arr.shape[0]
        return self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + delta ** 2
            * self.count
            * batch_count
            / tot_count
        )
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        return dataclasses.replace(
            self,
            mean=new_mean,
            var=new_var,
            count=new_count
        )
    
    def normalize(self, arr):
        return (arr - self.mean) / jnp.sqrt(self.var + 1e-5)

    def denormalize(self, arr):
        return arr * jnp.sqrt(self.var + 1e-5) + self.mean


if __name__ == "__main__":
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)
    
    # Generate some random data (e.g., 100 samples, each of dimension 3)
    num_samples = 100
    shape = (3,)
    data = jax.random.normal(key, (num_samples,) + shape)
    
    # Initialize the RunningMeanStd class
    rms = RunningMeanStd(mean=jnp.zeros(shape), var=jnp.ones(shape), count=0.0, epsilon=1e-4)

    # Update the RunningMeanStd instance with the data
    rms = rms.update(data)
    
    # Print the updated mean and variance
    print("Updated mean:", rms.mean)
    print("Updated variance:", rms.var)
    
    # Test normalization
    normalized_data = rms.normalize(data)
    print("Normalized data (first sample):", normalized_data[0])
    
    # Test denormalization
    denormalized_data = rms.denormalize(normalized_data)
    print("Denormalized data (first sample):", denormalized_data[0])
    
    # Check if the denormalized data is close to the original data
    assert jnp.allclose(denormalized_data, data, atol=1e-5), "Denormalization failed!"
    
    print("Test passed successfully!")
