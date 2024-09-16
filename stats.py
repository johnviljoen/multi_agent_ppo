import jax
import jax.numpy as jnp
import equinox as eqx

class RunningMeanStd(eqx.Module):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float

    def __init__(self, mean=None, var=None, epsilon=1e-4, shape=()):
        if mean is None:
            self.mean = jnp.zeros(shape)
        else:
            self.mean = mean
        if var is None:
            self.var = jnp.ones(shape)
        else:
            self.var = var
        self.count = epsilon

    def update(self, arr) -> None:
        arr = jax.lax.stop_gradient(arr)
        batch_mean = jnp.mean(arr, axis=0)
        batch_var = jnp.var(arr, axis=0)
        batch_count = arr.shape[0]
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
            + jnp.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        return RunningMeanStd(new_mean, new_var, new_count)

    def normalize(self, arr):
        return (arr - self.mean) / jnp.sqrt(self.var + 1e-5)

    def denormalize(self, arr):
        return arr * jnp.sqrt(self.var + 1e-5) + self.mean

