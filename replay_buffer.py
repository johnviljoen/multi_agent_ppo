import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class ReplayBufferEqx(eqx.Module):
    """A simple replay buffer for the AHAC algorithm which resets on every rollout so no long term memory needed"""
    obs: jnp.ndarray
    act: jnp.ndarray
    rew: jnp.ndarray
    v_next: jnp.ndarray
    done_mask: jnp.ndarray
    ptr: int
    size: int
    num_envs: int

    def __init__(self, first_init=True, **kwargs):
        """Initialize replay buffer with zeros."""
        if first_init is True:
            self.init_fresh(**kwargs)
        elif first_init is False:
            self.init_warm(**kwargs)
        else:
            raise Exception

    def init_fresh(self, horizon, num_envs, obs_size, act_size):
        self.num_envs = num_envs
        self.obs = jnp.zeros([horizon, num_envs, obs_size])
        self.act = jnp.zeros([horizon, num_envs, act_size])
        self.rew = jnp.zeros([horizon, num_envs])
        self.v_next = jnp.zeros([horizon, num_envs])
        self.done_mask = jnp.zeros([horizon, num_envs], dtype=bool)
        self.ptr, self.size = 0, 0

    def init_warm(self, obs, act, rew, v_next, done_mask, ptr, size, num_envs):
        self.obs = obs
        self.act = act
        self.rew = rew
        self.v_next = v_next
        self.done_mask = done_mask
        self.ptr = ptr
        self.size = size
        self.num_envs = num_envs

    def store(self, obs, act, rew, v_next, done_mask):
        """Store new experience."""
        return ReplayBufferEqx(
            first_init=False,
            obs=self.obs.at[self.ptr].set(obs),
            act=self.act.at[self.ptr].set(act),
            rew=self.rew.at[self.ptr].set(rew),
            v_next=self.v_next.at[self.ptr].set(v_next), 
            done_mask=self.done_mask.at[self.ptr].set(done_mask), 
            ptr=(self.ptr+1) % self.num_envs, 
            size=jnp.clip(self.size+1, max=self.num_envs), # self.size+1
            num_envs=self.num_envs
        )

    def sample_batch(self, batch_size, rng):
        """Sample past experience."""
        indexes = jr.randint(rng, shape=(batch_size,),
                                 minval=0, maxval=self.size)
        return self.obs[indexes], self.act[indexes], self.rew[indexes], self.v_next[indexes], self.done_mask[indexes]
