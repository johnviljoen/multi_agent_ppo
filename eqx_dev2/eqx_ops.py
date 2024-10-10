import equinox as eqx
import jax
from jax import lax
import equinox.internal as eqxi

def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    # Partition the initial carry and sequence inputs into dynamic and static parts
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)

    # Define the scanned function, handling the combination and partitioning
    def scanned_fn(carry_dynamic, x_dynamic):
        # Combine dynamic and static parts for the carry and input
        carry = eqx.combine(carry_dynamic, init_static)
        x = eqx.combine(x_dynamic, xs_static)

        # Apply the original function
        out_carry, out_y = f(carry, x)

        # Partition the outputs into dynamic and static parts
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        out_y_dynamic, out_y_static = eqx.partition(out_y, eqx.is_array)

        # Return dynamic outputs and wrap static outputs using Static to prevent tracing
        return out_carry_dynamic, (out_y_dynamic, eqxi.Static((out_carry_static, out_y_static)))

    # Use lax.scan with the modified scanned function
    final_carry_dynamic, (ys_dynamic, static_out) = lax.scan(
        scanned_fn, init_dynamic, xs_dynamic, length=length, reverse=reverse, unroll=unroll
    )

    # Extract static outputs
    out_carry_static, ys_static = static_out.value

    # Combine dynamic and static parts of the outputs
    final_carry = eqx.combine(final_carry_dynamic, out_carry_static)
    ys = eqx.combine(ys_dynamic, ys_static)

    return final_carry, ys

if __name__ == "__main__":

    import jax.numpy as jnp

    # Define a function to be scanned
    def f(carry, x):
        a, b = carry
        c, d = x
        new_carry = (a + c, [(b + d)[0]])
        y = a * c # + b * d
        return new_carry, y

    # Initial carry and sequence inputs
    init = (jnp.array(0.0), [1])  # Mix of array and non-array (static) data
    xs = (
        jnp.array([1.0, 2.0, 3.0]),
        [4, 5, 6]  # Static data in the sequence
    )

    # Apply filter_scan
    final_carry, ys = filter_scan(f, init, xs)

    print("Final Carry:", final_carry)
    print("Outputs:", ys)
