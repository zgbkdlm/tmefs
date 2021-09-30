import jax.numpy as jnp
import jax

from functools import partial
from typing import Callable, Tuple, Union


__all__ = ['disc_normal',
           'rmse',
           'vectorised_norm',
           'gen_sample']


@partial(jax.jit, static_argnums=(0,))
def disc_normal(m_and_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
                x0: jnp.ndarray, dts: jnp.ndarray, dws: jnp.ndarray) -> jnp.ndarray:
    r"""Simulate an SDE trajectory with Gaussian increments.

    .. math::

        X_k \approx f(X_{k-1}) + q_{k-1}(X_{k-1}),

    Parameters
    ----------
    m_and_cov : Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]]
        Mean and covariance approximation of SDE.
    x0 : jnp.ndarray
        Initial value.
    dts : jnp.ndarray
        Time intervals.
    dws : jnp.ndarray
        Increments of Wiener process.

    Returns
    -------
    jnp.ndarray
        An SDE trajectory.
    """
    def scan_body(carry, elem):
        x = carry
        dt, dw = elem

        m, cov = m_and_cov(x, dt)
        chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, sample = jax.lax.scan(scan_body, x0, (dts, dws))
    return sample


def rmse(x1: jnp.ndarray, x2: jnp.ndarray, reduce_sum: bool = True) -> Union[float, jnp.ndarray]:
    """Root mean square error.

    Parameters
    ----------
    x1 : jnp.ndarray
    x2 : jnp.ndarray
    reduce_sum : bool, default=True
        Let this be :code:`True` will take a sum of the RMSEs from all dimensions.
    """
    val = jnp.sqrt(jnp.mean((x1 - x2) ** 2, 0))
    if reduce_sum:
        return jnp.sum(val)
    else:
        return val


def _norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x, ord=2)


def vectorised_norm(x: jnp.ndarray) -> jnp.ndarray:
    """Compute L2 norm in batch.
    """
    return jax.vmap(_norm, in_axes=[0])(x)


def gen_sample(m_and_cov: Callable,
               x0: jnp.ndarray,
               dt: float, end_t: float, num_steps: int, int_steps: int,
               key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a sample of an SDE by calling :code:`disc_normal`.

    Parameters
    ----------
    m_and_cov
    x0
    dt
    end_t
    num_steps
    int_steps
    key

    Returns
    -------
    jnp.ndarray
        Times.
    jnp.ndarray
        An SDE trajectory at the times.
    """
    ts = jnp.linspace(dt / int_steps, end_t, num_steps * int_steps)
    dts = jnp.concatenate([jnp.array([dt]), jnp.diff(ts)], axis=0)
    dws = jax.random.normal(key, shape=(dts.size, x0.shape[0]))
    return ts[int_steps-1::int_steps], disc_normal(m_and_cov, x0, dts, dws)[int_steps-1::int_steps]
