import math
import numpy as np
import jax.numpy as jnp

from jax import jit
from typing import Callable, NamedTuple, Union, List, Tuple
from functools import partial

__all__ = ['rk4_m_P',
           'rk4_m_P_backward',
           'SigmaPoints']


@partial(jit, static_argnums=(0,))
def rk4_m_P(m_cov_ode: Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
            m: jnp.ndarray, P: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ad-hoc Runge--Kutta 4 for solving the mean and cov filtering ODE system.

    Parameters
    ----------
    m_cov_ode
    m
    P
    dt

    Returns
    -------

    """
    k1_m, k1_P = m_cov_ode(m, P)
    k2_m, k2_P = m_cov_ode(m + dt * k1_m / 2, P + dt * k1_P / 2)
    k3_m, k3_P = m_cov_ode(m + dt * k2_m / 2, P + dt * k2_P / 2)
    k4_m, k4_P = m_cov_ode(m + dt * k3_m, P + dt * k3_P)
    return m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6, \
           P + dt * (k1_P + 2 * k2_P + 2 * k3_P + k4_P) / 6


@partial(jit, static_argnums=(0,))
def rk4_m_P_backward(m_cov_ode: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                         Tuple[jnp.ndarray, jnp.ndarray]],
                     m: jnp.ndarray, P: jnp.ndarray,
                     mf: jnp.ndarray, Pf: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ad-hoc Runge--Kutta 4 for solving the mean and cov smoothing ODE system.

    Parameters
    ----------
    m_cov_ode
    m
    P
    mf
    Pf
    dt

    Returns
    -------

    """
    k1_m, k1_P = m_cov_ode(m, P, mf, Pf)
    k2_m, k2_P = m_cov_ode(m + dt * k1_m / 2, P + dt * k1_P / 2, mf, Pf)
    k3_m, k3_P = m_cov_ode(m + dt * k2_m / 2, P + dt * k2_P / 2, mf, Pf)
    k4_m, k4_P = m_cov_ode(m + dt * k3_m, P + dt * k3_P, mf, Pf)
    return m + dt * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6, \
           P + dt * (k1_P + 2 * k2_P + 2 * k3_P + k4_P) / 6


def euler(f: Callable, x0: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Euler's method (not used).
    """
    return x0 + f(x0) * dt


class SigmaPoints(NamedTuple):
    r"""Sigma-point integration.

    .. math::

        \int z(x) \mathrm{N}(x \mid m, P) dx \approx \sum^s_{i=1} z(\chi_i),

    where :math:`\chi_i = m + \sqrt{P} \, \xi_i`.

    Attributes
    ----------
    d : int
        Problem dimension (e.g., state dimension).
    n_points : int
        Number of sigma points.
    w : jnp.ndarray (s, )
        Weights.
    wc : jnp.ndarray (sc, )
        Additional weights (if has any).
    xi : jnp.ndarray (s, d)
        Pre-sigma points.
    """
    d: int
    n_points: int
    w: jnp.ndarray
    wc: Union[jnp.ndarray, None]
    xi: jnp.ndarray

    @staticmethod
    def _gen_hermite_poly_coeff(order: int) -> List[np.ndarray]:
        """Give the 0 to p-th order physician Hermite polynomial coefficients, where p is the
        order argument. The returned coefficients is ordered from highest to lowest.
        Also note that this implementation is different from the np.hermite method.

        Parameters
        ----------
        order : int
            The order of Hermite polynomial

        Returns
        -------
        H : List
            The 0 to p-th order Hermite polynomial coefficients in a list.
        """
        H0 = np.array([1])
        H1 = np.array([2, 0])

        H = [H0, H1]

        for i in range(2, order + 1):
            H.append(2 * np.append(H[i - 1], 0) -
                     2 * (i - 1) * np.pad(H[i - 2], (2, 0), 'constant', constant_values=0))
        return H

    @classmethod
    def cubature(cls, d: int):
        """A factory method for generating spherical cubature :code:`SigmaPoints`.

        Parameters
        ----------
        d : int
            State dimension.
        """
        n_points = 2 * d
        w = jnp.ones(shape=(n_points,)) / n_points
        xi = math.sqrt(d) * jnp.concatenate([jnp.eye(d), -jnp.eye(d)], axis=0)
        return cls(d=d, n_points=n_points, w=w, wc=None, xi=xi)

    @classmethod
    def gauss_hermite(cls, d: int, order: int = 3):
        """A factory method for generating Gauss--Hermite :code:`SigmaPoints`.

        Parameters
        ----------
        d : int
            State dimension.
        order : int, default=3
            Order of Hermite polynomial.
        """
        n_points = order ** d

        hermite_coeff = cls._gen_hermite_poly_coeff(order)
        hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

        table = np.zeros(shape=(d, order ** d))

        w_1d = np.zeros(shape=(order,))
        for i in range(order):
            w_1d[i] = (2 ** (order - 1) * np.math.factorial(order) * np.sqrt(np.pi) /
                       (order ** 2 * (np.polyval(hermite_coeff[order - 1],
                                                 hermite_roots[i])) ** 2))

        # Get roll table
        for i in range(d):
            base = np.ones(shape=(1, order ** (d - i - 1)))
            for j in range(1, order):
                base = np.concatenate([base,
                                       (j + 1) * np.ones(shape=(1, order ** (d - i - 1)))],
                                      axis=1)
            table[d - i - 1, :] = np.tile(base, (1, int(order ** i)))

        table = table.astype("int64") - 1

        s = 1 / (np.sqrt(np.pi) ** d)

        w = s * np.prod(w_1d[table], axis=0)
        xi = (math.sqrt(2) * hermite_roots[table]).T

        return cls(d=d, n_points=n_points, w=jnp.array(w), wc=None, xi=jnp.array(xi))

    def gen_sigma_points(self, m: jnp.ndarray, chol_of_P: jnp.ndarray) -> jnp.ndarray:
        r"""Generate sigma points :math:`\lbrace \chi_i = m + \sqrt{P} xi_i \rbrace^s_{i=1}`.
        """
        return m + jnp.einsum('ij,...j->...i', chol_of_P, self.xi)

    def expectation(self, v_f: Callable, chi: jnp.ndarray) -> jnp.ndarray:
        r"""Approximate expectation by using sigma points.

        Parameters
        ----------
        v_f : Callable (s, ...) -> (s, ???)
            Vectorised integrand function.
        chi : jnp.ndarray (s, ...)

        Returns
        -------

        """
        return jnp.einsum('i,i...->...', self.w, v_f(chi))

    def expectations(self, v_fs: Callable[[jnp.ndarray], Tuple[jnp.ndarray, ...]], chi: jnp.ndarray):
        """(Not used. generator cannot be jitted.)
        """
        vals = v_fs(chi)
        return (jnp.einsum('i,i...->...', self.w, val) for val in vals)

    def expectations_two(self, v_fs: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]], chi: jnp.ndarray):
        """Ad-hoc variation of :code:`expectation` for integrands that have two outputs.
        """
        vals = v_fs(chi)
        return jnp.einsum('i,i...->...', self.w, vals[0]), jnp.einsum('i,i...->...', self.w, vals[1])
