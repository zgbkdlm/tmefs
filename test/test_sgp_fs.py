"""
Unit test for sigma-point filters and smoothers
"""
import unittest
import math
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import tme.base_jax as tme

from tmefs.quadratures import SigmaPoints
from tmefs.filters_smoothers import sgp_filter, sgp_smoother, kf_rts
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)

dim_x = 1
dt = 0.01

# dx = A x dt + B dW
# TME should give asymtopically exact mean disc. I.e., Taylor expansion of exponential function.
a, b = 1., 1.
A = -a * jnp.eye(dim_x)
B = b * jnp.eye(dim_x)
Qw = 1. * jnp.eye(dim_x)

# x_k = F x_{k-1} + Q
F = math.exp(-a * dt) * jnp.eye(dim_x)
Q = b ** 2 / (2 * a) * (1 - math.exp(-2 * a * dt)) * jnp.eye(dim_x)

R = 0.01
H = jnp.ones((1, dim_x))

m0 = jnp.zeros((dim_x,))
P0 = 0.1 * jnp.eye(dim_x)


def drift(u):
    return A @ u


def dispersion(u):
    return B


@jit
def tme_m_cov(u, dt):
    return tme.mean_and_cov(x=u, dt=dt,
                            a=drift, b=dispersion, Qw=Qw, order=3)


@jit
def em_m_cov(u, dt):
    return u + drift(u) * dt, dispersion(u) @ Qw @ dispersion(u).T * dt


class TestFiltersSmoothers(unittest.TestCase):

    def setUp(self):
        # Simulate
        num_measurements = 1000
        xx = np.zeros((num_measurements, dim_x))
        yy = np.zeros((num_measurements,))
        x = np.array(m0).copy()
        for i in range(num_measurements):
            x = F @ x + np.sqrt(Q) @ np.random.randn(dim_x)
            y = H @ x + np.sqrt(R) * np.random.randn()
            xx[i] = x
            yy[i] = y

        self.true_signal = jnp.array(xx)
        self.ys = jnp.array(yy)

    def test_tme_sgpfs(self):
        # KFS
        kfs_mf, kfs_pf, kfs_ms, kfs_ps = kf_rts(np.array(F), np.array(Q), np.array(H),
                                                R, np.array(self.ys), np.array(m0), np.array(P0))

        # SGP
        for sgps in [SigmaPoints.gauss_hermite(d=dim_x, order=5), SigmaPoints.cubature(d=dim_x)]:

            @jit
            def jitted_filter(ys):
                return sgp_filter(f_Q=tme_m_cov, sgps=sgps,
                                  H=H, R=R,
                                  m0=m0, P0=P0, dt=dt, ys=ys)

            @jit
            def jitted_smoother(mfs, Pfs):
                return sgp_smoother(f_Q=tme_m_cov, sgps=sgps, mfs=mfs, Pfs=Pfs,
                                    dt=dt)

            filtering_results = jitted_filter(self.ys)
            smoothing_results = jitted_smoother(filtering_results[0], filtering_results[1])

            npt.assert_allclose(kfs_mf, filtering_results[0], atol=1e-7)
            npt.assert_allclose(kfs_ms, smoothing_results[0], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
