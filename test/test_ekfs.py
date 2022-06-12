"""
Unit test for ekfs
"""
import unittest
import math
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from tmefs.filters_smoothers import kf_rts, cd_ekf, cd_eks
from jax import jit
from jax.config import config

config.update("jax_enable_x64", True)

np.random.seed(666)

dim_x = 1
dt = 1e-3

# dx = A x dt + B dW
# CD-EKFS works only if the transition is very linear, as the ODEs are solved under numerical means (i.e., RK4 or Euler)
a, b = 0.1, 0.1
A = -a * jnp.eye(dim_x)
B = b * jnp.eye(dim_x)

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


class TestEKFS(unittest.TestCase):

    def setUp(self):
        # Simulate
        num_measurements = 10000
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

    def test_ekfs(self):
        # KFS
        kfs_mf, kfs_pf, kfs_ms, kfs_ps = kf_rts(np.array(F), np.array(Q), np.array(H),
                                                R, np.array(self.ys), np.array(m0), np.array(P0))

        @jit
        def jitted_ekf(ys):
            return cd_ekf(a=drift, b=dispersion, H=H, R=R, m0=m0, P0=P0, dt=dt, ys=ys)

        def jitted_eks(mfs, Pfs):
            return cd_eks(a=drift, b=dispersion, mfs=mfs, Pfs=Pfs, dt=dt)

        filtering_results = jitted_ekf(self.ys)
        smoothing_results = jitted_eks(filtering_results[0], filtering_results[1])

        npt.assert_allclose(kfs_mf, filtering_results[0], atol=1e-3)
        npt.assert_allclose(kfs_ms, smoothing_results[0], atol=1e-3)


if __name__ == '__main__':
    unittest.main()
