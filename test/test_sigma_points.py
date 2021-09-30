"""
Unit test for sigma-point integrations
"""
import unittest
import math
import jax.numpy as jnp
import numpy.testing as npt

from tmefs.quadratures import SigmaPoints
from jax import vmap
from jax.config import config

config.update("jax_enable_x64", True)


class TestKFSDiscrete(unittest.TestCase):

    def test_quadratures(self):
        d = 1
        gh_order = 5

        cub = SigmaPoints.cubature(d=d)
        gh = SigmaPoints.gauss_hermite(d=d, order=gh_order)

        # Test if weights normalise
        npt.assert_almost_equal(jnp.sum(cub.w), 1.)
        npt.assert_almost_equal(jnp.sum(gh.w), 1.)

        def f(x: jnp.ndarray):
            return jnp.sin(x)

        vf = vmap(f, [0])

        m = math.pi / 2 * jnp.ones(shape=(d,))
        P = jnp.eye(d)

        true_integral_val = jnp.reshape(jnp.sin(m) * jnp.exp(- P / 2), (-1,))

        integral_cub = cub.expectation(vf, cub.gen_sigma_points(m, jnp.sqrt(P)))
        integral_gh = gh.expectation(vf, gh.gen_sigma_points(m, jnp.sqrt(P)))

        npt.assert_allclose(true_integral_val, integral_cub, rtol=2e-1)
        npt.assert_almost_equal(true_integral_val, integral_gh, decimal=4)


if __name__ == '__main__':
    unittest.main()
