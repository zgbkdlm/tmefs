import math
import jax
import jax.numpy as jnp
import numpy as np
import tme.base_jax as tme
import argparse

from jax import jit
from jax.config import config

from tmefs.filters_smoothers import sgp_filter, sgp_smoother, cd_ekf, cd_eks
from tmefs.quadratures import SigmaPoints
from tmefs.tools import rmse, gen_sample

config.update("jax_enable_x64", True)

dim_x = 3
kappa = 10.
lam = 28.
mu = 2.
Qw = jnp.eye(3)
H = jnp.array([[1., 0., 0.]])
R = 2.


@jit
def drift(u):
    return jnp.array([kappa * (u[1] - u[0]),
                      u[0] * (lam - u[2]) - u[1],
                      u[0] * u[1] - mu * u[2]])


@jit
def dispersion(u):
    return 5. * jnp.eye(3)


@jit
def tme_m_cov_2(u, dt):
    return tme.mean_and_cov(x=u, dt=dt,
                            a=drift, b=dispersion, Qw=Qw, order=2)


@jit
def tme_m_cov_3(u, dt):
    return tme.mean_and_cov(x=u, dt=dt,
                            a=drift, b=dispersion, Qw=Qw, order=3)


@jit
def em_m_cov(u, dt):
    return u + drift(u) * dt, dispersion(u) @ Qw @ dispersion(u).T * dt


discs = {'EM': em_m_cov,
         'TME-2': tme_m_cov_2,
         'TME-3': tme_m_cov_3}

num_mc_trials = 1000

m0 = jnp.zeros((dim_x,))
P0 = 10. * jnp.eye(dim_x)
sgps = SigmaPoints.gauss_hermite(d=dim_x, order=3)

dt = 0.02
end_t = 2.
num_steps = 100


def run(filter_name: str, smoother_name: str):
    if filter_name == 'EKF':
        @jit
        def jitted_filter(ys):
            return cd_ekf(a=drift, b=dispersion, H=H, R=R, m0=m0, P0=P0, dt=dt, ys=ys)
    else:
        @jit
        def jitted_filter(ys):
            return sgp_filter(f_Q=discs[filter_name], sgps=sgps,
                              H=H, R=R,
                              m0=m0, P0=P0, dt=dt, ys=ys)

    if smoother_name == 'EKS':
        def jitted_smoother(mfs, Pfs):
            return cd_eks(a=drift, b=dispersion, mfs=mfs, Pfs=Pfs, dt=dt)
    else:
        @jit
        def jitted_smoother(mfs, Pfs):
            return sgp_smoother(f_Q=discs[smoother_name], sgps=sgps, mfs=mfs, Pfs=Pfs,
                                dt=dt)

    print(f'Running {filter_name} filter and {smoother_name} smoother.')

    rmses = np.zeros((num_mc_trials,))
    for mc in range(num_mc_trials):
        # Simulate data
        key = jax.random.PRNGKey(666 + mc)
        key, subkey = jax.random.split(key)
        x0 = jax.random.multivariate_normal(key=subkey, mean=m0, cov=P0)
        key, subkey = jax.random.split(key)
        ts, true_x = gen_sample(m_and_cov=em_m_cov, x0=x0, dt=dt, end_t=end_t,
                                num_steps=num_steps, int_steps=10000,
                                key=subkey)
        key, subkey = jax.random.split(key)
        ys = jnp.einsum('ij,...j->...i', H, true_x).reshape(-1) \
             + math.sqrt(R) * jax.random.normal(subkey, shape=(ts.size,))

        filtering_results = jitted_filter(ys)
        smoothing_results = jitted_smoother(filtering_results[0], filtering_results[1])

        rmses[mc] = rmse(true_x, smoothing_results[0])

    rmse_mean = np.mean(rmses)
    file_name = f'{filter_name}_{smoother_name}_{rmse_mean:.4f}.npy'
    np.save('./results/' + file_name, rmses)
    print('Results saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run filter and smoother.')
    parser.add_argument('-filter', type=str, help='Name of the filter.')
    parser.add_argument('-smoother', type=str, help='Name of the smoother.')
    args = parser.parse_args()
    run(args.filter, args.smoother)
