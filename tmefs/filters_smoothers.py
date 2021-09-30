import numpy as np
import jax.numpy as jnp
import scipy
import jax
import jax.scipy

from jax import jit, jacfwd, lax, vmap
from tmefs.quadratures import SigmaPoints, rk4_m_P, rk4_m_P_backward
from typing import Tuple, Callable

__all__ = ['cd_ekf',
           'cd_eks',
           'sgp_filter',
           'sgp_smoother']


@jit
def _linear_update(mp: jnp.ndarray, Pp: jnp.ndarray,
                   H: jnp.ndarray, R: float, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update for linear Gaussian measurement models (note that the dim of measurement is assumed to be 1).
    """
    S = H @ Pp @ H.T + R
    K = Pp @ H.T / S
    return mp + K @ (y - H @ mp), Pp - K @ K.T * S


def cd_ekf(a: Callable, b: Callable,
           H: jnp.ndarray, R: float,
           m0: jnp.ndarray, P0: jnp.ndarray,
           dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete extended Kalman filter with 4th order Runge--Kutta.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, w) -> (d, w)
        SDE dispersion coefficient.
    H : jnp.ndarray (dy, d)
        Measurement matrix.
    R : float
        Measurement noise variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Mean and covariance of the filtering estimates.
    """

    def jac_a(u):
        return jacfwd(a)(u)

    def odes(m, P):
        return a(m), P @ jac_a(m).T + jac_a(m) @ P + b(m) @ b(m).T

    def scan_cd_ekf(carry, elem):
        mf, Pf = carry
        y = elem

        mp, Pp = rk4_m_P(odes, mf, Pf, dt)
        mf, Pf = _linear_update(mp, Pp, H, R, y)

        return (mf, Pf), (mf, Pf)

    _, filtering_results = lax.scan(scan_cd_ekf, (m0, P0), ys)
    return filtering_results


def cd_eks(a: Callable, b: Callable,
           mfs: jnp.ndarray, Pfs: jnp.ndarray,
           dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete extended Kalman smoother with 4th order Runge--Kutta.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift coefficient.
    b : Callable (d, w) -> (d, w)
        SDE dispersion coefficient.
    mfs : jnp.ndarray (T, d)
        Filtering means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering covariances.
    dt : float
        Time interval

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Mean and covariance of the smoothing estimates.
    """
    dt = -dt

    def jac_a(u):
        return jacfwd(a)(u)

    def odes(m, P, mf, Pf):
        gamma = b(m) @ b(m).T
        c, low = jax.scipy.linalg.cho_factor(Pf)
        jac_and_gamma_and_chol = jac_a(m) + jax.scipy.linalg.cho_solve((c, low), gamma.T).T
        return a(m) + gamma @ jax.scipy.linalg.cho_solve((c, low), m - mf), \
               jac_and_gamma_and_chol @ P + P @ jac_and_gamma_and_chol.T - gamma

    def scan_cd_eks(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        ms, Ps = rk4_m_P_backward(odes, ms, Ps, mf, Pf, dt)

        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = lax.scan(scan_cd_eks, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]),
                             reverse=True)
    return jnp.vstack([mss, mfs[-1]]), jnp.vstack([Pss, Pfs[-1, None]])


def sgp_filter(f_Q: Callable, sgps: SigmaPoints,
               H: jnp.ndarray, R: float,
               m0: jnp.ndarray, P0: jnp.ndarray,
               dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-point filter by discretising the SDE.

    Parameters
    ----------
    f_Q : Callable
        Mean and covariance approximation of SDE.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    H : jnp.ndarray (dy, d)
        Measurement matrix.
    R : float
        Measurement noise variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Mean and covariance of the filtering estimates.
    """

    def ffT(x):
        f = f_Q(x, dt)[0]
        return jnp.outer(f, f)

    def v_f_Q(x): return vmap(f_Q, in_axes=[0, None])(x, dt)

    def v_ffT(x): return vmap(ffT, in_axes=[0])(x)

    @jit
    def sig_int_f_Q(chi): return sgps.expectations_two(v_f_Q, chi)

    @jit
    def sig_int_ffT(chi): return sgps.expectation(v_ffT, chi)

    def scan_sgp_filter(carry, elem):
        mf, Pf = carry
        y = elem

        chol_Pf = jax.scipy.linalg.cholesky(Pf, lower=True)
        chi = sgps.gen_sigma_points(mf, chol_Pf)
        mp, int_Q = sig_int_f_Q(chi)
        Pp = int_Q + sig_int_ffT(chi) - jnp.outer(mp, mp)

        mf, Pf = _linear_update(mp, Pp, H, R, y)
        return (mf, Pf), (mf, Pf)

    _, filtering_results = lax.scan(scan_sgp_filter, (m0, P0), ys)
    return filtering_results


def sgp_smoother(f_Q: Callable, sgps: SigmaPoints,
                 mfs: jnp.ndarray, Pfs: jnp.ndarray,
                 dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-point smoother by discretising the SDE.

    Parameters
    ----------
    f_Q : Callable
        Mean and covariance approximation of SDE.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    mfs : jnp.ndarray (T, d)
        Filtering means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering covariances.
    dt : float
        Time interval

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Mean and covariance of the smoothing estimates.
    """

    def ffT(x):
        f = f_Q(x, dt)[0]
        return jnp.outer(f, f)

    def xfT(x): return jnp.outer(x, f_Q(x, dt)[0])

    def v_f_Q(x): return vmap(f_Q, in_axes=[0, None])(x, dt)

    def v_ffT(x): return vmap(ffT, in_axes=[0])(x)

    def v_xfT(x): return vmap(xfT, in_axes=[0])(x)

    @jit
    def sig_int_f_Q(chi): return sgps.expectations_two(v_f_Q, chi)

    @jit
    def sig_int_ffT(chi): return sgps.expectation(v_ffT, chi)

    @jit
    def sig_int_xfT(chi): return sgps.expectation(v_xfT, chi)

    def scan_sgp_smoother(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        chol_Pf = jax.scipy.linalg.cholesky(Pf, lower=True)
        chi = sgps.gen_sigma_points(mf, chol_Pf)
        mp, int_Q = sig_int_f_Q(chi)
        Pp = int_Q + sig_int_ffT(chi) - jnp.outer(mp, mp)

        D = sig_int_xfT(chi) - jnp.outer(mf, mp)

        c, low = jax.scipy.linalg.cho_factor(Pp)
        G = jax.scipy.linalg.cho_solve((c, low), D.T).T
        ms = mf + G @ (ms - mp)
        Ps = Pf + G @ (Ps - Pp) @ G.T
        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = lax.scan(scan_sgp_smoother, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return jnp.vstack([mss, mfs[-1]]), jnp.vstack([Pss, Pfs[-1, None]])


def kf_rts(F: np.ndarray, Q: np.ndarray,
           H: np.ndarray, R: float,
           y: np.ndarray,
           m0: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kalman filter and RTS smoother in simple enough implementation. Used in unittest.

    x_k = F x_{k-1} + q_{k-1},
    y_k = H x_k + r_k,

    Parameters
    ----------
    F : np.ndarray
        State transition.
    Q : np.ndarray
        State covariance.
    H : np.ndarray
        Measurement matrix.
    R : float
        Measurement noise variance.
    y : np.ndarray
        Measurements.
    m0, P0 : np.ndarray
        Initial mean and cov.

    Returns
    -------
    ms, ps : np.ndarray
        Smoothing posterior mean and covariances.
    """
    dim_x = m0.size
    num_y = y.size

    mm = np.zeros(shape=(num_y, dim_x))
    pp = np.zeros(shape=(num_y, dim_x, dim_x))

    mm_pred = mm.copy()
    pp_pred = pp.copy()

    m = m0
    p = p0

    # Filtering pass
    for k in range(num_y):
        # Pred
        m = F @ m
        p = F @ p @ F.T + Q
        mm_pred[k] = m
        pp_pred[k] = p

        # Update
        S = H @ p @ H.T + R
        K = p @ H.T / S
        m = m + K @ (y[k] - H @ m)
        p = p - K @ S @ K.T

        # Save
        mm[k] = m
        pp[k] = p

    # Smoothing pass
    ms = mm.copy()
    ps = pp.copy()
    for k in range(num_y - 2, -1, -1):
        (c, low) = scipy.linalg.cho_factor(pp_pred[k + 1])
        G = pp[k] @ scipy.linalg.cho_solve((c, low), F).T
        ms[k] = mm[k] + G @ (ms[k + 1] - mm_pred[k + 1])
        ps[k] = pp[k] + G @ (ps[k + 1] - pp_pred[k + 1]) @ G.T

    return mm, pp, ms, ps
