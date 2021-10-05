# Gaussian filter and smoother with Taylor moment expansion in Python (Jax)

[![Unittests](https://github.com/zgbkdlm/tmefs/actions/workflows/tests.yml/badge.svg)](https://github.com/zgbkdlm/tmefs/actions/workflows/tests.yml)

In this repository, you can find a Python (Jax) implementation of the TME Gaussian filter and smoother for non-linear continuous-discrete filtering and smoothing problems. 

Alternatively, there is also a Matlab implementation in `https://github.com/zgbkdlm/tme/tree/main/matlab`.

# Installation

1. Git clone this repository.
2. Navigate to the cloned folder.
3. Open a terminal and run `python setup.py install` or `python setup.py develop` in your favourite python environment.

# Exmple

You can find a runnable Jupyter notebook in `./examples_plots_tables/demo_tme.ipynb` which does filtering and smoothing on a Lorenz model.

A skeleton for using the TME filter and smoother is shown in the following snippet.

```python
import jax
import tme.base_jax as tme
from tmefs.filters_smoothers import sgp_filter, sgp_smoother
from tmefs.quadratures import SigmaPoints

def drift(x):
    # Define your SDE drift function here.
    return ...

def dispersion(x):
    # Define your SDE dispersion function here.
    return ...

tme_order = 2

# Form the TME approximation for SDE mean and covariance
@jit
def tme_m_cov(x, dt):
    return tme.mean_and_cov(x=x, dt=dt,
                            a=drift, b=dispersion, Qw=Qw, order=tme_order)

# Make sigma points (Gauss--Hermite)
sigma_points = SigmaPoints.gauss_hermite(d=dim_of_state, order=3)

# Here is the filter which takes data `ys` as input
@jit
def tme_filter(ys):
    return sgp_filter(f_Q=tme_m_cov, sgps=sigma_points,
                      H=H, R=R, m0=m0, P0=P0, dt=dt, ys=ys)

# Here is the smoother which takes filtering means `mfs` and covariances `Pfs` as input
@jit
def tme_smoother(mfs, Pfs):
    return sgp_smoother(f_Q=discs[smoother_name], sgps=sgps, mfs=mfs, Pfs=Pfs,
                        dt=dt)

# Run
filtering_results = tme_filter(ys)
smoothing_results = tme_smoother(filtering_results[0], filtering_results[1])
```

# What are the files

1. `./examples_plot_tables`: This folder contains a demo for using TME filter and smoother, and scripts that generate the figures and tables in the manuscript.
2. `./test`: Unit tests.
3. `./tmefs`: Package.
4. `./triton`: This folder contains the files that produce the numerical results in the manuscript. This is helpful if you wish to reproduce our results exactly. Note that the scripts are executed in the Triton computational cluster (a slurm-based cluster), Aalto University, but you can still run them in your computer with small modifications.

# Citation

```bibtex
@article{ZhaoZ2021TMEsmoother,
	title={Non-linear {G}aussian smoothing with {T}aylor moment expansion},
	author={Zhao, Zheng and S{\"a}rkk{\"a}, Simo},
	journal={arXiv:2110.01396},
	year={2021}
}
```

# License
GNU General Public License version 3 or later.
