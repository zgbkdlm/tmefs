import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    sigmas = [0.2, 15.]

    file_names = [f'../triton/results/stability_TME-2_TME-2_{sigma}.npy' for sigma in sigmas]

    mses = [np.load(file_name) for file_name in file_names]

    mses_mean = [np.mean(mse, axis=0) for mse in mses]
    mses_std = [np.std(mse, axis=0) for mse in mses]

    ts = np.linspace(0.02, 2, 100)
    ks = np.linspace(1, 100, 100)

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': "san-serif",
        'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
        'font.size': 17})

    fig, axs = plt.subplots(nrows=len(sigmas), figsize=(10, 7), sharex=True)

    # Plot
    for id in range(len(sigmas)):
        axs[id].plot(ks, mses_mean[id], linewidth=3, c='black')
        axs[id].fill_between(ks,
                            mses_mean[id] - 1.96 * mses_std[id],
                            mses_mean[id] + 1.96 * mses_std[id],
                            color='black',
                            edgecolor='none',
                            alpha=0.15
                            )

        axs[id].grid(linestyle='--', alpha=0.3, which='both')
        axs[id].annotate(rf'$\sigma={sigmas[id]}$', (530, 180), xycoords='axes points',
                         fontsize=20,
                         bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round'))

    axs[-1].set_xlabel('$k$')
    axs[-1].set_xlim([0, 100])

    # Requires matplotlib >= 3.4
    fig.supylabel(r'Monte Carlo estimated $\mathbb{E}\big[\lVert X_k - m^s_k\rVert_2^2\big]$',
                  x=0., fontsize=19)

    plt.subplots_adjust(left=0.075, bottom=0.077, right=0.984, top=0.995, wspace=0.2, hspace=0.026)
    # plt.tight_layout(pad=0.1)

    # plt.show()
    plt.savefig('stability-sigma.pdf')
