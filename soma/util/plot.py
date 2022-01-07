from typing import Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_divergences(dimensions: np.ndarray, divergences: np.ndarray, mean: float, std: float,
                     ax: plt.Axes = None) -> plt.Figure:
    if ax is None:
        ax = plt.gca()
    ax.plot(dimensions, divergences[:, 0], label='PCA KL-Divergence', color='#4c72b0')
    ax.fill_between(dimensions, divergences[:, 0] - divergences[:, 1], divergences[:, 0] + divergences[:, 1],
                    label='PCA $\\pm \\sigma$', color='#6ea5ff')
    ax.axhline(mean, linestyle='--', c='red', label='Original KL-Divergence')
    ax.axhline(mean - std, linestyle='--', c='pink')
    ax.axhline(mean + std, linestyle='--', c='pink')
    ax.legend()
    return plt.gcf()


def plot_power(dimensions: np.ndarray, results: Dict[str, np.ndarray], *, alpha: float = 0.1):
    figsize = plt.rcParams['figure.figsize']
    fig, axes = plt.subplots(ncols=2, sharex=True, figsize=(figsize[0], figsize[1] / 2))
    for test_name, test_results in results.items():
        axes[0].plot(dimensions, test_results[:, 0], label=test_name)
        axes[1].plot(dimensions, test_results[:, 1], label=test_name)
    axes[1].legend()
    axes[0].axhline(alpha, linestyle='--', color='red')
    axes[0].set_xlabel('Dimensions')
    axes[0].set_ylabel('$\\alpha$')
    axes[1].set_xlabel('Dimensions')
    axes[1].set_ylabel('$\\beta$')
    axes[0].set_title('Type I Error')
    axes[1].set_title('Type II Error')
    plt.tight_layout()
    return fig
