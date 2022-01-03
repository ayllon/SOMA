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
