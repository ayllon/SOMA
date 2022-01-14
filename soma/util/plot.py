import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


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


def plot_errors(results: DataFrame, *, alpha: float = 0.1):
    tests = results.index.levels[0].values
    xval = results.index.levels[1].values
    xlabel = results.index.names[1].capitalize()

    figsize = plt.rcParams['figure.figsize']
    fig, axes = plt.subplots(ncols=2, sharex=True, figsize=(figsize[0], figsize[1] / 2))

    for test_name in tests:
        test_results = results.loc[test_name]
        axes[0].plot(xval, test_results['error1'], label=test_name)
        axes[1].plot(xval, test_results['error2'], label=test_name)
    axes[1].legend()
    axes[0].axhline(alpha, linestyle='--', color='red')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('$\\alpha$')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('$\\beta$')
    axes[0].set_title('Type I Error')
    axes[1].set_title('Type II Error')
    fig.tight_layout()
    return fig


def plot_time(results: DataFrame):
    tests = results.index.levels[0].values
    xval = results.index.levels[1].values
    xlabel = results.index.names[1].capitalize()

    fig = plt.figure()
    ax = fig.gca()
    for test_name in tests:
        test_results = results.loc[test_name]
        ax.plot(xval, test_results['time'], label=test_name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Seconds')
    fig.legend()
    fig.tight_layout()
    return fig
