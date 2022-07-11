from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas import DataFrame


def plot_divergences(dimensions: np.ndarray, divergences: np.ndarray, mean: float, std: float,
                     ax: plt.Axes = None, label: str = None) -> plt.Figure:
    if ax is None:
        ax = plt.gca()
    ax.plot(dimensions, divergences[:, 0], label=f'{label} KL-Divergence', color='#4c72b0')
    ax.fill_between(dimensions, divergences[:, 0] - divergences[:, 1], divergences[:, 0] + divergences[:, 1],
                    label=f'{label} $\\pm \\sigma$', color='#6ea5ff')
    ax.axhline(mean, linestyle='--', c='red', label='Original KL-Divergence')
    ax.axhline(mean - std, linestyle='--', c='pink')
    ax.axhline(mean + std, linestyle='--', c='pink')
    ax.legend()
    return plt.gcf()


def plot_errors(results: DataFrame, *, alpha: float = 0.1, logscale: bool = False, show_time: bool = False,
                legend: bool = True, fig: plt.Figure = None, axes: Optional[List[plt.Axes]] = None):
    tests = results.index.levels[0].values
    xval = results.index.levels[1].values
    xlabel = results.index.names[1].capitalize()

    figsize = plt.rcParams['figure.figsize']
    nrows = 3 if show_time else 2

    if fig is None:
        fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(figsize[0] / 2., figsize[1]))

    for test_name in tests:
        test_results = results.loc[test_name]
        axes[0].plot(xval, test_results['error1'], label=test_name)
        axes[1].plot(xval, test_results['error2'], label=test_name)
        if show_time:
            axes[2].plot(xval, test_results['time'], label=test_name)

    if legend:
        axes[1].legend()

    axes[0].axhline(alpha, linestyle='--', color='red')
    axes[0].set_ylabel('Type I')
    axes[1].set_ylabel('Type II')
    if show_time:
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel('Time (s)')
    axes[-1].set_xlabel(xlabel)

    if logscale:
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        if show_time:
            axes[2].set_xscale('log')

    for ax in axes[:-1]:
        ax.xaxis.set_ticklabels([])

    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def plot_time(results: DataFrame, logscale: bool = False):
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
    if logscale:
        ax.set_xscale('log')
    fig.legend()
    fig.tight_layout()
    return fig


def lock_axes(*axes):
    aux = np.array([ax.get_ylim() for ax in axes])
    ymin = np.min(aux[:, 0])
    ymax = np.max(aux[:, 1])
    for ax in axes:
        ax.set_ylim(ymin, ymax)


def plot_sample_and_dim(samples: DataFrame, dims: DataFrame, title: str) -> plt.Figure:
    gs = GridSpec(nrows=4, ncols=2, height_ratios=[0.1, 0.3, 0.3, 0.3], hspace=0.1)

    fig = plt.figure()

    axes_samples = [fig.add_subplot(gs[1 + i, 0]) for i in range(3)]
    _ = plot_errors(samples, logscale=True, show_time=True, legend=False, fig=fig, axes=axes_samples)

    axes_dim = [fig.add_subplot(gs[1 + i, 1]) for i in range(3)]
    _ = plot_errors(dims, logscale=True, show_time=True, legend=False, fig=fig, axes=axes_dim)

    for ax in axes_dim:
        ax.set_ylabel(None)

    lock_axes(axes_samples[0], axes_dim[0])
    lock_axes(axes_samples[1], axes_dim[1])

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.legend(handles=axes_dim[0].get_legend_handles_labels()[0], loc='center', ncol=4)
    ax_legend.set_axis_off()

    ax_legend.set_title(title)
    return fig
