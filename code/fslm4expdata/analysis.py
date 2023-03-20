from multiprocessing.sharedctypes import Value
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fslm.analysis import plot_iqrchanges, compare_iqr_ratios
from fslm4expdata.hh_simulator import HHSimulator
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.pyplot import Axes, cm
from matplotlib.image import AxesImage
from pandas import DataFrame
from matplotlib.cm import ScalarMappable
import matplotlib as mpl

from typing import Dict, Iterable, List, Optional, Tuple
from torch import Tensor
from matplotlib.pyplot import Axes

hh = HHSimulator()
ft_labels = hh.features()
param_labels = hh.parameters()


def plot_agg_fts_searches_bar(fts_searches: List[Dict], ax: Axes = None, add_cbar: bool = False, sort_by: str = "mean", monotone=False) -> Tuple[Axes, ScalarMappable]:
    """Plots several tree-searches. Order of features is colorcoded.

    Plots frequencies (y) with which a given feautre (x) was selected as the 1st,
    2nd, ... feature (z).

    Args:
        fts_searches: List that contains dictionaries with the results of each
            tree search, i.e. d1 = {0: kl_0, 2: kl_2, 1: kl_1}.
        ax: Axes object to plot onto.
        add_cbar: Whether to add a colorbar to the plot.
        sort_by: Change how to arange the barplot. 'mean': By average selection position
            during the tree search. 'max': Or in order of most frequently selected.
            'freq': by total frequency.
            as 1st, then 2nd, then 3rd, ...
        monotone: remove cmap from bars.
    Returns:
        Axes object.
        ScalarMappable object, i.e. to use for adding a colorbar later.
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6,4))
    
    fts = sum([sorted(fts) for fts in fts_searches], [])
    fts = list(set(fts))
    num_fts = len(fts)
    num_trials = len(fts_searches)
    depth = max([len(search) for search in fts_searches])
    selection_orders = torch.tensor([list(trial) for trial in fts_searches])
    counts = torch.vstack([order.bincount(minlength=23) for order in selection_orders.T])
    counts = counts[:,fts] # row = order, col = ft, cell = freq

    if "max" in sort_by.lower():
        sorted_df = DataFrame(counts).T.sort_values(by=list(range(depth)),ascending=False).T
        sorted_idx = sorted_df.columns.tolist()
        sorted_fts = torch.tensor(fts)[sorted_idx]
        sorted_counts = torch.from_numpy(sorted_df.values)
    elif "mean" in sort_by.lower():
        sorted_idx = torch.argsort(counts.T@torch.arange(depth+1,1,-1), descending=True)
        sorted_fts = torch.tensor(fts)[sorted_idx]
        sorted_counts = counts[:, sorted_idx]
    elif "freq" in sort_by.lower():
        sorted_df = DataFrame(torch.vstack([counts.sum(0),counts])).T.sort_values(by=list(range(depth)),ascending=False).T; sorted_df
        sorted_idx = sorted_df.columns.tolist()
        sorted_fts = torch.tensor(fts)[sorted_idx]
        sorted_counts = torch.from_numpy(sorted_df.loc[1:].values)
    else:
        raise ValueError("only 'max' and 'mean' are supported.")

    cumcounts = torch.vstack([torch.zeros((num_fts,)),sorted_counts.cumsum(dim=0)])
    
    if monotone:
        cmap = ListedColormap(['tab:blue'])
    else:
        cmap = cm.get_cmap("Reds_r")
        cmap = mpl.colors.LinearSegmentedColormap.from_list("reds_r 5", cmap((torch.linspace(0,0.8,depth).numpy())), 5)
    
    for i in range(depth):
        ax.bar(range(num_fts),sorted_counts[i], bottom=cumcounts[i], color=cmap(i/depth), width=0.9)
        ax.set_xticks(range(num_fts), [HHSimulator().features()[ft] for ft in sorted_fts.tolist()], rotation=90)
        ax.set_xlim(-0.5,len(fts) + 0.5)
        ax.set_xlabel("Selected Feature")
        ax.set_ylabel("Frequency")

    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,depth-1))

    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)

        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Selected as", rotation=90)
        cbar.ax.get_yaxis().labelpad = 10
        loc = torch.arange(0, depth, 1).numpy() + torch.linspace(0.5, -0.5, depth).numpy()
        cbar.set_ticks(loc)
        cbar.set_ticklabels(
        ["1$^{st}$", "2$^{nd}$", "3$^{rd}$"]
        + [f"{x}" + "$^{th}$" for x in range(4, depth+1)]
    )
    return ax, sm


def plot_agg_fts_searches_im(fts_searches: List[Dict], ax: Optional[Axes] = None, add_cbar: bool = False) -> Tuple[Axes, AxesImage]:
    """Plots several tree-searches. Order of features is colorcoded.

    Plots features (y) and their selection index (z) for different runs (x).
    (x,y,z) is plotted as imshow.

    Args:
        fts_searches: List that contains dictionaries with the results of each
            tree search, i.e. d1 = {0: kl_0, 2: kl_2, 1: kl_1}.
        ax: Axes object to plot onto.
        add_cbar: Whether to add a colorbar to the plot.
    Returns:
        Axes object.
        AxesImage object, i.e. for adding a colorbar later. 
    """
    fts_searches = torch.tensor([list(tree.keys()) for tree in fts_searches])
    rand_run = fts_searches[0]
    used_fts = sorted(rand_run)

    num_runs = len(fts_searches)
    num_fts = len(rand_run)
    hist = torch.zeros(num_fts, num_runs)
    for ft_idx, ft in enumerate(used_fts):
        for run in range(num_runs):
            hist[ft_idx, run] = (
                torch.arange(num_fts)[fts_searches[run] == ft]
            ).sum()  # WHY SUM?

    idx_hist = torch.hstack([torch.tensor([used_fts]).T, hist])
    new_idx = torch.argsort(idx_hist[:, 1:].mean(1), descending=True)
    sorted_hist, new_ft_idx = idx_hist[new_idx, 1:], idx_hist[new_idx, 0]
    new_idx = torch.argsort(sorted_hist[-4:, :].sum(0))
    resorted_hist, new_x_idx = sorted_hist[:, new_idx], sorted_hist[0, new_idx]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3))
    cmap = cm.get_cmap("RdYlBu", 10)
    im = ax.imshow(resorted_hist, cmap=cmap)
    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(r"Selected as", rotation=90)
        loc = torch.arange(0, 10, 1).numpy() + torch.linspace(0.5, -0.5, 10).numpy()
        cbar.set_ticks(loc)
        cbar.set_ticklabels(
            [r"1$^{st}$", r"2$^{nd}$", r"3$^{rd}$"]
            + [f"{x}" + r"$^{th}$" for x in range(4, 11)]
        )
    ax.set_yticks(range(num_fts))
    # ax.set_yticklabels([hh.features()[int(x)] for x in new_ft_idx])
    ax.set_yticklabels(
        [r"$\mathrm{" + ft_labels[int(x)][1:-1] + r"}$" for x in new_ft_idx]
    )
    ax.set_xticks(range(num_runs))
    ax.set_xticklabels(range(1, num_runs + 1))
    ax.set_xlabel(r"Run")
    ax.set_ylabel(r"Feature")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    return ax, im


def plot_ft_search_results(best_fts: Dict, baseline_fts: Optional[Dict] = None):
    """Plots trajectory of single tree search.

    Args:
        best_fts: Dictionary containing as keys the position at which a feature
            was seleted and as value its kl.
            Example: d1 = {0: kl_0, 2: kl_2, 1: kl_1}
        baseline_fts: Dictionary containing as keys the position at which a feature
            was seleted and as value its kl of a comparaitve baseline search."""
    xticklabels = [ft_labels[ft] for ft in best_fts.keys()]
    xticklabels = [r"$\mathrm{" + x[1:-1] + r"}$" for x in xticklabels]
    fig, ax1 = plt.subplots(figsize=(4.7, 2.1))
    ax1.plot(
        range(len(best_fts)), best_fts.values(), ".-", label="Best", c="black", ms=6
    )
    ax1.set_ylabel(r"$\mathcal{D}_{KL}[\hat{p}\vert\vert \hat{p}_{\backslash i}]$")
    ax1.set_xlabel(r"Selected features")

    ax1.set_xticks(range(len(best_fts)))
    ax1.set_xticklabels(xticklabels, rotation=90, color="black")
    ax1.set_xlim(-0.3, len(best_fts) + 0.3)

    if baseline_fts is not None:
        xticklabels_baseline = [ft_labels[ft] for ft in baseline_fts.keys()]
        xticklabels_baseline = [
            r"$\mathrm{" + x[1:-1] + r"}$" for x in xticklabels_baseline
        ]

        ax1.plot(
            range(len(baseline_fts)),
            baseline_fts.values(),
            ".-",
            label="Random",
            c="tab:blue",
            ms=6,
        )
        ax2 = ax1.twiny()
        ax2.set_xticks(range(len(best_fts)))
        ax2.set_xticklabels(xticklabels_baseline, rotation=90, color="tab:blue")
        ax2.set_xlim(-0.3, 9.3)

    ax1.legend()
    ax1.spines["right"].set_visible(True)
    ax1.spines["top"].set_visible(True)


def plot_dropout_results(
    samples: Tensor, base_sample: Tensor, fts, clims: Tuple[int, int] = [2, 20]
):
    """Plot iqr increases of posterior marginals obtained with a single feature
    missing compared to reference posterior.

    Args:
        samples: Batch of samples from posterior distributions that are missing
            one feature each in order of missing features.
        base_sample: Sample from reference distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    num_params = samples[0].shape[-1]
    num_fts = len(fts)

    hh = HHSimulator()
    fts_labels = hh.features()
    fts_labels = [fts_labels[ft] for ft in fts]

    im = plot_iqrchanges(
        samples, base_sample, zlims=clims, add_cbar=False, agg_with="median"
    )
    ax.set_xticks(range(num_fts))
    ax.set_yticks(range(num_params))
    ax.set_yticklabels(list(hh.parameters(include_units=False).keys()))
    ax.set_xticklabels(fts_labels, rotation=90)
    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="y", which="both", size=0)
    # fig.text(0.49, -0.11, "Removed feature", ha="center", va="center")
    yticktype = type(ax.get_yticks()[0])
    # fig.text(
    #     0.158,
    #     0.5,
    #     "Model parameters",
    #     ha="center",
    #     va="center",
    #     rotation=90,
    # )
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    plt.subplots_adjust(wspace=-0.21, hspace=-0.3)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label(
        r"$\frac{IQR(\hat{p}_{\backslash i})}{IQR(\hat{p})}$",
        rotation=0,
        ha="center",
        va="center",
        size=9,
    )
    vmin, vmax = im.get_clim()

    cticks = (
        torch.logspace(
            torch.log10(torch.tensor(vmin)), torch.log10(torch.tensor(vmax)), 4
        )
        .int()
        .numpy()
    )
    cbar.set_ticks(cticks + [0.2, 0.25, 0.1, -2.2])
    cbar.set_ticklabels(cticks)
    cbar.ax.minorticks_off()


def plot_batched_dropout_results(
    samples_batch: Tensor,
    fts: List[int],
    clims: Tuple[int, int] = [2, 20],
    n_colors: int = 10,
    agg_with: str = "mean",
    ax: Optional[Axes] = None,
) -> Tuple[Axes, AxesImage]:
    """Plot iqr increases of posterior marginals obtained with a single feature
    missing compared to reference posterior.

    Args:
        samples_batch: Batch of samples from posterior distributions that are missing
            one feature each in order of missing features. (batchsize,num_missing_fts,num_samples,num_params)
    fts: list of features used.
    clims: colorbar limits.
    n_colors: how many colors to divide the cbar into.
    agg_with: how to aggregate the batch. median or mean.
    ax: Axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    num_params = samples_batch.shape[-1]
    num_fts = len(fts)

    hh = HHSimulator()
    fts_labels = hh.features()
    fts_labels = [fts_labels[ft] for ft in fts]

    iqr_ratios = []
    for samples in samples_batch:
        iqr_ratios.append(compare_iqr_ratios(samples[1:], samples[0]))
    iqr_ratios = torch.stack(iqr_ratios)
    if "mean" in agg_with.lower():
        IQRs_agg = iqr_ratios.mean(dim=0)
    elif "median" in agg_with.lower():
        IQRs_agg = iqr_ratios.median(dim=0)[0]
    else:
        raise ValueError("Metric is not supported.")

    # ax = plt.gca()
    cmap = cm.get_cmap("Blues", n_colors)
    im = ax.imshow(
        IQRs_agg.numpy(),
        cmap=cmap,
        norm=LogNorm(vmin=clims[0], vmax=clims[1]),
    )

    ax.set_xticks(range(num_fts))
    ax.set_yticks(range(num_params))
    ax.set_yticklabels(list(hh.parameters(include_units=False).keys()))
    ax.set_xticklabels(fts_labels, rotation=90)
    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="y", which="both", size=0)
    # fig.text(0.49, -0.11, "Removed feature", ha="center", va="center")
    yticktype = type(ax.get_yticks()[0])
    # fig.text(
    #     0.158,
    #     0.5,
    #     "Model parameters",
    #     ha="center",
    #     va="center",
    #     rotation=90,
    # )
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)

    # plt.subplots_adjust(wspace=-0.21, hspace=-0.3)

    # cbar = fig.colorbar(im, ax=ax,pad=0.01)
    # cbar.ax.get_yaxis().labelpad = 10
    # cbar.set_label(r"$\frac{IQR(\hat{p}_{\backslash i})}{IQR(\hat{p})}$", rotation=0, ha="center", va="center", size=9)
    # vmin, vmax = im.get_clim()

    # cticks = (
    #     torch.logspace(
    #         torch.log10(torch.tensor(vmin)), torch.log10(torch.tensor(vmax)), 4
    #     )
    #     .int()
    #     .numpy()
    # )
    # cbar.set_ticks(cticks+[0.2,0.25,0.1,-2.2])
    # cbar.set_ticklabels(cticks)
    # cbar.ax.minorticks_off()
    return ax, im
