import re
from typing import Dict, List, Optional, Tuple, Any

import cycler
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.pyplot import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy import array as np_array
from numpy import histogram2d, linspace
from sbi.analysis import pairplot
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from scipy.stats import gaussian_kde

# types
from torch import Tensor

from fslm.metrics import sample_kl, unbiased_mmd_squared
from fslm.snle import ReducablePosterior
from fslm.utils import (
    ints_from_str,
    select_tag,
    skip_dims,
    sort_by_missing_dims,
)

color_theme = plt.cm.Greens(torch.linspace(0, 1, 10))
# color_theme[0] = torch.tensor([225, 0, 0, 255]).numpy() / 255  # ground truth / direct
# color_theme[1] = torch.tensor([31, 119, 180, 255]).numpy() / 255  # post-hoc
# color_theme[2] = torch.tensor([255, 127, 14, 255]).numpy() / 255  # full Orange
# color_theme[0] = torch.tensor([44, 160, 44, 255]).numpy() / 255  # x_o
# color_theme[3] = torch.tensor([23, 190, 207, 255]).numpy() / 255  # direct
# mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", color_theme)
# mpl.rc("image", cmap="Blues")


def contour_pairplot(
    data: List[Tensor] or Tensor,
    levels: int or List[float] = [0.02],
    nbins: int = 50,
    limits: Optional[Tensor] = None,
    figsize: Tuple = (10, 10),
    xlabels: List[str] = None,
):
    """Plots contours for pairs of dimensions for batch of samples drawn from a probability distribution.
    Similar to `sbi.analysis pairplot`.

    Args:
        data: Input data / samples to be plotted. (num_samples, num_dims)
        levels: Height or number of levels of the counter plot.
        nbins: number of bins for the histograms that approximate the distribution.
        limits: The x and y limits for the contour plots. (num_dims, 2) dim0 = min, dim1 = max.
        figsize: Size of the resulting figure.
        xlabels: Axis labels of the plots
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if type(data) != list:
        data = [data]

    if limits == None:
        lower_limits = torch.min(data[0], dim=0)[0]
        upper_limits = torch.max(data[0], dim=0)[0]
        limits = torch.vstack([lower_limits, upper_limits]).T

    nsamples, ndims = data[0].shape
    fig, ax = plt.subplots(ndims, ndims, figsize=figsize)
    for j in range(ndims):
        for i in range(ndims):
            if i == j:
                for x in data:
                    kde = gaussian_kde(x[:, i].numpy())
                    xvals = linspace(limits.numpy()[i, 0], limits.numpy()[i, 1], nbins)
                    ax[i, j].plot(xvals, kde(xvals))

                if xlabels != None:
                    ax[i, j].set_xlabel(xlabels[i])

                ax[i, j].set_xticks(limits.numpy()[i])
                ax[i, j].set_yticks([])

            if i < j:
                for idx, x in enumerate(data):
                    z, x, y = histogram2d(
                        x[:, i].numpy(),
                        x[:, j].numpy(),
                        bins=nbins,
                        range=limits[[i, j]],
                        density=True,
                    )
                    dx = x[1] - x[0]
                    dy = y[1] - y[0]
                    ax[i, j].contour(
                        x[:-1] + dx / 2,
                        y[:-1] + dy / 2,
                        z,
                        levels=levels,
                        colors=colors[idx],
                    )

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            if i > j:
                ax[i, j].set_axis_off()


def coordinate_mesh_2d(
    limits: Tensor, num_points: Tuple = (50, 50)
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create coordinate pairs on a 2D grid of points.

    Args:
        limits: Maximum and minimum values for the grid
            [[min_y, max_y],
             [min_x, max_x]].
        num_points: The number of grid points along each axis.
            If only one axis is specified through an int. It is assumed,
            that the other axis should have an equal number of grid points.

    Returns:
        xs: Grid points on the x-axis.  Shape = (,num_points[0])
        ys: Grid points on the y-axis.  Shape = (,num_points[1])
        coord_pairs: All coordinate pairs on the grid. (x_i,y_j) with (i,j) in
            {0,...,num_points[0]} X {0,...,num_points[1]}. Shape = (2,-1).
    """

    xs = torch.linspace(limits[1, 0], limits[1, 1], num_points[0])
    ys = torch.linspace(limits[0, 0], limits[0, 1], num_points[1])

    meshgrids = torch.meshgrid(ys, xs)

    coord_grid = torch.dstack(meshgrids)
    coord_pairs = coord_grid.reshape(-1, 2)

    return xs, ys, coord_pairs


def eval_grid_2d(
    distribution,
    limits: Tensor,
    context: Optional[Tensor] = None,
    num_grid_points: int or Tuple[int] = 50,
    xdims: List[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluate a posterior or likelihood on a 2D grid of coordinate pairs.

    Provided the likelihood / posterior instance posseses a log_prob method,
    it can be used to be evaluated on grid, given a specific context.

    Args:
        distribution: Distribution instance with .log_prob method. Could for
            example be a MCMCPosterior instance.
        context: Context x of the posterior distribution p(theta | x). If none is
            provided posterior.default_x will be used.
        limits: Maximum and minimum values for the mesh-grid
            [[min_y, max_y],
             [min_x, max_x]].
        num_grid_points: The number of grid points along each axis.
            If only one axis is specified through an int. It is assumed,
            that the other axis should have an equal number of grid points.
        xdims: If specific feature subsets are chosen, a `ReducableLikelihoodbasedPosterior`
            instance will be created and `.log_prob_reduced_features` will be used
            to evaluate the posterior based upon a reduced set of features. This
            only works for SNLE based posteriors at the moment.
    Returns:
        xs: Grid points on the x-axis. Shape = (,num_points[0])
        ys: Grid points on the y-axis. Shape = (,num_points[1])
        probs: Probabilities evaluated for all points on the grid. p(theta_i,theta_j | x)
             with (i,j) in {0,...,num_points[0]} X {0,...,num_points[1]}.
             Shape = (num_points[0],num_points[1]).
    """
    pass  # TODO:
    if type(num_grid_points) == int:
        num_grid_points = (num_grid_points, num_grid_points)

    if type(num_grid_points) == Tuple or type(num_grid_points) == List:
        num_grid_points = tuple(num_grid_points)

    xs, ys, gridpoints = coordinate_mesh_2d(limits, num_points=num_grid_points)

    if context == None:
        assert distribution.default_x != None, "No context has been set."
        context = distribution.default_x

    if xdims == None:
        log_probs = distribution.log_prob(gridpoints, context)
    else:
        assert isinstance(distribution, MCMCPosterior) or isinstance(
            distribution, RejectionPosterior
        ), "Only MCMCPosteriors can be marginalised."
        reducable_posterior = ReducablePosterior(distribution)
        reducable_posterior.marginalise(xdims)
        log_probs = reducable_posterior.log_prob(gridpoints, context=context)

    log_probs = log_probs.reshape(num_grid_points[0], num_grid_points[1])

    return xs, ys, torch.exp(log_probs)


def compare_kls(
    samples: List[Tensor], base_sample: Tensor, samplesize: int = 2000
) -> Tensor:
    """Computes estimate of the KL divergence between samples and samples from a
    reference distribution.

    Args:
        samples: List of samples to compare against a reference distribution.
        base_sample: Samples from a reference distribution to compare the var
            against.
        sample_size: Number of samples to use in estimate of the KL.

    Returns:
        KLs: Estimates of the KL divergence for each sample in the list.
    """
    KLs = []
    for sample in samples:
        if sample.shape[0] < samplesize:
            samplesize = int(sample.shape[0])
        KL_i = sample_kl(sample[:samplesize], base_sample[:samplesize])
        KLs.append(torch.tensor(KL_i))

    KLs = torch.hstack(KLs)
    return KLs


def compare_mmds(
    samples: List[Tensor], base_sample: Tensor, samplesize: int = 2000
) -> Tensor:
    """Computes estimate of the MMD between samples and samples from a
    reference distribution.

    Args:
        samples: List of samples to compare against a reference distribution.
        base_sample: Samples from a reference distribution to compare the var
            against.
        sample_size: Number of samples to use in calculation of the MMD.

    Returns:
        MMDs: Estimates of the MMD for each sample in the list.
    """
    MMDs = []
    for sample in samples:
        if sample.shape[0] < samplesize:
            samplesize = int(sample.shape[0])
        MMD_i = unbiased_mmd_squared(sample[:samplesize], base_sample[:samplesize])
        MMDs.append(MMD_i)

    MMDs = torch.hstack(MMDs)
    return MMDs


def compare_iqr_ratios(samples: List[Tensor], base_sample: Tensor) -> Tensor:
    """Computes ratio of sample sample inter quartile range (IQR) vs IQR of base
    sample for each dimension.

    Args:
        samples: List of samples to compare against a reference distribution.
        base_sample: Samples from a reference distribution to compare the IQR
            against.

    Returns:
        Ratios of inter quartiles for each dimension and sample.
    """
    # cov_base = cov(base_sample.clone())
    # var_base = cov_base.diag()
    iqr_base = base_sample.clone().quantile(0.75, dim=0) - base_sample.clone().quantile(
        0.25, dim=0
    )

    # var_ratios = []
    iqr_ratios = []
    for sample in samples:
        # cov_sample = cov(sample.clone())
        # var_sample = cov_sample.diag()
        iqr_sample = sample.clone().quantile(0.75, dim=0) - sample.clone().quantile(
            0.25, dim=0
        )

        # var_ratios.append(var_sample / var_base)
        iqr_ratios.append(iqr_sample / iqr_base)
    # return torch.vstack(var_ratios).T
    return torch.vstack(iqr_ratios).T


def plot_iqrchanges(
    samples: List[Tensor],
    base_sample: Tensor,
    yticklabels: Optional[str] = None,
    zlims: Optional[Tuple[float, float]] = (None, None),
    agg_with: str = "median",
    add_cbar: bool = True,
    plot_label: str = None,
    batchsize=0,
    n_colors: int = 10,
) -> Axes:
    """Plot changes in IQR per sample dim for a list of samples.

    Args:
        samples: List of samples to compare against a reference distribution.
        base_sample: Samples from a reference distribution to compare the var
            against.
        yticklabels: Label for the yticks, i.e. parameter labels.
        zlims: Sets the limits for the colorbar.
        agg_with: How to aggregate the samples. Mean or Median are valid.
        add_cbar: Whether to add a colorbar to the plot
        batchsize: In case multiple arguments are supplied. Have to be in order!
            i.e. [[0,1,2],[0,1,2]] would supply two samples of batchsize 3.

    Returns:
        ax: plot axes.
    """
    if batchsize > 0:
        # TODO: IF BASESAMPLE ALSO HAS BATCHES -> split and align them!
        batched_samples = samples.split(batchsize)
        IQRs = []
        for i, samples in enumerate(batched_samples):
            rel_IQR = compare_iqr_ratios(list(samples), base_sample)
            IQRs.append(rel_IQR)

        IQRs = torch.stack(IQRs)
        if "mean" in agg_with.lower():
            IQRs_agg = IQRs.mean(0)
        if "median" in agg_with.lower():
            IQRs_agg = IQRs.median(0)[0]
        # Vars_std = Vars.std(0)
    else:
        IQRs_agg = compare_iqr_ratios(samples, base_sample)

    ax = plt.gca()
    cmap = cm.get_cmap("Blues", n_colors)
    im = ax.imshow(
        IQRs_agg.numpy(), cmap=cmap, norm=LogNorm(vmin=zlims[0], vmax=zlims[1]),
    )
    # im = ax.imshow(Vars_agg.numpy(), vmin=1, vmax=zlims[1])
    ax.set_title(plot_label)
    if yticklabels != None:
        yrange = range(IQRs_agg.numpy().shape[0])
        plt.yticks(yrange, yticklabels)

    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
    return im


def plot_kls(
    samples: List[Tensor],
    base_sample: Tensor,
    samplesize: int = 2000,
    plot_label: Optional[str] = None,
    idx: int = 0,
    kind: str = "bar",
    agg_with: str = "mean",
    batchsize=0,
    n_idxs: int = 1,
) -> Axes:
    """Plot changes in variance per sample dim for a list of samples.

    Args:
        samples: List of samples to compare against a reference distribution.
        base_sample: Samples from a reference distribution to compare the var
            against.
        sample_size: Number of samples to use in estimate of the KL.
        plot_label: Name of the figure.
        idx: idx, decides translation of bars in plot.
        kind: Which kind of plot to use.
        agg_with: How to aggregate the samples. Mean or Median are valid.
        batchsize: In case multiple arguments are supplied. Have to be in order!
            i.e. [[0,1,2],[0,1,2]] would supply two samples of batchsize 3.

    Returns:
        ax: plot axes.
    """

    if batchsize > 0:
        # TODO: IF BASESAMPLE ALSO HAS BATCHES -> split and align them!
        batched_samples = samples.split(batchsize)
        KLs = []
        for i, samples in enumerate(batched_samples):
            rel_KL = compare_kls(list(samples), base_sample, samplesize)
            KLs.append(rel_KL)
        KLs = torch.vstack(KLs)
        if "mean" in agg_with.lower():
            KLs_agg = KLs.mean(0)
            KLs_disp = KLs.std(0)
        if "median" in agg_with.lower():
            KLs_agg = KLs.median(0)[0]
            KLs_disp_lower = KLs.median(0)[0] - KLs.quantile(0.25, 0)
            KLs_disp_upper = KLs.quantile(0.75, 0) - KLs.median(0)[0]
            KLs_disp = torch.vstack([KLs_disp_lower, KLs_disp_upper])
        if "box" in kind.lower():
            KLs_agg = KLs.T
    else:
        KLs_agg = compare_kls(samples, base_sample, samplesize)
        KLs_disp = torch.zeros(2, len(KLs_agg))

    N = len(KLs_agg)
    if "bar" in kind.lower():
        ax = plt.bar(
            (torch.arange(N) + idx / (n_idxs + 2) - 1 / (n_idxs + 1)).numpy(),
            height=KLs_agg.numpy(),
            label=plot_label,
            align="edge",
            width=1 / (n_idxs + 2),
            yerr=KLs_disp.numpy(),
        )
    elif "points" in kind.lower():
        ax = plt.errorbar(
            range(N),
            KLs_agg.numpy(),
            yerr=KLs_disp,
            ls="",
            marker=".",
            label=plot_label,
        )
    elif "box" in kind.lower():
        if batchsize > 0:
            ax = plt.boxplot(
                KLs_agg.T.numpy(),
                positions=(torch.arange(N) + idx / (n_idxs) - 1 / (n_idxs + 1)).numpy(),
                widths=1 / (n_idxs),
                patch_artist=True,
                # boxprops={"color":color_theme[idx], "lw":1.5},
                # whiskerprops={"color":"black", "lw":1.5},
                # capprops={"color":color[idx], "lw":1.5},
                # flierprops={"markeredgecolor":color[idx]},
                medianprops={"color": "black"},
            )
            for patch in ax["boxes"]:
                patch.set_facecolor(color_theme[idx])
        else:
            ax = plt.plot(
                (torch.arange(N) + idx / (n_idxs + 2) - 1 / (n_idxs + 1)).numpy(),
                KLs_agg.numpy(),
                ls="",
                marker=">",
                c=color_theme[idx],
                mew=1,
                ms=6,
                label=plot_label,
            )
    plt.ylabel(r"$D_{KL}$")
    return ax


def plot_change_in_uncertainties(
    samples: List[Tensor] or Dict[str, Tensor],
    tags: Optional[List[str]] = None,
    metric: str = "KL",
    agg_with: str = "mean",
    base_sample: Optional[Tensor] = "auto",
    figsize: Tuple[int, int] = (10, 10),
    plot_title: str = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plot changes in variance per sample dim for a list of samples.
    Args:
        samples: List or Dictionary of samples to compare against a reference distribution.
        tags: List of dict labels to select samples from. If samples is a list
            the tags are supplied to a dict constructor together with samples.
        metric: Which metric to measure increased uncertainy against a base dist
            with.
        agg_with: How to aggregate the samples. Mean or Median are valid.
        base_sample: Samples from a reference distribution to compare the var
            against. If 'auto' is selected, reference distribution is chosen as
            the one with the most feature dimensions, based on its label.
        figsize: Specify size of the figure.
        plot_title: Set title for figure.
        kwargs: Includes kwargs that are passed down to `plot_iqrchanges` or
        `plot_kls` depending on the metric that was chosen.

    Returns:
        axes: plot axes.
    """
    includes_base_dist = base_sample != "auto"
    # caches number of features and base dist dict key
    max_range = 0
    base_key = None

    if type(samples) == list:
        if tags == None:
            # auto geenerate tags assuming missing dims and ordered list
            dims = list(range(len(samples)))
            skipped_dims = skip_dims(dims, 1)
            tags = ["auto_tagged_" + str(dims) for dims in skipped_dims]
        else:
            assert len(tags) == len(
                samples
            ), "There needs to be a tag associated with every sample."
        samples = dict(zip(tags, samples))

    elif type(samples) == dict:
        if tags == None:
            tags = list(samples.keys())

    for tag in tags:
        samples_selected = select_tag(samples, tag)

        # base bist assumed to be dist with most dims
        if not includes_base_dist:
            maxlen = 0
            minlen = 0
            for i, label in enumerate(samples_selected.keys()):
                match = re.search(tag, label)
                ndims = len(ints_from_str(label[match.span()[1] :]))
                if ndims > maxlen:
                    maxlen = ndims
                    base_key = label
                    if max_range < maxlen:
                        max_range = maxlen
                if ndims < minlen or i == 0:
                    minlen = ndims
            includes_base_dist += minlen != maxlen
            base_sample = samples[base_key]

    assert (
        includes_base_dist == 1
    ), "The selected samples include none or more than 1 base distributions to\
        compare to. Please select one or remove the {} of them.".format(
        includes_base_dist - 1
    )

    ncols = 2
    nrows = 1
    # if "var" in metric.lower():
    #     if len(tags) > 2:
    #         plot_sizes = torch.arange(10) ** 2
    #         diffs = plot_sizes - len(tags)
    #         idx = torch.min(torch.arange(10)[diffs >= 0])
    #         nrows = int(torch.sqrt(plot_sizes[idx]))
    #         ncols = nrows
    #     else:
    #         ncols = len(tags)
    #     nrows = len(tags)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if ncols > 1:
        axes = axes.reshape(-1)

    # COUNT BATCHES
    for i, tag in enumerate(tags):
        batchsize = 0
        samples_selected = select_tag(samples, tag)
        old_keys = list(samples_selected.keys())
        for old_key in old_keys:
            match = re.search(tag, old_key)
            new_key = old_key[match.span()[1] :]
            k = 0
            idx_str = "(%s)" % k
            while idx_str + new_key in samples_selected.keys():
                k += 1
                idx_str = "(%s)" % k
            samples_selected[idx_str + new_key] = samples_selected.pop(old_key)
        labels, data = sort_by_missing_dims(samples_selected, range(max_range))
        if labels.shape[0] > max_range + 1:
            batchsize = max_range

        if -1 in labels:
            data = data[torch.tensor(labels) != -1]
        xticklabels = [r"$x_{}$".format("{%s}" % i) for i in range(max_range)]
        yticklabels = [r"$\theta_{}$".format(i) for i in range(base_sample.shape[1])]

        # if "plot_label" not in kwargs:
        #     kwargs["plot_label"] = tag[:-1]
        # else:
        #     if i == 0:
        #         legend_labels = kwargs["plot_label"]
        #     kwargs["plot_label"] = legend_labels[i]

        if "var" in metric.lower():
            ax = plt.axes(axes[i])
            if "zlims" in kwargs:
                im = plot_iqrchanges(
                    data,
                    base_sample,
                    add_cbar=False,
                    batchsize=batchsize,
                    agg_with=agg_with,
                    **kwargs,
                )
            else:
                plot_iqrchanges(
                    data, base_sample, batchsize=batchsize, agg_with=agg_with, **kwargs
                )
            fig.text(0.5, 0.04, "missing feature", ha="center", va="center", size=14)
            ax.set_xticks(range(max_range))
            ax.set_xticklabels(xticklabels)
            yticktype = type(ax.get_yticks()[0])
            fig.text(
                0.05,
                0.5,
                "model parameters",
                ha="center",
                va="center",
                rotation=90,
                size=14,
            )
            if yticktype != str:
                ax.set_yticks(range(base_sample.shape[1]))
                ax.set_yticklabels(yticklabels)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(ylabel, size=14)
            ax.tick_params(axis="both", which="major", labelsize=13)

        elif "kl" in metric.lower():
            # ax = plt.axes(axes[i])
            plot_kls(
                data,
                base_sample,
                idx=i,
                batchsize=batchsize,
                agg_with=agg_with,
                **kwargs,
            )
            ax.legend(legend_labels)
            ax.set_xticks(range(max_range))
            plt.suptitle(plot_title)
            ax.set_xlabel("missing feature", size=14)
            ax.set_xticklabels(xticklabels)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(ylabel, size=14)

    if "zlims" in kwargs:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(r"$\frac{Var(\bar{p})}{Var(p)}$", size=16, rotation=0)
        vmin, vmax = im.get_clim()

        cticks = (
            torch.logspace(
                torch.log10(torch.tensor(vmin)), torch.log10(torch.tensor(vmax)), 4
            )
            .int()
            .numpy()
        )
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(cticks)
        cbar.ax.minorticks_off()

    for i in range(len(tags), ncols * nrows):
        fig.delaxes(axes[i])

    # if batchsize > 0 and "plot_label" in kwargs and "kl" in metric.lower():
    #     if "mean" in agg_with.lower(): legend_labels += [r"$\pm \sigma$"]
    #     if "median" in agg_with.lower(): legend_labels += [r"$Q_3$"]
    #     ax.legend([], legend_labels)

    return fig, axes


def plot_feature_space(
    xs: List[Tensor] or Tensor,
    figsize: Tuple[int, int] = (10, 10),
    xlimits: Optional[List[Tuple]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Plots histograms for each dimension of a sample.

    Args:
        xs: Sample of shape (num_samples, num_dims).
        figisze: Determines size of figure.
        limits: edges of the plots.
        kwargs: kwargs passed to `plt.hist`.

    Returns:
        fig: Figure object.
        axes: Axes object."""

    if type(xs) != list:
        xs = [xs]

    # select smalles possible n by n subplot arangement
    nsamples, ndims = xs[0].shape
    plot_sizes = torch.arange(10) ** 2
    diffs = plot_sizes - ndims
    idx = torch.min(torch.arange(10)[diffs >= 0])
    m = int(torch.sqrt(plot_sizes[idx]))

    fig, axes = plt.subplots(m, m, figsize=figsize)
    axes = axes.reshape(-1)
    if xlimits == None:
        xlimits = [None] * ndims

    for x in xs:
        for i, ax in enumerate(axes[:ndims]):
            ax.hist(x[:, i].numpy(), range=xlimits[i], **kwargs)
            ax.set_yticks([])
    for ax in axes[ndims:]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    return fig, axes


def contour_kde_pairplot(
    data: Tensor or List[Tensor],
    limits: Optional[Tuple[float, float]] = None,
    theta_o: Tensor = None,
    labels: List[str] = None,
    prior: Any = None,
    legend_labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    suptitle: Optional[str] = None,
    levels: List[float] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """ "Creates contour plots and marginal kde plots, for list of samples.

    Args:
        data: List of samples from different distributions (num_samples, num_dims).
        limits: xlimits for the marginals can be added explicitely in the form of
            [(lbound1, ubound1), (lbound2, ubound2), ...].
        theta_o: Can be provided to plot points on the contours.
        labels: Labels for the different axis / dims.
        legend_labels: Label for each sample is put into the legend.
        figisze: Determines size of figure.
        suptitle: Name of the whole figure.
        levels: Levels for each contour line that is plot.
        kwargs: kwargs passed to `sbi.analysis.pairplot`.

    Returns:
        fig: Figure object.
        axes: Axes object.
    """

    if prior != None:
        lbound = prior.base_dist.low
        ubound = prior.base_dist.high
        limits = list(zip(lbound, ubound))

    if levels == None:
        levels = [0.2, 0.6, 0.99]
    fig, axes = pairplot(
        data,
        upper="contour",
        diag="kde",
        limits=limits,
        ticks=limits,
        points=theta_o,
        labels=labels,
        points_offdiag={"markersize": 8, "mew": 2.5, "marker": "x", "lw": 1.5},
        points_colors=["black"],
        contour_offdiag={"levels": levels},
        figsize=figsize,
        **kwargs,
    )
    for axe in axes:
        for ax in axe:
            plt.setp(ax.spines.values(), linewidth=2.5)
            plt.setp(ax._get_lines(), linewidth=20.5)
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right", wrap=True)
            ax.tick_params(axis="x", width=2.5, labelsize=13, length=6)
            ax.xaxis.label.set_size(15)
    if legend_labels != None:
        fig.legend(legend_labels, loc=(0.1, 0.1), fontsize=12)
    plt.suptitle(suptitle, size=16)
    return fig, axes


def plot_uncertainty_change_avg(
    tagged_samples: Dict,
    tag: Optional[str] = "",
    subset_size: Optional[int] = None,
    feature_labels: Optional[List[str]] = None,
    param_labels: Optional[List[str]] = None,
    metric: str = "var",
    agg_with: str = "median",
    skip_features: Optional[List[int]] = [],
    remove_features: Optional[List[int]] = [],
    figsize: Tuple[int] = (12, 5),
    print_feature_counts: bool = True,
) -> Tuple[Figure, Axes]:
    """Pltos accumulated uncertainty changes for all features and parameters.
    Depending on the metric used, uncertainty changes calculated with `compare_iqr_ratios`
    or `compare_kls` are accumulated over several subsets of featurs and averaged
    over features in general. The final plot enables to compare feature contributions
    over a large set of features even though they have been obtained only on
    subsets.
    Args:
        tagged_samples: Dictionary of posterior samples with tags containing:
            1. Idxs of features used, i.e. [12,15,18]
            2. Idxs of features that are part of the posterior estimate, i.e. [0,2]
            The tags can also contain some other descriptive string.
        tag: Picks up on other descriptive tags and only selects those, i.e. direct vs posthoc.
        subset_size: Only use features subsets of a fixed size, if the sample contains
            subsets of different sizes as well. If not selected, all subset sizes
            will be selected.
        feature_labels: Can be used to supply x labels to the plot.
        param_labels: Can be used to supply y labels to the plot.
        metric: Wether to use `compare_iqr_ratios` or `compare_kls` to gauge feature
            contributions.
        agg_with: Choose method to aggregate different estimates with.
            1. mean
            2. binned
            3. median
        skip_features: Determines if featuresets that contain certain features
            are omitted. This allows to itteratively drop the most prominent feature.
        remove_features: Removes features from the plot, only at the end! This is
            not the same as skip_features!
        figsize: Resize plot.
        print_feature_counts: prints out the counts of samples that include each
            feature.

    Returns:
        fig: Figure.
        Axes: Axes object.
    """
    used_twice = [x for x in remove_features if x in skip_features]
    assert (
        len(used_twice) == 0
    ), "%s are used in remove_fts and skip_fts! This does not work." % str(used_twice)

    feature_sets = []
    num_params = 0
    for key in tagged_samples.keys():
        match = re.findall("(\[.+\]).*(\[.+\])$", key)
        matching_tags = re.search("(\[.+\]).*(\[.+\])$", key)
        if num_params == 0 and matching_tags != None:
            num_params = tagged_samples[matching_tags.string].shape[1]
        if num_params != 0 and matching_tags != None:
            assert tagged_samples[matching_tags.string].shape[1] == num_params, (
                "Not all samples have param dim %s." % num_params
            )

        if match != []:
            feature_sets.append(match[0][0])

    feature_sets = list(set(feature_sets))
    feature_sets = [ints_from_str(subset) for subset in feature_sets]

    new_feature_sets = feature_sets.copy()
    for ft in skip_features:
        new_feature_sets = [subset for subset in new_feature_sets if ft not in subset]
    feature_sets = new_feature_sets.copy()

    cum_features = [y for x in feature_sets for y in x]
    all_features = list(set(cum_features))
    feature_counts = [cum_features.count(i) for i in all_features].copy()
    list2regex = lambda l: "\[" + str(l)[1:-1] + "\]"

    if print_feature_counts:
        stacked_counts = torch.vstack(
            [torch.tensor(all_features), torch.tensor(feature_counts)]
        )
        print(stacked_counts)

    M = {k: [] for k in all_features}
    Xs_centre = torch.zeros(1, len(all_features))
    Xs_disp = torch.zeros(2, 1, len(all_features))
    if "var" in metric.lower():
        Xs_centre = Xs_centre.repeat(num_params, 1)
        Xs_disp = Xs_disp.repeat(1, num_params, 1)

    if subset_size == None:
        subset_sizes = [len(subset) for subset in feature_sets]
    else:
        subset_sizes = [subset_size]

    for size in subset_sizes:
        for features in feature_sets:
            if len(features) == size:
                base_tag = tag + ".+" + list2regex(features)
                selection = select_tag(tagged_samples, base_tag)
                base_selection = select_tag(
                    tagged_samples,
                    list2regex(features) + ".+" + list2regex(list(range(size))),
                )
                try:
                    selection.pop(list(base_selection.keys())[0])
                except KeyError:
                    base_key = list(base_selection.keys())[0]
                    # print("basesamples selected from different key: %s." %base_key[:base_key.find("[")])

                old_keys = list(selection.keys())
                for old_key in old_keys:
                    match = re.search(base_tag, old_key)
                    new_key = old_key[match.span()[1] :]
                    k = 0
                    idx_str = "(%s)" % k
                    while idx_str + new_key in selection.keys():
                        k += 1
                        idx_str = "(%s)" % k
                    selection[idx_str + new_key] = selection.pop(old_key)

                num_samples = [
                    data.shape[0] for data in {**selection, **base_selection}.values()
                ]
                min_len = min(num_samples)
                for key, vals in selection.items():
                    selection[key] = vals[:min_len]
                for key, vals in base_selection.items():
                    base_selection[key] = vals[:min_len]
                # print(features)
                labels, data = sort_by_missing_dims(selection, range(size))

                if "var" in metric.lower():
                    x = compare_iqr_ratios(data, list(base_selection.values())[0])

                if "kl" in metric.lower():
                    x = compare_kls(
                        data, list(base_selection.values())[0], samplesize=min_len
                    )
                    x = x.view(1, -1)

                if "bin" in agg_with.lower():
                    x_greater_mean = x > x.mean()
                    x[x_greater_mean] = 1
                    x[~x_greater_mean] = 0
                if (
                    x.shape[1] == size
                ):  # TODO: THIS IS VERY HACKY AND ONLY TEMPORARY !!!!! FILTERS OUT MISTAKES MADE FROM SELECTION OF LABELS AND EXTRACTION OF MISSING FEATURES
                    for idx, ft in enumerate(features):
                        M[ft].append(x[:, idx])

    if "median" in agg_with.lower():
        for idx, key in enumerate(M.keys()):
            x = torch.vstack(M[key])
            Xs_centre[:, idx] = x.median(0)[0]
            Xs_disp_lower = x.median(0)[0] - x.quantile(0.25, 0)
            Xs_disp_upper = x.quantile(0.75, 0) - x.median(0)[0]
            Xs_disp[:, :, idx] = torch.vstack([Xs_disp_lower, Xs_disp_upper])

    if "mean" in agg_with.lower():
        for idx, key in enumerate(M.keys()):
            x = torch.vstack(M[key])
            Xs_centre[:, idx] = x.mean(0)
            Xs_disp[0, :, idx] = x.std(0)
            Xs_disp[1, :, idx] = x.std(0)

    plot_fts = []
    for idx, ft in enumerate(all_features):
        if ft not in remove_features:
            plot_fts.append(idx)

    fig, ax = plt.subplots(figsize=figsize)
    if "var" in metric.lower():
        cmap = cm.get_cmap("coolwarm", 10)
        zmax = torch.round(Xs_centre[:, plot_fts].max())
        im = ax.imshow(
            Xs_centre[:, plot_fts].numpy(),
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=zmax),
            vmin=1,
            vmax=zmax,
        )
        if param_labels == None:
            param_labels = [r"$\theta_{%s}$" % i for i in range(num_params)]

        ax.set_yticks(range(num_params))
        ax.set_yticklabels(param_labels)
        # ax.set_title("MISSING TITLE")

        cbar = fig.colorbar(im, ax=ax, pad=0.03)
        cbar.ax.get_yaxis().labelpad = 16
        cbar.set_label(
            r"$\frac{Var(\bar{p})}{Var(p)}$", size=14, rotation=0
        )  # TODO: FIND BETTER LABEL
        vmin, vmax = im.get_clim()
        cticks = (
            torch.logspace(
                torch.log10(torch.tensor(vmin)), torch.log10(torch.tensor(vmax)), 4
            )
            .int()
            .numpy()
        )
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(cticks)
        cbar.ax.minorticks_off()

    if "kl" in metric.lower():
        ax.bar(
            range(len(plot_fts)),
            Xs_centre.view(-1)[plot_fts].numpy(),
            yerr=Xs_disp.squeeze()[:, plot_fts].numpy(),
        )
        # ax.set_title("MISSING TITLE")

    degree = 90
    if feature_labels == None:
        degree = 0
        feature_labels = [r"$x_{%s}$" % i for i in range(len(plot_fts))]
    elif len(feature_labels) > len(plot_fts):
        feature_labels = feature_labels.copy()
        for ft in sorted(skip_features + remove_features)[
            ::-1
        ]:  # so idxs are not changed b4 val is popped
            feature_labels.pop(ft)

    ax.set_xticks(range(len(plot_fts)))
    ax.set_xticklabels(feature_labels, rotation=degree)

    ax.set_ylabel("model parameter", size=14)
    ax.set_xlabel("missing feature", size=14)

    return fig, ax


def plot_vtrace_comparison(
    t: Tensor,
    Vt: Tensor,
    figsize: Tuple[int, int] = (10, 3),
    title: str = "",
    timewindow: Tuple[int, int] = None,
    style: str = "stacked",
    show_scale: bool = True,
    show_stim: bool = True,
    compare2sample0: Optional[str] = "",
    line_colors: List[str] = ["grey"],
    line_widths: List[int] = [1],
    **plot_kwargs,
) -> Tuple[Figure, Axes]:
    """Plot voltage and possibly current trace.

    Args:
        t: Time axis in steps of dt.
        Vt: Membrane voltage.
        It: Stimulus.
        figsize: Changes the figure size of the plot.
        title: Adds a custom title to the figure.
        timewindow: The voltage and current trace will only be plotted
            between (t1,t2). To be specified in secs.
        style: Defines the style of the plot.
            - stacked
            - overlayed
            - window
            - hybrid1
            - hybrid2
        show_scale: Whether to show scale bars.
        compare2sample0: Plot observational trace on top of:
            1. full
            2. window
            3. full, window
        line_colors: Specify the line colors of the traces. Last color specified
            will be repeated if more samples are plotted, than colors specified.
        line_widths: Specify the line widths of the traces. Last width specified
            will be repeated if more samples are plotted, than widths specified.
    Returns:
        fig: plt.Figure.
        axes: plt.Axes.
    """

    def plot_sample0(ax, col, **plot_kwargs):
        ax.plot(
            t[0][T[col, 0]].numpy(),
            Vt[0][T[col, 0]].numpy(),
            c=line_fmt["color"][0],
            lw=line_fmt["lw"][0],
            alpha=line_fmt["alpha"][0],
            **plot_kwargs,
        )

    def add_stimbar(ax, size, label):
        asb = AnchoredSizeBar(
            ax.transData,
            size,
            label,
            borderpad=0,
            sep=3,
            frameon=False,
            size_vertical=2,
            label=label,
            loc=3,
            pad=-3,
        )
        ax.add_artist(asb)

    window = (timewindow != None) and (
        ("window" in style.lower()) or ("hybrid" in style.lower())
    )

    def set_plot_params(t, timewindow, style, figsize, line_colors, line_widths):
        n_samples, n_points = t.shape
        axes_params = {"figsize": figsize}
        T = torch.ones_like(t, dtype=torch.bool).repeat(2, 1, 1)
        if "overlay" in style.lower():
            axes_params["nrows"] = 1
        if "stack" in style.lower():
            axes_params["nrows"] = n_samples
        if window:
            axes_params["ncols"] = 1
            axes_params["gridspec_kw"] = {"width_ratios": [1]}
            T[1] = torch.logical_and(t > timewindow[0], t < timewindow[1])
        elif timewindow != None and not "window" in style.lower():
            axes_params["ncols"] = 2
            axes_params["gridspec_kw"] = {"width_ratios": [3, 1]}
            T[1] = torch.logical_and(t > timewindow[0], t < timewindow[1])
        else:
            axes_params["ncols"] = 1
            axes_params["gridspec_kw"] = {"width_ratios": [1]}
        if "hybrid" in style.lower():
            axes_params["ncols"] = 2
            axes_params["nrows"] = n_samples
            if "1" in style.lower():
                axes_params["gridspec_kw"] = {"width_ratios": [3, 1]}
            if "2" in style.lower():
                axes_params["gridspec_kw"] = {"width_ratios": [3, 4]}

        fig, axes = plt.subplots(
            axes_params["nrows"],
            axes_params["ncols"],
            figsize=axes_params["figsize"],
            gridspec_kw=axes_params["gridspec_kw"],
        )
        try:
            axes = axes.reshape(axes_params["nrows"], axes_params["ncols"])
        except AttributeError:
            axes = np_array([[axes]])

        if "hybrid1" in style.lower():
            gs = axes[0, 0].get_gridspec()
            for ax in axes[:, 0]:
                ax.remove()
            ax_fused = fig.add_subplot(gs[:, 0])
        elif "hybrid2" in style.lower():
            gs = axes[0, 1].get_gridspec()
            for ax in axes[:, 1]:
                ax.remove()
            ax_fused = fig.add_subplot(gs[:, 1])
        else:
            ax_fused = None

        if len(line_colors) < n_samples:
            line_colors = line_colors + [line_colors[-1]] * (
                n_samples - len(line_colors)
            )
        if len(line_widths) < n_samples:
            line_widths = line_widths + [line_widths[-1]] * (
                n_samples - len(line_widths)
            )

        line_fmt = {
            "color": line_colors,
            "lw": line_widths,
            "alpha": [1] + [1] * (n_samples - 1),
        }

        return fig, axes, T, line_fmt, ax_fused

    fig, axes, T, line_fmt, ax_fused = set_plot_params(
        t, timewindow, style, figsize, line_colors, line_widths
    )
    N = t.shape[0]
    window_params = [0.65, 0.4, 0.3, 0.7]

    for i, ax_i in enumerate(axes):
        for j, ax_ij in enumerate(ax_i):
            if "stacked" in style.lower():
                ax_ij.plot(
                    t[i, T[j, i]].numpy(),
                    Vt[i, T[j, i]].numpy(),
                    c=line_fmt["color"][i],
                    lw=line_fmt["lw"][i],
                    **plot_kwargs,
                )
                if i != 0 and "full" in compare2sample0.lower():
                    plot_sample0(ax_ij, j)
                if window:
                    ax_inset = ax_ij.inset_axes(window_params)
                    ax_inset.plot(
                        t[i, T[1, i]].numpy(),
                        Vt[i, T[1, i]].numpy(),
                        c=line_fmt["color"][i],
                        lw=line_fmt["lw"][i],
                        **plot_kwargs,
                    )
                    if i != 0 and "window" in compare2sample0.lower():
                        plot_sample0(ax_inset, 1)
                    ax_inset.tick_params(axis="both", which="both", length=0, width=0)
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    # ax_inset.patch.set_alpha(0.7)
            if "hybrid" in style.lower():
                if "1" in style.lower() and (i, j) == (0, 0):
                    small, large = (1, 0)
                    if show_scale:
                        add_scalebar(
                            ax_fused,
                            loc=3,
                            barwidth=2,
                            labelx=f"{100} ms",
                            labely=f"{30} mV",
                            sizex=100,
                            sizey=30,
                            matchx=False,
                            matchy=False,
                            pad=-0.1,
                            borderpad=-3,
                            bbox_to_anchor=(40, 45),
                        )
                    if show_stim:
                        add_stimbar(ax_fused, 600, "ON")
                if "2" in style.lower() and (i, j) == (0, 0):
                    small, large = (0, 1)
                    if show_scale:
                        add_scalebar(
                            ax_fused,
                            loc=3,
                            barwidth=2,
                            labelx=f"{10} ms",
                            labely=f"{30} mV",
                            sizex=10,
                            sizey=30,
                            matchx=False,
                            matchy=False,
                            pad=0,
                            borderpad=-3,
                            bbox_to_anchor=(195, 40),
                        )
                if j == small:
                    ax_ij.plot(
                        t[i, T[j, i]].numpy(),
                        Vt[i, T[j, i]].numpy(),
                        c=line_fmt["color"][i],
                        lw=line_fmt["lw"][i],
                        **plot_kwargs,
                    )
                    if i != 0 and "window" in compare2sample0.lower():
                        plot_sample0(ax_ij, j)
                if j == large:
                    for k, (t_k, Vt_k) in enumerate(zip(t, Vt)):
                        ax_fused.plot(
                            t_k[T[j, k]].numpy(),
                            Vt_k[T[j, k]].numpy(),
                            c=line_fmt["color"][k],
                            lw=line_fmt["lw"][k],
                            alpha=line_fmt["alpha"][k],
                            **plot_kwargs,
                        )
                    if i != 0 and "full" in compare2sample0.lower():
                        plot_sample0(ax_fused, j)

                ax_fused.axis("off")
                ax_fused.margins(0.05)

            if "overlay" in style.lower():
                N = 1
                if window and "window" in style.lower():
                    ax_inset = ax_ij.inset_axes(window_params)
                    ax_inset.tick_params(axis="both", which="both", length=0, width=0)
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    ax_inset.patch.set_alpha(0.7)

                for k, (t_k, Vt_k) in enumerate(zip(t, Vt)):
                    ax_ij.plot(
                        t_k[T[j, k]].numpy(),
                        Vt_k[T[j, k]].numpy(),
                        c=line_fmt["color"][k],
                        lw=line_fmt["lw"][k],
                        alpha=line_fmt["alpha"][k],
                        **plot_kwargs,
                    )
                    if window and "window" in style.lower():
                        ax_inset.plot(
                            t_k[T[1, k]].numpy(),
                            Vt_k[T[1, k]].numpy(),
                            c=line_fmt["color"][k],
                            lw=line_fmt["lw"][k],
                            alpha=line_fmt["alpha"][k],
                            **plot_kwargs,
                        )
                if i != 0 and "full" in compare2sample0.lower():
                    plot_sample0(ax_ij, j)
                    if window:
                        ax_inset.plot(
                            t_k[T[1, k]].numpy(),
                            Vt_k[T[1, k]].numpy(),
                            c=line_fmt["color"][k],
                            lw=line_fmt["lw"][k],
                            alpha=line_fmt["alpha"][k] ** plot_kwargs,
                        )
                        ax_inset.tick_params(
                            axis="both", which="both", length=0, width=0
                        )
                        if i != 0 and "full" in compare2sample0.lower():
                            plot_sample0(ax_inset, 1)
                        ax_inset.set_xticks([])
                        ax_inset.set_yticks([])
                        ax_inset.patch.set_alpha(0.7)

            ymin, ymax = ax_ij.get_ylim()
            ax_ij.set_xlim(t[i, T[j, i]][0], t[i, T[j, i]][-1])
            ax_ij.set_ylim(ymin, ymax)
            ax_ij.axis("off")
            ax_ij.margins(0.05)
            if (N - 1, 0) == (i, j):
                if show_scale:
                    add_scalebar(
                        ax_ij,
                        loc="lower left",
                        barwidth=2,
                        labelx=f"{100} ms",
                        labely=f"{50} mV",
                        sizex=100,
                        sizey=50,
                        matchx=False,
                        matchy=False,
                        pad=-0.5,
                        borderpad=-2,
                        bbox_to_anchor=(30, 35),
                    )
                if show_stim:
                    add_stimbar(ax_ij, 600, "ON")
            if (N - 1, 1) == (i, j):
                if show_scale and not "hybrid1" in style.lower():
                    add_scalebar(
                        ax_ij,
                        loc="lower left",
                        barwidth=2,
                        labelx=f"{10} ms",
                        labely=None,
                        sizex=10,
                        sizey=0,
                        matchx=False,
                        matchy=False,
                        pad=-1.0,
                        borderpad=-0.5,
                        bbox_to_anchor=(310, 25),
                    )
                if show_scale and "hybrid1" in style.lower():
                    add_scalebar(
                        ax_ij,
                        loc=3,
                        barwidth=2,
                        labelx=f"{10} ms",
                        labely=f"{30} mV",
                        sizex=10,
                        sizey=30,
                        matchx=False,
                        matchy=False,
                        pad=-3,
                        borderpad=0,
                        bbox_to_anchor=(280, 45),
                    )

    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    return fig, axes


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.1,
        borderpad=0.1,
        sep=2,
        prop=None,
        barcolor="black",
        barwidth=None,
        **kwargs,
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.offsetbox import (
            AuxTransformBox,
            DrawingArea,
            HPacker,
            TextArea,
            VPacker,
        )
        from matplotlib.patches import Rectangle

        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(
                Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none")
            )
        if sizey:
            bars.add_artist(
                Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none")
            )

        if sizex and labelx:
            self.xlabel = TextArea(labelx)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs,
        )


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs["sizex"] = f(ax.xaxis)
        kwargs["labelx"] = str(kwargs["sizex"])
    if matchy:
        kwargs["sizey"] = f(ax.yaxis)
        kwargs["labely"] = str(kwargs["sizey"])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)
    if hidex and hidey:
        ax.set_frame_on(False)

    return sb
