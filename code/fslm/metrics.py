# from sbi.utils.metrics import c2st, unbiased_mmd_squared_hypothesis_test
from typing import Callable, Dict, List, Optional, Tuple

import torch
from sbi.utils.metrics import (
    biased_mmd,
    biased_mmd_hypothesis_test,
    c2st,
    unbiased_mmd_squared,
    unbiased_mmd_squared_hypothesis_test,
)
from torch import Tensor

from fslm.utils import ensure_same_batchsize, nearest_neighbours

# Suite of diverse 2 Sample Tests to evaluate the quality of the
# obtained posterior estimates relative to each other or relative to
# a reference posterior.


# Is the X close to Y?
# Is X as accurate as Y?
# - MEDDIST
# Are the distributions alike?
# - KL

# Is X as constrained as Y?
# Ratio of Vars

# def test(X,Y, test_params):
#     return "how close is X to Y"
#     return "how much more uncertainty does Y have compared to X"


def mmd(X: Tensor, Y: Tensor) -> float:
    """MMD squared between two distributions P and Q.

    X ~ P
    Y ~ Q

    Args:
        X: Samples from P.
        Y: Samples from Q.

    Returns:
        MMD^2
    """
    return float(unbiased_mmd_squared(X, Y))


def sample_kl(X: Tensor, Y: Tensor) -> float:
    """Uses nearest neighbour search to estimate the KL divergence of X and Y
    coming from 2 distributions P and Q.
    Args:
        X: Samples from P.
        Y: Samples from Q.

    Returns:
        kl: Estimate of the KL divergence.
    """
    n, d = X.shape
    m, d = Y.shape
    _, minXX = nearest_neighbours(X, X)
    _, minXY = nearest_neighbours(X, Y)

    kl1 = d / n * torch.sum(torch.log(minXY / minXX), dim=0)
    kl2 = torch.log(torch.tensor(m) / (torch.tensor(n) - 1))
    kl = kl1 + kl2
    return float(kl)

# def sample_js(X: Tensor, Y: Tensor) -> float:
#     """Estimates the Jensen-Shannon (JS) divergence between two samples X and Y.
#     Args:
#         X: Samples from P.
#         Y: Samples from Q.

#     Returns:
#         js: Estimate of the js divergence.
#     """

#     Z = torch.vstack([X,Y])
#     Z = Z[torch.randperm(len(Z))]
#     Z = Z[:len(X)] # I AM NOT SURE IF THIS IS MATHEMATICALLY CORRECT !!!
    
#     js = sample_kl(X, Z) + sample_kl(Y, Z)

#     return js


def cov(X: Tensor, rowvar: bool = False) -> Tensor:
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        X: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `X` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    X = X.clone()
    if X.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if X.dim() < 2:
        X = X.view(1, -1)
    if not rowvar and X.size(0) != 1:
        X = X.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (X.size(1) - 1)
    X -= torch.mean(X, dim=1, keepdim=True)
    Xt = X.t()  # if complex: mt = m.t().conj()
    return fact * X.matmul(Xt).squeeze()


def cov_ratio(X: Tensor, Y: Tensor) -> Tensor:
    """ "Computes the ratio of covariance
    of two different samples.

    Args:
        X: First sample.
        Y: Second sample.

    Returns:
        ratio_of_covs: Difference in sample covariances.
    """
    cov_X = cov(X)
    cov_Y = cov(Y)
    ratio_of_covs = cov_X / cov_Y
    return ratio_of_covs


def montecarlo_kl(X: Tensor, logpP: Callable, logpQ: Callable) -> float:
    """ "Computes the Monte-Carlo approximation of the KL divergence of two
    distributions P and Q given a set of samples from P.

    P and Q are represented by a corresponding logprob function that
    evaluates the probability of x.

    Args:
        X: Samples from P.
        logP: logprob associated with P.
        logQ: logprob associated with Q.
    Returns:
        kl: Estimate of the KL-Divergence.
    """
    two = torch.tensor(2)
    logpP = logpP(X) / torch.log(two)
    logpQ = logpQ(X) / torch.log(two)

    kl_terms = logpP - logpQ
    kl = 1 / len(X) * torch.sum(kl_terms)
    return float(kl)


def gaussian_kl(muA: Tensor, precA: Tensor, muB: Tensor, precB: Tensor) -> float:
    """Computes analytic expression of the KL divergence for 2 multivariate Gaussians.
    Args:
        muA: Mean of distribution A.
        precA: Precision Matrix of distribution A.
        muB: Mean of distribution B.
        precB: Precision Matrix of distribution B.
    Returns:
        kl: KL-Divergence of A and B.
    """
    SigmaA = torch.inverse(precA)
    SigmaB = torch.inverse(precB)
    d = muA.shape[1]

    det_frac = torch.det(SigmaB) / torch.det(SigmaA)

    kl = (
        1
        / 2
        * (
            torch.log(det_frac)
            - d
            + torch.trace(precB @ SigmaA)
            + (muB - muA) @ precB @ (muB - muA).T
        )
    )
    return float(kl)


def avg_neg_log_prob(log_prob: Callable, X: Tensor) -> float:
    """Calculates the average of the negative log probabilities for
    a given probability distribution with samples X and probability
    measure `log_prob`.

    `log_prob` can also be:
    ```
    log_prob = lambda theta: posterior.log_prob(theta, x_o)
    ```
    Args:
        log_prob: Log of a probability measure.
        X: Sample of points from a distribution.

    Returns:
        The average of the negative log probabilities of X.
    """
    return float(-1 * log_prob(X).mean())


def meddist(X: Tensor, x_o: Tensor, dist: Callable or str = "Euclidean") -> float:
    """Calculates the median of the distances between samples X and an
    observed data point x_o.

    The distance metric employed as default is the Euclidian distance,
    however any distance measure of the form d(x,y) can be supplied.

    Args:
        X: Sample of points.
        x_o: Single point of interest.
        dist: Distance metric (function) or string that specifies
            pre-implemented distance metrics, i.e. Euclidean.

    Returns:
        Median of the distances between X and x_o.
    """
    if type(dist) is str:
        if "eucl" in dist.lower():
            dist = lambda X, Y: torch.sqrt(((X - Y) ** 2).sum())
        else:
            raise ValueError("Not a valid metric.")
    else:
        assert type(dist) is Callable

    x_o = x_o.expand_as(X)
    d = dist(X, x_o)

    return float(d.median())


def avg_meddist(X: Tensor, X_o: Tensor, dist: Callable or str = "Euclidean") -> float:
    """Computes the average median distance between a set of samples X and
    a set of observations X_o = (x_{o_i})_{i=1:N}.

    The average is taken over i for d(X,x_{o_i}).
    Args:
        X: Set of input points.
        X_o: Set of observed points.
        dist: Distance metric (function) or string that specifies
            pre-implemented distance metrics, i.e. Euclidean.

    Returns:
        Average of the median distances between X and x_o.
    """
    cum_meddist = 0
    for x_o in X_o:
        cum_meddist += meddist(X, x_o, dist)
    return cum_meddist / X_o.shape[0]


def ratio_of_vars(X: Tensor, Y: Tensor) -> Tensor:
    """Calculates the ratio of variances for 2 sets of points.

    Returns a vector of variance ratios.

    Args:
        X: Var(X) = numerator.
        Y: Var(Y) = denominator.

    Returns:
        Ratio of Var(X)/Var(Y)
    """
    return X.var(dim=0) / Y.var(dim=0)

def ratio_of_iqrs(X: Tensor, Y: Tensor) -> Tensor:
    """Calculates the ratio of iqrs for 2 sets of points.

    Returns a vector of variance ratios.

    Args:
        X: IQR(X) = numerator.
        Y: IQR(Y) = denominator.

    Returns:
        Ratio of IQR(X)/IQR(Y)
    """
    iqr = lambda x: (x.quantile(0.75, dim=0) - x.quantile(0.25, dim=0))

    return  iqr(X)/ iqr(Y)


def kl_estimate(X: Tensor, Y: Tensor, **kwargs) -> float:
    """Wrapper for current best estimator of KL(P || Q).
    Args:
        X: Samples from distribution P.
        Y: Samples from distribution Q.

    Returns:
        Estimate of the KL Divergence
    """
    return sample_kl(X, Y)


# C2ST

# MMD


def sample_quality_check(X: Tensor, Y: Tensor, x_o: Optional[Tensor] = None) -> Dict:
    r"""Compares how closely two random variables are related from samples.

    Applies 4+1 different metrics to assess how closely two random variables are
    related based on samples from their respective probability distributions.
    The metrics compuated are:
    - Classifier 2 Sample Test (C2ST)
    - Maximum Mean Discrepancy (MMD)
    - Ratio of Marginal Variances
    - Sample based estimate of the KL divergence
    In Addition a ground truth value can be supplied, which is used to calculate
    further metrics
    - Median Distance (MEDDIST) from x_o for both samples seperately.


    Args:
        X: Samples from random variable X.
        Y: Samples from random variable Y.
        x_o: Ground truth value for either X or Y.

    Returns:
        metrics: Dictionary containing the calculated metrics for X and Y.
    """
    metrics = {}
    metrics["c2st"] = c2st(X, Y)
    metrics["mmd"] = unbiased_mmd_squared(X, Y)
    metrics["var_ratios"] = ratio_of_vars(X, Y)
    metrics["kl"] = sample_kl(X, Y)

    if x_o is not None:
        metrics["meddist_X"] = meddist(X, x_o)
        metrics["meddist_Y"] = meddist(Y, x_o)

    return metrics


def agg_sample_quality_checks(
    Xs: Tensor or List, Ys: Tensor or List, xs_o: Tensor or List
) -> Tuple[Dict, Dict]:
    r"""Aggregates comparative metricsfor different pairs of random variables (X,Y).

    Applies 4+1 different metrics to assess how closely two random variables are
    related based on samples from their respective probability distributions.
    The metrics compuated are:
    - Classifier 2 Sample Test (C2ST)
    - Maximum Mean Discrepancy (MMD)
    - Ratio of Marginal Variances
    - Sample based estimate of the KL divergence
    In Addition a ground truth value can be supplied, which is used to calculate
    further metrics
    - Median Distance (MEDDIST) from x_o for both samples seperately.

    The metrics are then collected for batches of (X,Y) and mean and std are
    computed.

    Xs or Ys or xs_o can be of different batchsize, i.e. 10, 1, 1. In this case
    the batches will be expanded such that they match in size, i.e. 10, 10, 10.

    Args:
        Xs: List or Tensor containing batches of samples from distribution p(x).
        Ys: List or Tensor containing batches of samples from distribution p(y).
        xs_o: List or Tensor of different ground truths matching the Xs or Ys.

    Returns:
        agg_metrics: The aggregated metrics, i.e. means and stds.
        acc_metrics: The accumulated metrics, before aggregation.
    """
    Xs, Ys, xs_o = ensure_same_batchsize(Xs, Ys, xs_o)

    accumulated_metrics = {}
    agg_metrics = {}
    for X, Y, x_o in zip(Xs, Ys, xs_o):
        metrics = sample_quality_check(X, Y, x_o)

        if accumulated_metrics == {}:
            accumulated_metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                accumulated_metrics[key].append(value)

    for key, value in accumulated_metrics.items():
        try:
            vals = torch.vstack(value)
        except TypeError:
            # if key == "kl":
            #     value = [kl for kl in value if kl != float("-inf")]
            vals = torch.tensor(value)

        accumulated_metrics[key] = vals.view(-1).tolist()
        agg_metrics[key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    return agg_metrics, accumulated_metrics
