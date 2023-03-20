from copy import deepcopy
from math import pi  # for pi
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributions.transforms as torch_tf
from numpy import ndarray
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.samplers.mcmc.init_strategy import proposal_init, sir_init
from sbi.utils import mcmc_transform
from sbi.utils.torchutils import BoxUniform
from scipy.signal import find_peaks
from scipy.stats import describe
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from fslm.utils import sample_posterior_potential

try:
    from ephys_helper.hh_simulator import HHSimulator
except ModuleNotFoundError:
    class DummyHHSimulator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ephys_helper not found. Required for HH simulation.")

    HHSimulator = DummyHHSimulator
    print("ephys_helper not found. Required for HH simulation.")

class GM:
    """Gaussian Model (GM) for testing Simulation Based Inference (SBI).

    Implements basic GM functionalities such as sampling
    and log_prob functionalities for the joint probability distribution,
    likelihood, and posterior distributions of a GM.

    The model i.e. $p(x|\theta)$ is provided via the `parameters` method.
    It takes theta as an argument and returns the corresponding mean and
    covariance tensors. This method can be replaced with any other, to
    allow for custom GMs.

    Supports marginalisation of likelihood, joint and posterior distributions.

    Args:
        prior: A prior parameter distribution for theta. Can be uniform, i.e.
        sbi.utils.torchutils.BoxUniform or a multivariate Gaussian, i.e.
        torch.distributions.multivariate_normal.MultivariateNormal.
        seed: Sets a new torch.Seed

    Attributes:
        Prior: Holds the prior distribution in order to sample from the joint.
        xdims: List of dimensions of the observations x.
    """

    def __init__(self, prior: Distribution, seed: int = None):
        self.Prior = prior
        self.seed = seed
        if seed != None:
            torch.manual_seed(seed)

        m, S = self.model_parameters(prior.sample((1,)))
        self.xdims = list(range(m.shape[-1]))

    def __str__(self):
        return "GM"

    def parameters(self):
        """Parameter dummy function for interfacing with LikelihoodBasedPotential."""
        yield torch.tensor(0)

    def _ensure_correct_dims(self, xdims: List[int]) -> List[int]:
        """Ensures all input dimensions are selected if none are provided.

        Args:
            xdims: Dimensions input into the function and which to keep
                in marginal distribution.
        Returns:
            Either all or the xdims provided as function input.
        """

        if xdims == None:
            return self.xdims
        else:
            return xdims

    @staticmethod
    def model_parameters(theta: Tensor, sigma_sq: float = 1) -> Tuple[Tensor, Tensor]:
        """Implements the likelihood function for the GM.

        Depending on theta, distribution parameters m(theta) and
        S(theta) are returned such that:
        x ~ N(x|m,S)

        Args:
            theta: Parameters that influence the observations of the GM.
        Returns:
            m: Mean tensor of the resulting GM.
            S: Covariance matrix of the resulting GM.
        """

        S = sigma_sq * torch.eye(3)
        m = torch.zeros((1, 3))
        return m, S

    @staticmethod
    def __ensure_batch_dims(*items: Tensor) -> List[Tensor]:
        """Ensures all items have the same batch dimension.

        Takes the item with the largest batch dimension and
        repeats all other items to have the same number of
        items in their batch.

        Args:
            items: Items to adjust the batch dimension for.

        Returns:
            List of items with adjusted batch dimensions.
        """
        items = list(items)
        batch_size = max([item.shape[0] for item in items])

        for i, item in enumerate(items):
            if item.shape[0] != batch_size:
                items[i] = item.repeat_interleave(batch_size, 0)
        return items

    @staticmethod
    def cholesky_sample(m: Tensor, S: Tensor, k_samples: int) -> Tensor:
        """Obtain samples from Gaussian via Cholesky decomposition.

        Get 1 sample for multiple means and covariances
        -> m.shape[0] == k_samples, or multiple samples for
        the same distribution with m.shape[0] == 1.

        Args:
            m: Mean of the Gaussian distribution.
            S: Covariance of the Gaussian distribution.
            k_samples: Number of samples to obtain.
        Returns:
            x: Samples obtained.
        """
        out_dims = m.shape[-1]
        std_normal_sample = torch.randn(out_dims, k_samples).view(
            k_samples, out_dims, 1
        )
        chol_factor = torch.linalg.cholesky(S)
        x = m + (chol_factor @ std_normal_sample).transpose(1, 2)
        return x

    @classmethod
    def log_prob_normal(gm, X: Tensor, mean: Tensor, prec: Tensor) -> Tensor:
        """Evaluate samples from Gaussian.
        Args:
            X: Samples to evaluate.
            mean: Mean of the Gaussian distribution.
            prec: Precision matrix of the Gaussian distribution.
        Returns:
            log_prob: logarithmic probabilities of each sample.
        """

        batch_size, dims = X.shape
        X, mean, prec = gm.__ensure_batch_dims(X, mean, prec)

        logits = torch.zeros((batch_size, 1))
        prec = prec.view(-1, 1, dims, dims)
        # precfs = torch.cholesky(prec)
        # log_diag = torch.log(torch.diagonal(precfs, dim1=3, dim2=2))
        # sumlogdiag = torch.sum(log_diag,dim=2).view(-1,1)
        sumlogdiag = torch.log(torch.det(prec)) / 4
        return mdn.log_prob_mog(X, logits, mean, prec, sumlogdiag)

    def log_prob(
        self, inputs: Tensor, context: Tensor, xdims: List[int] = None
    ) -> Tensor:
        """Evaluate the likelihood of the GM for a set of obervations.
        Args:
            inputs: Samples to evaluate the likelihood for.
            context: Specific set of model parameters.
            xdims: List of marginals to consider.
        Returns:
            log_prob: logarithmic probabilities of each sample.
        """

        xdims = self._ensure_correct_dims(xdims)
        m, S = self.model_parameters(context, xdims)
        prec = torch.inverse(S)

        return self.log_prob_normal(inputs, m, prec)

    def sample(
        self,
        theta_o: Tensor,
        sample_shape: Tuple[int, int] = (1,),
        xdims: List[int] = None,
    ) -> Tensor:
        """Draw samples from the likelihood of the GM given a set of parameters.
        Args:
            theta_o: Specific set of model parameters.
            sample_shape: The shape of the sample to be drawn.
            xdims: List of marginals to consider.
        Returns:
            x: Samples from the likelihood distribution.
        """
        xdims = self._ensure_correct_dims(xdims)
        if theta_o.shape[0] <= 1:
            k_samples = torch.Size(sample_shape).numel()
            theta_o = theta_o.repeat(k_samples, 1)
        else:
            sample_shape += (theta_o.shape[0],)
            k_samples = torch.Size(sample_shape).numel()
            theta_o = theta_o.repeat(sample_shape[0], 1)

        m, S = self.model_parameters(theta_o, xdims)

        x = self.cholesky_sample(m, S, k_samples)

        return x.reshape((*sample_shape, -1))

    def sample_evidence(
        self, sample_shape: Tuple[int, int] = (1,), xdims: List[int] = None
    ):
        """Draw samples from the marginal likelihood of the GM.

        Args:
            sample_shape: The shape of the sample to be drawn.
            xdims: List of marginals to consider.

        Returns:
            x: Samples from the likelihood distribution conditioned on the
                parameter samples from the prior.
        """
        theta = self.Prior.sample(sample_shape)
        xdims = self._ensure_correct_dims(xdims)
        k_samples = torch.Size(sample_shape).numel()
        m, S = self.model_parameters(theta, xdims)

        x = self.cholesky_sample(m, S, k_samples)

        return x.reshape((*sample_shape, -1))

    def sample_joint(
        self,
        sample_shape: Tuple[int, int] = (1,),
        return_theta: bool = True,
        xdims: List[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Draw samples from the joint distribution of the GM.
        Args:
            sample_shape: The shape of the sample to be drawn.
            return_theta: Whether to return theta along with observations x.
            xdims: List of marginals to consider.
        Returns:
            theta: Parameter samples.
            x: Samples from the likelihood distribution conditioned on the
                parameter samples from the prior.
        """
        theta = self.Prior.sample(sample_shape)
        xdims = self._ensure_correct_dims(xdims)
        k_samples = torch.Size(sample_shape).numel()
        m, S = self.model_parameters(theta, xdims)

        x = self.cholesky_sample(m, S, k_samples)

        if return_theta:
            return theta, x.reshape((*sample_shape, -1))
        else:
            return x.reshape((*sample_shape, -1))

    def sample_posterior(
        self,
        sample_shape: Tuple[int, int] = (1,),
        context: Tensor = None,
        xdims: List[int] = None,
        **kwargs
    ) -> Tensor:
        """Draws samples from contextualised posterior distribution.
        Args:
            context: Condition, x_o, for the posterior function p(theta|x_o).
            sample_shape: The shape of the sample to be drawn.
            xdims: List of marginals to consider.
        Returns:
            sample: Posterior samples.
        """
        assert context != None, "No context has been provided."
        return self.sample_posterior_potential(sample_shape, context, xdims, **kwargs)

    def sample_posterior_potential(
        self,
        sample_shape: Tuple[int, int] = (1,),
        context: Tensor = None,
        xdims: List[int] = None,
        mcmc_method: str = "slice_np_vectorized",
        init_strategy: str = "sir",
        enable_transform: bool = True,
        num_chains: int = 1,
    ) -> Tensor:
        """Draws samples from contextualised posterior distribution.

        Args:
            context: Condition, x_o, for the posterior function p(theta|x_o).
            sample_shape: The shape of the sample to be drawn.
            xdims: List of marginals to consider.
        Returns:
            sample: Posterior samples.
        """
        assert context != None, "No context has been provided."
        k_samples = torch.Size(sample_shape).numel()
        xdims = self._ensure_correct_dims(xdims)
        if context.shape[1] != len(xdims):
            context = context[:, xdims]

        self_copy = deepcopy(self)
        self_copy.log_prob = lambda inputs, context: self.log_prob(
            inputs, context, xdims
        )

        transform = mcmc_transform(self.Prior, 1000, enable_transform)

        pfp_init = LikelihoodBasedPotential()

        if not isinstance(transform, torch_tf.IndependentTransform):
            transform = torch_tf.IndependentTransform(
                transform, reinterpreted_batch_ndims=1
            )

        potential_fn = pfp_init(self.Prior, self_copy, context, mcmc_method, transform)

        init_func = None
        if init_strategy == "sir":
            init_func = lambda: sir_init(self.Prior, potential_fn, transform=transform)
        if init_strategy == "prior":
            init_func = lambda: proposal_init(self.Prior, transform=transform)

        posterior_samples = sample_posterior_potential(
            (k_samples,),
            potential_fn,
            init_func,
            mcmc_method=mcmc_method,
            num_chains=num_chains,
        )
        posterior_samples = transform.inv(posterior_samples)
        return posterior_samples.reshape((*sample_shape, -1))


class SLCP(GM):
    """Simple Likelihood Complex Posterior (SLCP) model for testing
    Simulation Based Inference (SBI).

    Inherits functionalities from the GM such as sampling
    and log_prob functionalities for the joint probability distribution,
    likelihood, and posterior distributions.

    The model is constructed using a uniform prior $\theta$ ~ U(-3,3)
    and x_i ~ N(x|m(theta), S(theta)). Where X = [x0,x1,x2,x3].

    m = [theta0,theta1].
    S = [[s1,s2],[s3,s4]], where s are nonlinear functions of theta with
    rho = tanh(theta4), s1 = theta2**4, s4 = theta3**4,
    s2=s3=rho*theta2**2*theta3**2.

    Args:
        prior: A uniform prior parameter distribution for theta, i.e.
        sbi.utils.torchutils.BoxUniform

    Attributes:
        Prior: Holds the prior distribution in order to sample from the joint.
        xdims: List of dimensions of the observations x.
        alpha: scaling factors of covariance matrices.
    """

    def __init__(self, prior, seed: int = None):
        self.alpha = torch.ones(4)

        super().__init__(prior, seed)

    def __str__(self):
        return "SLCP"

    def model_parameters(self, theta, xdims=[0, 1, 2, 3, 4, 5, 6, 7]):
        """Model, where theta is 5-dimensional,
        and x is a set of four 2-dimensional points
        (or an 8-dimensional vector) sampled from
        a Gaussian whose mean m and covariance matrix
        S are functions of theta (and alpha in the case of S).

        Parameters
        ----------
        theta : torch.tensor with size (1,5)
            Model parameters.
        alpha : torch.tensor with size (1,4)
            Controls information content of x_i.

        Returns
        -------
        x : torch.tensor
            Observation."""

        k_samples = theta.shape[0]
        m = theta[:, :2].view(k_samples, 1, 2).repeat(1, 1, 4)

        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rho = torch.tanh(theta[:, 4])

        S1 = s1 ** 2
        S2 = rho * s1 * s2
        S3 = rho * s1 * s2
        S4 = s2 ** 2

        S = torch.zeros(k_samples, 8, 8)
        for d in torch.arange(0, 8, 2):
            S[:, d : d + 2, d : d + 2] = float(self.alpha[int(d / 2)]) * torch.dstack(
                [S1, S2, S3, S4]
            ).reshape(k_samples, 2, 2)

        # NUMERICALLY STABELISE SLCP
        S = S + 1e-4 * torch.eye(S.shape[-1])

        S = S[:, xdims]
        S = S[:, :, xdims]
        m = m[:, :, xdims]
        return m, S


class SLCP2(GM):
    """Simple Likelihood Complex Posterior (SLCP) model for testing
    Simulation Based Inference (SBI).

    Inherits functionalities from the GM such as sampling
    and log_prob functionalities for the joint probability distribution,
    likelihood, and posterior distributions.

    The model is constructed using a uniform prior $\theta$ ~ U(-3,3)
    and x_i ~ N(x|m_i(theta), S_i(theta)). Where X = [x0,x1].

    m_1 = [theta0,theta1].
    m_2 = [theta1,theta2].
    S_1 = [[s1,s2],[s3,s4]], S_2 = [[s1,s2],[s3,s4]], where s are nonlinear
    functions of theta with.
    Args:
        prior: A uniform prior parameter distribution for theta, i.e.
        sbi.utils.torchutils.BoxUniform

    Attributes:
        Prior: Holds the prior distribution in order to sample from the joint.
        xdims: List of dimensions of the observations x.
    """

    def __init__(self, prior, seed=None):
        super().__init__(prior, seed)

    def __str__(self):
        return "SCLP2"

    def model_parameters(self, theta, xdims=[0, 1, 2, 3]):
        """Model, where theta is 5-dimensional,
        and x is a set of two 2-dimensional points
        (or an 4-dimensional vector) sampled from
        a Gaussians whose mean m and covariance matrix
        S are functions of theta.

        Parameters
        ----------
        theta : torch.tensor with size (1,5)
            Model parameters.

        Returns
        -------
        x : torch.tensor
            Observation."""

        k_samples = theta.shape[0]
        m1 = theta[:, :2].view(k_samples, 1, 2)
        m2 = theta[:, 1:3].view(k_samples, 1, 2)
        m = torch.dstack([m1, m2])

        s1_1 = theta[:, 3] ** 2
        s2_1 = theta[:, 4] ** 2
        rho_1 = torch.tanh(theta[:, 5])

        S1_1 = s1_1 ** 2
        S2_1 = rho_1 * s1_1 * s2_1
        S3_1 = rho_1 * s1_1 * s2_1
        S4_1 = s2_1 ** 2

        s1_2 = theta[:, 3] ** 2
        s2_2 = theta[:, 6] ** 2
        rho_2 = torch.tanh(theta[:, 7])

        S1_2 = s1_2 ** 2
        S2_2 = rho_2 * s1_2 * s2_2
        S3_2 = rho_2 * s1_2 * s2_2
        S4_2 = s2_2 ** 2

        S = torch.zeros(k_samples, 4, 4)
        S[:, :2, :2] = torch.dstack([S1_1, S2_1, S3_1, S4_1]).reshape(k_samples, 2, 2)
        S[:, 2:4, 2:4] = torch.dstack([S1_2, S2_2, S3_2, S4_2]).reshape(k_samples, 2, 2)

        S = S[:, xdims]
        S = S[:, :, xdims]
        m = m[:, :, xdims]
        return m, S


class LGM(GM):
    """Linear Gaussian Model (LGM) for testing Simulation Based Inference (SBI).

    Implements sampling and log_prob functionality
    for the joint probability distribution, likelihood, and
    posterior distributions of a LGM with parameters:

    x ~ N(x| m, S)

    m = mu_x + L@theta
    S = sig^2*1

    with mu_x being a translation and L a linear transformation.
    By default,

        |1 0 0 0|
    L = |0 1 1 0|
        |0 0 1 0|

    which means x_0(theta_0), x_1(theta_1), x_2=(theta_1, theta_2), x_3=noise.

    Supports marginalisation of likelihood, joint and posterior distributions.

    Args:
        prior: A prior parameter distribution for theta. Can be uniform, i.e.
        sbi.utils.torchutils.BoxUniform or a multivariate Gaussian, i.e.
        torch.distributions.multivariate_normal.MultivariateNormal.
        seed: Sets a new torch.Seed

    Attributes:
        Prior: Holds the prior distribution in order to sample from the joint.
    """

    def __init__(self, prior: Optional[Distribution] = None, seed: int = None, noise: float = 0.1):
        if prior is None:
            prior = BoxUniform([-5, -5, -5], [5, 5, 5])
        LGM.noise = noise

        super().__init__(prior, seed)

    def __str__(self):
        return "LGM"

    @staticmethod
    def default_LM(
        in_dims: int, k_samples: int, xdims=[0, 1, 2, 3]
    ) -> Tuple[Tensor, Tensor]:
        """Returns parameters for a simple implementation
        of a linear model.

        m = mu_x + L@theta

        Creates dependencies such that

        x_0 = theta_0, -> x_0 independent of x_i
        x_1 = theta_1, -> x_1 depends on x_2
        x_2 = theta_1 + theta_2, -> x_2 depends on x_1
        x_3 = noise
        Args:
            in_dims: Dimensionality of parameters theta.
            k_samples: How many samples to draw/evaluate.
            xdims: List of marginals to consider.
        Returns:
            mu_x: Translational parameter.
            L: Linear Mapping applied to theta.
        """
        mu_x = torch.zeros((k_samples, 4))

        L = torch.eye(in_dims, 4)
        L[1, 2] = 1  # x_1 = theta_1 + theta_2
        L = L[:, xdims]
        mu_x = mu_x[:, xdims]

        return mu_x, L

    @classmethod
    def model_parameters(
        lgm, theta: Tensor, xdims2keep: List[int] = [0, 1, 2, 3]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns parameters of the Gaussian likelihood distribution.

        Args:
            theta: Condition for the likelihood function p(x|theta).
            xdims2keep: List of marginals to consider.
        Returns:
            m: Mean of LGM.
            S: Covariance matrix of LGM.
            L: Linear Mapping applied to theta.
            mu_x: Translational parameter.
        """
        out_dims = len(xdims2keep)
        k_samples, in_dims = theta.shape
        mu_x, L = lgm.default_LM(in_dims, k_samples, xdims2keep)

        # LM
        m = mu_x + theta @ L
        m = m.view(k_samples, 1, out_dims)
        S = lgm.noise * torch.eye((out_dims)).repeat(k_samples, 1, 1)

        return m, S

    @classmethod
    def posterior_lgm_params(
        lgm, prior: Distribution, context: Tensor, xdims: List[int] = [0, 1, 2, 3]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns parameters of the Gaussian posterior distribution.

        Depending on the prior distribution chosen, the parameters
        for the new Gaussians are computed for use in sampling and
        log_prob methods.

        Args:
            prior: Prior distribution.
            context: Condition, x_o, for the posterior function p(theta|x_o).
            xdims: List of marginals to consider.
        Returns:
            mean: Mean of Gaussian posterior.
            cov: Covariance matrix of Gaussian posterior.
            support: Bounds of the prior support.
        """

        mean = None
        cov = None
        support = None

        theta = prior.sample((1,))
        pdim = theta.shape[-1]
        m, S = lgm.model_parameters(torch.zeros_like(theta), xdims2keep=xdims)
        mu_x, L = lgm.default_LM(pdim, m.shape[0], xdims)
        sigma_sq = S[0, 0, 0]

        if type(prior) is BoxUniform or type(prior) is Uniform:
            cov = torch.inverse(1 / sigma_sq * L @ L.T + 1e-6 * torch.eye(pdim))
            mean = (1 / sigma_sq * context @ L.T) @ cov

            ubound = prior.support.base_constraint.upper_bound
            lbound = prior.support.base_constraint.lower_bound
            support = torch.vstack([lbound, ubound])

        if type(prior) is MultivariateNormal:
            covp = prior.covariance_matrix
            precp = torch.inverse(covp).view(1, pdim, pdim)
            meanp = prior.mean

            ubound = float("inf")
            lbound = -ubound
            bounds = torch.tensor([[lbound], [ubound]])

            cov = torch.inverse(precp + 1 / sigma_sq * L @ L.T)
            mean = (precp @ meanp + 1 / sigma_sq * context @ L.T) @ cov
            support = bounds.repeat(1, pdim)

        return mean.view(1, 1, -1), cov.view(1, pdim, pdim), support

    @staticmethod
    def cholesky_sample(m: Tensor, S: Tensor, k_samples: int) -> Tensor:
        """Obtain samples from Gaussian via Cholesky decomposition.

        Get 1 sample for multiple means and covariances
        -> m.shape[0] == k_samples, or multiple samples for
        the same distribution with m.shape[0] == 1.

        Args:
            m: Mean of the Gaussian distribution.
            S: Covariance of the Gaussian distribution.
            k_samples: Number of samples to obtain.
        Returns:
            x: Samples obtained.
        """
        out_dims = m.shape[-1]
        std_normal_sample = torch.randn(out_dims, k_samples).view(
            k_samples, out_dims, 1
        )
        chol_factor = torch.linalg.cholesky(S)
        x = m + (chol_factor @ std_normal_sample).transpose(1, 2)
        return x

    def log_prob_posterior(
        self, theta: Tensor, context: Tensor, xdims: List[int] = [0, 1, 2, 3]
    ) -> Tensor:
        """Evaluate the posterior of the LGM for a set of obervations.
        Args:
            theta: Paramter samples to evaluate the posterior for.
            context: Specific set of observations.
            xdims: List of marginals to consider.
        Returns:
            log_prob: logarithmic probabilities of each sample.
        """
        mean, cov, support = self.posterior_lgm_params(
            self.Prior, context[:, xdims], xdims
        )
        prec = torch.inverse(cov)

        within_support = self.is_supported(support, theta)
        log_prob = self.log_prob_normal(theta, mean, prec)
        log_prob[~within_support] = float("-inf")

        return log_prob

    @staticmethod
    def is_supported(support: Tensor, theta: Tensor) -> Tensor:
        """Determines whether parameters are within the prior support.
        Takes a set of points theta and upper and lower support boundaries.
        Checks, each point for its prior support.

        Args:
            support: 2 x D Tensor. Contains the boundaries of the prior support.
            theta: Contains a set of multidimensional points to check
                against the prior support.

        Returns:
            within_support: Boolean Tensor representing, whether a sample is
            within the prior support or not.
        """
        lbound = support[0].view(1, -1)
        ubound = support[1].view(1, -1)

        dim = theta.shape[-1]
        theta = theta.view(-1, dim)

        within_support = torch.logical_and(lbound < theta, theta < ubound)

        return torch.all(within_support, dim=1)

    @classmethod
    def rejection_sampling(
        lgm, mean: Tensor, cov: Tensor, support: Tensor, k_samples: int,
    ) -> Tensor:
        """Draws/redraws k samples until all are within their prior support.

        Draws samples from a Gaussian distribution. After each draw, all
        samples outside of the prior support are rejected. Recursively redraws
        the samples that have been rejected.

        Args:
            mean: Mean of the Gaussian distribution.
            cov: Covariance of the Gaussian distribution.
            support: 2 x D Tensor. Contains the boundaries of the prior support.
            k_samples: Number of samples to obtain.
        Returns:
            samples: All accepted samples.
        """
        sample_cache = []
        while k_samples != 0:
            sample = lgm.cholesky_sample(mean, cov, k_samples)
            within_support = lgm.is_supported(support, sample).view(-1)
            sample_cache.append(sample[within_support])

            k_samples = int(sum(~within_support))
            print("sampling ...", end="\r")

        return torch.vstack(sample_cache)

    @classmethod
    def sample_posterior_within_prior(
        lgm, k_samples: int, prior: Distribution, context: Tensor, xdims: List[int]
    ) -> Tensor:
        """Draws/redraws k samples until all are within their prior support.
        Wrapps the rejection_sampling recursion.
        Args:
            k_samples: Number of samples to obtain.
            prior: Prior distribution.
            context: Condition, x_o, for the posterior function p(theta|x_o).
            xdims: List of marginals to consider.
        Returns:
            sample: All accepted samples.
        """
        mean, cov, support = lgm.posterior_lgm_params(prior, context, xdims)

        # sample while rejecting samples outside of prior support
        sample = lgm.rejection_sampling(mean, cov, support, k_samples)

        return sample

    def sample_posterior(
        self,
        sample_shape: Tuple[int, int] = (1,),
        context: Tensor = None,
        xdims: List[int] = [0, 1, 2, 3],
    ) -> Tensor:
        """Draws samples from contextualised posterior distribution.
        Args:
            context: Condition, x_o, for the posterior function p(theta|x_o).
            sample_shape: The shape of the sample to be drawn.
            xdims: List of marginals to consider.
        Returns:
            sample: Posterior samples.
        """
        assert context != None, "No context has been set."

        k_samples = torch.Size(sample_shape).numel()
        sample = self.sample_posterior_within_prior(
            k_samples, self.Prior, context[:, xdims], xdims
        )
        return sample.reshape((*sample_shape, -1))


class MoG:
    """Mixture of Gaussians (MoG) with a mixture of K distributions
    of arbitrary dimensionality.

    Implements sampling and log_prob functionality.

    p(x) = \sum_{k=1}^K \p_k \mathcal{N}(x;m_k,S_k)

    Args:
        p: The mixture coefficients. (nclusters, )
        m: The mean vectors for each Gaussian. (nclusters, ndims)
        S: The covariance matrices for each Gaussian. (nlusters, ndims, ndims)

    Attributes:
        p: The mixture coefficients. (nclusters, )
        m: The mean vectors for each Gaussian. (nclusters, ndims)
        S: The covariance matrices for each Gaussian. (nlusters, ndims, ndims)
        dim: Number of dimensions of each multivariate normal distribution.
    """

    def __init__(self, p: Tensor, m: Tensor, S: Tensor):
        self.p = p
        self.m = m
        self.S = S
        self.dim = m.shape[1]

    def sample(self, sample_shape: Tuple = (1,)) -> Tensor:
        """Draws a samples from the MoG.
        Args:
            sample_shape: Number of samples per batch and number of batches.
                (n,) 1 batch of n samples.
                (k,n) k batches of n samples.

        Returns:
            samples: Samples drawn from the MoG with parameters p,m,S.
        """
        n_samples = torch.Size(sample_shape).numel()
        ks = self.p.multinomial(num_samples=n_samples, replacement=True)

        cum_count = 0
        samples = torch.empty((*sample_shape, 1, self.dim))
        for n, m, S in zip(ks.bincount(), self.m, self.S):
            if n != 0:
                samples[cum_count : cum_count + n] = GM.cholesky_sample(m, S, n)
                cum_count += n
        return samples.squeeze()

    def log_prob(self, X: Tensor) -> Tensor:
        """Evaluates the MoG for a batch of points.
        Args:
            X: Batch of points (npoints,ndims).

        Returns:
            log_prob: log(p(X)). (npoints, )
        """
        MD = lambda X, m, S: (X - m) @ S.inverse() @ (X - m).transpose(1, 2)
        Z = lambda S: torch.sqrt((2 * pi) ** self.dim * torch.det(S))
        MVN_prob = lambda X, m, S: 1 / Z(S) * torch.exp(-1 / 2 * MD(X, m, S))

        prob = torch.zeros((len(X), 1, 1))
        for p, m, S in zip(self.p, self.m, self.S):
            prob += p * MVN_prob(X.unsqueeze(1), m, S)

        return torch.log(prob).squeeze()
        

class Lotka_Volterra:
    """Implements the Prey-Predator or Lotka-Volterra equations.

    $$
    dx/dt = \alpha x - \beta xy \\
    dy/dt = \delta xy - \gamma y
    $$

    - x is the number of prey.
    - y is the number of predators.
    - $\alpha, \beta, \gamma, \delta$ are positive, real parameters,
    that describe the interactions between the two species.

    Args:
        theta: The model parameters $[\alpha, \beta, \gamma, \delta]$. (1,4)
            Several models can be run in parallel if multiple sets of
            parameters are provided. (nbatches, 4)
        make_torch_compatible: I/O is supplied as Tensors.
        seed: sets numpy random seed for repeatability.

    Attributes:
        theta: The model parameters. (nbatches, 4)
        state: The current internal state of the model. (nbatches, 2)
        t: The current timepoint.
    """

    def __init__(
        self,
        theta: ndarray or Tensor = None,
        make_torch_compatible: bool = False,
        seed: int = 0,
    ):
        self.model_init(theta)
        self.t = 0
        self.torch_io = make_torch_compatible
        np.random.seed(seed)

    def model_init(self, theta: ndarray or Tensor):
        """Helper function to initialise theta.
        Args:
            theta: The model parameters. (nbatches, 4)
        """
        if theta == None:
            theta = np.array([0, 0, 0, 0])
            print("No parameters were provided. Provide them at runtime!")
        self.theta = np.array(theta)
        if theta.ndim == 1:
            self.theta = np.expand_dims(self.theta, axis=1)
        self.state = np.zeros((self.theta.shape[0], 2))

    def f(self, state: ndarray, theta: ndarray, noise_lvl: float = 0) -> ndarray:
        """Implements the Lotka-Voltera dynamical ODE system.

        Args:
            state: The current state vector [x,y]. (nbatches, 2)
            theta: The model parameters [a, b ,c ,d]. (nbatches, 4)
            noise_lvl: Controls the amount of noise in the model.

        Returns:
            Derivatives [dx/dt, dy/dt] of f(x,y).
        """
        x_dot = theta[:, 0] * state[:, 0] - theta[:, 1] * state[:, 0] * state[:, 1]
        y_dot = theta[:, 0] * state[:, 0] * state[:, 1] - theta[:, 1] * state[:, 1]
        noise = np.random.normal(scale=noise_lvl, size=self.state.shape)
        return np.vstack([x_dot, y_dot]).T + abs(noise)  # state musst be > 0 !!!

    def RK4(self, f: Callable, X: ndarray, dt: float) -> ndarray:
        """4th order Runge Kutta method.

        Args:
            f: Function to integrate.
            X: Current state vector [x,y].
            dt: Stepsize in the timedomain.
        Returns:
            X: The next state of X according to X_{t+dt} = RK4[f](X_t).
        """
        k1 = dt * f(X)
        k2 = dt * f(X + 0.5 * k1)
        k3 = dt * f(X + 0.5 * k2)
        k4 = dt * f(X + k3)

        X = X + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return X

    def run(
        self,
        initial_state: ndarray or Tensor,
        T: float,
        dt: float,
        model_noise: float = 0,
        measurement_noise: float = 0,
        t0: Optional[float] = 0,
        theta: Optional[Tensor or ndarray] = None,
        batchsize: int = 1000,
    ) -> Tuple[ndarray, ndarray] or Tuple[Tensor, Tensor]:
        """Integrates the LV model and records the trajectories.
        Args:
            initial_state: Sets the initial state [x0, y0]. (nbatches, 2)
            T: Integration time interval.
            dt: Stepsize.
            model_noise: How much noise there is in f(x) + \eps.
            measurement_noise: How much noise there is in measuring x_t.
            t0: Option to start not from 0 but from a different timepoint.
            theta: The model parameters [a, b ,c ,d]. (nbatches, 4)
            batchsize: How many samples are simulated simultaneaously.
        Returns:
            ts: time axis. (ntimepoints, )
            states: States. (ntimepoints, nbatches, 2)
        """

        if theta != None:
            self.model_init(theta)

        ts = np.arange(t0, T, dt)
        states = np.zeros((len(ts), *self.state.shape))

        N = len(self.theta)
        if N > batchsize:
            n_batches = int(N / batchsize)
        else:
            n_batches = 1
            batchsize = N

        states_batched = np.array_split(states, n_batches, axis=1)
        theta_batched = np.array_split(self.theta, n_batches)

        self.t = t0
        self.dt = dt

        f = lambda X: self.f(X, self.theta, model_noise)

        for batch_idx in range(n_batches):
            self.state = np.array(initial_state).reshape(-1, 2)
            self.theta = theta_batched[batch_idx]
            for idx, t in enumerate(ts):
                noise = np.random.normal(scale=measurement_noise, size=self.state.shape)
                self.state = self.RK4(f, self.state, dt) + abs(
                    noise
                )  # state musst be > 0 !!!
                self.t = t
                states_batched[batch_idx][idx] = self.state
            print("{}/{}".format((batch_idx + 1) * batchsize, N))

        states = np.hstack(states_batched)
        if self.torch_io:
            ts = torch.from_numpy(ts)
            states = torch.from_numpy(states)
        return ts, states

    def extract_freq(self, Xs: ndarray or Tensor) -> List:
        """Extracs frequencies along axis 0 from a set of timeseries.

        Makes use of discrete Fourier transform and selects most prominent
        peak with lowest frequency.

        Args:
            Xs: Set of timeseries of shape (N, n_samples, 2).

        Returns:
            freqs: List of frquencies, for each timeseries. (n_samples, 2)
        """
        Xs = np.array(Xs)

        n = len(Xs)  # length of the signal
        k = np.arange(n)
        T = n / 100
        frq = k / T  # two sides frequency range
        frq = frq[: len(frq) // 2]  # one side frequency range

        Y = np.fft.fft(Xs, axis=0) / n  # dft and normalization
        Y = Y[: n // 2]

        freqs = np.zeros((Xs.shape[1], 1))
        for i in range(Xs.shape[1]):
            # threshold and height params were chosen based on experimentation
            idx_x, peak_properties_x = find_peaks(
                abs(Y[:, i, 0]), threshold=0.01, height=0.001
            )
            idx_y, peak_properties_y = find_peaks(
                abs(Y[:, i, 1]), threshold=0.01, height=0.001
            )
            peak_heights_x = peak_properties_x["peak_heights"]
            peak_heights_y = peak_properties_y["peak_heights"]
            idxs = np.hstack([idx_x, idx_y])

            # take freq at heighest peak
            peak_heights = np.hstack([peak_heights_x, peak_heights_y])
            if len(peak_heights) != 0:
                f_idx = idxs[np.argmax(peak_heights)]
                freqs[i] = frq[f_idx]
            else:
                freqs[i] = 0
        return freqs

    def summarise(
        self, S: ndarray or Tensor, batchsize: int = 1000
    ) -> ndarray or Tensor:
        """Summarises the outputs of the Lotka Voltera model outputs.

        Considers: {
                    min_x, min_y,
                    max_x, max_y,
                    mean_x, mean_y,
                    var_x, var_y,
                    skew_x, skew_y,
                    kurt_x, kuert_y,
                    freq,
                    x(T), y(T)
                    }

        Args:
            Xs: Set of timeseries of shape (N, n_samples, 2).
            batchsize: How many samples are simulated simultaneaously.


        Returns:
            stats: Array of summary statistics. (n_samples, 15)
        """
        S = np.array(S)

        N = S.shape[1]
        n_batches = int(N / batchsize)

        if N > batchsize:
            S_batched = np.array_split(S, n_batches, axis=1)
        else:
            S_batched = [S]

        stats = []
        for i, S_batch in enumerate(S_batched):
            print("{}/{}".format(i * batchsize, N))
            _, minmax, *moments = describe(S_batch, axis=0)
            A = np.hstack(minmax)
            B = np.hstack(moments)

            C = np.hstack([A, B])
            D = self.extract_freq(S_batch)
            E = S_batch[-1]
            batch_stats = np.hstack([C, D, E])
            stats.append(batch_stats)

        stats = np.vstack(stats)
        if self.torch_io:
            return torch.from_numpy(stats)
        else:
            return stats


class SimpleHH(HHSimulator):
    """Implements a basic HH version with 5 parameters, that can be used for model testing.

    The model also includes a prior already.

    The model parameters are as follows:
        53.0,  # ENa [mV] : Reversal potential of sodium.
        -107.0,  # EK [mV] : Reversal potential of potassium.
        131.1,  # ECa [mV] : Reversal potential of calcium.
        36.0,  # T_1 [°C] :  36 °C from paper MartinPospischil et al.
        34.0,  # T_2 [°C] :  Experimental temperature.
        3.0,  # Q10 : temperature coeff.
        10.0,  # tau [ms] : Membrane time constant.
        100.0,  # input_res [Mohm] : Membrane input resistance.
        1.0,  # C [uF/cm^2] : Membrane capcitance per area.
        float("nan"),  # gNa [mS] : Channel conductance of sodium.
        float("nan"),  # gK [mS] : Channel conductance of potassium.
        0.07,  # gM [mS] : Channel conductance for adpt. K currents.
        float("nan"),  # gLeak [mS] : Leak conductance.
        0.0,  # gL [mS] : Channel conductance for Calcium current.
        600,  # tau_max [s] :
        float("nan"),  # VT [mV] : Threshold voltage.
        float("nan"),  # Eleak  [mV] : Reversal potential of leak currents.
        1.0,  # rate_to_SS_factor : Correction factor.
        70 # V0 [mV] : The starting voltage.

    Along with all other methods this class supports `__call__`, which runs the
    `simulate_and_summarise` method of `HHSimulator`, with V0=70.

    Args:
        cythonise: Whether to comile with cython or to use numpy.
    """

    def __init__(self, cythonise: bool = True):
        super().__init__(cythonise=cythonise)
        self.model_params = torch.tensor(
            [
                [
                    53.0,  # ENa [mV] : Reversal potential of sodium.
                    -107.0,  # EK [mV] : Reversal potential of potassium.
                    131.1,  # ECa [mV] : Reversal potential of calcium.
                    36.0,  # T_1 [°C] :  36 °C from paper MartinPospischil et al.
                    34.0,  # T_2 [°C] :  Experimental temperature.
                    3.0,  # Q10 : temperature coeff.
                    10.0,  # tau [ms] : Membrane time constant.
                    100.0,  # input_res [Mohm] : Membrane input resistance.
                    1.0,  # C [uF/cm^2] : Membrane capcitance per area.
                    float("nan"),  # gNa [mS] : Channel conductance of sodium.
                    float("nan"),  # gK [mS] : Channel conductance of potassium.
                    0.07,  # gM [mS] : Channel conductance for adpt. K currents.
                    float("nan"),  # gLeak [mS] : Leak conductance.
                    0.0,  # gL [mS] : Channel conductance for Calcium current.
                    600,  # tau_max [s] :
                    float("nan"),  # VT [mV] : Threshold voltage.
                    float("nan"),  # Eleak  [mV] : Reversal potential of leak currents.
                    1.0,  # rate_to_SS_factor : Correction factor.
                ]
            ]
        )

        # instantiate simulator and set a stimulus
        stimulus_protocol = {
            "dt": 0.04,
            "duration": 800,
            "stim_end": 700,
            "stim_onset": 100,
            "I": 200,
        }
        self.set_stimulus(protocol_params=stimulus_protocol)

        lower, upper = [
            [5.0000e-01, 1.0000e-04, 1.0000e-04, -9.0000e01, -1.1000e02],
            [80.0000, 30.0000, 0.8000, -40.0000, -50.0000],
        ]
        self.prior = BoxUniform(lower, upper)

        self.x_o = torch.tensor(
            [
                [
                    -4.4565e01,
                    9.3398e01,
                    1.3600e00,
                    -8.5289e00,
                    -4.3625e01,
                    8.9088e01,
                    1.3200e00,
                    -1.1349e01,
                    1.7000e01,
                    5.0000e00,
                    6.0000e00,
                    1.0000e01,
                    7.0000e00,
                    -6.9745e-02,
                    4.9409e-01,
                    -4.1154e00,
                    1.8164e-01,
                    -1.0903e00,
                    1.6938e00,
                    -6.4249e01,
                    -5.0359e01,
                    3.0874e02,
                    4.1981e00,
                ]
            ]
        )
        self.theta_o = torch.tensor([[25.4444, 0.9454, 0.0870, -61.0779, -57.4112]])

    def simulate(
        self, theta: Optional[Tensor] = None, n_workers: int = 12, batch_size: int = 50, seed: int = 0
    ) -> Tensor:
        """Runs `simulate_and_summarise` of `HHSimulator.

        Args:
            theta: simulator parameters. If none is provided, theta_o will be
                used as default. 
            n_workers: How many threads to use for simulation.
            batch_size: How many simulations per thread.

        Returns:
            x: Summary statistics for the simulated parameter set.
        """
        V0 = 70
        if theta is None:
            theta = self.theta_o
        return self.simulate_and_summarise(
            theta, V0=V0, n_workers=n_workers, batch_size=batch_size, rnd_seed=seed,
        )
