from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from nflows.flows import Flow
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior

# types
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.utils.data import DataLoader, TensorDataset

import fslm.utils as fi_utils


def build_reducable_posterior(inference_obj, **kwargs):
    posterior = inference_obj.build_posterior(**kwargs)
    return ReducablePosterior(posterior)


def ReducablePosterior(posterior: RejectionPosterior or MCMCPosterior):
    r"""Factory function to wrap `MCMCPosterior` or `RejectionPosterior`.

    Provides passthrough to wrap the posterior instance in its respective
    reducable class wrapper `ReducableRejectionPosterior` or
    `ReducableMCMCPosterior` depending on the instance's class.

    Example:
        ```
        posterior = infer(simulator, prior, "SNLE_A", num_simulations)
        reducable_posterior = ReducablePosterior(posterior)
        reducable_posterior.marginalise(list_of_dims_to_keep)
        reducable_posterior.sample()
        ```
    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Returns:
        ReducableRejectionPosterior` or `ReducableMCMCPosterior`
    """
    if isinstance(posterior, RejectionPosterior):
        return ReducableRejectionPosterior(deepcopy(posterior))
    elif isinstance(posterior, MCMCPosterior):
        return ReducableMCMCPosterior(deepcopy(posterior))


class ReducableBasePosterior:
    r"""Provides marginalisation functionality to for `MCMCPosterior` and
    `RejectionPosterior`.


    Args:
        posterior: sbi `MCMCPosterior` or `RejectionPosterior` instance that has
            been trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior or RejectionPosterior) -> None:
        self._wrapped_posterior = posterior

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_posterior, attr)

    def marginalise(
        self, dims: List[int], inplace: bool = True
    ) -> Optional[ReducablePosterior]:
        r"""Marginalise likelihood distribution of the likelihood-based posterior.

        Marginalises the MDN-based likelihood $p(x_1, ..., x_N|\theta)$ such that
        $$
        p(\theta|x_{subset}) \propto p(x_{subset}|\theta) p(\theta)
        $$
        , where $x_{susbet} \susbet (x_1, ..., x_N)$

        Args:
            dims: Feature dimensions to keep.
            inplace: Whether to return a marginalised copy of self, or to
                marginalise self directly.

        Returns:
            red_posterior: If inplace=False, returns a marginalised copy of self.
        """
        likelihood_estimator = self.potential_fn.likelihood_estimator
        marginal_likelihood = ReducableLikelihoodEstimator(likelihood_estimator, dims)
        if inplace:
            self.potential_fn.likelihood_estimator = marginal_likelihood
        else:
            red_posterior = deepcopy(self)
            red_posterior.potential_fn.likelihood_estimator = marginal_likelihood
            return red_posterior


class ReducableRejectionPosterior(ReducableBasePosterior, RejectionPosterior):
    r"""Wrapper for `RejectionPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features p(\theta|x1)
    of x_o = (x1, x2).

    Args:
        posterior: `RejectionPosterior` instance trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: RejectionPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableMCMCPosterior(ReducableBasePosterior, MCMCPosterior):
    r"""Wrapper for `MCMCPosterior` that make use of a
    MDN as their density estimator. Implements functionality to evaluate
    and sample the posterior based on a reduced subset of features $p(\theta|x1)$
    of $x_o = (x1, x2)$.

    Args:
        posterior: `MCMCPosterior` instance that trained using an MDN.

    Attributes:
        Inherits all its attributes from posterior instance.
    """

    def __init__(self, posterior: MCMCPosterior) -> None:
        self._wrapped_posterior = posterior


class ReducableLikelihoodEstimator:
    r"""Adds marginalisation functionality to mdn based likelihood estimators.

    Supports `.log_prob` of the MoG likelihood.

    Its main purpose is to emulate the likelihood estimator employed in
    `LikelihoodbasedPotential`.

    Args:
        likelihood_estimator: Flow instance that was trained using a MDN.
        marginal_dims: List of x dimensions to consider. Dimensions not
            in `marginal_dims` are marginalised out.

    Attributes:
        likelihood_net: Conditional density estimator for the likelihood.
        dims: List of x dimensions to consider in the evaluation.
    """

    def __init__(
        self, likelihood_estimator: Flow, marginal_dims: Optional[List[int]] = None
    ) -> None:
        self.likelihood_net = likelihood_estimator
        self.dims = marginal_dims

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self.likelihood_net.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self.likelihood_net.eval()

    def marginalise(
        self, context: Tensor, dims: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Marginalise MoG and return new mixture parameters.

        Args:
            context: Condition in $p(x|\theta)$.
            dims: List of dimensions to keep.

        Returns:
            logits: log-weights for each component of the marginal distributions.
            mu_x: means of the marginal distributution for each component.
            precfs_xx: precision factors of the marginal distribution for
                each component.
            sumlogdiag: Sum of the logarithms of the diagonal elements of the
                precision factors of the marginal distributions for each component.
        """

        # reset to unmarginalised params
        logits, means, precfs, _ = fi_utils.extract_and_transform_mog(
            self.likelihood_net, context
        )

        mask = torch.zeros(means.shape[-1], dtype=bool)
        mask[dims] = True

        # Make a new precisions with correct ordering
        mu_x = means[:, :, mask]
        precfs_xx = precfs[:, :, mask]
        precfs_xx = precfs_xx[:, :, :, mask]

        # set new GMM parameters
        sumlogdiag = torch.sum(
            torch.log(torch.diagonal(precfs_xx, dim1=2, dim2=3)), dim=2
        )
        return logits, mu_x, precfs_xx, sumlogdiag

    def log_prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Evaluate the Mixture of Gaussian (MoG)
        probability density function at a value x.

        Args:
            inputs: Values at which to evaluate the MoG pdf.
            context: Conditiones the likelihood distribution.

        Returns:
            Log probabilities at values specified by theta.
        """
        logits, means, precfs, sumlogdiag = self.marginalise(context, self.dims)
        prec = precfs.transpose(3, 2) @ precfs

        return mdn.log_prob_mog(inputs[:, self.dims], logits, means, prec, sumlogdiag)


class NaNCalibration(nn.Module):
    r"""Learns calibration bias to compensate for NaN observations.

    Logistic regressor predicts which parameters cause valid (no NaNs) 
    observations. The model is optimised with a binary cross entropy loss.

    The forward pass computes $p(valid|\theta)$.

    For reference (https://openreview.net/pdf?id=kZ0UYdhqkNY)

    Args:
        input_dim: Number of parameters in $\theta$.
        device: On which device to train.

    Attributes:
        linear: Linear layer.
        device: Which device is used.
        n_params: Number of input params.
    """

    def __init__(self, input_dim: int, device: str = "cpu"):
        super(NaNCalibration, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.device = device
        self.n_params = input_dim

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Implements logistic regression

        Args:
            inputs: theta.

        Returns:
            outputs: outputs of the forward pass. $p(valid|\theta)$
        """
        outputs = torch.sigmoid(self.linear(inputs))
        return outputs

    def log_prob(self, theta: Tensor) -> Tensor:
        r"""log of the forward pass, i.e. $p(valid|\theta)$.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            $log(p(valid|\theta))$.
        """
        probs = self.forward(theta.view(-1, self.n_params))
        return torch.log(probs)

    def train(
        self,
        theta_train: Tensor,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> NaNCalibration:
        r"""Trains classifier to predict the probability of valid observations.

        Returns self for calls such as:
        `nan_likelihood = NaNCallibration(n_params).train(theta, y)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Observation labels. 0 if x_i contains no NaNs, 1 if does.
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation 
                improvement.
            val_ratio: How any training_examples to split of for validation.
            
        Returns:
            self
        """
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_split = int(len(theta_train) * val_ratio)

        train_data = TensorDataset(theta_train[n_split:], y_train[n_split:])
        val_data = TensorDataset(theta_train[:n_split], y_train[:n_split])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        best_val_log_prob = 0
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            train_log_probs_sum = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                theta_batch, y_batch = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )
                train_losses = criterion(self.forward(theta_batch), y_batch)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                optimizer.step()

                train_log_prob_average = train_log_probs_sum / (
                    len(train_loader) * train_loader.batch_size
                )

            val_log_probs_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, y_batch = (
                        batch[0].to(self.device),
                        batch[1].to(self.device),
                    )
                    val_losses = criterion(self.forward(theta_batch), y_batch)
                    val_log_probs_sum -= val_losses.sum().item()

            val_log_prob_average = val_log_probs_sum / (
                len(val_loader) * val_loader.batch_size
            )

            if epoch == 0 or val_log_prob_average > best_val_log_prob:
                best_val_log_prob = val_log_prob_average
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > stop_after_epoch:
                print(f"converged after {epoch} epochs.")
                break

            if verbose:
                print(
                    "\r[{}] train loss: {:.5f}, val_loss: {:.5f}".format(
                        epoch, train_log_prob_average, val_log_prob_average
                    ),
                    end="",
                )
        return self


class CalibratedPrior(Distribution):
    r"""Prior distribution that can be calibrated to compensate for the
    likelihood bias, that is caused by ignoring non-valid observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given prior distribution $p(\theta)$, we can hence obtain a calibratedd
    prior $\tilde{p}(\theta)$ according to:
    $$
    \tilde{p}(\theta) \propto p(\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Its has a modified `.log_prob()` method compared to the base_prior. While,
    sampling is passed on to the base prior.
    The support is inherited from the base prior distribution.


    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_prior = CallibratedPrior(inference._prior).train(theta,x)
    posterior = inference.build_posterior(prior=calibrated_prior, sample_with='rejection')
    ```

    Args:
        prior: A base_prior distribution.
        device: Which device to use.

    Attributes:
        base_prior: The prior distribution that is optimised.
        dim: The dimensionality of $\theta$.
        nan_likelihood: Classifier to predict if $\theta$ will produce NaNs in
            observation.
    """

    def __init__(self, prior: Any, device: str = "cpu"):
        r"""
        Args:
            prior: A prior distribution that supports `.log_prob()` and
                `.sample()`.
            device: Which device to use. Should be same as for `prior`.
        """
        self.base_prior = prior
        self.dim = prior.sample((1,)).shape[1]
        self.nan_calibration = None
        self.device = device
        self._mean = prior.mean
        self._variance = prior.variance

    @property
    def mean(self):
        if self.nan_calibration is None:
            return self.base_prior.mean
        else:
            return self._mean

    @property
    def variance(self):
        if self.nan_calibration is None:
            return self.base_prior.variance
        else:
            return self._variance

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.base_prior.arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.base_prior.support

    def log_prob(self, theta: Tensor) -> Tensor:
        """Prob of calibrated prior.

        Args:
            theta: Simulator parameters, (batch_size, num_params),

        Returns:
            p_nan: log_probs for $p(valid|\theta)$
        """
        if self.nan_calibration is None:
            warnings.warn(
                "Evaluating non calibrated prior! To calibrate, call .train() first!"
            )
            return self.base_prior.log_prob(theta)
        else:
            p_no_nan = self.nan_calibration(theta.view(-1, self.dim)).view(-1)
            p = self.base_prior.log_prob(theta)
            return p + p_no_nan

    def sample(self, sample_shape: Tuple) -> Tensor:
        """Sample the calibrated prior distribution.

        Samples the calibrated prior via rejection sampling in the support of
        the original prior.

        Args:
            sample_shape: Shape of the sample. (n_batches, batch_size)

        Returns:
            Samples from the calibrated prior.
        """
        if self.nan_calibration is None:
            warnings.warn(
                "Sampling non calibrated prior! To optimise, call .train() first!"
            )
            return self.base_prior.sample(sample_shape)
        else:
            n_samples = torch.Size(sample_shape).numel()
            n = 0
            samples = []
            # rejection sampling | could be replaced by sbi's rejection_sample
            # return rejection_sample(self, self.base_prior, num_samples)

            while n < n_samples:
                theta = self.base_prior.sample((n_samples - n,))
                acceptance_probs = self.nan_calibration(theta).view(-1)
                accepted = acceptance_probs > torch.rand_like(acceptance_probs)
                samples.append(theta[accepted])
                n = len(torch.vstack(samples))
            return torch.vstack(samples).reshape((*sample_shape, -1))

    def train(
        self,
        theta_train: Tensor,
        x_train: Tensor = None,
        y_train: Tensor[bool] = None,
        lr: float = 1e-3,
        batch_size: int = 1000,
        max_epochs: int = 5000,
        val_ratio=0.1,
        stop_after_epoch: int = 50,
        verbose: bool = True,
    ) -> CalibratedPrior:
        r"""Trains classifier to predict which parameters produce valid observations.

        Calibration factor is then added to log_prob of `base_prior`.

        The model is a logistic regressor optimised with a binary cross entropy
        loss.

        Returns self for calls such as:
        `trained_prior = CalibratedPrior(prior).train(theta, x)`

        Args:
            theta_train: Sets of parameters for training.
            x_train: Set of training observations that some of which include
                NaN features. Will be used to create labels y_train, depending
                on presence of NaNs (0 if x_i contains NaN, else 1).
            y_train: Labels whether corresponding theta_train produced an
                observation x_train that included NaN features. valid = 1
            lr: Learning rate.
            num_epochs: Number of epochs to train for.
            batch_size: Training batch size.
            batch_size: Training batch size.
            stop_after_epoch: Stop after number of epochs without validation 
                improvement.
            val_ratio: How any training_examples to split of for validation.

        Returns:
            self trained calibrated prior.
        """
        if x_train is not None:
            has_nan = fi_utils.includes_nan(x_train).view(-1, 1)
        elif y_train is not None:
            has_nan = y_train.view(-1, 1)
        else:
            raise ValueError("Please provide y_train or x_train.")

        self.nan_calibration = NaNCalibration(self.dim, device=self.device).train(
            theta_train,
            (~has_nan).float(),
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            stop_after_epoch=stop_after_epoch,
            verbose=verbose,
        )

        # estimating mean and variance from samples
        samples = self.sample((10000,))
        self._mean = samples.mean()
        self._variance = samples.var()

        return self


class CalibratedLikelihoodEstimator:
    r"""Modifies the likelihood by a calibration factor.

    Wraps the trained likelihood estimator and applies a calibration term to
    compensate for discarded training data due to NaN observations.

    Since the likelihood obtained through SNLE is biased according to
    $\ell_ψ(x_o |θ) \approx 1/Z p(x_o|θ)/p(valid|θ)$, we can account for this
    bias by estimating $c_ζ(θ)=p(valid|θ)$, such that:
    $$
    \ell_ψ(x_o|θ)\ell_ψ(x_o|θ)p(θ)c_ζ(θ) \propto p(x_o|θ)p(θ) \propto p(θ|x_o )
    $$

    For a given likelihood distribution $p(x|\theta)$, we can hence obtain a
    calibrated likelihood $\tilde{p}(x|\theta)$ according to:
    $$
    \tilde{p}(x|\theta) \propto p(x|\theta)c_ζ(θ).
    $$
    Here $c_ζ(θ)$ is a logistic regressor that predicts whether a set of
    parameters $\theta$ produces valid observations features.

    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    calibrated_likelihood = calibrate_likelihood_estimator(estimator, theta, x)
    posterior = inference.build_posterior(density_estimator=calibrated_likelihood, sample_with='rejection')
    ```

    Args:
        likelihood_estimator: A likelihood estimator (a Flow).
        calibration_f: A trained callibration network that has learned
            $p(valid|\theta)$.

    Attributes:
        calibration_f: calibration factor for likelihood that has been trained
        partially on NaNs.
    """

    def __init__(self, likelihood_estimator: Flow, calibration_f: NaNCalibration):
        self._wrapped_estimator = likelihood_estimator
        self.calibration_f = calibration_f

    def __getattr__(self, attr):
        """Forward attrs to wrapped object if not existant in self."""
        if attr == "log_prob":
            return getattr(self._wrapped_estimator, attr)
        elif attr in self.__dict__:
            return getattr(self, attr)

    def parameters(self) -> Generator:
        """Provides pass-through to `parameters()` of self.likelihood_net.

        Used for infering device.

        Returns:
            Generator for the model parameters.
        """
        return self._wrapped_estimator.parameters()

    def eval(self) -> Flow:
        """Provides pass-through to `eval()` of self.likelihood_net.

        Returns:
            Flow model.
        """
        return self._wrapped_estimator.eval()

    def log_prob(self, inputs: Tensor, context: Optional[Tensor] = None, track_gradients=False) -> Tensor:
        r"""calibrated likelihood.

        Adds the calibration log_prob $p(valid|\theta)$ on top of the likelihood
        log_prob.

        Args:
            inputs: where to evaluate the likelihood.
            context: context of the likelihood.

        Returns:
            calibrated likelihoods.
        """
        return self._wrapped_estimator.log_prob(inputs, context) + torch.log(
            self.calibration_f(context)
        )


def calibrate_likelihood_estimator(
    estimator: Flow, theta: Tensor, x: Tensor, **train_kwargs
) -> CalibratedLikelihoodEstimator:
    r"""Calibrates the likelihood estimator if partially trained on NaNs.

    It learns a calibration function $c_ζ(θ)=p(valid|θ)$ on the training data an applies it to the
    likelihood.

    Args:
        estimator: likelihood estimator with log_prob method.
        theta: parameters.
        x: observations.

    Returns:
        a calibrated likelihood function.
    """
    calibration_f = NaNCalibration(int(theta.shape[1]))
    y = fi_utils.includes_nan(x).float()
    calibration_f.train(theta, y, **train_kwargs)
    return CalibratedLikelihoodEstimator(deepcopy(estimator), calibration_f)
