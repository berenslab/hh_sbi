from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Generator, Optional, Tuple

import os
import torch
import torch.nn as nn
from nflows.flows import Flow
from pyknos.nflows.nn import nets
from sbi.utils.sbiutils import Standardize, standardizing_net
from torch import Tensor, relu
from torch.distributions import Distribution, constraints
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import numpy as np
from pynwb import NWBHDF5IO

def get_path2obs_nwb(sample_name, root_dir):
    fpaths = (os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames)
    fpaths = (f for f in fpaths if os.path.splitext(f)[1] == '.nwb')

    for fpath in fpaths:
        if sample_name.replace("_", "-") in fpath:
            return fpath

    raise ValueError("This does not point to a valid file")

def get_time_voltage_current_currindex0(nwb):
    """taken from hh_sbi/code/ephys_utils.py by Yves Bernaerts"""
    df = nwb.sweep_table.to_dataframe()
    voltage = np.zeros((len(df['series'][0][0].data[:]), int((df.shape[0]+1)/2)))
    time = np.arange(len(df['series'][0][0].data[:]))/df['series'][0][0].rate
    #time = np.array([round(c, 5) for c in time])
    voltage[:, 0] = df['series'][0][0].data[:]
    current_initial = df['series'][1][0].data[12000]*df['series'][1][0].conversion
    curr_index_0 = int(-current_initial/20) # index of zero current stimulation
    current = np.linspace(current_initial, (int((df.shape[0]+1)/2)-1)*20+current_initial, \
                         int((df.shape[0]+1)/2))
    for i in range(curr_index_0):   # Find all voltage traces from minimum to  0 current stimulation
        voltage[:, i+1] = df['series'][0::2][(i+1)*2][0].data[:]
    for i in range(curr_index_0, int((df.shape[0]+1)/2)-1):   # Find all voltage traces from 0 to highest current stimulation
        voltage[:, i+1] = df['series'][1::2][i*2+1][0].data[:]
    voltage[:, curr_index_0] = df.loc[curr_index_0*2][0][0].data[:]    # Find voltage trace for 0 current stimulation
    return time, voltage, current, curr_index_0

def get_Vt_o(sample_name, root_dir):
    with warnings.catch_warnings(): # this relates to some subfields in the .nwb object that aren't there so a warning is throwed everytime
        warnings.simplefilter("ignore")
        data = NWBHDF5IO(get_path2obs_nwb(sample_name, root_dir), "r", load_namespaces=True).read()
    t_o, Vt_o, It_o, _ = get_time_voltage_current_currindex0(data)

    Vt_o = Vt_o[:,np.where(It_o == 300)].reshape(-1)

    liquid_junction_potential=15.4
    Vt_o-=liquid_junction_potential

    return t_o[:20000], Vt_o[:20000]

class BiasEstimator(nn.Module):
    r"""Learns calibration term to compensate for simulations withheld during
    training of a neural likelihood estimate.

    Logistic regressor predicts which parameters cause simulations to be
    considered valid and approved for training according to some rule.
    The model is optimised with a cross entropy loss.

    The forward pass computes $p(valid |\theta)$. Where N is the closest
    N points to x_o.

    Args:
        input_dim: Number of parameters in $\theta$.
        device: On which device to train.
        summarywriter: Logs metrics during training.
        model: The selected model.

    Attributes:
        logreg: Linear layer with sigmoid activation for logistic regression.
        mlp: mlp for logistic regression.
        model: The selected model.
        device: Which device is used.
        n_params: Number of input params.
        summarywriter: Logs metrics during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim=50,
        model: str = "mlp",
        device: str = "cpu",
        summarywriter=None,
    ):
        super(BiasEstimator, self).__init__()
        self.device = device
        self.n_params = input_dim
        self.set_summarywriter(summarywriter)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.model = self.build_model(model, hidden_dim)

    def z_score_inputs(self, batch_theta: Tensor):
        """Prepend z-score layer to classifier.

        Args:
            theta_batch: input batch to use for calculating the z-transform.
        """
        input_layer = standardizing_net(batch_theta, False)
        input_layer.to(self.device)
        key, layer = next(iter(self.model._modules.items()))
        if isinstance(layer, Standardize):
            self.model_modules[key] = input_layer
        else:
            self.model = nn.Sequential(input_layer, self.model)

    def build_model(self, model: str, hidden_dim: int) -> nn.Module:
        """Build the classifier.

        Args:
            model: String that specifies model.
            hidden_dim: Number of hidden units.

        Returns:
            model

        Raises:
            ValueError if input string is not recognised."""
        if model == "mlp":
            model = nn.Sequential(
                nn.Linear(self.n_params, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        elif model == "logreg":
            model = nn.Sequential(nn.Linear(self.n_params, 2))
        elif model == "resnet":
            model = nets.ResidualNet(
                in_features=self.n_params,
                out_features=2,
                hidden_features=hidden_dim,
                context_features=None,
                num_blocks=2,
                activation=relu,
                dropout_probability=0.5,
                use_batch_norm=True,
            )
        else:
            raise ValueError("Select a supported model")

        model.to(self.device)
        return model

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Implements logistic regression

        Args:
            inputs: theta.

        Returns:
            outputs: outputs of the forward pass. $p(valid |\theta)$
        """
        outputs = self.model(inputs)
        return self.logsoftmax(outputs)

    def log_prob(
        self, theta: Tensor, valid: int = 1, track_gradients: bool = False
    ) -> Tensor:
        r"""log of the forward pass, i.e. $p(valid|\theta)$.

        Args:
            theta: Simulator parameters, (batch_size, num_params),
            valid: Whether theta produces valid sims. valid = 1.

        Returns:
            $log(p(valid |\theta))$.
        """
        self.model.eval()

        with torch.set_grad_enabled(track_gradients):
            log_probs = self.forward(theta.view(-1, self.n_params))
        return log_probs[:, valid]

    def set_summarywriter(self, summarywriter: Any):
        """Sets summary writer for the training logs.

        If None, then logging will be sunk.

        summarywriter needs to be an object with a `.log` method, that takes
        a dictionary as input.

        Args:
            summarywriter: object with `.log` to record training metrics.
        """

        class DummyWriter:
            def __init__(self):
                pass

            def log(self, dct: Dict):
                pass

        if summarywriter is None:
            dummy = DummyWriter()
            self.summarywriter = dummy
        else:
            self.summarywriter = summarywriter

    def get_normalisation(self) -> Tensor:
        """Return normalisation constant Z = p(valid)."""
        return self._norm

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
    ) -> BiasEstimator:
        r"""Trains classifier to predict the probability of observations being
        valid.

        Returns self for calls such as:
        `bias_estimate = BiasEstimator(n_params).train(theta, y)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Observation labels. valid = 1, invalid = 0.
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
        y_train = y_train.view(-1)
        valid_one_hot = torch.vstack([~y_train, y_train])

        Z = torch.log(valid_one_hot.count_nonzero(axis=1) / len(y_train))
        self._norm = Z  # lambda function for norm / Z provokes pickling error

        y_train = y_train.long()  # bool -> long

        # logsoftmax + nll = softmax + cross entropy
        # weights according to prior acceptance probs
        criterion = torch.nn.NLLLoss(weight=1 - torch.exp(Z))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        n_split = int(len(theta_train) * val_ratio)

        train_data = TensorDataset(theta_train[n_split:], y_train[n_split:])
        val_data = TensorDataset(theta_train[:n_split], y_train[:n_split])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        best_val_log_prob = 0
        epochs_without_improvement = 0

        self.model.train(True)
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

            train_info = {
                "train_loss": train_log_prob_average,
                "val_loss": val_log_prob_average,
            }
            self.summarywriter.log(train_info)

            if verbose:
                print(
                    "\r[{}] train loss: {:.5f}, val_loss: {:.5f}".format(
                        epoch, train_log_prob_average, val_log_prob_average
                    ),
                    end="",
                )
        self.model.train(False)

        return self


class CalibratedPrior(Distribution):
    r"""Prior distribution that can be calibrated to compensate for the
    likelihood bias, that is caused by discarding selected simulations from the
    prior.

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

    Its has a modified `.log_prob()` method compared to the base_prior.
    The support is inherited from the base prior distribution.


    Example:
    ```
    inference = SNLE_A(prior, show_progress_bars=True, density_estimator="mdn")
    estimator = inference.append_simulations(theta, x).train()
    cal_prior = CalibratedPrior(inference._prior).train(theta,x)
    posterior = inference.build_posterior(prior=cal_prior, sample_with='rejection')
    ```

    Args:
        prior: A base_prior distribution.
        device: Which device to use.

    Attributes:
        base_prior: The prior distribution that is optimised.
        dim: The dimensionality of $\theta$.
        bias_calibration: Classifier to predict if $\theta$ will produce
        observations close enough to x_o.
    """

    def __init__(
        self,
        prior: Any,
        trained_bias_estimator: Optional[BiasEstimator] = None,
        device: str = "cpu",
    ):
        r"""
        Args:
            prior: A prior distribution that supports `.log_prob()` and
                `.sample()`.
            device: Which device to use. Should be same as for `prior`.
        """
        self.base_prior = prior
        self.dim = prior.sample((1,)).shape[1]
        self.bias = None
        self.device = device
        self._mean = prior.mean
        self._variance = prior.variance

        if trained_bias_estimator is not None:
            self.insert_trained_estimator(trained_bias_estimator)

    @property
    def mean(self):
        if self.bias is None:
            return self.base_prior.mean
        else:
            return self._mean

    @property
    def variance(self):
        if self.bias is None:
            return self.base_prior.variance
        else:
            return self._variance

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.base_prior.arg_constraints

    @property
    def support(self) -> constraints.Constraint:
        return self.base_prior.support

    def log_prob(self, theta: Tensor, valid: int = 1) -> Tensor:
        """Prob of calibrated prior.

        Args:
            theta: Simulator parameters, (batch_size, num_params).
            valid: Whether theta produces valid sims. valid = 1.

        Returns:
            log_p_valid: log_probs for $p(valid|\theta)$
        """
        if self.bias is None:
            warnings.warn(
                "Evaluating non calibrated prior! To calibrate, call .train() first!"
            )
            return self.base_prior.log_prob(theta)
        else:
            with torch.no_grad():
                p_valid = self.bias.log_prob(theta.view(-1, self.dim), valid)
                p = self.base_prior.log_prob(theta)
                Z = self.Z(valid)
            return (p_valid + p - Z).squeeze()

    def Z(self, valid: int = 1) -> Tensor:
        """Normalisation constant p(valid).

        Args:
            valid: Prior log_prob of producing invalid outputs = 0,
                vs. valid outputs = 1.

        Returns:
            Marginal log_prob of producing NaNs.
        """
        return self.bias.get_normalisation()[valid]

    def sample(
        self, sample_shape: Tuple = (1,), valid: int = 1, show_progress=False
    ) -> Tensor:
        """Sample the calibrated prior distribution.

        Samples the calibrated prior via rejection sampling in the support of
        the original prior.

        Args:
            sample_shape: Shape of the sample. (n_batches, batch_size)
            valid: Whether theta produces valid sims. valid = 1.

        Returns:
            Samples from the optimised prior.
        """
        if self.bias is None:
            warnings.warn(
                "Sampling non optimised prior! To optimise, call .train() first!"
            )
            return self.base_prior.sample(sample_shape)
        else:
            n_samples = torch.Size(sample_shape).numel()
            n = 0
            samples = []
            # rejection sampling | could be replaced by sbi's rejection_sample
            # return rejection_sample(self, self.base_prior, num_samples)

            pbar = tqdm(
                disable=not show_progress,
                total=n_samples,
                desc=f"Drawing {n_samples} prior samples",
            )
            while n < n_samples:
                theta = self.base_prior.sample((n_samples - n,))
                p_accept = torch.exp(self.bias.log_prob(theta, valid)).view(-1)
                accepted = p_accept > torch.rand_like(p_accept)
                samples.append(theta[accepted])
                n = len(torch.vstack(samples))
                if show_progress:
                    pbar.update(n)

            pbar.close()
            return torch.vstack(samples).reshape((*sample_shape, -1))

    def insert_trained_estimator(self, estimator: BiasEstimator):
        """Insert an externally trained bias estimator.

        Args:
            estimator: Estimator of the bias for neural likelihood estimate
                with excluded simulations."""
        self.bias = estimator

        # estimating mean and variance from samples
        samples = self.sample((10000,), show_progress=False)
        self._mean = samples.mean()
        self._variance = samples.var()

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
    ) -> CalibratedPrior:
        r"""Trains classifier to predict which parameters produce observations
        close enough to the observed values.

        Calibration factor is then added to log_prob of `base_prior`.

        The model is a logistic regressor optimised with a binary cross entropy
        loss.

        Returns self for calls such as:
        `trained_prior = CalibratedPrior(prior).train(theta, x)`

        Args:
            theta_train: Sets of parameters for training.
            y_train: Labels whether corresponding theta_train produced an
                observation x_train that is discarded.
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

        is_invalid = y_train.view(-1, 1)

        self.bias = BiasEstimator(self.dim, device=self.device).train(
            theta_train,
            is_invalid,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            stop_after_epoch=stop_after_epoch,
            verbose=verbose,
        )

        # estimating mean and variance from samples
        samples = self.sample((10000,), show_progress=False)
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

    def __init__(self, likelihood_estimator: Flow, calibration_f: BiasEstimator):
        self._wrapped_estimator = deepcopy(likelihood_estimator)
        self.calibration_f = calibration_f

    def __getattr__(self, attr):
        """Forward attrs to wrapped object if not existant in self."""
        if attr not in self.__dict__:
            return getattr(self._wrapped_estimator, attr)
        elif attr in self.__dict__:
            return getattr(self, attr)
        else:
            raise AttributeError(
                f"Both {self.__class__.__name__} and wrapped {self._wrapped_estimator.__class__.__name__} objects have no attribute '{attr}'"
            )

    def __reduce__(self):
        return (self.__class__, (self._wrapped_estimator, self.calibration_f))

    def log_prob(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tensor:
        r"""calibrated likelihood.

        Adds the calibration log_prob $p(valid|\theta)$ on top of the likelihood
        log_prob.

        Args:
            inputs: where to evaluate the likelihood.
            context: context of the likelihood.

        Returns:
            calibrated likelihoods.
        """
        log_p = self._wrapped_estimator.log_prob(inputs, context)
        log_p_valid = self.calibration_f.log_prob(context, valid=1)
        return log_p + log_p_valid


# def includes_nan(X: Tensor) -> Tensor:
#     """Checks if obsevation contains NaNs.

#     Args:
#         X: Batch of observations x_i, (batch_size, num_features).

#     Returns:
#         True if x_i contains NaN feature.
#     """
#     has_inf = torch.any(X.isinf(), axis=1)
#     has_nan = torch.any(X.isnan(), axis=1)
#     return torch.logical_or(has_inf, has_nan)

# class ZTransform:
#     def __init__(self, X):
#         self.m = X.mean(dim=0)
#         self.s = X.std(dim=0)

#     def __call__(self, x):
#         return (x-self.m)/self.s

#     def params(self):
#         return self.m, self.s

# def is_to_far(X, X_o, n_closest_per_x_o=1000):
#     tf = ZTransform(X)
#     X = tf(X)
#     X_o = tf(X_o)
#     d = torch.norm(X[:,None,:]-X_o, p=2, dim=-1, keepdim=True).squeeze().T
#     idxs = torch.argsort(d)[:, :n_closest_per_x_o]
#     closest_idxs = torch.unique(idxs)
#     close_enough = torch.zeros(len(X), dtype=bool)
#     close_enough[closest_idxs] = True
#     return ~close_enough

# test4validity = lambda X, X_o: is_to_far(X, X_o) or includes_nan(X)

# def closest_idxs(X, X_o, num_points=1000):
#     tf = ZTransform(X)
#     X = tf(X).numpy()
#     X_o = tf(X_o).numpy()

#     idxs = np.empty((len(X_o), num_points))
#     for i, x_o in enumerate(X_o):
#         print(f"{i+1} / {len(X_o)}")
#         ds = np.square(X-x_o).sum(axis=-1)
#         closetst_idxs = np.argsort(ds)[:num_points]
#         idxs[i] = closetst_idxs
#     return idxs.long()


def import_tree_data(path: str) -> Dict:
    """Import data from tree search.
    
    Args:
        path: path to list of fts and kl.
    
    Returns:
        Dictionary containing fts as keys and their corresponding kls as values.
    """
    with open(path, "r") as f:
        data_str = f.read()
    data_list = data_str.replace(" ", "").split("\n")
    if data_list[-1] == "":
        data_list = data_list[:-1]
    return {int(line.split(":")[0]):float(line.split(":")[1]) for line in data_list}
