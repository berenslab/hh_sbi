import re
from math import isnan
from typing import Callable, Dict, List, Optional, Tuple

import brian2 as br2
import torch
from joblib import Parallel, delayed
from scipy import stats as spstats
from torch import Tensor
from torch._C import dtype
from tqdm.auto import tqdm

# ephys extraction
import fslm4expdata.ephys_extractor as efex
import fslm4expdata.ephys_features as ft
from fslm4expdata.ephys_utils import constant_stimulus, sigmoid

# scipy bugfix slurm https://github.com/rusty1s/pytorch_geometric/issues/1001


class HHSimulator:
    """Higher level functionality for interfacing with Brian2 HH Simulator.

    Run HH models with specified stimuli and summarise their results.
    This can be done in parallel and with caching.

    Args:
        input_res: Input resistance of a cell [MOhm].
        tau: Membrane time constant of a cell [ms].
        T: Temperature at which the experiment was carried out [°C].
        E_Na: Reversal potential of sodium [mV].
        E_K: Reversal potential of potassium [mV].
        E_Ca: Reversal potential of calcium [mV].

    Attributes:
        input_res: Input resistance of a cell [MOhm].
        tau: Membrane time constant of a cell [ms].
        T: Temperature at which the experiment was carried out [°C].
        E_Na: Reversal potential of sodium [mV].
        E_K: Reversal potential of potassium [mV].
        E_Ca: Reversal potential of calcium [mV].
        cythonise: Whether to compile code to C with cython or use numpy instead.

        model_params: All parameters that are static for the HH simulator rather than dynamic.
            ENa [mV] : Reversal potential of sodium.
            EK [mV] : Reversal potential of potassium.
            ECa [mV] : Reversal potential of calcium.

            T_1 [°C] :  21 °C from paper Etay Hay et al.
            T_1 [°C] :  36 °C from paper Martin Pospischil et al.
            T_2 [°C] :  Experimental temperature.
            Q10 : temperature coeff.

            C [uF/cm^2] : Membrane capcitance per area.
            input_res [Mohm] : Membrane input resistance.
            tau [ms] : Membrane time constant.

            gNa [mS] : Channel conductance of sodium.
            gNa2 [mS] : Channel conductance of sodium.
            gK [mS] : Channel conductance of potassium.
            gM [mS] : Channel conductance for adaptive potassium currents.
            gKv31 [mS] : Channel conductance of potassium.
            gL [mS] : Channel conductance for Calcium current.
            Eleak  [mV] : Reversal potential of leak currents.

            tau_max [s] :
            VT [mV] : Threshold voltage.
            rate_to_SS_factor : Correction factor.

        t: Time axis in steps of dt [ms].
        V_t: Membrane voltage [mV].
        I_t: Stimulus [pA].
    """

    def __init__(
        self,
        input_res: Optional[float] = float("nan"),
        tau: Optional[float] = float("nan"),
        T: int or float = 34,
        E_Na: float = 71.1,
        E_K: float = -101.1,
        E_Ca: float = 131.1,
        cythonise=True,
    ):
        if cythonise:
            br2.prefs.codegen.target = "cython"  # for C compile.
        else:
            br2.prefs.codegen.target = "numpy"

        self.model_params = torch.tensor(
            [
                [
                    E_Na,  # ENa [mV] : Reversal potential of sodium.
                    E_K,  # EK [mV] : Reversal potential of potassium.
                    E_Ca,  # ECa [mV] : Reversal potential of calcium.
                    21.0,  # T_1 [°C] :  21 °C from paper Etay Hay et al.
                    36.0,  # T_1 [°C] :  36 °C from paper Martin Pospischil et al.
                    T,  # T_2 [°C] :  Experimental temperature.
                    2.3,  # Q10 : temperature coeff.
                    float("nan"),  # C [uF/cm^2] : Membrane capcitance per area.
                    input_res,  # input_res [Mohm] : Membrane input resistance.
                    tau,  # tau [ms] : Membrane time constant.
                    float("nan"),  # gNa [mS] : Channel conductance of sodium.
                    float("nan"),  # gNa2 [mS] : Channel conductance of sodium.
                    float("nan"),  # gK [mS] : Channel conductance of potassium.
                    float("nan"),  # gM [mS] : Channel conductance for adpt. K currents.
                    float("nan"),  # gKv31 [mS] : Channel conductance of potassium.
                    float("nan"),  # gL [mS] : Channel conductance for Calcium current.
                    float("nan"),  # Eleak  [mV] : Reversal potential of leak currents.
                    float("nan"),  # tau_max [s] :
                    float("nan"),  # VT [mV] : Threshold voltage.
                    float("nan"),  # rate_to_SS_factor : Correction factor.
                ]
            ]
        )

        self.t = None
        self.V_t = None
        self.I_t = None

    def set_static_parameters(
        self,
        new_model_params: Optional[Tensor] = None,
    ):
        """Setter for the static model parameters.

        0. ENa [mV] : Reversal potential of sodium.
        1. EK [mV] : Reversal potential of potassium.
        2. ECa [mV] : Reversal potential of calcium.
        3. T_1 [°C] :  21 °C from paper Etay Hay et al.
        4. T_1 [°C] :  36 °C from paper Martin Pospischil et al.
        5. T_2 [°C] :  Experimental temperature.
        6. Q10 : temperature coeff.
        7. C [uF/cm^2] : Membrane capcitance per area.
        8. input_res [Mohm] : Membrane input resistance.
        9. tau [ms] : Membrane time constant.
        10. gNa [mS] : Channel conductance of sodium.
        11. gNa2 [mS] : Channel conductance of sodium.
        12. gK [mS] : Channel conductance of potassium.
        13. gM [mS] : Channel conductance for adpt. K currents.
        14. gKv31 [mS] : Channel conductance of potassium.
        15. gL [mS] : Channel conductance for Calcium current.
        16. Eleak  [mV] : Reversal potential of leak currents.
        17. tau_max [s] :
        18. VT [mV] : Threshold voltage.
        19. rate_to_SS_factor : Correction factor.

        Non static parameters are set to NaN.

        Args:
            new_model_params: Tensor with new static and variable model parameters.
                Shape = (1, 18). Variable inputs are signaled with NaNs.
        """
        if new_model_params != None:
            self.model_params = new_model_params

    def set_stimulus(
        self,
        ts: Optional[Tensor] = None,
        It: Optional[Tensor] = None,
        protocol: str = "constant_current",
        protocol_params: Optional[dict] = None,
        noise_threshold: float = 0.10,
    ):
        """Sets a stimulus current for the HH model.

        Takes pairs of time and current values or protocol parameters, to
        initialise the input current to the HH model.

        Args:
            ts: Time axis [ms].
            It: Current values taken at each t_i [pA].
            protocol: Specifies the current protocol.
            protocol_params: Specifies the parameters of the stimulus.
                dt, duration, stim_onset, stim_end, I
            noise_threshold: Determines magnitude of fluctuations in current,
                that are attributed to noise vs. stimulus to determine.
                stim_onset, stim_end in case ts and It are provided only.
        """
        if It != None and ts != None:
            self.I_t = It
            self.t = ts
            self.dt = float(ts[1] - ts[0])
            self.t_start = 0.0
            self.t_end = float(ts[-1] + self.dt)

            noise_lvl = (
                noise_threshold * It.mean()
            )  # Potentially required better method

            self.stim_onset = float(ts[abs(It) > noise_lvl][0]) - self.dt
            self.stim_end = float(ts[abs(It) > noise_lvl][-1]) + self.dt
        else:
            assert (
                protocol_params != None
            ), "No Current or Protocol Parameters were specified."

            self.t_start = 0.0
            self.dt = protocol_params["dt"]
            self.t_end = protocol_params["duration"]
            self.stim_onset = protocol_params["stim_onset"]
            self.stim_end = protocol_params["stim_end"]

            self.t = torch.arange(self.t_start, self.t_end, self.dt)

            if "const" in protocol.lower():
                self.I_t = constant_stimulus(
                    self.t_end,
                    self.dt,
                    self.stim_onset,
                    self.stim_end,
                    protocol_params["I"],
                )

    def features(self) -> dict:
        """Produces a Dictionary with fummary features.

        Can be valuable resource for debugging the HH model.
        Makes it easy to dump available summary stats to file.

        Returns:
            feature_dict: Dictionary containing feature indexes
                and their abbreviations.
        """
        feature_dict = {
            0: "AP threshold",
            1: "AP amplitude",
            2: "AP width",
            3: "AHP",
            4: "3rd AP threshold",
            5: "3rd AP amplitude",
            6: "3rd AP width",
            7: "3rd AHP",
            8: "AP count",
            9: "AP count 1st 8th",
            10: "AP count 1st quarter",
            11: "AP count 1st half",
            12: "AP count 2nd half",
            13: "AP amp adapt",
            14: "AP average amp adapt",
            15: "AP CV",
            16: "ISI adapt",
            17: "ISI CV",
            18: "Latency",
            19: "Rest $V_m$ mean",
            20: "$V_m$ mean",
            21: "$V_m$ std",
            22: "$V_m$ skewness",
        }
        return feature_dict

    def parameters(
        self,
        include_units=True,
        show_variable_params=True,
        show_static_params=False,
        show_stimulus_params=False,
    ) -> dict:
        """Produces a Dictionary with Model Parameters.

        Can be valuable resource for debugging the HH model.
        Makes it easy to dump model settings to file.

        Args:
            include_units: Whether to show units in parameter dictionary.
            show_variable_params: Whether to show variable parameters.
            show_static_params: Whether to show fixed parameters.
            show_stimulus_params: Whether to show stimulus parameters.

        Returns:
            model_params: Dictionary containing parameter names
                and their set values.
        """
        model_params = {}

        param_values = self.model_params[0].tolist()
        param_keys = [
            r"$E_{Na}$",
            r"$E_{K}$",
            r"$E_{Ca}$",
            "T_1_hay",
            "T_1_pospischil",
            "T_2",
            "Q10",
            r"$C$",
            r"$R_{in}$",
            r"$\tau$",
            r"$g_{Na}$",
            r"$g_{Na2}$",
            r"$g_{K}$",
            r"$g_{M}$",
            r"$g_{Kv31}$",
            r"$g_{L}$",
            r"$E_{leak}$",
            r"$\tau_{max}$",
            r"$V_{T}$",
            r"$r_{SS}$",
        ]
        if include_units:
            param_keys = [
                r"$E_{Na}$ [mV]",
                r"$E_{K}$ [mV]",
                r"$E_{Ca}$ [mV]",
                "T_1_hay [°C]",
                "T_1_pospischil [°C]",
                "T_2 [°C]",
                "Q10 [1]",
                r"C $[\mu F/cm^2]$",
                r"$R_{in}$ $[M\Omega]$",
                r"$\tau$ [ms]",
                r"$g_{Na}$ [mS]",
                r"$g_{Na2}$ [mS]",
                r"$g_{K}$ [mS]",
                r"$g_{M}$ [mS]",
                r"$g_{Kv31}$ [mS]",
                r"$g_{L}$ [mS]",
                r"$E_{leak}$  [mV]",
                r"$\tau_{max}$ [s]",
                r"$V_{T}$ [mV]",
                r"$r_{SS}$ [1]",
            ]

        param_info = dict(zip(param_keys, param_values))
        if show_static_params:
            param_update = {
                key: value for key, value in param_info.items() if not isnan(value)
            }
            model_params.update(param_update)
        if show_variable_params:
            param_update = {
                key: value for key, value in param_info.items() if isnan(value)
            }
            model_params.update(param_update)

        try:
            stimulus_values = [
                self.t_start,
                self.t_end,
                self.dt,
                self.stim_onset,
                self.stim_end,
                float(max(self.I_t)),
            ]
            if include_units:
                stimulus_keys = [
                    "t_start [ms]",
                    "t_end [ms]",
                    "dt [ms]",
                    "stim_onset [ms]",
                    "stim_end [ms]",
                    "I [pA]",
                ]
            else:
                stimulus_keys = [
                    "t_start",
                    "t_end",
                    "dt",
                    "stim_onset",
                    "stim_end",
                    "I",
                ]
            stimulus_info = dict(zip(stimulus_keys, stimulus_values))
            if show_stimulus_params:
                model_params.update(stimulus_info)

        except AttributeError:
            pass

        return model_params

    def dict_init(self, dct: Dict):
        """Initialise the simualtor model with parameters imported from a dictionary.

        Args:
            dct: Dictionary with key value pairs for parameters. Dict needs to contain
                the following keys for the static model params:

                ["ENa", "EK", "ECa", "T_1", "T_1", "T_2", "Q10", "C", "input_res",
                "tau", "gNa", "gNa2", "gK", "gM", "gKv31", "gL","Eleak", "tau_max",
                "VT", "rate_to_SS_factor"]

                OR

                [r"$E_{Na}$", r"$E_{K}$", r"$E_{Ca}$", "T_1", "T_1",
                "T_2", "Q10", r"C", r"$R_{in}$",
                r"$\tau$", r"$g_{Na}$", r"$g_{Na2}$", r"$g_{K}$",r"$g_{M}$",
                r"$g_{Kv31}$", r"$g_{L}$", r"$E_{leak}$", r"$\tau_{max}$",
                r"$V_{T}$", r"$r_{SS}$"]

                and

                ["t_start", "t_end", "dt", "stim_onset", "stim_end", "I"]

                for the stimulus params.
        """

        mparam_labels_old = [
            "ENa",
            "EK",
            "ECa",
            "T_1",
            "T_2",
            "Q10",
            "tau",
            "input_res",
            "C",
            "gNa",
            "gK",
            "gM",
            "gL",
            "tau_max",
            "VT",
            "Eleak",
            "rate_to_SS_factor",
        ]

        mparam_labels_new = [
            r"$E_{Na}$",
            r"$E_{K}$",
            r"$E_{Ca}$",
            "T_1_hay [°C]",
            "T_1_pospischil [°C]",
            "T_2 [°C]",
            "Q10 [1]",
            r"C",
            r"$R_{in}$",
            r"$\tau$",
            r"$g_{Na}$",
            r"$g_{Na2}$",
            r"$g_{K}$",
            r"$g_{M}$",
            r"$g_{Kv31}$",
            r"$g_{L}$",
            r"$E_{leak}$",
            r"$\tau_{max}$",
            r"$V_{T}$",
            r"$r_{SS}$",
        ]

        mparam_labels = mparam_labels_new
        if "ENa [mV]" in dct.keys():
            mparam_labels = mparam_labels_old
        mparam_vals_ordered = []

        stimparam_labels = ["t_start", "t_end", "dt", "stim_onset", "stim_end", "I"]
        stimparam_vals_ordered = []

        for param in mparam_labels:
            mparam_vals_ordered += [
                dct[key] for key in dct.keys() if param + " " in key
            ]  # " " p revents tau and tau_max being confused
        mparams = torch.tensor([mparam_vals_ordered])

        for param in stimparam_labels:
            stimparam_vals_ordered += [
                float(dct[key]) for key in dct.keys() if param in key
            ]

        stimparams = dict(zip(stimparam_labels, stimparam_vals_ordered))
        stimparams["duration"] = stimparams["t_end"] - stimparams["t_start"]
        self.set_static_parameters(new_model_params=mparams)
        self.set_stimulus(protocol_params=stimparams)

    def simulate(
        self,
        theta: Tensor,
        V0: float or Tensor = None,
        stimulus_noise=10,
        n_workers: int = 1,
        batch_size: int = "auto",
        rnd_seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Simulates the membrane current of a HH model with the specified inputs.

        Args:
            theta: Holds the model parameters. Static and variable ones. (batch_size, num_parameters)
            V0: Initial value for the simulator. (batch_size, 1) or (1)
            stimulus_noise: std of normal noise added to the input current in PA.
            n_workers: Number threads to distribute the simulations over.
            batch_size: How many neurons to simulate in parallel per thread.
                batch_size here is not the same as in (batch_size, 1).
                The latter regards the whole input, the former takes subsamples of this.
            rnd_seed: Seed pytorch's random generator for repeatable stimulus noise.

        Returns:
            ts: Time axis.
            Vs: Membrane voltage.
            It: Input current.
        """
        N, D = theta.shape
        batch_size = self._adjust_batchsize(N, batch_size)

        if rnd_seed != None:
            torch.manual_seed(rnd_seed)

        params_batched = self._splice_model_params(theta)

        if V0 != None:
            if type(V0) is float or type(V0) is int:
                V0 = torch.tensor(V0).repeat(theta.shape[0], 1)
            params_batched = torch.hstack([params_batched, V0])

        It = self.I_t.repeat(N, 1)
        It += stimulus_noise * torch.randn_like(It)

        runHH = lambda x: HH_Br2(x, None, It, self.dt)

        results = self.split_work(runHH, params_batched, n_workers, batch_size)

        ts = torch.vstack([ts for (ts, Vs, Is) in results])
        Vs = torch.vstack([Vs for (ts, Vs, Is) in results])
        Is = torch.vstack([Is for (ts, Vs, Is) in results])

        return ts, Vs, Is

    def _splice_model_params(self, theta: Tensor) -> Tensor:
        params_batched = self.model_params.repeat(theta.shape[0], 1)
        mask = params_batched.isnan()
        params_batched[mask] = theta.view(-1)
        return params_batched

    def _adjust_batchsize(self, n_inputs: int, batch_size: int, MAX: int = 25) -> int:
        """Adjusts batch size based on the number of inputs.

        Args:
            n_inputs: Number of inputs.
            batch_size: Currently set batch size.
            MAX: Batch size limit.

        Returns:
            batch_size: Adjusted batch size."""

        if batch_size == "auto":
            if n_inputs > MAX:
                batch_size = MAX
            else:
                batch_size = n_inputs
            print("batch size was automatically set to {}.".format(batch_size))
        if batch_size == 1 and n_inputs > 50:
            batch_size = MAX
            print("batch size was automatically adjusted to {}.".format(batch_size))
        return batch_size

    @staticmethod
    def split_work(
        func: Callable, arg: Tensor, n_workers: int, batch_size: int
    ) -> Tuple[Tensor]:
        """Distributes task that takes large input tensor into batches and across threads.

        Given a function `func` with an input argument `arg`, `pathos.Parallel` is used to
        parallelise workload.

        Args:
            func: Function to be run in parallel.
            arg: Input tensor to be worked on in seperate processes.
            n_workers: Number threads to distribute the simulations over.
            batch_size: How many neurons to simulate in parallel per thread.

        Returns:
            results: Tuple that contains the results of each subprocess.
        """

        batches = None
        if type(arg) == Tensor:
            batches = torch.split(arg, batch_size)
        if type(arg) == list:
            batches = [arg[i : i + batch_size] for i in range(0, len(arg), batch_size)]

        with Parallel(n_jobs=n_workers) as parallel:
            results = parallel(delayed(func)(batch)
            for batch in tqdm(
                batches,
                disable=False,
                desc=f"Running {len(arg)} tasks with {len(batches)} batches per worker.",
                total=len(batches),
            ))
            
        return results

    @staticmethod
    def prune_simulations(
        theta: Tensor, x: Tensor, verbose: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Removes simulations resulting in NaNs and infs.

        Args:
            theta: Simulator parameters. (batch_size, num_parameters)
            x: Observations, i.e. summarised membrane voltages.
                (batch_size, num_features)

        Returns:
            theta: Simulator parameters without NaNs or infs.
                (batch_size, num_parameters)
            x: Observations, i.e. summarised membrane voltages without NaNs or infs.
                (batch_size, num_features)
        """

        not_numeric = torch.logical_or(x.isnan(), abs(x) == float("inf"))
        is_val = ~torch.any(not_numeric, dim=1)
        if verbose:
            print("{} simulations left after removing NaNs.".format(torch.sum(is_val)))
        return theta[is_val], x[is_val]

    @staticmethod
    def clean_simulations(
        x: Tensor,
        strategy: str = "recode",
        strategy_params: dict = {
            "real": "mean + 15stds",
            "pos": None,
            "noise_factor": 5,
        },
    ) -> Tuple[Tensor]:
        """Cleans simulations resulting in NaNs and infs.

        This is done by replacing NaNs and infs with numerical values.
        By default positive features they are set to -10, for real feautures,
        they are set to the feature mean + 10*stds. 1 sample std of noise is added.

        Args:
            x: Observations, i.e. summarised membrane voltages.
                (batch_size, num_features)
            strategy: Determines which strategy or combination of strategies
                is used for cleaning.
                - recode
                - noise
                - spike_feature
                - zero
            strategy_params: specifies how to deal with real valued features or
                positive integer features for example.
                - real 'How to recode real values'
                - pos 'How to recode positive values', value or None
                    in case of None, pos values are treated like real values
                - noise_factor 'Noise added to recoded features'

        Returns:
            x: Observations, i.e. summarised membrane voltages without NaNs or infs.
                (batch_size, num_features)
        """
        n_samples, n_features = x.shape
        not_numeric = torch.logical_or(x.isnan(), abs(x) == float("inf"))
        is_val = ~torch.any(not_numeric, dim=1)
        x_pruned = x[is_val]
        is_int = torch.all((x_pruned % 1) == 0, dim=0)
        assert len(x_pruned) > 0, "Batch contains only 'bad' samples."
        stds = x_pruned.std(dim=0)
        means = x_pruned.mean(dim=0)

        moveby = int(re.findall("\d+", strategy_params["real"])[0])

        x = x.clone()  # prevents overriding original observations

        if "recode" in strategy.lower():
            for i in range(n_features):
                ft_min = torch.min(x_pruned[:, i])
                nan_i = not_numeric[:, i]

                if strategy_params["pos"] == None:
                    # treat pos values like other values
                    ft_min = -1

                if ft_min > 0:
                    if is_int[i]:
                        x[nan_i, i] = strategy_params["pos"]
                        x[nan_i, i] -= torch.poisson(
                            strategy_params["noise_factor"]
                            * stds[i] ** 2
                            * torch.ones_like(x[nan_i, i])
                        )
                    else:
                        gamma = torch.distributions.gamma.Gamma(
                            (means[i] + moveby * stds[i]) ** 2
                            / (strategy_params["noise_factor"] * stds[i]) ** 2,
                            (means[i] + moveby * stds[i])
                            / (strategy_params["noise_factor"] * stds[i]) ** 2,
                        )
                        x[nan_i, i] = strategy_params["pos"]
                        if strategy_params["noise_factor"] > 0:
                            x[nan_i, i] -= gamma.sample((len(x[nan_i, i]),))
                else:
                    x[nan_i, i] = means[i] - moveby * stds[i]
                    x[nan_i, i] += (
                        strategy_params["noise_factor"]
                        * stds[i]
                        * torch.randn_like(x[nan_i, i])
                    )
                    if is_int[i]:
                        x[nan_i, i] = torch.round(x[nan_i, i])

        if "zero" in strategy.lower():
            for i in range(n_features):
                ft_min = torch.min(x_pruned[:, i])
                nan_i = not_numeric[:, i]
                x[nan_i, i] = 0

        if "noise" in strategy.lower():
            ft_min = torch.min(x_pruned, dim=0)[0]  # TODO: check dim
            ft_max = torch.max(x_pruned, dim=0)[0]  # TODO: check dim
            for i in range(n_features):
                nan_i = not_numeric[:, i]
                noise = ft_min[i] + (
                    (ft_max[i] - ft_min[i]) * torch.rand_like(x[nan_i, i])
                )
                if torch.all((x_pruned[:, i] % 1) == 0):
                    x[nan_i, i] = noise.floor()
                else:
                    x[nan_i, i] = noise

        if "spike_feature" in strategy.lower():
            spike_presence = torch.zeros(len(x), 1, dtype=bool)
            spike_presence[is_val] = True
            x = torch.hstack([x, spike_presence])

        print(
            "{} simulations out of {} had to be cleaned.".format(
                torch.sum(~is_val), n_samples
            )
        )

        return x

    def __simulate_and_cache_summaries(
        self,
        theta: Tensor,
        selected_features: List[int] = None,
        stimulus_noise=10,
        cache_results: bool = True,
        cache_file: str = "./cached_results",
        rnd_seed: Optional[int] = None,
    ):
        """Helper method that allows to cache intermediate results during parallel simulations.

        Args:
            theta: Simulator parameters. (batch_size, num_parameters)
            stimulus_noise: std of normal noise added to the input current in pA.
            selected_features: Which features to return by index
            cache_results: Whether to cache the results or not.
            cache_file: Location of cached files.
            rnd_seed: Seed pytorch's random generator for repeatable stimulus noise.

        Returns:
            stats: Summary statistics of simulator outputs. (batch_size, num_features)
        """
        if rnd_seed != None:
            torch.manual_seed(rnd_seed)

        It = self.I_t.repeat(theta.shape[0], 1)
        It += stimulus_noise * torch.randn_like(It)

        ts, Vs, Is = HH_Br2(theta, None, It, self.dt)
        stats = self.summarise(ts, Vs, Is, selected_features)
        stats_copy = stats.clone()

        if cache_results:
            try:
                saved_stats = torch.load(cache_file + "_stats")
                stats_copy = torch.vstack([saved_stats, stats_copy])

                saved_params = torch.load(cache_file + "_params")
                theta = torch.vstack([saved_params, theta])
            except FileNotFoundError:
                pass

            torch.save(stats_copy, cache_file + "_stats")
            torch.save(theta, cache_file + "_params")

        return stats

    def summarise(
        self, ts: Tensor, Vs: Tensor, Is: Tensor, selected_features: List[int] = None
    ) -> Tensor:
        """Calculate summary statistics for membrane voltages.

        AP_threshold # 0
        AP_amplitude # 1
        AP_width # 2
        AHP # 3
        AP_threshold_3 # 4
        AP_amplitude_3 # 5
        AP_width_3 # 6
        AHP_3 # 7
        AP_count # 8
        AP_count_1st_8th # 9
        AP_count_1st_quarter # 10
        AP_count_1st_half # 11
        AP_count_2nd_half # 12
        AP_amp_adapt # 13
        AP_amp_adapt_average # 14
        AP_cv # 15
        AI # 16
        cv # 17
        latency # 18
        resting_pot_mean # 19
        V_m_mean # 20
        std # 21
        skewness # 22

        Args:
            ts: Time axis. (batch_size, 1/dt)
            Vs: Membrane voltage. (batch_size, 1/dt)
            It: Input current. (batch_size, 1/dt)
            selected_features: Which features to return by index.

        Returns:
            Summary stats for batch of inputs. (batch_size, num_features)
        """
        n_mom = 3
        t_on = self.stim_onset
        t_off = self.stim_end
        dts = ts[:, 1] - ts[:, 0]

        sum_stats = []

        for dt, t, V, I in zip(dts, ts, Vs, Is):
            # -------- #
            # 1st part: features that electrophysiologists are actually interested in #
            vtrace = efex.EphysSweepFeatureExtractor(
                t=t.numpy() / 1e3,
                v=V.numpy(),
                i=I.numpy(),
                start=t_on / 1e3,
                end=t_off / 1e3,
                filter=10,
            )
            vtrace.process_spikes()

            AP_count_1st_8th = float("nan")
            AP_count_1st_quarter = float("nan")
            AP_count_1st_half = float("nan")
            AP_count_2nd_half = float("nan")
            AP_count_1st_quarter = float("nan")
            AP_count = float("nan")
            # fano_factor = float("nan")
            cv = float("nan")
            AI = float("nan")
            # AI_adapt_average = float("nan")
            latency = float("nan")
            AP_amp_adapt = float("nan")
            AP_amp_adapt_average = float("nan")
            AHP = float("nan")
            AP_threshold = float("nan")
            AP_amplitude = float("nan")
            AP_width = float("nan")
            # UDR = float("nan")
            AHP_3 = float("nan")
            AP_threshold_3 = float("nan")
            AP_amplitude_3 = float("nan")
            AP_width_3 = float("nan")
            # UDR_3 = float("nan")
            # AP_fano_factor = float("nan")
            AP_cv = float("nan")
            # SFA = float("nan")

            vtrace_df = vtrace._spikes_df

            if vtrace_df.size:
                vtrace_df["peak_height"] = (
                    vtrace_df["peak_v"].values - vtrace_df["threshold_v"].values
                )
                AP_count_1st_8th = (
                    vtrace_df["threshold_t"]
                    .values[vtrace_df["threshold_t"].values < 0.175]
                    .size
                )
                AP_count_1st_quarter = (
                    vtrace_df["threshold_t"]
                    .values[vtrace_df["threshold_t"].values < 0.25]
                    .size
                )
                AP_count_1st_half = (
                    vtrace_df["threshold_t"]
                    .values[vtrace_df["threshold_t"].values < 0.4]
                    .size
                )
                AP_count_2nd_half = (
                    vtrace_df["threshold_t"]
                    .values[vtrace_df["threshold_t"].values >= 0.4]
                    .size
                )
                AP_count = vtrace_df["threshold_i"].values.size

            if not vtrace_df.empty:  # There are APs and in the positive current regime
                if False in list(
                    vtrace_df["clipped"]
                ):  # There should be spikes that are also not clipped

                    # Add the Fano Factor of the interspike intervals (ISIs), a measure of the dispersion of a
                    # probability distribution (std^2/mean of the isis)
                    # fano_factor = vtrace._sweep_features['fano_factor']

                    # Add the coefficient of variation (std/mean, 1 for Poisson firing Neuron)
                    cv = vtrace._sweep_features["cv"]
                    # if cv<=0:
                    #    print('cv<0')
                    # And now the same for AP heights in the trace
                    # AP_fano_factor = vtrace._sweep_features['AP_fano_factor']
                    AP_cv = vtrace._sweep_features["AP_cv"]
                    # if AP_cv<0:

                    # Add the AP AHP, threshold, amplitude, width and UDR (upstroke-to-downstroke ratio) of the
                    # first fired AP in the trace
                    AHP = (
                        vtrace_df.loc[0, "fast_trough_v"]
                        - vtrace_df.loc[0, "threshold_v"]
                    )
                    AP_threshold = vtrace_df.loc[0, "threshold_v"]
                    AP_amplitude = vtrace_df.loc[0, "peak_height"]
                    AP_width = vtrace_df.loc[0, "width"] * 1000
                    # UDR = vtrace_df.loc[0, 'upstroke_downstroke_ratio']

                    if AP_count > 2:

                        AHP_3 = (
                            vtrace_df.loc[2, "fast_trough_v"]
                            - vtrace_df.loc[2, "threshold_v"]
                        )
                        AP_threshold_3 = vtrace_df.loc[2, "threshold_v"]
                        AP_amplitude_3 = vtrace_df.loc[2, "peak_height"]
                        AP_width_3 = vtrace_df.loc[2, "width"] * 1000
                        # UDR_3 = vtrace_df.loc[2, 'upstroke_downstroke_ratio']
                        # if np.sum(vtrace_df['threshold_index'] < half_stim_index)!=0:
                        #    SFA = np.sum(vtrace_df['threshold_index'] > half_stim_index) / \
                        #          np.sum(vtrace_df['threshold_index'] < half_stim_index)

                    half_stim_index = ft.find_time_index(
                        t / 1000, float(0.1 + (0.7 - 0.1) / 2)
                    )
                    APs_in_2ndhalf = torch.where(
                        torch.tensor(vtrace_df["threshold_index"]) > half_stim_index
                    )[0]

                    if APs_in_2ndhalf.size == torch.Size([0]):
                        AP1_in_2ndhalf = APs_in_2ndhalf.numpy()[0]
                        # print('@time: ', vtrace_df['threshold_t'][AP1_in_2ndhalf])
                        AHP_2ndhalf = (
                            vtrace_df.loc[AP1_in_2ndhalf, "fast_trough_v"]
                            - vtrace_df.loc[AP1_in_2ndhalf, "threshold_v"]
                        )
                        AP_threshold_2ndhalf = vtrace_df.loc[
                            AP1_in_2ndhalf, "threshold_v"
                        ]
                        AP_amplitude_2ndhalf = vtrace_df.loc[
                            AP1_in_2ndhalf, "peak_height"
                        ]
                        AP_width_2ndhalf = vtrace_df.loc[AP1_in_2ndhalf, "width"] * 1000

                    # Add the (average) adaptation index
                    AI = vtrace._sweep_features["isi_adapt"]
                    # AI_adapt_average = vtrace._sweep_features['isi_adapt_average']

                    # Add the latency
                    latency = vtrace._sweep_features["latency"] * 1000
                    if (latency + 0.4) <= 0:
                        latency = float("nan")
                    # Add the AP amp (average) adaptation (captures changes in AP amplitude during stimulation time)
                    AP_amp_adapt = vtrace._sweep_features["AP_amp_adapt"]
                    AP_amp_adapt_average = vtrace._sweep_features[
                        "AP_amp_adapt_average"
                    ]

            # -------- #
            # 2nd part: features that derive standard stat moments, possibly good to perform inference
            t_stim = (t > t_on) & (t < t_off)

            std_pw = torch.pow(
                torch.std(V[t_stim]), torch.linspace(3, n_mom, n_mom - 2)
            )
            std_pw = torch.cat((torch.ones(1), std_pw))
            moments = list(
                spstats.moment(V[t_stim], torch.linspace(2, n_mom, n_mom - 1))
                / std_pw  # 21, 22
            )

            spike_stats = [
                torch.tensor(AP_threshold),  # 0
                torch.tensor(AP_amplitude),  # 1
                torch.tensor(AP_width),  # 2
                torch.tensor(AHP),  # 3
                torch.tensor(AP_threshold_3),  # 4
                torch.tensor(AP_amplitude_3),  # 5
                torch.tensor(AP_width_3),  # 6
                torch.tensor(AHP_3),  # 7
                torch.tensor(AP_count),  # 8
                torch.tensor(AP_count_1st_8th),  # 9
                torch.tensor(AP_count_1st_quarter),  # 10
                torch.tensor(AP_count_1st_half),  # 11
                torch.tensor(AP_count_2nd_half),  # 12
                torch.log(torch.tensor(AP_amp_adapt)),  # 13
                sigmoid(
                    torch.tensor(AP_amp_adapt_average), offset=1, steepness=50
                ),  # 14
                torch.log(torch.tensor(AP_cv)),  # 15
                torch.log(torch.tensor(AI)),  # 16
                torch.log(torch.tensor(cv)),  # 17
                torch.log(torch.tensor(latency + 0.4)),  # 18
            ]

            pot_stats = [torch.mean(V[~t_stim]), torch.mean(V[t_stim])]  # 19, 20

            # concatenation of summary statistics
            sum_stats_tensor = torch.hstack(spike_stats + pot_stats + moments)

            sum_stats.append(sum_stats_tensor)

        batch_stats = torch.vstack(sum_stats).float()
        if selected_features == None:
            return batch_stats
        else:
            return batch_stats[:, selected_features]

    def simulate_and_summarise(
        self,
        theta: Tensor,
        V0: float or Tensor = None,
        selected_features: List[int] = None,
        stimulus_noise=0.1,
        clean: bool = False,
        prune: bool = False,
        n_workers: int = 1,
        batch_size: int = "auto",
        cache_results: bool = False,
        cache_file: str = "./cached_results",
        rnd_seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor] or Tensor:
        """Simulate membrane voltages of HH model and summarise them.

        Options to cache results and also remove NaNs/infs.

        Args:
            theta: Holds the model parameters. Static and variable ones. (batch_size, num_parameters)
            V0: Initial value for the simulator. (batch_size, 1) or (1)
            selected_features: Select subsets to be returned.
            n_workers: Number threads to distribute the simulations over.
            batch_size: How many neurons to simulate in parallel per thread.
                batch_size here is not the same as in (batch_size, 1).
                The latter regards the whole input, the former takes subsamples of this.
            stimulus_noise: std of normal noise added to the input current.
            prune: Removes summarised observations that contain NaNs.
            clean: Replaces NaNs with different numeric values.
            cache_results: Whether to cache the results or not.
            cache_file: Location of cached files.
            rnd_seed: Seeds the torch random generator of the stimulus noise.

        Returns:
            theta: Model parameters. (batch_size, num_parameters)
            stats: Summary statistics of simulator outputs. (batch_size, num_features)
        """
        assert (
            not clean or not prune
        ), "Simulations can only be cleaned OR pruned, not both!"

        params_batched = self._splice_model_params(theta)
        if V0 != None:
            if type(V0) is float or type(V0) is int:
                V0 = torch.tensor(V0).repeat(theta.shape[0], 1)
            params_batched = torch.hstack([params_batched, V0])

        summariseHH = lambda x: self.__simulate_and_cache_summaries(
            x,
            selected_features,
            stimulus_noise,
            cache_results,
            cache_file,
            rnd_seed,
        )
        batch_size = self._adjust_batchsize(theta.shape[0], batch_size)
        results = self.split_work(summariseHH, params_batched, n_workers, batch_size)
        stats = torch.vstack(results)

        if clean:
            return self.clean_simulations(stats)
        if prune:
            return self.prune_simulations(theta, stats)
        else:
            return stats


def HH_Br2(
    model_params: Tensor, V0: float or Tensor, It: Tensor, dt: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run the Hodgkin Huxley model with provided parameters and stimulation
    protocol. Brian2 Implementation.

    Model parameters (in order) that can be manipulated include:
    ENa [mV] : Reversal potential of sodium.
    EK [mV] : Reversal potential of potassium.
    ECa [mV] : Reversal potential of calcium.

    T_1 [°C] :  25 °C from paper Etay Hay et al.
    T_1 [°C] :  36 °C from paper Martin Pospischil et al.
    T_2 [°C] :  Experimental temperature.
    Q10 : temperature coeff.

    C [uF/cm^2] : Membrane capcitance per area.
    input_res [Mohm] : Membrane input resistance.
    tau [ms] : Membrane time constant.

    gNa [mS] : Channel conductance of sodium.
    gNa2 [mS] : Channel conductance of sodium.
    gK [mS] : Channel conductance of potassium.
    gM [mS] : Channel conductance for adaptive potassium currents.
    gKv31 [mS] : Channel conductance of potassium.
    gL [mS] : Channel conductance for Calcium current.
    Eleak  [mV] : Reversal potential of leak currents.

    tau_max [s] :
    VT [mV] : Threshold voltage.
    rate_to_SS_factor : Correction factor.

    Adpated from code courtesy of @ybernaerts.

    Args:
        model_params : Parameter vector that contains the HH model parameters.
        V0: Contains the initial value for the integration of the HH DEQ [mV].
        It: Stimulation current, sampled in time intervalls of dt [ms].
        dt: Integration timesteps in ms.

    Returns:
        Timepoints, membrane voltage and the injected currents are returned."""

    model_params = model_params.numpy()
    N, dims = model_params.shape

    ####################
    # Setting up the injection current, the model equations, further initialisations and run the model
    if It.shape[0] != N:
        It = It.repeat(N, 1)
    It = It.T

    I = br2.TimedArray(It.numpy(), dt=dt * br2.ms)

    # The conductance-based model
    eqs = """
            dVm/dt = -(gNa*m**3*h*(Vm - ENa) + gNa2*m2**3*h2*(Vm - ENa) + gK*n**4*(Vm - EK) + gleak*(Vm - Eleak) 
            + gM*p*(Vm - EK) + gKv31*v*(Vm - EK) + gL*q**2*r*(Vm - ECa) - I_inj)/ (C * area) : volt
            
            I_inj = I(t, i)*pA : amp
            
            dm/dt = (alpham*(1-m) - betam*m) * t_adj_factor_hay : 1
            dh/dt = (alphah*(1-h) - betah*h) * t_adj_factor_hay : 1
            dp/dt = ((p_inf - p)/tau_p) * t_adj_factor_pos : 1
            dv/dt = ((v_inf - v)/tau_v) * t_adj_factor_hay : 1
            dq/dt = (alphaq*(1-q) - betaq*q) : 1
            dr/dt = (alphar*(1-r) - betar*r) : 1
            dm2/dt = (alpham2*(1-m2) - betam2*m2) * t_adj_factor_pos  / rate_to_SS_factor : 1
            dh2/dt = (alphah2*(1-h2) - betah2*h2) * t_adj_factor_pos  / rate_to_SS_factor : 1
            dn/dt = (alphan*(1-n) - betan*n) * t_adj_factor_pos / rate_to_SS_factor : 1

            alpham = (0.182/mV) * (Vm + 38.*mV) / (1-exp((-(Vm + 38.*mV))/(6.*mV)))/ms : Hz
            betam = (-0.124/mV) * (Vm + 38.*mV) / (1-exp((Vm + 38.*mV)/(6.*mV)))/ms : Hz
            alphah = (-0.015/mV) * (Vm + 66.*mV) / (1-exp((Vm + 66.*mV)/(6.*mV)))/ms : Hz 
            betah = (0.015/mV) * (Vm + 66.*mV) / (1-exp((-(Vm + 66.*mV))/(6.*mV)))/ms : Hz
            
            alphan = (-0.032/mV) * (Vm - VT - 15.*mV) / (exp((-(Vm - VT - 15.*mV)) / (5.*mV)) - 1.)/ms : Hz
            betan = 0.5*exp(-(Vm - VT - 10.*mV) / (40.*mV))/ms : Hz
            
            alpham2 = (-0.32/mV) * (Vm - VT - 13.*mV) / (exp((-(Vm - VT - 13.*mV))/(4.*mV)) - 1.)/ms : Hz
            betam2 = (0.28/mV) * (Vm - VT - 40.*mV) / (exp((Vm - VT - 40.*mV)/(5.*mV)) - 1.)/ms : Hz
            alphah2 = 0.128 * exp(-(Vm - VT - 17.*mV) / (18.*mV))/ms : Hz
            betah2 = 4./(1. + exp((-(Vm - VT - 40.*mV)) / (5.*mV)))/ms : Hz                         
            
            p_inf = 1./(1. + exp(-(Vm + 35.*mV)/(10.*mV))) : 1
            tau_p = (tau_max/1000.)/(3.3 * exp((Vm + 35.*mV)/(20.*mV)) + exp(-(Vm + 35.*mV)/(20.*mV))) : second
            
            v_inf = 1. / (1. + exp((-(Vm - 18.7*mV))/(9.7*mV))) : 1
            tau_v = 4.*ms / (1. + exp((-(Vm + 56.56*mV))/(44.14*mV))) : second

            alphaq = (-0.055/mV) * (27.*mV + Vm) / (exp((-(27.*mV + Vm))/(3.8*mV)) - 1.)/ms : Hz
            betaq = 0.94*exp(-((75.*mV + Vm)) / (17.*mV))/ms : Hz
            alphar = 0.000457*exp((-(13.*mV + Vm))/(50.*mV))/ms : Hz
            betar = 0.0065/(1. + exp((-(15.*mV + Vm))/(28.*mV)))/ms : Hz
            
            area : metre**2
            gNa : siemens
            gNa2 : siemens
            gKv31 : siemens
            gK : siemens
            gleak : siemens
            gM : siemens
            gL : siemens
            tau_max : second
            
            C : farad/metre**2 
            VT : volt
            Eleak : volt
            rate_to_SS_factor : 1

            ENa : volt
            ECa : volt
            EK : volt
            t_adj_factor_hay : 1
            t_adj_factor_pos : 1
            """
    neurons = br2.NeuronGroup(
        N, eqs, method="exponential_euler", name="neurons", dt=dt * br2.ms
    )

    ###################
    # Model parameter initialisations
    # Some are set, some are performed inference on
    # Inspired by Martin Pospischil et al. "Minimal Hodgkin-Huxley type models for
    # different classes of cortical and thalamaic neurons".

    neurons.ENa = model_params[:, 0] * br2.mV
    neurons.EK = model_params[:, 1] * br2.mV
    neurons.ECa = model_params[:, 2] * br2.mV

    # It is important to adapt your kinetics to the temperature of your experiment
    # temperature coeff., https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)
    T_1_hay = model_params[:, 3]  # °C, from paper Etay Hay et al.
    T_1_pos = model_params[:, 4]  # °C, from paper Martin Pospischil et al.
    T_2 = model_params[:, 5]  # °C, experiment was actually done at 34 °C
    Q10 = model_params[:, 6]  # temperature coeff.

    C_ = model_params[:, 7] # -> 0.
    input_res = model_params[:, 8] # -> 1.
    tau = model_params[:, 9] # -> 2.
    gleak_ = 1/(input_res*1e3)/((tau*1e3)/(C_*input_res*1e6))

    area_ = tau / (C_ * input_res) * 1e5 * br2.umetre ** 2  # um2

    neurons.area = area_
    neurons.t_adj_factor_hay = Q10 ** ((T_2 - T_1_hay) / 10.0)
    neurons.t_adj_factor_pos = Q10 ** ((T_2 - T_1_pos) / 10.0)
    neurons.C = C_ * br2.uF / br2.cm ** 2
    neurons.gNa = model_params[:, 10] * br2.mS / br2.cm ** 2 * area_ # -> 3.
    neurons.gNa2 = model_params[:, 11] * br2.mS / br2.cm ** 2 * area_  # -> 4.

    neurons.gK = model_params[:, 12] * br2.mS / br2.cm ** 2 * area_ # -> 5.
    neurons.gM = model_params[:, 13] * br2.mS / br2.cm ** 2 * area_ # -> 6.
    neurons.gKv31 = model_params[:, 14] * br2.mS / br2.cm ** 2 * area_ # -> 7.

    neurons.gleak = gleak_ * br2.mS / br2.cm ** 2 * area_
    neurons.gL = model_params[:, 15] * br2.mS / br2.cm ** 2 * area_ # -> 8.
    neurons.Eleak = model_params[:, 16] * br2.mV # -> 9.
    neurons.tau_max = model_params[:, 17] * br2.second # -> 10.
    neurons.VT = model_params[:, 18] * br2.mV # -> 11.
    neurons.rate_to_SS_factor = model_params[:, 19] # -> 12.

    # set monitoring
    Vm_mon = br2.StateMonitor(
        neurons, ["Vm", "I_inj"], record=True, name="Vm_mon"
    )  # Specify what to record

    # init
    if V0 != None:
        neurons.Vm = V0 * br2.mV  # V0
    else:
        neurons.Vm = "Eleak"  # V0
    neurons.m = "1.0/(1.0 + betam/alpham)"  # Would be the solution when dm/dt = 0
    neurons.h = "1.0/(1.0 + betah/alphah)"  # Would be the solution when dh/dt = 0
    neurons.m2 = "1.0/(1.0 + betam2/alpham2)"  # Would be the solution when dm/dt = 0
    neurons.h2 = "1.0/(1.0 + betah2/alphah2)"  # Would be the solution when dh/dt = 0
    neurons.n = "1.0/(1.0 + betan/alphan)"  # Would be the solution when dn/dt = 0
    neurons.p = "p_inf"  # Would be the solution when dp/dt = 0
    neurons.v = "v_inf"  # Would be the solution when dp/dt = 0
    neurons.q = "1.0/(1.0 + betaq/alphaq)"  # Would be the solution when dq/dt = 0
    neurons.r = "1.0/(1.0 + betar/alphar)"  # Would be the solution when dr/dt = 0
    # run simulation
    br2.run(len(It) * dt * br2.ms)

    return (
        torch.from_numpy(Vm_mon.t / br2.ms).repeat(N, 1),
        torch.from_numpy(Vm_mon.Vm / br2.mV),
        torch.from_numpy(Vm_mon.I_inj / br2.pA),
    )
