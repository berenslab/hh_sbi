# Ephys utilities

import numpy as np
from scipy import stats as spstats

import fslm4expdata.ephys_extractor as efex
import fslm4expdata.ephys_features as ft

import re
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from scipy.io import loadmat
from torch import Tensor

# return sigmoid of input x
def sigmoid(x: Tensor, offset: int = 1, steepness: int = 1) -> Tensor:
    """Implements the sigmoid function.

    Args:
        x: Where to evaluate the sigmoid.
        offset: translation of x.
        steepness: time constant of the exponential.

    Returns:
        Sigmoid evaluated at x.
    """
    # offset to shift the sigmoid centre to 1
    return 1 / (1 + torch.exp(-steepness * (x - offset)))


def constant_stimulus(
    duration: float or int,
    dt: float,
    stim_onset: float or int,
    stim_end: float or int,
    magn: float or int,
    noise: float = 0.0,
    return_ts: bool = False,
) -> Tuple[Tensor, Tensor] or Tensor:
    """Creates a constant stimulus current.

    Based on the input parameters, a stimulus current is generated, that can be fed
    to a HH simulator in order to compute the stimulus response of a HH neuron.

    Args:
        duration: Duration of the whole current pulse in ms.
        dt: Time steps in ms, 1/dt = sampling frequency in MHz.
        stim_onset: At which timepoint [ms] the stimulus is applied, i.e. I != 0.
        stim_end: At which timepoint [ms] the stimulus is stopped, i.e. I = 0 again.
        magn: The magnitude of the constant stimulus in pA.
        noise: Noise added ontop of the input current. Measured in mV
        return_ts: Whether to also return the time axis along with the current.

    Returns:
        t: Corresponding time axis of the stimulus [ms].
        I_t: Stimulus current [pA].
    """
    t_start = 0.0
    t_end = duration

    t = torch.arange(t_start, t_end, dt)
    I_t = torch.zeros_like(t)
    stim_at = torch.logical_and(t > stim_onset, t < stim_end)
    I_t[stim_at] = magn
    I_t += noise * torch.randn(len(I_t)) / (dt ** 0.5)

    if return_ts:
        return t, I_t
    else:
        return I_t


def ints_from_str(string: str) -> List[int]:
    """Extracts list of integers present in string."""
    number_strs = re.findall(r"\d+", string)
    return [int(s) for s in number_strs]


def import_and_select_trace(
    path2mat: str, Iinj: dict = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """Import electrophysiological recordings stored in .mat files into t, V(t) Tensors.

    This function takes the Path to a .mat file as input and optionally the
    current parameters of the stimulation protocol 'stim_start', 'stim_end', 'duration'
    of the whole I(t). If provided, I(t) will be returned along with V(t), otherwise
    it will be left blank.

    WARNING: Some Values in this function are hard coded, thus works
    with specific recordings only at the moment.

    Args:
        path2mat: Path to the file location.
        Iinj: Dictionary containing current specifics.

    Returns:
        t: Time axis in steps of dt in ms.
        Vt: Membrane voltage in mV.
        It: Stimulation current in pA.
    """
    data = loadmat(path2mat)
    trace_keys = [key for key in data.keys() if "Trace" in key]
    trace_tags = torch.vstack(
        [torch.tensor(ints_from_str(x)) for x in trace_keys if ints_from_str(x) != []]
    )
    # print(tags)
    num_electrodes = int(torch.max(trace_tags[:, -1]))
    num_samples = int(len(trace_tags) / num_electrodes)
    num_bins = len(list(data.values())[10])  # arbitratry

    t = torch.zeros(num_electrodes, num_samples, num_bins)
    Vt = torch.zeros(num_electrodes, num_samples, num_bins)
    It = torch.zeros(num_electrodes, num_samples, num_bins)

    for tags, key in zip(trace_tags, trace_keys):
        trace = torch.tensor(data[key])
        elec_idx = tags[-1] - 1
        sample_idx = tags[-2] - 1
        if trace.ndim > 1:
            t[elec_idx, sample_idx, :], Vt[elec_idx, sample_idx, :] = trace.T

    if Iinj != None:
        It = torch.ones_like(Vt)
        dt = (t[:, :, 1] - t[:, :, 0]) * 1000
        if "stim_onset" in Iinj.keys():
            t1 = torch.max((Iinj["stim_onset"] / dt).long())
            t2 = torch.max((Iinj["stim_end"] / dt).long())
        else:
            t1 = torch.max((Iinj["stim_onset [ms]"] / dt).long())
            t2 = torch.max((Iinj["stim_end [ms]"] / dt).long())

        if "duration" in Iinj.keys():
            T = torch.max((Iinj["duration"] / dt).long())
        else:
            if "t_start [ms]" in Iinj.keys():
                T = torch.max(((Iinj["t_end [ms]"] - Iinj["t_start [ms]"]) / dt).long())

        I = torch.arange(
            -200, num_samples * 20 - 200, 20
        )  # params are hardcoded for now
        It[:, :, :t1] = 0
        It[:, :, t2:] = 0
        It[:, :, t1:t2] = I.reshape(1, num_samples, 1)

        t = t[:, :, :T]
        Vt = Vt[:, :, :T]
        It = It[:, :, :T]

    t = t * 1000
    Vt = Vt * 1000
    return t, Vt, It


def plot_vtrace(
    t: Tensor,
    Vt: Tensor,
    It: Optional[Tensor] = None,
    figsize: Tuple = (6, 4),
    title="",
    timewindow=None,
    **plot_kwargs
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
    Returns:
        fig: plt.Figure.
        axes: plt.Axes.
    """

    start, end = (0, -1)

    if timewindow != None:
        dt = t[1] - t[0]
        start, end = (torch.tensor(timewindow) / dt).int()

    Vts = Vt[:, start:end]
    ts = t[:, start:end]

    if It == None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
        for t_i, Vt_i in zip(ts, Vts):
            axes[0].plot(t_i.numpy(), Vt_i.numpy(), lw=2, **plot_kwargs)

    else:
        Its = It[:, start:end]

        fig, axes = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        for t_i, Vt_i in zip(ts, Vts):
            axes[0].plot(t_i.numpy(), Vt_i.numpy(), lw=2, **plot_kwargs)

        for t_i, It_i in zip(ts, Its):
            axes[1].plot(t_i.numpy(), It_i.numpy(), lw=2, c="grey")

    for i, ax in enumerate(axes):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

        ax.set_xticks(torch.linspace(t[0, 0], t[0, -1], 3).numpy())
        ax.set_xticklabels(torch.linspace(t[0, 0], t[0, -1], 3).numpy())
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        ax.xaxis.set_tick_params(width=2)

        ax.set_yticks(
            torch.linspace(
                torch.round(torch.min(Vt)), torch.round(torch.max(Vt)), 2
            ).numpy()
        )
        ax.set_yticklabels(torch.linspace(torch.min(Vt), torch.max(Vt), 2).numpy())
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
        ax.yaxis.set_tick_params(width=2)

        if i == 0:
            ax.set_title(title)
            ax.set_ylabel("voltage (mV)", fontsize=12)
            if len(axes) == 1:
                ax.set_xlabel("time (ms)", fontsize=12)
            else:
                ax.xaxis.set_ticks_position("none")
            if "label" in plot_kwargs.keys():
                ax.legend(loc=1)
        if i == 1:
            ax.set_xlabel("time (ms)", fontsize=12)
            ax.set_ylabel("input (pA)", fontsize=12)
            ax.set_yticks([0, torch.max(It)])
    plt.tight_layout()

    return fig, axes


# Normalises current with experimentally derived area (micrometer^2). Setting up.
def syn_current(duration=800, dt=0.04, t_on = 100,
                curr_level = 1e-4, area=10000, seed=None):

    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration+dt, dt)
    A_soma=area/1e8 #convert to cm^2
    I = np.zeros_like(t)
    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = curr_level/A_soma # muA/cm^2
    
    return I, t_on, t_off, dt, t, A_soma


# Normalises current with experimental input resistance and membrane time constant. Setting up.
def syn_current_start(duration=800, dt=0.04, t_on = 100,
                curr_level = 1e-4, exp_input_res=270, exp_tau=10, seed=None):
    # exp_input_res in MOhms!
    # exp_tau in ms!
    duration = duration
    t_off = duration - t_on
    t = np.arange(0, duration+dt, dt)

    # external current
    r_m = exp_tau*1e3            # specific membrane resistance (Ohm*cm**2) (so that r_m*c_m= exp_tau, 
                                 # c_m assumed to be 1 microF/cm**2)

    A_soma = r_m/(exp_input_res*1e6)  # cm2
    I = np.zeros_like(t)
    I[int(np.round(t_on/dt)):int(np.round(t_off/dt))] = curr_level/A_soma # muA/cm2
    return I, t_on, t_off, dt, t, A_soma


# prepares the data from raw format
def data_preparation(data, el_num = 2, current_step = 20):
    """Analyse the data in dictionary format (assumes a certain outlook of the data) and return the voltage traces, 
    stimulus current magnitudes for all traces and the time and the current index for which the current magnitude
    equals 0 pA.

    Parameters
    ----------
    data : dictionary full of voltage (V) and time (s) traces
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    
    Returns
    -------
    voltage : numpy 2D array of voltage traces (mV)
    time : numpy 1D array of time points (s)
    current : numpy 1D array of current stimulus magnitudes
    curr_index_0 : integer of current index where the current = 0 pA
    """
    
    
    
    # All the keys in the dictionary should have the form "Trace_1_j_c_el" where j represents a certain number which could
    # be different between cells/samples, therefore we will store this number as num. c represents the current stimulus
    # magnitude index, el the electrode number. They usually always record from electrode number el = 2.

    
    # Random initialisations
    num = 2
    n_samp = 22500
    dt = 4e-05
    
    big_data = True
    
    if len(data.keys()) == 83:                     # We're in the special case of 4 datafiles where raw data is one
                                                   # electrode only
            big_data = False
    
    for key in list(data)[1:6]:
        if key.split('_')[0] == "Trace":
            num = key.split('_')[2]
            n_samp = data[key].shape[0] # Amount of timepoints
            dt = data[key][1, 0] - data[key][0, 0] # Time step
            
            if not big_data:
                el_num = key.split('_')[4]
            break
    time = np.arange(0,n_samp*dt,dt)
    
    
    # Check whether we have hyperpolarisation in the beginning, otherwise the electrode where APs have been recorded
    # must have been a different one
    
    
    object_ephys = efex.EphysSweepFeatureExtractor(t = time, v = 1000*data['Trace_{}_{}_{}_{}'.format(1, num, 1, el_num)][:, 1], \
                                                   start = 0.1, end = 0.7, filter = 10)
    voltage_deflection_v, _ = object_ephys.voltage_deflection()
    Vm = object_ephys._get_baseline_voltage()
    V_defl = voltage_deflection_v
    if  np.abs(Vm - V_defl) < 2:
        el_num = 1
        while np.abs(Vm - V_defl) < 2:
            object_ephys = efex.EphysSweepFeatureExtractor(t = time, v = 1000*data['Trace_{}_{}_{}_{}'.format(1, num, 1, el_num)][:, 1], \
                                                   start = 0.1, end = 0.7, filter = 10)
            Vm = object_ephys._get_baseline_voltage()
            V_defl, _ = object_ephys.voltage_deflection()
            if el_num == 1: # We already investigated el_num = 2
                el_num = 3
            else: el_num += 1
        el_num = el_num - 1
    
    Amount_Of_Electrodes = np.max([int(g.split('_')[-1]) if (g.split('_')[0] == 'Trace') else 0 for g in list(data.keys())])
    # stim_paradigm_num is the number of current stimulus magnitudes that were used
    stim_paradigm_num = (np.array(list(data)).size - 3)/Amount_Of_Electrodes
                                    # 3 keys should just be '__globals__', '__header__', and '__version__'
 
    curr_index_0 = 0 # Current stimulus magnitude index that corresponds to stimulating the cell with 0 pA
    
    # Trace with the least amount of variance in the trace is assumed to be the trace corresponding to stimulating the cell
    # with 0 pA stimulation current
    best = np.var(1000*data['Trace_1_{}_{}_{}'.format(num, 1, el_num)][:, 1])
    #best = np.abs(np.mean(1000*data['Trace_1_{}_{}_{}'.format(num, 1, el_num)][:, 1]) - Vm)
    for i in np.arange(2, stim_paradigm_num + 1, 1, dtype = 'int'):
        best_temp = np.var(1000*data['Trace_1_{}_{}_{}'.format(num, i, el_num)][:, 1])
        #best_temp = np.abs(np.mean(1000*data['Trace_1_{}_{}_{}'.format(num, i, el_num)][:, 1]) - Vm)
        if  best_temp < best:
            best = best_temp
            curr_index_0 = i
    
    start_current_impulse = -(curr_index_0 - 1)*current_step # - 1 since in the dictionary we start at 1 (not at zero)    
    stop_current_impulse = start_current_impulse + (current_step*stim_paradigm_num)
    current = np.arange(start_current_impulse, stop_current_impulse, current_step)
    #current = current[current < 800] #After that the cell might die and it's probably unnecessary for the analysis
    
    # voltage will give us the voltage response for all different current steps
    voltage = np.zeros((n_samp, len(current)))
    for c, C in enumerate(current):
        voltage[:,c] = 1000*data['Trace_1_{}_{}_{}'.format(num, c+1, el_num)][:,1] # c+1: goes from 1 to end of possible current
                                                                                   # stimulation magnitudes
    
    return time, current, voltage, (curr_index_0 - 1) # -1 since the current and voltage vector start at 0



# calculate summary statistics from simulations
def calculate_summary_statistics(x, use_feature_list=None):
    """Calculate summary statistics
    
    Parameters
    ----------
    x : output of the simulator
    use_feature_list : only select certain features from the summary statistic vector (provided with a list of indices, optional)

    Returns
    -------
    np.array, summary statistics
    """
    n_mom = 3
    t_on=100
    t_off=700
    t = x["time"]
    dt = x["dt"]
    I = x["I"]
    
    sum_stats=[]
    
    for v in x['data']:
        v=v.reshape(-1)
        # -------- #
        # 1st part: features that electrophysiologists are actually interested in #
        EphysObject = efex.EphysSweepFeatureExtractor(t = t/1000, v = v, \
                                                      i = I, start = 0.1, \
                                                      end = 0.7, filter = 10)
        EphysObject.process_spikes()
        AP_count_1st_8th = np.nan
        AP_count_1st_quarter = np.nan
        AP_count_1st_half = np.nan
        AP_count_2nd_half = np.nan
        AP_count = np.nan
        #fano_factor = np.nan
        cv = np.nan
        AI = np.nan
        #AI_adapt_average = np.nan
        latency = np.nan
        AP_amp_adapt = np.nan
        AP_amp_adapt_average = np.nan
        AHP = np.nan
        AP_threshold = np.nan
        AP_amplitude = np.nan
        AP_width = np.nan
        #UDR = np.nan
        AHP_3 = np.nan
        AP_threshold_3 = np.nan
        AP_amplitude_3 = np.nan
        AP_width_3 = np.nan
        #UDR_3 = np.nan
        #AP_fano_factor = np.nan
        AP_cv = np.nan
        #SFA = np.nan

        if EphysObject._spikes_df.size:
            EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                   EphysObject._spikes_df['threshold_v'].values
            AP_count_1st_8th = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.175].size
            AP_count_1st_quarter = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.25].size
            AP_count_1st_half = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.4].size
            AP_count_2nd_half = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values>=0.4].size
            AP_count = EphysObject._spikes_df['threshold_i'].values.size
            
            # AP counts are zero-inflated for simulations. Thus comparisons with truth are oftentimes small, much smaller than differences
            # according to other features. A log transformation helps.
            AP_count=np.log(AP_count+3)
            AP_count_1st_half=np.log(AP_count_1st_half+3)
            AP_count_2nd_half=np.log(AP_count_2nd_half+3)
            AP_count_1st_quarter=np.log(AP_count_1st_quarter+3)
            AP_count_1st_8th=np.log(AP_count_1st_8th+3)
            
        if not EphysObject._spikes_df.empty: # There are APs and in the positive current regime
            if False in list(EphysObject._spikes_df['clipped']): # There should be spikes that are also not clipped

                # Add the Fano Factor of the interspike intervals (ISIs), a measure of the dispersion of a
                # probability distribution (std^2/mean of the isis)
                #fano_factor = EphysObject._sweep_features['fano_factor']

                # Add the coefficient of variation (std/mean, 1 for Poisson firing Neuron)
                cv = EphysObject._sweep_features['cv']

                # And now the same for AP heights in the trace
                #AP_fano_factor = EphysObject._sweep_features['AP_fano_factor']
                AP_cv = EphysObject._sweep_features['AP_cv']

                # Adding spike frequency adaptation (ratio of spike frequency of second half to first half for the highest
                # frequency count trace)

                if AP_count > 2: # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here

                    AHP_3 = EphysObject._spikes_df.loc[2, 'fast_trough_v'] - EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_threshold_3 = EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_amplitude_3 = EphysObject._spikes_df.loc[2, 'peak_height']
                    AP_width_3 = EphysObject._spikes_df.loc[2, 'width']*1000
                    #UDR_3 = EphysObject._spikes_df.loc[2, 'upstroke_downstroke_ratio']

                # Add the (average) adaptation index
                AI = EphysObject._sweep_features['isi_adapt']
                #AI_adapt_average = EphysObject._sweep_features['isi_adapt_average']

                # Add the latency
                latency = EphysObject._sweep_features['latency']*1000
                if (latency+0.4)<=0:
                    #print('latency+0.4<0')
                    latency=np.nan
                # Add the AP amp (average) adaptation (captures changes in AP amplitude during stimulation time)
                AP_amp_adapt = EphysObject._sweep_features['AP_amp_adapt']
                AP_amp_adapt_average = EphysObject._sweep_features['AP_amp_adapt_average']


                # Add the AP AHP, threshold, amplitude, width and UDR (upstroke-to-downstroke ratio) of the
                # first fired AP in the trace
                AHP = EphysObject._spikes_df.loc[0, 'fast_trough_v'] - EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_threshold = EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_amplitude = EphysObject._spikes_df.loc[0, 'peak_height']
                AP_width = EphysObject._spikes_df.loc[0, 'width']*1000
                #UDR = EphysObject._spikes_df.loc[0, 'upstroke_downstroke_ratio']

        # -------- #
        # 2nd part: features that derive standard stat moments, possibly good to perform inference
        std_pw = np.power(
            np.std(v[(t > t_on) & (t < t_off)]), np.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = np.concatenate((np.ones(1), std_pw))
        moments = (
            spstats.moment(
                v[(t > t_on) & (t < t_off)], np.linspace(2, n_mom, n_mom - 1)
            )
            / std_pw
        )

        rest_pot = np.mean(v[(t < t_on) | (t > t_off)])

        # concatenation of summary statistics
        sum_stats_vec = np.concatenate(
            (
                np.array([AP_threshold, AP_amplitude, AP_width, AHP, \
                          AP_threshold_3, AP_amplitude_3, AP_width_3, AHP_3, \
                          AP_count, AP_count_1st_8th, AP_count_1st_quarter, AP_count_1st_half, AP_count_2nd_half, \
                          np.log(AP_amp_adapt), sigmoid(AP_amp_adapt_average, offset=1, steepness=50), \
                          np.log(AP_cv), np.log(AI), np.log(cv), np.log(latency+0.4)]),
                np.array(
                    [rest_pot, np.mean(v[(t > t_on) & (t < t_off)])]
                ),
                moments,
            )
        )
        # sum_stats_vec = sum_stats_vec[0:n_summary]

        if use_feature_list is not None:
            sum_stats_vec=sum_stats_vec[use_feature_list]
        
        sum_stats.append(sum_stats_vec)
    
    return np.array(sum_stats)
