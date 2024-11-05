# Ephys utilities

import numpy as np
from scipy import stats as spstats

import ephys_extractor as efex
import ephys_features as ft

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

# return sigmoid of input x
def sigmoid(x, offset = 1, steepness = 1):
    # offset to shift the sigmoid centre to 1
    return 1/(1 + np.exp(-steepness*(x-offset)))


# prepares the data from raw format
def get_time_voltage_current_currindex0(nwb):
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
    voltage[:, curr_index_0] = df.iloc[curr_index_0*2][0][0].data[:]    # Find voltage trace for 0 current stimulation
    return time, voltage, current, curr_index_0



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
