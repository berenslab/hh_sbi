# Ephys utilities

import numpy as np
import pandas as pd
from scipy import stats as spstats

import ephys_extractor as efex
import ephys_features as ft

# regression utility
from sklearn import linear_model
ransac = linear_model.RANSACRegressor()

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
    for i in range(curr_index_0):   # Find all voltage traces from minimum to 0 current stimulation
        voltage[:, i+1] = df['series'][0::2][(i+1)*2][0].data[:]
    for i in range(curr_index_0, int((df.shape[0]+1)/2)-1):   # Find all voltage traces from 0 to highest current stimulation
        voltage[:, i+1] = df['series'][1::2][i*2+1][0].data[:]
    voltage[:, curr_index_0] = df.iloc[curr_index_0*2][0][0].data[:]    # Find voltage trace for 0 current stimulation
    return time, voltage, current, curr_index_0

def get_time_voltage_current_currindex0_gouwens(nwb):
    # Here we're simply going to find a suitable voltage trace,
    # and remember the current amplitude that was applied 
    df = nwb.sweep_table.to_dataframe()
    time = np.linspace(0,1.2,60000) # 50 kHz sampling freq, 0 to 1.2 seconds
    voltage = []
    current = []
    currents_applied = []
    for i in range(df['sweep_number'].shape[0]):
        stim_descr = df['series'][i][0].stimulus_description
        #print(stim_descr)
        if stim_descr=='C1LSFINEST150112_DA_0':
            # 45000:105000 corresponds to 0.1s, 1s stim, 0.1s
            if i%2==0:
                voltage.append(df['series'][i][0].data[45000:105000][:,np.newaxis])
            else:
                current.append(df['series'][i][0].data[45000:105000][:,np.newaxis])
                currents_applied.append(np.max(df['series'][i][0].data[:]).item())
        elif stim_descr=='X4PS_SupraThresh_DA_0':
            # 25000:85000 corresponds to 0.1s, 1s stim, 0.1s :
            if i%2==0:
                voltage.append(df['series'][i][0].data[25000:85000][:,np.newaxis])
            else:
                current.append(df['series'][i][0].data[25000:85000][:,np.newaxis])
                currents_applied.append(np.max(df['series'][i][0].data[:]).item())

    if len(voltage)==0:
        return np.nan, np.nan, np.nan, np.nan
    else:
        voltage=np.concatenate(voltage,axis=1)
        current=np.concatenate(current,axis=1)
        return time, voltage[:,-1], current[:,-1], currents_applied[-1]


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


def calculate_summary_statistics_gouwens(x, use_feature_list=None):
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
    t_off=1100
    t = x["time"]
    I = x["I"]
    
    sum_stats=[]
    
    for v in x['data']:
        v=v.reshape(-1)
        # -------- #
        # 1st part: features that electrophysiologists are actually interested in #
        EphysObject = efex.EphysSweepFeatureExtractor(t = t/1000, v = v, \
                                                      i = I, start = 0.1, \
                                                      end = 1.1, filter = 10)
        EphysObject.process_spikes()
        AP_count_1st_8th = np.nan
        AP_count_1st_quarter = np.nan
        AP_count_1st_half = np.nan
        AP_count_2nd_half = np.nan
        AP_count = np.nan
        cv = np.nan
        AI = np.nan
        latency = np.nan
        AP_amp_adapt = np.nan
        AP_amp_adapt_average = np.nan
        AHP = np.nan
        AP_threshold = np.nan
        AP_amplitude = np.nan
        AP_width = np.nan
        AHP_3 = np.nan
        AP_threshold_3 = np.nan
        AP_amplitude_3 = np.nan
        AP_width_3 = np.nan
        AP_cv = np.nan

        if EphysObject._spikes_df.size:
            EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                   EphysObject._spikes_df['threshold_v'].values
            AP_count_1st_8th = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.225].size
            AP_count_1st_quarter = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.35].size
            AP_count_1st_half = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values<0.6].size
            AP_count_2nd_half = EphysObject._spikes_df['threshold_t'].values[EphysObject._spikes_df['threshold_t'].values>=0.6].size
            AP_count_ = EphysObject._spikes_df['threshold_i'].values.size
            
            # AP counts are zero-inflated for simulations. Thus comparisons with truth are oftentimes small, much smaller than differences
            # according to other features. A log transformation helps.
            AP_count=np.log(AP_count_+3)
            AP_count_1st_half=np.log(AP_count_1st_half+3)
            AP_count_2nd_half=np.log(AP_count_2nd_half+3)
            AP_count_1st_quarter=np.log(AP_count_1st_quarter+3)
            AP_count_1st_8th=np.log(AP_count_1st_8th+3)
            
        if not EphysObject._spikes_df.empty: # There are APs and in the positive current regime
            if False in list(EphysObject._spikes_df['clipped']): # There should be spikes that are also not clipped

                # Add the coefficient of variation (std/mean, 1 for Poisson firing Neuron)
                cv = EphysObject._sweep_features['cv']

                # And now the same for AP heights in the trace
                AP_cv = EphysObject._sweep_features['AP_cv']

                if AP_count_ > 2: # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here

                    AHP_3 = EphysObject._spikes_df.loc[2, 'fast_trough_v'] - EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_threshold_3 = EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_amplitude_3 = EphysObject._spikes_df.loc[2, 'peak_height']
                    AP_width_3 = EphysObject._spikes_df.loc[2, 'width']*1000

                # Add the (average) adaptation index
                AI = EphysObject._sweep_features['isi_adapt']

                # Add the latency
                latency = EphysObject._sweep_features['latency']*1000
                if (latency+0.4)<=0:
                    #print('latency+0.4<0')
                    latency=np.nan
                # Add the AP amp (average) adaptation (captures changes in AP amplitude during stimulation time)
                AP_amp_adapt = EphysObject._sweep_features['AP_amp_adapt']
                AP_amp_adapt_average = EphysObject._sweep_features['AP_amp_adapt_average']


                # Add the AP AHP, threshold, amplitude, width and UDR of the
                # first fired AP in the trace
                AHP = EphysObject._spikes_df.loc[0, 'fast_trough_v'] - EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_threshold = EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_amplitude = EphysObject._spikes_df.loc[0, 'peak_height']
                AP_width = EphysObject._spikes_df.loc[0, 'width']*1000

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

        if use_feature_list is not None:
            sum_stats_vec=sum_stats_vec[use_feature_list]
        
        sum_stats.append(sum_stats_vec)
    
    return np.array(sum_stats)


def cell_features(data_tuple, names, ephys_features, current_val=300, liquid_junction_potential = 15.4, \
                  el_num = 2, current_step = 20, start=0.1, end=0.7):
    """ Analyses a stream of cell dictionaries and outputs all the cell's features in a concatenated DataFrame
    Parameters
    ----------
    data_tuple : tuple of dictionaries of data full of voltage (V) and time (s) traces for different cells
    names : tuple of names of the samples
    ephys_features : list of ephys feature names being extracted
    current_val : int, current value for the trace you'd like to extract features from (optional, 300 pA by default)
    liquid_junction_potential : float, potential to be substracted from all traces (optional, 15.4 mV by default)
    el_num : integer, from which electrode number has been measured (optional, 2 by default)
    current_step : float, which current step (pA) has been used between consecutive experiments (optional, 20 by default)
    start : float (s), start of the stimulation interval (optional, 0.1 by default)
    end: float (s), end of the stimulation interval (optional, 0.7 by default)
    
    Returns
    -------
    All_Cells_Features : DataFrame with values for all required features mentioned in get_cell_features
    """
    All_Cells_Features = pd.DataFrame()
    print('Extracting ephys properties cell by cell:')
    for data, name in zip(data_tuple, names):
        #print(name)
        print('.', end='')
        
        # Extract relevant time, voltage and current information and then set up the observation corresponding to
        # injecting current_val pA in the cell
        time_obs, voltage_obs, current_obs, curr_index_0_obs = get_time_voltage_current_currindex0(data)
        voltage_obs-=liquid_junction_potential
        start_index = (np.abs(time_obs - 0.1)).argmin() # Find closest index where the injection current starts
        end_index = (np.abs(time_obs - 0.7)).argmin() # Find closest index where the injection current ends
        
        
        # ------------------------------ #
        # Extract input resistance and membrane time constant #
        
        filter_ = 10
        if (1/time_obs[1]-time_obs[0]) < 20e3:
            filter_ = (1/time_obs[1]-time_obs[0])/(1e3*2)-0.5
        
        ####################################
        # Input resistance
        ####################################
        
        df_related_features = pd.DataFrame()
        for c, curr in enumerate(current_obs[curr_index_0_obs-4:curr_index_0_obs+4]):
            current_array = curr*np.ones_like(time_obs)
            current_array[:start_index] = 0
            current_array[end_index:len(current_array)] = 0
            EphysObject = efex.EphysSweepFeatureExtractor(t = time_obs, v = voltage_obs[:, curr_index_0_obs-4+c], \
                                                          i = current_array, start = start, \
                                                          end = end, filter = filter_)
            
            # Some easily found features
            df_features = EphysObject._sweep_features
            
            # Adding current (pA)
            df_features.update({'current': curr})
            
            # Adding minimal/maximal voltage deflection
            if curr < 0:
                v_peak_, peak_index = EphysObject.voltage_deflection("min")
                df_features.update({'deflection': v_peak_})
            elif curr==0:
                df_features.update({'deflection':np.average(voltage_obs[:,curr_index_0_obs]
                                                            [start_index:ft.find_time_index(time_obs, 0.2)])})
            elif curr>0:
                EphysObject.process_spikes()
                if EphysObject._spikes_df.empty:
                    v_peak_, peak_index = EphysObject.voltage_deflection("max")
                    df_features.update({'deflection': v_peak_})
            
            # Concatenating
            df_related_features = pd.concat([df_related_features, pd.DataFrame([df_features])], sort = True)
        
        indices = ~np.isnan(df_related_features['deflection'].values)
        ransac.fit(df_related_features['current'].values[indices].reshape(-1, 1), \
                   df_related_features['deflection'].values[indices].reshape(-1, 1))
        slope = ransac.estimator_.coef_[0][0]
        R_input = slope*1000
        
        
        ####################################
        # Extract firing trace related features and (pre-spike) membrane time constant
        ####################################
        
        tau=np.nan
        Vi=np.nan
        
        if current_val not in current_obs:
            x_o = np.ones((len(ephys_features)-2))*np.nan
        else:
            current_array = current_val*np.ones_like(time_obs)
            current_array[:start_index] = 0
            current_array[end_index:len(current_array)] = 0
            curr_index = np.where(current_obs==current_val)[0][0]
            Vi=voltage_obs[0,curr_index]
            EphysObject = efex.EphysSweepFeatureExtractor(t = time_obs, v = voltage_obs[:, curr_index], \
                                                          i = current_array, start = start, \
                                                          end = end, filter = filter_)
            EphysObject.process_spikes()
            
            if not EphysObject._spikes_df.empty: # There are APs and in the positive current regime
                if False in list(EphysObject._spikes_df['clipped']): # There should be spikes that are also not clipped
                    
                    time_first_spike=EphysObject._spikes_df['threshold_t'].values[0]
                    if time_first_spike>start:
                        while True:
                            try:
                                tau=ft.fit_prespike_time_constant(voltage_obs[:,np.where(current_obs==current_val)[0][0]],
                                                          time_obs,
                                                          0.1,
                                                          time_first_spike)*1000
                                break
                            except ValueError: # Pre-spike time cte could just not be reliably estimated
                                break
            
            I, t_on, t_off, dt, t, A_soma = syn_current_start(exp_input_res=R_input, exp_tau=tau, curr_level=3e-4)
            observation = {'data': voltage_obs[:20001, curr_index][np.newaxis,:], \
                           'time': time_obs[:20001]*1e3, 'dt':4*1e-5*1e3, 'I': I}

            # calculate summary statistics from the observation
            x_o = calculate_summary_statistics(observation)[0,:]
    
        # Calculate a tentative membrane time constant and input resistance derived 1-comp area
        # provided membrane capacitance would be 1
        area=tau*1e3/(R_input*1e6)*1e8
        
        
        # ------------------------------ #
        # Concatenating it all #        
        x_o=np.concatenate([x_o,np.array([Vi, area, R_input, tau])],axis=0)
        cell_features_obs = dict(zip(ephys_features, list(x_o)))
        Cell_Features_obs = pd.DataFrame([cell_features_obs])
        Cell_Features_obs = Cell_Features_obs.reindex(columns = ephys_features)
        All_Cells_Features = pd.concat([All_Cells_Features, Cell_Features_obs], sort = True)
    All_Cells_Features.insert(0, 'name sample', names)
    return All_Cells_Features


def cell_features_Gouwens(data_tuple, names, ephys_features, liquid_junction_potential = 14, start=0.1, end=1.1):
    """ Analyses a stream of cell dictionaries and outputs all the cell's features in a concatenated DataFrame
    Parameters
    ----------
    data_tuple : tuple of dictionaries of data full of voltage (V) and time (s) traces for different cells
    names : tuple of names of the samples
    ephys_features : list of ephys feature names being extracted
    liquid_junction_potential : float, potential to be substracted from all traces (optional, 14 mV by default for Gouwens)
    start : float (s), start of the stimulation interval (optional, 0.1 by default)
    end: float (s), end of the stimulation interval (optional, 1.1 by default)
    
    Returns
    -------
    All_Cells_Features : DataFrame with values for all required features mentioned in get_cell_features
    """
    All_Cells_Features = pd.DataFrame()
    print('Extracting ephys properties cell by cell:')
    for i, (data, name) in enumerate(zip(data_tuple, names)):
        #print(name)
        if i%100==0:
            print('\n',i)
        print('.', end='')
        
        # Extract relevant time, voltage and current of highest injected square pulse
        time_obs, voltage_obs, current_obs, curr_obs_applied = get_time_voltage_current_currindex0_gouwens(data)
        if np.isnan(np.sum(time_obs)):
            cell_features_obs = dict(zip(ephys_features, [np.nan]*len(ephys_features)))
        else:
            voltage_obs-=liquid_junction_potential
        
            observation = {'data': voltage_obs[np.newaxis,:], \
                           'time': time_obs*1e3, 'dt':2e-5*1e3, 'I': current_obs}

            # calculate summary statistics from the observation
            x_o = calculate_summary_statistics_gouwens(observation)[0,:]
        
            # ------------------------------ #
            # Concatenating it all #        
            cell_features_obs = dict(zip(ephys_features, list(x_o)+[curr_obs_applied]))
        Cell_Features_obs = pd.DataFrame([cell_features_obs])
        Cell_Features_obs = Cell_Features_obs.reindex(columns = ephys_features)
        All_Cells_Features = pd.concat([All_Cells_Features, Cell_Features_obs], sort = True)
    All_Cells_Features.insert(0, 'name sample', names)
    return All_Cells_Features
