import numpy as np
import math
from scipy import stats as spstats

# sbi
import sbi.utils as utils

# warning handlings
import warnings

# import saver utilities
import pickle

# parallel processing
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pathos_Pool

# keep track of some time
import time

# ephys extraction
import ephys_extractor as efex
import ephys_features as ft
from ephys_utils import sigmoid

import os
from brian2 import *
import torch


# Helper functions constructed for Brian2
def initialize_parameter(variableview, value):
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name
def set_parameter_value(identifier, value):
    np.atleast_1d(value).tofile(os.path.join(device.project_dir,
                                             'static_arrays',
                                             identifier))
def run_again():
    device.run(device.project_dir, with_output=False, run_args=[])


class EphysModel:
    """
    Set up a class that contains the true experimental observation and the capabilities to infer conductance-based model parameters which,
    provided to the model, reproduces the observation closely.
    """
    
    def __init__(self, name, T, E_Na, E_K, E_Ca, start=100, end=700, dt=0.04, ephys_features=None,
                 label_params=None, use_feature_list=None, use_param_list=None, area_Vi_list=None,
                 ephys_limits=None, n_processes=None, verbose=True, use_pathos=False, noise_factor=10, seed=None, chunk_size=None,
                 save_chunks=False):
        """
        Initializing the ephys object.
        
        Parameters
        ----------
        name: dataset name (String)
        T: temperature (°C)
        E_Na: Nernst potential for sodium (mV)
        E_K: Nernst potential for potassium (mV)
        E_Ca: Nernst potential for calcium (mV)
        start: start of current clamp protocol (ms, optional, default=100)
        end: end of current clamp protocol (ms, optional, default=700)
        dt: time step (ms, optional, default=0.04)
        ephys_features: list of names of summary statistics (optional, default=None)
        label_params: list of names of the parameters (optional, default=None)
        use_feature_list: list of summary statistic indices that correspond to features you'd like to keep for the inference
                          (optional, default=None)
        use_param_list: list of parameter indices that correspond to parameters you'd like to keep for the inference
                          (optional, default=None)
        area_Vi_list: list of parameter indices that correspond to the area and initial membrane voltage you'd like to convert to
                          observed summary statistics (optional, default=None)
        ephys_limits: dictionary with upper and lower limits for every ephys feature
        n_processes: amount of parallel workers available (optional, default=None)
        verbose: set to False if you'd not like to see printed information (optional, default=True)
        use_pathos: if True we try parallelization with pathos (optional, default=False)
        noise_factor: current noise (optional, default=2pA)
        seed: set seed for adding current noise to the model (optional, default=None)
        chunk_size: size of batch of simulations (optional, default=None)
        save_chunks: save chunks of simulated data if True
        """
        self.name=name
        self.T=T
        self.E_Na=E_Na
        self.E_K=E_K
        self.E_Ca=E_Ca
        self.start=start
        self.end=end
        self.dt=dt
        self.ephys_features=ephys_features
        self.label_params=label_params
        self.use_feature_list=use_feature_list
        self.use_param_list=use_param_list
        self.area_Vi_list=area_Vi_list
        self.ephys_limits=ephys_limits
        self.n_processes=n_processes
        self.verbose=verbose
        self.use_pathos=use_pathos
        self.noise_factor=noise_factor
        self.seed=seed
        self.chunk_size=chunk_size
        self.save_chunks=save_chunks
        
        self.obs=None
        self.xo=None
        self.theta=None
        self.stats=None
        self.used_ephys_features=None
        self.used_params=None
        self.pruned_ephys=False
        self.posterior=None
    
        
    def sim(self, params):
        """
        Create a training set of parameters and summary statistics of simulations corresponding to setting up the model with those
        parameters. This code is designed to help parallelize the simulations across cpu's and to save memory along the way while
        saving what's necessary.
        
        Parameters
        ----------
        params: 2d torch array of parameter sets
        """
        self.theta=params
        n_samples=self.theta.shape[0]
        chunk_size=self.chunk_size
        
        if self.verbose:
            if chunk_size is not None:
                print('Chunks:')
        
        if self.n_processes is None:
            if chunk_size is None:
                self.stats = self._simulation_wrapper(params)
            else:
                # Create chunks
                fit_params_=[]
                for i in range(math.ceil(n_samples/chunk_size)):
                    fit_params_.append(params[i*chunk_size:(i+1)*chunk_size])
                if n_samples%chunk_size!=0:
                    fit_params_[-1]=params[(math.ceil(n_samples/chunk_size)-1)*chunk_size:
                                          (math.ceil(n_samples/chunk_size)-1)*chunk_size+n_samples%chunk_size]

                # Get summ stats for every chunk
                summstats_=[]
                for chunk, sim_params in enumerate(fit_params_):
                    summstats_.append(self._simulation_wrapper(sim_params))
                    if self.verbose:
                        if (chunk+1)%10==0:
                            print(chunk+1,end='')
                        else:
                            print('.',end='')
                    if self.save_chunks:
                        np.savez('./save_sims/{}_chunks/{}.npz'.format(self.name, chunk+1),
                                 theta=sim_params,
                                 stats=summstats_[-1].to(torch.float32).numpy(),
                                )
                self.stats=torch.cat(summstats_).to(torch.float32)
   
        elif not self.use_pathos:
            if chunk_size is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with Pool(self.n_processes) as pool:
                        simulation_outputs = pool.map(self._simulation_wrapper,
                                                      np.reshape(self.theta.numpy(),
                                                                 (self.n_processes,
                                                                  int(n_samples/self.n_processes),
                                                                  self.theta.shape[1])
                                                                )
                                                     )

                self.stats = torch.cat(simulation_outputs).to(torch.float32)
            else:
                # ToDo: chunk size now needs to dividable by self.n_processes (make more flexible, or make pathos default)
                # Create chunks
                fit_params_=[]
                for i in range(math.ceil(n_samples/chunk_size)):
                    fit_params_.append(params[i*chunk_size:(i+1)*chunk_size])
                if n_samples%chunk_size!=0:
                    fit_params_[-1]=params[(math.ceil(n_samples/chunk_size)-1)*chunk_size:
                                          (math.ceil(n_samples/chunk_size)-1)*chunk_size+n_samples%chunk_size]
                
                # Get summ stats for every chunk
                summstats_=[]
                for chunk, sim_params in enumerate(fit_params_):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with Pool(self.n_processes) as pool:
                            simulation_outputs = pool.map(self._simulation_wrapper,
                                                          np.reshape(sim_params.numpy(),
                                                                     (self.n_processes,
                                                                      int(sim_params.shape[0]/self.n_processes),
                                                                      self.theta.shape[1])
                                                                    )
                                                         )

                    simulation_outputs = torch.cat(simulation_outputs).to(torch.float32)
                    summstats_.append(simulation_outputs)
                    if self.verbose:
                        if (chunk+1)%10==0:
                            print(chunk+1,end='')
                        else:
                            print('.',end='')
                    if self.save_chunks:
                        np.savez('./save_sims/{}_chunks/{}.npz'.format(self.name, chunk+1),
                                 theta=sim_params,
                                 stats=summstats_[-1].to(torch.float32).numpy(),
                                )
                self.stats=torch.cat(summstats_).to(torch.float32)
        
        elif self.use_pathos:
            if chunk_size is None:
                # Calculate runs per cpu (last cpu is used for remainder of division)
                singlejobs = round(np.shape(self.theta.numpy())[0] / (self.n_processes - 1))
                # build a new parameter list, every index corresponds to a parameter set 'set' for a specific cpu
                fitparams = []
                for i in range(0, self.n_processes - 1):
                    fitparams.append(self.theta.numpy()[i * singlejobs: (i+1) * singlejobs])
                fitparams.append(self.theta.numpy()[(i+1)*singlejobs:])
                if singlejobs==0:
                    fitparams = fitparams[-1]
                if singlejobs!=0:
                    if len(fitparams) > self.theta.numpy().shape[0]/singlejobs: # Let us not start simulating with empty parameter sets
                        fitparams = fitparams[:math.ceil(self.theta.numpy().shape[0]/singlejobs)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with Pathos_Pool(processes=self.n_processes) as pool:
                            simulation_outputs = pool.map(self._simulation_wrapper, \
                                                        tuple(fitparams)
                                                         )

                self.stats = torch.cat(simulation_outputs).to(torch.float32)
            else:
                # Create chunks
                fit_params_=[]
                for i in range(math.ceil(n_samples/chunk_size)):
                    fit_params_.append(params[i*chunk_size:(i+1)*chunk_size])
                if n_samples%chunk_size!=0:
                    fit_params_[-1]=params[(math.ceil(n_samples/chunk_size)-1)*chunk_size:
                                          (math.ceil(n_samples/chunk_size)-1)*chunk_size+n_samples%chunk_size]                
                
                # Get summ stats for every chunk
                summstats_=[]
                for chunk, sim_params in enumerate(fit_params_):
                    # Calculate runs per cpu (last cpu is used for remainder of division)
                    singlejobs = round(np.shape(sim_params.numpy())[0] / (self.n_processes - 1))
                    # build a new parameter list, every index corresponds to a parameter set 'set' for a specific cpu
                    fitparams = []
                    for i in range(0, self.n_processes - 1):
                        fitparams.append(sim_params.numpy()[i * singlejobs: (i+1) * singlejobs])
                    fitparams.append(sim_params.numpy()[(i+1)*singlejobs:])
                    if singlejobs==0:
                        fitparams = fitparams[-1]
                    if singlejobs!=0:
                        if len(fitparams) > sim_params.numpy().shape[0]/singlejobs: # Let us not start simulating with empty parameter sets
                            fitparams = fitparams[:math.ceil(sim_params.numpy().shape[0]/singlejobs)]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with Pathos_Pool(processes=self.n_processes) as pool:
                                simulation_outputs = pool.map(self._simulation_wrapper, \
                                                            tuple(fitparams)
                                                             )

                    simulation_outputs = torch.cat(simulation_outputs).to(torch.float32)
                    summstats_.append(simulation_outputs)
                    if self.verbose:
                        if (chunk+1)%10==0:
                            print(chunk+1,end='')
                        else:
                            print('.',end='')
                    if self.save_chunks:
                        np.savez('./save_sims/{}_chunks/{}.npz'.format(self.name, chunk+1),
                                 theta=sim_params,
                                 stats=summstats_[-1].to(torch.float32).numpy(),
                                )
                        
                self.stats=torch.cat(summstats_).to(torch.float32)
            
            if self.verbose:
                print('\nTraining set of parameters and summary statistics has been made:')
                print('parameters shape: ', self.theta.shape)
                print('summary statistics shape: ', self.stats.shape)        
        
                    
    def _simulation_wrapper(self, params): 
        obs = self._run_HH_model(params)
        summstats = torch.as_tensor(self._calculate_summary_statistics(obs))
        return summstats
        
    def _run_HH_model(self, params):

        t, v, I = self._sim_Brian2(params)
        return dict(data=v, time=t, dt=self.dt, I=I.reshape(-1))
        
    def _sim_Brian2(self, params):
        
        params = np.asarray(params, float)
    
        if params.ndim == 1:
            return self._sim_Brian2(params[np.newaxis, :])
        else:

            ###################
            # Brian 2 initiliasations to make it run faster in C++

            device.reinit() 
            # 'cython' could make this code run faster (check paper, converting to C++ blocks of code)
            prefs.codegen.target = 'cython'
            set_device('cpp_standalone', clean = True, directory = None)
            defaultclock.dt = self.dt*ms

            n_sims=0
            voltage_list=[]
            
            #params[:,:8][params[:,:8]<0]=0
            
            for run_id, theta in enumerate(params):
                #print(run_id, end='')
                if n_sims==0: # We set the body once, for other parameter combinations we simply swap parameter values
                    #print('.', end='')
                    
                    ###################
                    # Model parameter initialisations
                    # Some are set, some are performed inference on
                    # Inspired by Martin Pospischil et al. "Minimal Hodgkin-Huxley type models for
                    # different classes of cortical and thalamaic neurons", and Hay et al. "Models of neocortical layer
                    # 5b pyramidal cells capturing a wide range of dendritic and perisomatic active properties".
                    
                    # Nernst potentials
                    ECa = self.E_Ca*mV
                    ENa = self.E_Na*mV
                    EK = self.E_K*mV
                    rate_to_SS_factor=1
                    
                    # It is important to adapt your kinetics to the temperature of your experiment
                    # temperature coeff., https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)
                    T_1 = 21.          # °C, from paper Etay Hay et al.
                    T_2 = self.T      # °C, experiment was actually done at 25 °C
                    Q10=2.3           # temperature coeff.
                    t_adj_factor_Hay = Q10**((T_2 - T_1)/10)

                    T_1 = 36.          # °C, from paper Pospichil et al.
                    T_2 = self.T      # °C, experiment was actually done at 25 °C
                    Q10=2.3           # temperature coeff.
                    t_adj_factor_Pos = Q10**((T_2 - T_1)/10)

                    ####################
                    # Setting up the injection+noise current, the model equations, further initialisations and run the model
                    
                    current_val=300
                    I = TimedArray(([0]+[current_val]*6+[0])*pA, dt=100*ms)
                    if self.noise_factor is None:
                        I_noise = TimedArray([0]*pA, dt=self.end*ms)
                    else:
                        if self.seed is not None:
                            rng=np.random.RandomState(seed)
                        else:
                            rng=np.random.RandomState()
                        I_noise = TimedArray(self.noise_factor*rng.randn(round((self.end+100)/self.dt))*pA, dt=self.dt*ms)
                            
                    ####################
                    # The conductance-based model
                    eqs = '''

                         dVm/dt = - (
                                    gNat*m**3*h*(Vm - ENa) + gNa*m_**3*h_*(Vm - ENa) + gKd*n**4*(Vm - EK) + 
                                    gM*p*(Vm - EK) + gKv31*v*(Vm - EK) + gL*q**2*r*(Vm - ECa) + gleak*(Vm - El) -
                                    I_inj - I_inj_noise
                                    ) / (C * area) : volt
                         
                         I_inj = I(t): amp
                         I_inj_noise = I_noise(t): amp
                         
                         dm/dt = (alpham*(1-m) - betam*m) * t_adj_factor_Hay : 1
                         dh/dt = (alphah*(1-h) - betah*h) * t_adj_factor_Hay : 1
                         dp/dt = ((p_inf - p)/tau_p) * t_adj_factor_Pos : 1
                         dv/dt = ((v_inf - v)/tau_v) * t_adj_factor_Hay : 1
                         dq/dt = alphaq*(1-q) - betaq*q : 1
                         dr/dt = alphar*(1-r) - betar*r : 1
                         dm_/dt = (alpham_*(1-m_) - betam_*m_) * t_adj_factor_Pos / rate_to_SS_factor : 1
                         dh_/dt = (alphah_*(1-h_) - betah_*h_) * t_adj_factor_Pos / rate_to_SS_factor : 1
                         dn/dt = (alphan*(1-n) - betan*n) * t_adj_factor_Pos / rate_to_SS_factor : 1
                         

                         alpham = (0.182/mV) * (Vm + 38.*mV) / (1-exp((-(Vm + 38.*mV))/(6.*mV)))/ms : Hz
                         betam = (-0.124/mV) * (Vm + 38.*mV) / (1-exp((Vm + 38.*mV)/(6.*mV)))/ms : Hz
                         alphah = (-0.015/mV) * (Vm + 66.*mV) / (1-exp((Vm + 66.*mV)/(6.*mV)))/ms : Hz 
                         betah = (0.015/mV) * (Vm + 66.*mV) / (1-exp((-(Vm + 66.*mV))/(6.*mV)))/ms : Hz
                         
                         alphan = (-0.032/mV) * (Vm - VT - 15.*mV) / (exp((-(Vm - VT - 15.*mV)) / (5.*mV)) - 1.)/ms : Hz
                         betan = 0.5*exp(-(Vm - VT - 10.*mV) / (40.*mV))/ms : Hz
                         
                         alpham_ = (-0.32/mV) * (Vm - VT - 13.*mV) / (exp((-(Vm - VT - 13.*mV))/(4.*mV)) - 1.)/ms : Hz
                         betam_ = (0.28/mV) * (Vm - VT - 40.*mV) / (exp((Vm - VT - 40.*mV)/(5.*mV)) - 1.)/ms : Hz
                         alphah_ = 0.128 * exp(-(Vm - VT - 17.*mV) / (18.*mV))/ms : Hz
                         betah_ = 4./(1. + exp((-(Vm - VT - 40.*mV)) / (5.*mV)))/ms : Hz                         
                         
                         p_inf = 1./(1. + exp(-(Vm + 35.*mV)/(10.*mV))) : 1
                         tau_p = (tau_max/1000.)/(3.3 * exp((Vm + 35.*mV)/(20.*mV)) + exp(-(Vm + 35.*mV)/(20.*mV))) : second
                         
                         v_inf = 1. / (1. + exp((-(Vm - 18.7*mV))/(9.7*mV))) : 1
                         tau_v = 4.*ms / (1. + exp((-(Vm + 56.56*mV))/(44.14*mV))) : second


                         alphaq = (-0.055/mV) * (27.*mV + Vm) / (exp((-(27.*mV + Vm))/(3.8*mV)) - 1.)/ms : Hz
                         betaq = 0.94*exp(-((75.*mV + Vm)) / (17.*mV))/ms : Hz
                         alphar = 0.000457*exp((-(13.*mV + Vm))/(50.*mV))/ms : Hz
                         betar = 0.0065/(1. + exp((-(15.*mV + Vm))/(28.*mV)))/ms : Hz                         
                         
                         C : farad/metre**2 (shared)
                         area : metre**2 (shared)
                         gNat : siemens (shared)
                         gNa : siemens (shared)
                         gKd : siemens (shared)
                         gM : siemens (shared)
                         gKv31 : siemens (shared)
                         gL : siemens (shared)
                         gleak : siemens (shared)
                         El: volt (shared)
                         tau_max : second (shared)
                         VT : volt (shared)
                        
                         '''

                    neurons = NeuronGroup(1, eqs, method='exponential_euler', name='neurons')
                    Vm_mon = StateMonitor(neurons, ['Vm', 'I_inj'], record = True, name = 'Vm_mon') # Specify what to record
                    
                    param_C = initialize_parameter(neurons.C, theta[0]*uF/cm**2)
                    input_res = theta[1] #MOhm
                    tau = theta[2] #ms
                    A_1comp = ((tau*1e3)/(theta[0]*input_res*1e6))*1e8*umetre**2 # mirom^2
                    g_leak = 1/(input_res*1e3)/((tau*1e3)/(theta[0]*input_res*1e6)) # mS/cm^2
                    param_area = initialize_parameter(neurons.area, A_1comp)
                    param_gleak = initialize_parameter(neurons.gleak, g_leak*mS/cm**2*A_1comp)
                    param_gNat = initialize_parameter(neurons.gNat, theta[3]*mS/cm**2*A_1comp)
                    param_gNa = initialize_parameter(neurons.gNa, theta[4]*mS/cm**2*A_1comp)
                    param_gKd = initialize_parameter(neurons.gKd, theta[5]*mS/cm**2*A_1comp)
                    param_gM = initialize_parameter(neurons.gM, theta[6]*mS/cm**2*A_1comp)
                    param_gKv31 = initialize_parameter(neurons.gKv31, theta[7]*mS/cm**2*A_1comp)
                    param_gL = initialize_parameter(neurons.gL, theta[8]*mS/cm**2*A_1comp)
                    param_El = initialize_parameter(neurons.El, theta[9]*mV)
                    param_tau_max = initialize_parameter(neurons.tau_max, theta[10]*second)
                    param_VT = initialize_parameter(neurons.VT, theta[11]*mV)

                    neurons.Vm = 'El'
                    neurons.m = '1/(1 + betam/alpham)'         # Would be the solution when dm/dt = 0
                    neurons.h = '1/(1 + betah/alphah)'         # Would be the solution when dh/dt = 0
                    neurons.m_ = '1/(1 + betam_/alpham_)'      # Would be the solution when dm_/dt = 0
                    neurons.h_ = '1/(1 + betah_/alphah_)'      # Would be the solution when dh_/dt = 0                 
                    neurons.n = '1/(1 + betan/alphan)'         # Would be the solution when dn/dt = 0
                    neurons.p = 'p_inf'                        # Would be the solution when dp/dt = 0
                    neurons.v = 'v_inf'                        # Would be the solution when dv/dt = 0
                    neurons.q = '1/(1 + betaq/alphaq)'         # Would be the solution when dq/dt = 0                 
                    neurons.r = '1/(1 + betar/alphar)'         # Would be the solution when dr/dt = 0
                    
                    run((self.end+100)*ms)
                    n_sims+=1

                    voltage_list.append(Vm_mon.Vm/mV)

                else:
                    #print('.', end='')
                    # Run it again, swap some parameters
                    set_parameter_value(param_C, theta[0]*uF/cm**2)
                    input_res = theta[1] #MOhm
                    tau = theta[2] #ms
                    A_1comp = ((tau*1e3)/(theta[0]*input_res*1e6))*1e8*umetre**2 # mirom^2
                    g_leak = 1/(input_res*1e3)/((tau*1e3)/(theta[0]*input_res*1e6)) # mS/cm^2
                    set_parameter_value(param_area, A_1comp)
                    set_parameter_value(param_gleak, g_leak*mS/cm**2*A_1comp)
                    set_parameter_value(param_gNat, theta[3]*mS/cm**2*A_1comp)
                    set_parameter_value(param_gNa, theta[4]*mS/cm**2*A_1comp)
                    set_parameter_value(param_gKd, theta[5]*mS/cm**2*A_1comp)
                    set_parameter_value(param_gM, theta[6]*mS/cm**2*A_1comp)
                    set_parameter_value(param_gKv31, theta[7]*mS/cm**2*A_1comp)
                    set_parameter_value(param_gL, theta[8]*mS/cm**2*A_1comp)
                    set_parameter_value(param_El, theta[9]*mV)
                    set_parameter_value(param_tau_max, theta[10]*second)
                    set_parameter_value(param_VT, theta[11]*mV)
                    run_again()
                    n_sims+=1
                    voltage_list.append(Vm_mon.Vm/mV)
                    
        return Vm_mon.t/ms, np.array(voltage_list), Vm_mon.I_inj/pA
    
    
    def _calculate_summary_statistics(self, x):
        """Calculate summary statistics.
    
        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """
        n_mom = 3
        t_on=self.start
        t_off=self.end
        t = x["time"]
        dt = x["dt"]
        I = x["I"]

        sum_stats=[]

        for v in x['data']:
            #print(i, end='')

            # -------- #
            # 1st part: features that electrophysiologists are actually interested in #
            v=v.reshape(-1)
            EphysObject = efex.EphysSweepFeatureExtractor(t = t/1e3, v = v, \
                                                          i = I, start = t_on/1e3, \
                                                          end = t_off/1e3, filter = 10)
            EphysObject.process_spikes()
            AP_count_1st_8th = np.nan
            AP_count_1st_quarter = np.nan
            AP_count_1st_half = np.nan
            AP_count_2nd_half = np.nan
            AP_count_1st_quarter = np.nan
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
                
                
                # AP counts are zero-inflated for simulations. Thus comparisons with truth are oftentimes small,
                # much smaller than differences according to other features. A log transformation helps.
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
                    #if cv<=0:
                    #    print('cv<0')
                    # And now the same for AP heights in the trace
                    #AP_fano_factor = EphysObject._sweep_features['AP_fano_factor']
                    AP_cv = EphysObject._sweep_features['AP_cv']
                    #if AP_cv<0:
                    #    print('AP_cv<0')

                    # Add the AP AHP, threshold, amplitude, width and UDR (upstroke-to-downstroke ratio) of the
                    # first fired AP in the trace
                    AHP = EphysObject._spikes_df.loc[0, 'fast_trough_v'] - EphysObject._spikes_df.loc[0, 'threshold_v']
                    AP_threshold = EphysObject._spikes_df.loc[0, 'threshold_v']
                    AP_amplitude = EphysObject._spikes_df.loc[0, 'peak_height']
                    AP_width = EphysObject._spikes_df.loc[0, 'width']*1000
                    #UDR = EphysObject._spikes_df.loc[0, 'upstroke_downstroke_ratio']
                    
                    if AP_count > 2:

                        AHP_3 = EphysObject._spikes_df.loc[2, 'fast_trough_v'] - EphysObject._spikes_df.loc[2, 'threshold_v']
                        AP_threshold_3 = EphysObject._spikes_df.loc[2, 'threshold_v']
                        AP_amplitude_3 = EphysObject._spikes_df.loc[2, 'peak_height']
                        AP_width_3 = EphysObject._spikes_df.loc[2, 'width']*1000
                        #UDR_3 = EphysObject._spikes_df.loc[2, 'upstroke_downstroke_ratio']
                        #if np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)!=0:
                        #    SFA = np.sum(EphysObject._spikes_df['threshold_index'] > half_stim_index) / \
                        #          np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)
                    
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
            
            rest_pot = np.mean(v[(t < t_on)])

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

            sum_stats.append(sum_stats_vec)

        return np.array(sum_stats)
        