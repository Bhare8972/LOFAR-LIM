#!/usr/bin/env python3

""" This is fundamental peice of autoCorrelator3 that plots a pulse"""

import numpy as np
import matplotlib.pyplot as plt

import h5py

from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD, even_antName_to_odd
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.read_pulse_data import read_station_delays


def plot_stations(timeID, polarization, input_file_name, source_XYZT, known_station_delays, stations="all", referance_station="CS002", min_antenna_amplitude=10, skip_stations=[], antennas_to_exclude=[], seperation_factor=1, 
                  plot_real=False, plot_peak_time=True):
    """Plot all pulses on all stations in an autocorrelator file. polarization should be 0 or 1. stations should be "all", "RS", or "CS" """

    if polarization == 0:
        dataset_index = 1 ##1 for even HE, 3 for odd HE
        peaktime_pol = "PolE_peakTime" ## even or odd peak times 
        antOffset_pol = 'PolE_timeOffset'
    elif polarization == 1:
        dataset_index = 3 ##1 3 for odd HE
        peaktime_pol = "PolO_peakTime" ## even or odd peak times 
        antOffset_pol = 'PolO_timeOffset'
    
    if referance_station not in known_station_delays:
        known_station_delays[referance_station] = 0                                                      
    known_station_delays = {station:T-known_station_delays[referance_station] for station,T in known_station_delays.items()}
    
    data_file = h5py.File(input_file_name, "r")
    
    station_names = [sname for sname in data_file.keys() if sname not in skip_stations]
    known_station_delays = {sname:known_station_delays[sname] if sname in known_station_delays else 0.0 for sname in station_names}

    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True) for sname in station_names }
    
    stat_i=0
    for (sname, stat_group) in data_file.items():
        if (stations!="all" and stations!=sname[:2]) or sname in skip_stations:
            continue
        
        stat_data = raw_data_files[sname]
        ant_loc_dict = {ant_name:ant_loc for ant_name,ant_loc in zip(stat_data.get_antenna_names(),stat_data.get_LOFAR_centered_positions())}
        
        
        trace_max = 0.0
        for ant_name, ant_dataset in stat_group.items():
            trace = np.array( ant_dataset[ dataset_index ])
            if np.max(trace)>trace_max:
                trace_max = np.max(trace)
                
        trace_max*=2
        
        
        for ant_name, ant_dataset in stat_group.items():
            
            trace = np.array( ant_dataset[ dataset_index ])
            if np.max(trace)<min_antenna_amplitude or ant_name in antennas_to_exclude:
                continue
            
            ant_loc = ant_loc_dict[ant_name]
            travel_delay = np.sqrt( (ant_loc[0]-source_XYZT[0])**2 + (ant_loc[1]-source_XYZT[1])**2 + (ant_loc[2]-source_XYZT[2])**2 )/v_air
            total_correction = travel_delay + source_XYZT[3] + known_station_delays[sname]
            
            step = stat_i*seperation_factor
            
            peak_time = ant_dataset.attrs[ peaktime_pol ] - total_correction
            color=None
            if plot_peak_time:
                p = plt.plot((peak_time,peak_time), (step, step+0.5))
                color = p[0].get_color()
            
            
            trace_time = np.arange(len(trace))*5.0E-9 + ant_dataset.attrs['starting_index']*5.0E-9 + ant_dataset.attrs[ antOffset_pol ] - total_correction
            plt.plot(trace_time, trace/trace_max + step, color = color)
            
            if plot_real:
                rt = np.array( ant_dataset[ dataset_index-1 ])
                plt.plot(trace_time, rt/trace_max + step, color = color)
                
        stat_i += 1
            
        plt.annotate(sname, xy=(0.0,step+0.5))
    plt.axvline(x=0)
    plt.show()
    
    
def plot_one_station(timeID, polarization, input_file_name, source_XYZT, known_station_delays, station, referance_station="CS002", min_antenna_amplitude=10, plot_real=False):
    """Plot all pulses on all stations in an autocorrelator file. polarization should be 0 or 1. station should be station name """

    if polarization == 0:
        dataset_index = 1 ##1 for even HE, 3 for odd HE
        peaktime_pol = "PolE_peakTime" ## even or odd peak times 
        antOffset_pol = 'PolE_timeOffset'
    elif polarization == 1:
        dataset_index = 3 ##1 for even HE, 3 for odd HE
        peaktime_pol = "PolO_peakTime" ## even or odd peak times 
        antOffset_pol = 'PolO_timeOffset'
    
    if referance_station not in known_station_delays:
        known_station_delays[referance_station] = 0                                                      
    known_station_delays = {station:T-known_station_delays[referance_station] for station,T in known_station_delays.items()}
    
    data_file = h5py.File(input_file_name, "r")
    
    station_names = list(data_file.keys())
    known_station_delays = {sname:known_station_delays[sname] if sname in known_station_delays else 0.0 for sname in station_names}

    raw_fpaths = filePaths_by_stationName(timeID)
    stat_group = data_file[station]
    stat_data = MultiFile_Dal1(raw_fpaths[station], force_metadata_ant_pos=True)
    
    current_amp = 0.0
    for ant_name, ant_loc in zip(stat_data.get_antenna_names(),stat_data.get_LOFAR_centered_positions()):
        if ant_name not in stat_group:
            continue
        
        ant_dataset = stat_group[ant_name]
        trace = np.array( ant_dataset[dataset_index] )
        trace_amp = np.max(trace)
        
        if trace_amp<min_antenna_amplitude:
            continue
        
        
        travel_delay = np.sqrt( (ant_loc[0]-source_XYZT[0])**2 + (ant_loc[1]-source_XYZT[1])**2 + (ant_loc[2]-source_XYZT[2])**2 )/v_air
        total_correction = travel_delay + source_XYZT[3] + known_station_delays[station]
        
        peak_time = ant_dataset.attrs[peaktime_pol] - total_correction
        
        p = plt.plot((peak_time,peak_time), (current_amp, current_amp+trace_amp))
        print(ant_name, peak_time, ant_dataset.attrs[antOffset_pol])
        
        trace_time = np.arange(len(trace))*5.0E-9 + ant_dataset.attrs['starting_index']*5.0E-9 + ant_dataset.attrs[antOffset_pol] - total_correction
        plt.plot(trace_time, trace + current_amp, color = p[0].get_color())
        
        if plot_real:
            rt = np.array( ant_dataset[ dataset_index-1 ])
            plt.plot(trace_time, rt + current_amp, color = p[0].get_color())
        
        plt.annotate(ant_name, xy=(0.0, current_amp+0.5*trace_amp))
        
        current_amp += trace_amp
    plt.axvline(x=0)
    plt.show()
    
def plot_one_station_allData(timeID, input_file_name, source_XYZT, known_station_delays, station, referance_station="CS002", min_antenna_amplitude=10):
    """Plot all pulses on all stations in an autocorrelator file. polarization should be 0 or 1. station should be station name """

    if referance_station not in known_station_delays:
        known_station_delays[referance_station] = 0                                                      
    known_station_delays = {station:T-known_station_delays[referance_station] for station,T in known_station_delays.items()}
    
    data_file = h5py.File(input_file_name, "r")
    
    station_names = list(data_file.keys())
    known_station_delays = {sname:known_station_delays[sname] if sname in known_station_delays else 0.0 for sname in station_names}

    raw_fpaths = filePaths_by_stationName(timeID)
    stat_group = data_file[station]
    stat_data = MultiFile_Dal1(raw_fpaths[station], force_metadata_ant_pos=True)
    
    current_amp = 0.0
    for ant_name, ant_loc in zip(stat_data.get_antenna_names(),stat_data.get_LOFAR_centered_positions()):
        if ant_name not in stat_group:
            continue
        
        ant_dataset = stat_group[ant_name]
        even_trace = np.array( ant_dataset[0] )
        even_HE = np.array( ant_dataset[1] )
        odd_trace = np.array( ant_dataset[2] )
        odd_HE = np.array( ant_dataset[3] )
        trace_amp = max(np.max(even_HE), np.max(odd_HE))
        
        if trace_amp<min_antenna_amplitude:
            continue
        
        travel_delay = np.sqrt( (ant_loc[0]-source_XYZT[0])**2 + (ant_loc[1]-source_XYZT[1])**2 + (ant_loc[2]-source_XYZT[2])**2 )/v_air
        total_correction = travel_delay + source_XYZT[3] + known_station_delays[station]
        
        trace_time = np.arange(len(even_trace))*5.0E-9 + ant_dataset.attrs['starting_index']*5.0E-9  - total_correction
        even_trace_time = trace_time + ant_dataset.attrs['PolE_timeOffset']
        plt.plot(even_trace_time, even_trace + current_amp, 'g')
        plt.plot(even_trace_time, even_HE + current_amp, 'g')
        
        odd_trace_time = trace_time + ant_dataset.attrs['PolO_timeOffset']
        plt.plot(odd_trace_time, odd_trace + current_amp, 'm')
        plt.plot(odd_trace_time, odd_HE + current_amp, 'm')
        
        plt.annotate(ant_name, xy=(0.0, current_amp+0.5*trace_amp))
        
        current_amp += trace_amp*2
    plt.axvline(x=0)
    plt.show()
    