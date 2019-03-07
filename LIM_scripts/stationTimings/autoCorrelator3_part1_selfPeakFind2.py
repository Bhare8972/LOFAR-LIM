#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq
from scipy.signal import resample
from matplotlib import pyplot as plt

import h5py

##my packages
from LoLIM.utilities import log, processed_data_dir, v_air
from LoLIM.porta_code import code_logger
from LoLIM.IO.binary_IO import write_long, write_double_array, write_string, write_double
from LoLIM.getTrace_fromLoc import getTrace_fromLoc
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.signal_processing import parabolic_fit

TODO: dble check this file for correctness. Correct for polarization offsets. save correct time to pulse files
    
def save_Pulse(timeID, output_folder, event_index, pulse_time, station_delays, skip_stations=[], window_width=7E-6, pulse_width=50, block_size=2**16, 
               min_ant_amp=5.0, guess_flash_location=None, referance_station="CS002", polarization_flips="polarization_flips.txt", 
               bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt"):
    """Desined to be used in conjuction with plot_multiple_data_all_stations.py. Given a pulse_time, and window_width, it saves a pulse on every antenna that is 
    centered on the largest peak in the window, with a width of pulse_width. station_delays doesn't need to be accurate, just should be the same as used in plot_multiple_data_all_stations.
    Same for referance_station, and guess_flash_location. Note that this will save the pulse under an index, and will overwrite any other pulse saved under that index in the output folder"""
    
    
    if referance_station in station_delays:
        ref_delay = station_delays[referance_station]
        station_delays = {sname:delay-ref_delay for sname,delay in station_delays.items()}
    
    #### setup directory variables ####
    processed_data_folder = processed_data_dir(timeID)
    
    data_dir = processed_data_folder + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_event_"+str(event_index)+".txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    print("pulse time:", pulse_time)
    print("window_width:", window_width)
    print("pulse_width:", pulse_width)
    print("skip stations:", skip_stations)
    print("guess_flash_location:", guess_flash_location)
    print("input delays:")
    print( station_delays)
    print()
    
        ##station data
    print()
    print("opening station data")
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
        
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname in station_delays.keys()}
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_delays.keys()}
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters, {sname:0.0 for sname in station_delays.keys()} )
    
    
    if guess_flash_location is not None:
        guess_flash_location = np.array(guess_flash_location)
        
        ref_stat_file = raw_data_files[ referance_station ]
        ant_loc = ref_stat_file.get_LOFAR_centered_positions()[0]
        ref_delay = np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_stat_file.get_nominal_sample_number()*5.0E-9
        
        for sname, data_file in raw_data_files.items():
            if sname not in station_delays:
                station_delays[sname] = 0.0
            
            data_file = raw_data_files[sname]
            ant_loc = data_file.get_LOFAR_centered_positions()[0]
            station_delays[sname] += (np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_delay) 
            station_delays[sname] -= data_file.get_nominal_sample_number()*5.0E-9
    
    print()
    print("used apparent delays:")
    print(station_delays)
    print()
    
    
    out_fname = data_dir + "/potSource_"+str(event_index)+".h5"
    out_file = h5py.File(out_fname, "w")
    
    half_pulse_width = int(0.5*pulse_width)
    window_width_samples = int(window_width/5.0E-9)
    for sname in station_delays.keys():
        if (not np.isfinite(station_delays[sname])) or sname in skip_stations:
            continue
        
        h5_statGroup = out_file.create_group( sname )
        
        antenna_names = raw_data_files[sname].get_antenna_names()
        
        data_arrival_index = int((pulse_time + station_delays[sname] - window_width*0.5)/5.0E-9)
        
        print()
        print(sname)
        
        for even_ant_i in range(0, len(antenna_names), 2):
            
            #### get window and find peak ####
            throw, even_total_time_offset, throw, even_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i], window_width_samples)
            throw, odd_total_time_offset, throw, odd_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i+1], window_width_samples)

            even_HE = np.abs(even_trace)
            even_good = False
            if np.max( even_HE ) > min_ant_amp:
                even_good = True
                even_para_fit = parabolic_fit( even_HE )
            
            odd_HE = np.abs(odd_trace)
            odd_good = False
            if np.max( odd_HE ) > min_ant_amp:
                odd_good = True
                odd_para_fit = parabolic_fit( odd_HE )
                
            if (not even_good) and (not odd_good):
                continue
            elif even_good and (not odd_good):
                center_index = int(even_para_fit.peak_index)
            elif (not even_good) and odd_good:
                center_index = int(odd_para_fit.peak_index)
            else: ## both good
                even_amp = even_HE[ int(even_para_fit.peak_index) ]
                odd_amp = odd_HE[ int(odd_para_fit.peak_index) ]
                if even_amp > odd_amp:
                    center_index = int(even_para_fit.peak_index)
                else:
                    center_index = int(odd_para_fit.peak_index)

            
            #### now get data centered on peak. re-get data, as peak may be near edge ####
            throw, even_total_time_offset, throw, even_trace = trace_locator.get_trace_fromIndex( data_arrival_index+center_index-half_pulse_width, antenna_names[even_ant_i], half_pulse_width*2)
            throw, odd_total_time_offset , throw, odd_trace  = trace_locator.get_trace_fromIndex( data_arrival_index+center_index-half_pulse_width, antenna_names[even_ant_i+1], half_pulse_width*2)
                  
#            even_trace = even_trace[center_index-half_pulse_width:center_index+half_pulse_width]
#            odd_trace = odd_trace[center_index-half_pulse_width:center_index+half_pulse_width]
            even_HE = np.abs(even_trace)
            odd_HE = np.abs(odd_trace)
                
            h5_Ant_dataset = h5_statGroup.create_dataset(antenna_names[even_ant_i], (4, half_pulse_width*2), dtype=np.double)
            h5_Ant_dataset[0] = np.real( even_trace )
            h5_Ant_dataset[1] = even_HE
            h5_Ant_dataset[2] = np.real( odd_trace )
            h5_Ant_dataset[3] = odd_HE
                        
            WARNING: not sure this is correct. even_total_time_offset includes station delay
            starting_index = data_arrival_index+center_index-half_pulse_width
            h5_Ant_dataset.attrs['starting_index'] = starting_index
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -even_total_time_offset ## NOTE: due to historical reasons, there is a sign flip here
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -odd_total_time_offset ## NOTE: due to historical reasons, there is a sign flip here
               
            if np.max(even_HE) > min_ant_amp:
#                even_HE = resample(even_HE, len(even_HE)*8 )
#                even_para_fit = parabolic_fit( even_HE )
#                h5_Ant_dataset.attrs['PolE_peakTime'] = (even_para_fit.peak_index/8.0 + starting_index)*5.0E-9 - even_total_time_offset
                even_para_fit = parabolic_fit( even_HE )
                h5_Ant_dataset.attrs['PolE_peakTime'] = (even_para_fit.peak_index + starting_index)*5.0E-9 - even_total_time_offset
            else:
                h5_Ant_dataset.attrs['PolE_peakTime'] = np.nan
                 
            if np.max(odd_HE) > min_ant_amp:
#                odd_HE = resample(odd_HE, len(odd_HE)*8 )
#                odd_para_fit = parabolic_fit( odd_HE )
#                h5_Ant_dataset.attrs['PolO_peakTime'] = (odd_para_fit.peak_index/8.0 + starting_index)*5.0E-9 - odd_total_time_offset
                odd_para_fit = parabolic_fit( odd_HE )
                h5_Ant_dataset.attrs['PolO_peakTime'] = (odd_para_fit.peak_index + starting_index)*5.0E-9 - odd_total_time_offset
            else:
                h5_Ant_dataset.attrs['PolO_peakTime'] = np.nan
                
            
            print("   ", antenna_names[even_ant_i], (data_arrival_index+center_index)*5.0E-9 - station_delays[sname], h5_Ant_dataset.attrs['PolE_peakTime'], h5_Ant_dataset.attrs['PolO_peakTime'] )
            
    out_file.close()
                