#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq
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



if __name__=="__main__":
    
    timeID = "D20180813T153001.413Z"
    output_folder = "correlate_foundPulses"
    
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    event_index = 1
    
    ## time delay, in seconds, used to align the peaks
    used_station_delays = {
"CS002": 0.0,
            
            "RS208":  6.97951240511e-06 , ##??
            "RS210": 0.0, ##??
            
'RS305' :  7.1934989871e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues
            
'CS001' :  2.24401172494e-06 , ## diff to guess: -1.97549250145e-12
'CS003' :  1.4070479021e-06 , ## diff to guess: -2.43902860006e-11
'CS004' :  4.40075444977e-07 , ## diff to guess: -2.42217738996e-11
'CS005' :  -2.17678756484e-07 , ## diff to guess: -2.21583592931e-11
'CS006' :  4.30647454293e-07 , ## diff to guess: -1.65542771957e-11
'CS007' :  3.99377465507e-07 , ## diff to guess: -2.02129216007e-11
'CS011' :  -5.8955385854e-07 , ## diff to guess: -1.23462674702e-12
'CS013' :  -1.81293364466e-06 , ## diff to guess: -2.14074126082e-11
'CS017' :  -8.4551582793e-06 , ## diff to guess: 4.31470974773e-11
'CS021' :  9.23663075135e-07 , ## diff to guess: 0.0
'CS024' :  2.33045304324e-06 , ## diff to guess: 8.82976926023e-12
'CS026' :  -9.26723147045e-06 , ## diff to guess: -9.26723147045e-06
'CS030' :  -2.73965173619e-06 , ## diff to guess: 2.41776414432e-11
'CS031' :  6.28836357272e-07 , ## diff to guess: 6.28836357272e-07
'CS032' :  -1.54223879178e-06 , ## diff to guess: -2.35580980099e-06
'CS101' :  -8.21535661203e-06 , ## diff to guess: 1.21635324342e-10
'CS103' :  -2.85878989548e-05 , ## diff to guess: 7.47105624697e-10
'CS201' :  -1.0510320435e-05 , ## diff to guess: 1.70364032269e-10
'CS301' :  -6.87236667019e-07 , ## diff to guess: -2.36495962731e-11
'CS302' :  -5.26832572532e-06 , ## diff to guess: 3.69871327386e-10
'CS501' :  -9.63031916796e-06 , ## diff to guess: -1.77630731779e-11
'RS106' :  6.78633242596e-06 , ## diff to guess: 1.575478568e-08
'RS205' :  7.09137015323e-06 , ## diff to guess: 9.84835165259e-10 
'RS306' :  7.41317947913e-06 , ## diff to guess: 1.6935786672e-08
'RS307' :  7.80951637749e-06 , ## diff to guess: 5.67451737617e-08
'RS310' :  7.96312229684e-06 , ## diff to guess: 4.82651956563e-07
'RS406' :  6.96541956339e-06 , ## diff to guess: 8.9825482633e-09
'RS407' :  6.83033534647e-06 , ## diff to guess: 3.5607628373e-09
'RS409' :  7.4346353712e-06 , ## diff to guess: 1.55561380515e-07
'RS503' :  6.9238247372e-06 , ## diff to guess: 2.51105167164e-10
'RS508' :  6.36331650134e-06 , ## diff to guess: 2.40127126525e-09
            }
    
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    guess_flash_location = [ 2.43390916381 , -50825.1364969 , 0.0 , 1.56124908739 ] ##ONLY USE THIS IF USED IN plot_multiple_data_all_stations.py!!!
    
    pulse_time = 1.1516532
    window_width = 7E-6 ## in s, centered on pulse_time
    pulse_width = 50 ## in data points
    block_size= 2**16
    min_ant_amp = 5.0
    
    skip_stations = ["RS208", "RS210"]
    
        
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
    print( used_station_delays)
    print()
    
        ##station data
    print()
    print("opening station data")
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
        
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname in used_station_delays.keys()}
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in used_station_delays.keys()}
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters, {sname:0.0 for sname in used_station_delays.keys()} )
    
    
    if guess_flash_location is not None:
        guess_flash_location = np.array(guess_flash_location)
        
        ref_stat_file = raw_data_files[ referance_station ]
        ant_loc = ref_stat_file.get_LOFAR_centered_positions()[0]
        ref_delay = np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_stat_file.get_nominal_sample_number()*5.0E-9
        
        for sname, data_file in raw_data_files.items():
            if sname not in used_station_delays:
                used_station_delays[sname] = 0.0
            
            data_file = raw_data_files[sname]
            ant_loc = data_file.get_LOFAR_centered_positions()[0]
            used_station_delays[sname] += (np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_delay) 
            used_station_delays[sname] -= data_file.get_nominal_sample_number()*5.0E-9
    
    print()
    print("used apparent delays:")
    print(used_station_delays)
    print()
    
    
    out_fname = data_dir + "/potSource_"+str(event_index)+".h5"
    out_file = h5py.File(out_fname, "w")
    
    half_pulse_width = int(0.5*pulse_width)
    window_width_samples = int(window_width/5.0E-9)
    for sname in used_station_delays.keys():
        if (not np.isfinite(used_station_delays[sname])) or sname in skip_stations:
            continue
        
        h5_statGroup = out_file.create_group( sname )
        
        antenna_names = raw_data_files[sname].get_antenna_names()
        
        data_arrival_index = int((pulse_time + used_station_delays[sname] - window_width*0.5)/5.0E-9)
        
        for even_ant_i in range(0, len(antenna_names), 2):
            throw, even_total_time_offset, throw, even_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i], window_width_samples)
#            plt.plot( np.abs(even_trace), 'r' )
            
            throw, odd_total_time_offset, throw, odd_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i+1], window_width_samples)
#            plt.plot( np.abs(odd_trace), 'g' )
            
            
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
                  
            even_trace = even_trace[center_index-half_pulse_width:center_index+half_pulse_width]
            odd_trace = odd_trace[center_index-half_pulse_width:center_index+half_pulse_width]
                
                
            h5_Ant_dataset = h5_statGroup.create_dataset(antenna_names[even_ant_i], (4, pulse_width), dtype=np.double)
            h5_Ant_dataset[0] = np.real( even_trace )
            h5_Ant_dataset[1] = np.abs( even_trace )
            h5_Ant_dataset[2] = np.real( odd_trace )
            h5_Ant_dataset[3] = np.abs( odd_trace )
            
            even_para_fit = parabolic_fit( np.abs(even_trace) )
            odd_para_fit = parabolic_fit( np.abs(odd_trace) )
                        
            starting_index = data_arrival_index+center_index-half_pulse_width
            h5_Ant_dataset.attrs['starting_index'] = starting_index
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -even_total_time_offset
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -odd_total_time_offset ## NOTE: due to historical reasons, there is a sign shift here
               
            if even_good:
                even_para_fit = parabolic_fit( np.abs(even_trace) )
                h5_Ant_dataset.attrs['PolE_peakTime'] = (even_para_fit.peak_index + starting_index)*5.0E-9 - even_total_time_offset
            else:
                h5_Ant_dataset.attrs['PolE_peakTime'] = np.nan
                 
            if odd_good:
                odd_para_fit = parabolic_fit( np.abs(odd_trace) )
                h5_Ant_dataset.attrs['PolO_peakTime'] = (odd_para_fit.peak_index + starting_index)*5.0E-9 - odd_total_time_offset
            else:
                h5_Ant_dataset.attrs['PolO_peakTime'] = np.nan
            
    out_file.close()
                