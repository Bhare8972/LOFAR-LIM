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
    
    event_index = 0
    
    #approx time of peak, from beginning of file
    station_peak_times = {
            "CS001":1.14841985,
            "CS002":1.1484190,
            "CS003":1.14841935,
            "CS004":1.1484187,
            "CS005":1.1484186,
            "CS006":1.1484240,
            "CS007":1.14842425,
            "CS011":1.1484236,
            "CS013":1.14842515,
            "CS017":1.1484192,
            "CS021":np.nan,
            "CS024":1.1484229,
            "CS026":1.14841525,
            "CS030":1.14842675,
            "CS031":1.1484257,
            "CS032":1.1484232,
            "CS101":1.1484268, 
            "CS103":1.1484244,
            "CS201":1.1484126,
            "CS301":1.1484207,
            "CS302":1.1484193, 
            "CS501":1.1484186,
            "RS106":1.14843375,
            "RS205":1.1484329,
            
            "RS208":np.nan, ##??
            "RS210":np.nan, ##??
            "RS305":1.1484254,
            "RS306":1.1484491,
            "RS307":1.1484287,
            "RS310":1.14843255,
            "RS406":1.1484324,
            "RS407":1.1484376,
            "RS409":1.14843045,
            "RS503":1.1484305,
            "RS508":1.1484269, 
            
            }
    
    ## time delay, in seconds, used to align the peaks
    used_station_delays = {
            "CS001": 0.0,
            "CS002":  0.0,
            "CS003":  1.40436380151e-06 ,
            "CS004":  4.31343360778e-07 ,
            "CS005":  -2.18883924536e-07 ,
            "CS006":  4.33532992523e-07 ,
            "CS007":  3.99644095007e-07 ,
            "CS011":  -5.85451477265e-07 ,
            "CS013":  -1.81434735154e-06 ,
            "CS017":  -8.4398374875e-06 ,
            "CS021": 9.23663075135e-07 ,
            "CS024": 9.23663075135e-07 ,
            "CS026": 0.0,
            "CS030":  -2.74255354078e-06,
            "CS031":    0.0,
            "CS032":  -1.57305580305e-06,
            "CS101":  -8.17154277682e-06,
            "CS103":  -2.85194082718e-05,
            "CS201":  0.0,
            "RS106":  -1.14843+1.14824,
            "RS205":  -1.14843+1.14832,
            "RS208":  6.97951240511e-06 , ##??
            "RS210": 0.0, ##??
            "CS301":  -7.15482701536e-07 ,
            "CS302":  -5.35024064624e-06 ,
            "CS501":  0.0,
            "RS305":  -1.14843+1.14815,
            "RS306":  7.04283154727e-06-1.14843+1.14782,
            "RS307":  6.96315727897e-06-1.14843+1.14811 ,
            "RS310":  7.04140267551e-06-1.14843+1.14811,
            "RS406":  6.96866309712e-06-1.14843+1.14839 -1.14843+1.14822,
            "RS407":  -1.14843+1.14849,
            "RS409":  7.02251772331e-06-1.14843+1.14848, 
            "RS503":  6.93934919654e-06-1.14843+1.14802,
            "RS508":  6.98208245779e-06-1.14843+1.148,
            }
    
    pulse_width = 100 ## in data points
    block_size= 2**16
    
        
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
    
    
    print("used times:")
    print(station_peak_times)
    print()
    print()
    print("used apparent delays:")
    print(used_station_delays)
    print()
    
        ##station data
    print()
    print("opening station data")
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
        
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname in station_peak_times.keys()}
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_peak_times.keys()}
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters, {sname:0.0 for sname in station_peak_times.keys()} )
    
    out_fname = data_dir + "/potSource_"+str(event_index)+".h5"
    out_file = h5py.File(out_fname, "w")
    
    for sname in station_peak_times.keys():
        h5_statGroup = out_file.create_group( sname )
        
        antenna_names = raw_data_files[sname].get_antenna_names()
        
        if not np.isfinite(station_peak_times[sname]):
            continue
        
        data_arrival_index = int((station_peak_times[sname] + used_station_delays[sname])/5.0E-9) - int(pulse_width*0.5)
        
        for even_ant_i in range(0, len(antenna_names), 2):
            throw, even_total_time_offset, throw, even_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i], pulse_width)
#            plt.plot( np.abs(even_trace), 'r' )
            
            throw, odd_total_time_offset, throw, odd_trace = trace_locator.get_trace_fromIndex( data_arrival_index, antenna_names[even_ant_i+1], pulse_width)
#            plt.plot( np.abs(odd_trace), 'g' )
            
            even_para_fit = parabolic_fit( np.abs(even_trace) )
            odd_para_fit = parabolic_fit( np.abs(odd_trace) )
            
            
            h5_Ant_dataset = h5_statGroup.create_dataset(antenna_names[even_ant_i], (4, pulse_width), dtype=np.double)
            h5_Ant_dataset[0] = np.real( even_trace )
            h5_Ant_dataset[1] = np.abs( even_trace )
            h5_Ant_dataset[2] = np.real( odd_trace )
            h5_Ant_dataset[3] = np.abs( odd_trace )
                        
            h5_Ant_dataset.attrs['starting_index'] = data_arrival_index
            h5_Ant_dataset.attrs['PolE_peakTime'] =  (even_para_fit.peak_index + data_arrival_index)*5.0E-9 - even_total_time_offset
            h5_Ant_dataset.attrs['PolO_peakTime'] =  (odd_para_fit.peak_index + data_arrival_index)*5.0E-9 - odd_total_time_offset
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -even_total_time_offset
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -odd_total_time_offset ## NOTE: due to historical reasons, there is a sign shift here
               
    out_file.close()
                