#!/usr/bin/env python3

from os import mkdir
from os.path import isdir, isfile
from itertools import chain

import numpy as np
import h5py
from scipy.signal import resample

from LoLIM.utilities import logger, processed_data_dir
from LoLIM.getTrace_fromLoc import getTrace_fromLoc    
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.signal_processing import parabolic_fit

#if __name__ == "__main__":
#    
#    timeID = "D20170929T202255.000Z"
#    
#    output_folder = "autoCorrelator3_fromLOCA"
#    num_points_width = 50
#    
#    XYZT = np.array([-15254.54018532 ,  8836.1191801,    3126.99536402, 1.17993760262 ] )
#    output_i = 10
#    
##    
##    station_delay_file = "station_delays.txt"
##    station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
#    
##    station_timing_offsets = {  
##"CS002":  0.0,
##"CS003":  1.40436380151e-06 ,
##"CS004":  4.31343360778e-07 ,
##"CS005":  -2.18883924536e-07 ,
##"CS006":  4.33532992523e-07 ,
##"CS007":  3.99644095007e-07 ,
##"CS011":  -5.85451477265e-07 ,
##"CS013":  -1.81434735154e-06 ,
##"CS017":  -8.4398374875e-06 ,
##"CS021":  9.23663075135e-07 ,
##"CS030":  -2.74255354078e-06,
##"CS032":  -1.57305580305e-06,
##"CS101":  -8.17154277682e-06,
##"CS103":  -2.85194082718e-05,
##"RS208":  6.97951240511e-06 ,
##"CS301":  -7.15482701536e-07 ,
##"CS302":  -5.35024064624e-06, 
##"RS306":  7.04283154727e-06,
##"RS307":  6.96315727897e-06 ,
##"RS310":  7.04140267551e-06,
##"CS401":  -9.5064990747e-07 ,
##"RS406":  6.96866309712e-06,
##"RS409":  7.02251772331e-06,
##"CS501":  -9.61256584076e-06 ,
##"RS503":  6.93934919654e-06 ,
##"RS508":  6.98208245779e-06 ,
##"RS509":  7.01900854365e-06,
##        }
#    
#    
#    polarization_flips = "polarization_flips.txt"
#    bad_antennas = "bad_antennas.txt"
#    additional_antenna_delays = "ant_delays.txt"
    
def save_EventByLoc(timeID, XYZT, station_timing_delays, pulse_index, output_folder, pulse_width=50, min_ant_amp=5, upsample_factor=0,
                    polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt"):
    
    processed_data_folder = processed_data_dir(timeID)
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
    data_dir = processed_data_folder + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log = logger()
    log.set(logging_folder + "/log_out_"+str(pulse_index)+"_.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    print("saving traces to file", pulse_index)
    print("location:", XYZT)
    print("upsample_factor:", upsample_factor)
    
    print()
    print()
    print("station timing offsets")
    print(station_timing_delays)
    
    print()
    print()
    print("pol flips")
    print(polarization_flips)
    
    print()
    print()
    print("bad antennas")
    print(bad_antennas)
    
    print()
    print()
    print("additional antenna delays")
    print(additional_antenna_delays)
    
    
    
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays,
                                           station_delay=station_timing_delays[sname]) for sname,fpaths in raw_fpaths.items() if sname in station_timing_delays}
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_timing_delays}
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters )
    
    print()
    print()
    print("data opened")
    
    out_fname = data_dir + "/potSource_"+str(pulse_index)+".h5"
    out_file = h5py.File(out_fname, "w")
    for sname in station_timing_delays.keys():
        TBB_file = raw_data_files[ sname ]
        TBB_file.find_and_set_polarization_delay()
        antenna_names = TBB_file.get_antenna_names()
        
        
        h5_statGroup = out_file.create_group( sname )
        print()
        print(sname)
        
        for even_ant_i in range(0,len(antenna_names),2):
            even_ant_name = antenna_names[ even_ant_i ]
            odd_ant_name = antenna_names[ even_ant_i+1 ]
            
            starting_index, PolE_offset, throw, PolE_trace = trace_locator.get_trace_fromLoc(XYZT, even_ant_name, pulse_width, do_remove_RFI=True, do_remove_saturation=True)
            throw, PolO_offset, throw, PolO_trace = trace_locator.get_trace_fromIndex(starting_index, odd_ant_name, pulse_width, do_remove_RFI=True, do_remove_saturation=True)
            
            PolE_HE = np.abs(PolE_trace)
            PolO_HE = np.abs(PolO_trace)
            
            
            h5_Ant_dataset = h5_statGroup.create_dataset(even_ant_name, (4, pulse_width ), dtype=np.double)
            
            h5_Ant_dataset[0] = np.real(PolE_trace)
            h5_Ant_dataset[1] = PolE_HE
            h5_Ant_dataset[2] = np.real(PolO_trace)
            h5_Ant_dataset[3] = PolO_HE
            
            ### note that peak time should NOT account for station timing offsets!
            
            h5_Ant_dataset.attrs['starting_index'] = starting_index
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -(PolE_offset - station_timing_delays[sname])
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -(PolO_offset - station_timing_delays[sname])
            h5_Ant_dataset.attrs['PolE_timeOffset_CS'] = (PolE_offset - station_timing_delays[sname])
            h5_Ant_dataset.attrs['PolO_timeOffset_CS'] = (PolO_offset - station_timing_delays[sname])
            
            
            
            
            sample_time = 5.0E-9
            if upsample_factor > 1:
                PolE_HE = resample(PolE_HE, len(PolE_HE)*upsample_factor )
                PolO_HE = resample(PolO_HE, len(PolO_HE)*upsample_factor )
                
                sample_time /= upsample_factor
            
            
            if np.max(PolE_HE)> min_ant_amp:
                
                
                PolE_peak_finder = parabolic_fit( PolE_HE )
                h5_Ant_dataset.attrs['PolE_peakTime'] =  starting_index*5.0E-9 - (PolE_offset-station_timing_delays[sname]) + PolE_peak_finder.peak_index*sample_time
            else:
                h5_Ant_dataset.attrs['PolE_peakTime'] = np.nan
            
            if np.max(PolO_HE)> min_ant_amp:
                
                
                PolO_peak_finder = parabolic_fit( PolO_HE )
                h5_Ant_dataset.attrs['PolO_peakTime'] =  starting_index*5.0E-9 - (PolO_offset-station_timing_delays[sname]) + PolO_peak_finder.peak_index*sample_time
            else:
                h5_Ant_dataset.attrs['PolO_peakTime'] =  np.nan
            
            print("  ", even_ant_name, h5_Ant_dataset.attrs['PolE_peakTime'], h5_Ant_dataset.attrs['PolO_peakTime'])
                
            
    out_file.close()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    