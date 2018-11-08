#!/usr/bin/env python3

#### just plots a block of data on all antennas
#### usefull for health checks

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from matplotlib.transforms import blended_transform_factory

from scipy.signal import hilbert

def plot_blocks(timeID, block_size, block_starts, guess_delays, guess_location = None, bad_stations=[], polarization_flips="polarization_flips.txt", 
                bad_antennas = "bad_antennas.txt", additional_antenna_delays = "ant_delays.txt", do_remove_saturation = True, do_remove_RFI = True, 
                positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 5, 
                referance_station = "CS002"):
    """plot multiple blocks, for guessing initial delays and finding pulses. If guess_location is None, then guess_delays should be apparent delays,
    if guess_location is a XYZT location, then guess_delays should be real delays. If a station isn't in guess_delays, its' delay is assumed to be zero.
    A station is only not plotted if it is bad_stations. If referance station is in guess_delays, then its delay is subtract from all stations"""
    
    if referance_station in guess_delays:
        ref_delay = guess_delays[referance_station]
        guess_delays = {sname:delay-ref_delay for sname,delay in guess_delays.items()}
    
    processed_data_folder = processed_data_dir(timeID)
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
        
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname in raw_fpaths.keys() if sname not in bad_stations}
    
    if guess_location is not None:
        guess_location = np.array(guess_location)
        
        ref_stat_file = raw_data_files[ referance_station ]
        ant_loc = ref_stat_file.get_LOFAR_centered_positions()[0]
        ref_delay = np.linalg.norm(ant_loc-guess_location[:3])/v_air - ref_stat_file.get_nominal_sample_number()*5.0E-9
        
        for sname, data_file in raw_data_files.items():
            if sname not in guess_delays:
                guess_delays[sname] = 0.0
            
            data_file = raw_data_files[sname]
            ant_loc = data_file.get_LOFAR_centered_positions()[0]
            guess_delays[sname] += (np.linalg.norm(ant_loc-guess_location[:3])/v_air - ref_delay) 
            guess_delays[sname] -= data_file.get_nominal_sample_number()*5.0E-9
    
    RFI_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in raw_fpaths.keys() if sname not in bad_stations}
    
    data = np.empty(block_size, dtype=np.double)
    
    height = 0
    t0 = np.arange(block_size)*5.0E-9
    transform = blended_transform_factory(plt.gca().transAxes, plt.gca().transData)
    sname_X_loc = 0.0
    for sname, data_file in raw_data_files.items():
        print(sname)
        
        station_delay = 0.0
        if sname in guess_delays:
            station_delay = guess_delays[sname] 
        
        station_delay_points = int(station_delay/5.0E-9)
            
        RFI_filter = RFI_filters[sname]
        ant_names = data_file.get_antenna_names()
        
        num_antenna_pairs = int( len( ant_names )/2 )
        peak_height = 0.0
        for point in block_starts:
            T = t0 + point*5.0E-9
            for pair in range(num_antenna_pairs):
                data[:] = data_file.get_data(point+station_delay_points, block_size, antenna_index=pair*2)
                
                if do_remove_saturation:
                    remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
                if do_remove_RFI:
                    filtered_data = RFI_filter.filter( data )
                else:
                    filtered_data = hilbert(data)
                even_HE = np.abs(filtered_data)
                
                data[:] = data_file.get_data(point+station_delay_points, block_size, antenna_index=pair*2+1)
                if do_remove_saturation:
                    remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
                if do_remove_RFI:
                    filtered_data = RFI_filter.filter( data )
                else:
                    filtered_data = hilbert(data)
                odd_HE = np.abs(filtered_data)
                
                plt.plot(T, even_HE + height, 'r')
                plt.plot(T, odd_HE + height, 'g' )
                
                max_even = np.max(even_HE)
                if max_even > peak_height:
                    peak_height = max_even
                max_odd = np.max(odd_HE)
                if max_odd > peak_height:
                    peak_height = max_odd
                    
#        plt.annotate(sname, (points[-1]*5.0E-9+t0[-1], height))
        plt.annotate(sname, (sname_X_loc, height), textcoords=transform, xycoords=transform)
        height += 2*peak_height
            
    plt.show()