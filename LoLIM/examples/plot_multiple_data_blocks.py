#!/usr/bin/env python3

#### just plots a block of data on all antennas
#### usefull for health checks

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays

from scipy.signal import hilbert



if __name__ == "__main__":
    timeID = "D20180813T153001.413Z"
    station = "CS002"
    points = [ int(i*(2**16)) for i in range(2292,2302) ]
    block_size = 2**16
    polarization = 0
    
    do_remove_saturation = True
    do_remove_RFI = True
    
    positive_saturation = 2046
    negative_saturation = -2047
    saturation_post_removal_length = 50
    saturation_half_hann_length = 50
    
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    processed_data_folder = processed_data_dir(timeID)
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
    raw_fpaths = filePaths_by_stationName(timeID)
    TBB_data = MultiFile_Dal1( raw_fpaths[station], polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays )
    RFI_filter = window_and_filter(timeID=timeID, sname=station)
    
    data = np.empty(block_size, dtype=np.double)
    
    ant_names = TBB_data.get_antenna_names()
    num_antenna_pairs = int( len( ant_names )/2 )
    t0 = np.arange(block_size)*5.0E-9
    for point in points:
        T = t0 + point*5.0E-9
        for pair in range(num_antenna_pairs):
            data[:] = TBB_data.get_data(point, block_size, antenna_index=pair*2)
            
            if do_remove_saturation:
                remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
            if do_remove_RFI:
                filtered_data = RFI_filter.filter( data )
            else:
                filtered_data = hilbert(data)
            even_HE = np.abs(filtered_data)
            
            data[:] = TBB_data.get_data(point, block_size, antenna_index=pair*2+1)
            if do_remove_saturation:
                remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
            if do_remove_RFI:
                filtered_data = RFI_filter.filter( data )
            else:
                filtered_data = hilbert(data)
            odd_HE = np.abs(filtered_data)
            
#            peak = max( np.max(even_HE), np.max(odd_HE) )
            
            if polarization == 0:
                plt.plot(T, even_HE)
            else:
                plt.plot(T, odd_HE )
            
    plt.show()