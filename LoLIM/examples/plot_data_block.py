#!/usr/bin/env python3

#### just plots a block of data on all antennas
#### usefull for health checks

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter



## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
# utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_raw_data_loc = "/home/brian/local_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20190424T194432.504Z"
    station = "CS002"
    point = int( 797*(2**16) )
    block_size = 2**16
    
    # bad_antennas = 'bad_antennas.txt'
    # polarization_flips = 'polarization_flips.txt'
    # additional_antenna_delays = 'ant_delays.txt'
    cal_file = '/OlafCal_202201311810.txt'
    
    
    positive_saturation = 2046
    negative_saturation = -2047
    saturation_post_removal_length = 50
    saturation_half_hann_length = 50
    
    processed_data_folder = processed_data_dir(timeID)

    # polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    # bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    # additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
    raw_fpaths = filePaths_by_stationName(timeID)
    TBB_data = MultiFile_Dal1( raw_fpaths[station],  total_cal= processed_data_folder+cal_file)
                              # polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays )
    # RFI_filter = window_and_filter(timeID=timeID, sname=station)
    
    #RFI_data = RFI_filter.RFI_data
    #print("num total channels 30-80 MHz:", np.sum(  np.logical_and(RFI_filter.FFT_frequencies>30e6, RFI_filter.FFT_frequencies<80e6 )) )
   # DC = RFI_filter.FFT_frequencies[RFI_data["dirty_channels"]]
   # print('num dirty channels 30-80 MHz:', np.sum(  np.logical_and(DC>30e6, DC<80e6 )) )
    
    RFI_filter = window_and_filter(blocksize=block_size)

    
    data = np.empty(block_size, dtype=np.double)
    
    ant_names = TBB_data.get_antenna_names()
    num_antenna_pairs = int( len( ant_names )/2 )
    t0 = np.arange(block_size)
    H = 0
    T = t0 + point
    for pair in range(num_antenna_pairs):
        data[:] = TBB_data.get_data(point, block_size, antenna_index=pair*2)
        remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
        filtered_data = RFI_filter.filter( data )
        even_HE = np.abs(filtered_data)
        even_real = np.real(filtered_data)
        
        #plt.plot(data)
        
        data[:] = TBB_data.get_data(point, block_size, antenna_index=pair*2+1)
        remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
        filtered_data = RFI_filter.filter( data )
        odd_HE = np.abs(filtered_data)
        
        peak = max( np.max(even_HE), np.max(odd_HE) )
#        peak = np.max(even_HE)
        
        print(ant_names[pair*2], ":", H )
        plt.plot(T, even_HE+H, 'r' )
        plt.plot(T, odd_HE+H, 'g' )
        H += peak
#        plt.plot(T, even_HE+H, 'C1', linewidth=3 )
#        plt.plot(T, even_real+H, 'C0', linewidth=3 )
#        H += 3*peak
            
    plt.show()
