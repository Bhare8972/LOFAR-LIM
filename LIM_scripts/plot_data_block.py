#!/usr/bin/env python3

#### just plots a block of data on all antennas
#### usefull for health checks

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter



if __name__ == "__main__":
    timeID = "D20180728T135703.246Z"
    station = "RS205"
    antenna_index = 1
    initial_point = int( 2500*(2**16) )
    block_size = 2**16
    
    
    positive_saturation = 2046
    negative_saturation = -2047
    saturation_post_removal_length = 50
    saturation_half_hann_length = 50
    
    
    
    raw_fpaths = filePaths_by_stationName(timeID)
    TBB_data = MultiFile_Dal1( raw_fpaths[station] )
    RFI_filter = window_and_filter(timeID=timeID, sname=station)
    
    data = np.empty(block_size, dtype=np.double)
    
    ant_names = TBB_data.get_antenna_names()
    num_antenna_pairs = int( len( ant_names )/2 )
    H = 0
    for pair in range(num_antenna_pairs):
        data[:] = TBB_data.get_data(initial_point, block_size, antenna_index=pair*2)
        remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
        filtered_data = RFI_filter.filter( data )
        even_HE = np.abs(filtered_data)
        
        data[:] = TBB_data.get_data(initial_point, block_size, antenna_index=pair*2+1)
        remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
        filtered_data = RFI_filter.filter( data )
        odd_HE = np.abs(filtered_data)
        
        peak = max( np.max(even_HE), np.max(odd_HE) )
        
        print(ant_names[pair*2], ":", H )
        
        plt.plot( even_HE+H, 'r' )
        plt.plot( odd_HE+H, 'g' )
        H += peak
        
    plt.show()