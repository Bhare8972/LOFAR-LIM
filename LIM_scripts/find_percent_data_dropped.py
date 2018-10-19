#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import num_double_zeros


if __name__ == "__main__":
    timeID = "D20180809T141413.250Z"
    block_size = 2**16
    
    initial_block = 5000
    final_block = 6000
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for station, fpaths in raw_fpaths.items():
        print( station )
        
        TBB_data = MultiFile_Dal1( fpaths )
        
        for i,ant_name in enumerate(TBB_data.get_antenna_names()):
            num_data_points = 0
            num_zeros = 0
            for block in range(initial_block,final_block):
            
                data = TBB_data.get_data(block*block_size, block_size, antenna_index=i)
                
                num_zeros += num_double_zeros( data )
                num_data_points += block_size
                
            print(i, ant_name, 100.0*float(num_zeros)/num_data_points)
            
        print()