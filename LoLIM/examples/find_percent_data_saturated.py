#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir, logger
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import num_double_zeros

from os import mkdir
from os.path import isdir


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20200814T122532.788Z"
    block_size = 2**16
    output_folder = "/find_percent_data_saturated"
    initial_block = 2500
    final_block = 3500
    
    is_flash_thrash = 250
    
    positive_saturation = 2046
    negative_saturation = -2047
    
    raw_fpaths = filePaths_by_stationName(timeID)
 
    processed_data_dir = processed_data_dir(timeID)
    
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    log = logger()
    log.set(output_fpath+'/log.txt', True)
    log.take_stdout()
        
    total_saturated = 0
    total_flash_points = 0
    total_points = 0
   
    for station, fpaths in raw_fpaths.items():
        print( station )
        
        TBB_data = MultiFile_Dal1( fpaths )
        
        
        
        for i,ant_name in enumerate(TBB_data.get_antenna_names()):
            num_flash_points = 0
            num_saturated = 0
            for block in range(initial_block,final_block):
            
                data = TBB_data.get_data(block*block_size, block_size, antenna_index=i)
                
                saturated = np.sum( data>=positive_saturation ) + np.sum( data<=negative_saturation )
                in_flash = np.sum( data>=is_flash_thrash ) + np.sum( data<= -is_flash_thrash )
                
                num_saturated += saturated
                num_flash_points += in_flash
                
                total_saturated += saturated
                total_flash_points += in_flash
                total_points += block_size
                
            print(i, ant_name, 100.0*float(num_saturated)/num_flash_points)
            
    print()
    print('totals:')
    print('staturated:', total_saturated, "in flash:", total_flash_points)
    print('percent flash saturated:', 100.0*float(total_saturated)/total_flash_points )
    print('percent is flash:', 100.0*float(total_flash_points)/total_points )
        
        
