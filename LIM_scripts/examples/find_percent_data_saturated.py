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
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    block_size = 2**16
    output_folder = "/find_percent_data_saturated"
    initial_block = 3400
    final_block = 4400
    
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
        
   
    for station, fpaths in raw_fpaths.items():
        print( station )
        
        TBB_data = MultiFile_Dal1( fpaths )
        
        for i,ant_name in enumerate(TBB_data.get_antenna_names()):
            num_data_points = 0
            num_saturated = 0
            for block in range(initial_block,final_block):
            
                data = TBB_data.get_data(block*block_size, block_size, antenna_index=i)
                
                num_saturated += np.sum( data>=positive_saturation )
                num_saturated += np.sum( data<=negative_saturation )
                num_data_points += block_size
                
            print(i, ant_name, 100.0*float(num_saturated)/num_data_points)
            
        print()
